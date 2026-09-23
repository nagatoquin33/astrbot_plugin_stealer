export function createSourceActions({
    apiFetch,
    applySourceInspection,
    resetSourceCategoryMap,
    fetchSources,
    refreshView,
    showConfirm,
    t,
    sourceLoading,
    sourceError,
    sourceFile,
    sourceUploadedPath,
    sourceSelected,
    sourceInspection,
    sourceCategoryMap,
    sourceForm,
    sourceDefaults,
    sourceJob,
}) {
    let pollInterval = null;

    const stopPolling = () => {
        if (pollInterval) clearInterval(pollInterval);
        pollInterval = null;
    };

    const handleSourceFile = async (event) => {
        const file = event?.target?.files?.[0] || null;
        sourceFile.value = file;
        sourceUploadedPath.value = '';
        sourceSelected.value = null;
        sourceInspection.value = null;
        sourceForm.endpoint = '';
        sourceForm.github = '';
        if (!file) return;
        sourceLoading.value = true;
        sourceError.value = '';
        try {
            const form = new FormData();
            form.append('file', file);
            const res = await apiFetch('api/sources/upload', { method: 'POST', body: form });
            const data = await res.json();
            if (!data?.success) throw new Error(data?.error || 'Pack upload failed');
            sourceUploadedPath.value = data.path || '';
            applySourceInspection(data.inspection);
        } catch (error) {
            sourceError.value = error.message || String(error);
        } finally {
            sourceLoading.value = false;
        }
    };

    const inspectExternalApi = async () => {
        const endpoint = String(sourceForm.endpoint || '').trim();
        if (!endpoint) return;
        sourceLoading.value = true;
        sourceError.value = '';
        sourceSelected.value = null;
        sourceUploadedPath.value = '';
        sourceForm.github = '';
        try {
            const res = await apiFetch('api/sources/inspect', {
                method: 'POST',
                body: JSON.stringify({ source_type: 'http_json', endpoint }),
            });
            const data = await res.json();
            if (!data?.success) throw new Error(data?.error || 'API preflight failed');
            applySourceInspection(data.inspection);
        } catch (error) {
            sourceError.value = error.message || String(error);
        } finally {
            sourceLoading.value = false;
        }
    };

    const inspectGitHubSource = async () => {
        const repository = String(sourceForm.github || '').trim();
        if (!repository) return;
        sourceLoading.value = true;
        sourceError.value = '';
        sourceSelected.value = null;
        sourceUploadedPath.value = '';
        sourceForm.endpoint = '';
        try {
            const res = await apiFetch('api/sources/inspect', {
                method: 'POST',
                body: JSON.stringify({ source_type: 'github', repository }),
            });
            const data = await res.json();
            if (!data?.success) throw new Error(data?.error || 'GitHub preflight failed');
            applySourceInspection(data.inspection);
        } catch (error) {
            sourceError.value = error.message || String(error);
        } finally {
            sourceLoading.value = false;
        }
    };

    const sourcePayloadFor = (source = sourceSelected.value) => {
        let payload;
        if (sourceUploadedPath.value) {
            payload = { source_type: 'meme_pack', path: sourceUploadedPath.value };
        } else if (source) {
            payload = source.discovered
                ? { source_type: source.source_type, path: source.endpoint }
                : { source_id: source.source_id };
        } else {
            const github = String(sourceForm.github || '').trim();
            payload = github
                ? { source_type: 'github', repository: github }
                : {
                    source_type: 'http_json',
                    endpoint: String(sourceForm.endpoint || '').trim(),
                };
        }
        const mapping = {};
        for (const [key, value] of Object.entries(sourceCategoryMap)) {
            if (value) mapping[key] = value;
        }
        return {
            ...payload,
            category_map: mapping,
            review: !!sourceForm.review,
            scope_mode: sourceForm.scope_mode,
            origin_target: String(sourceForm.origin_target || '').trim(),
            character: sourceForm.assign_character
                ? String(sourceForm.character || '').trim()
                : '',
            create_character: !!sourceForm.assign_character,
        };
    };

    const inspectSource = async (source) => {
        sourceLoading.value = true;
        sourceError.value = '';
        sourceUploadedPath.value = '';
        sourceFile.value = null;
        try {
            const payload = source.discovered
                ? { source_type: source.source_type, path: source.endpoint }
                : { source_id: source.source_id };
            const res = await apiFetch('api/sources/inspect', {
                method: 'POST',
                body: JSON.stringify(payload),
            });
            const data = await res.json();
            if (!data?.success) throw new Error(data?.error || 'Source preflight failed');
            applySourceInspection(data.inspection, source);
        } catch (error) {
            sourceError.value = error.message || String(error);
        } finally {
            sourceLoading.value = false;
        }
    };

    const pollSourceJob = async () => {
        const jobId = sourceJob.value?.job_id;
        if (!jobId) return;
        try {
            const res = await apiFetch(`api/sources/jobs?job_id=${encodeURIComponent(jobId)}`);
            const data = await res.json();
            if (!data?.success || !data.job) return;
            sourceJob.value = data.job;
            if (['completed', 'failed', 'cancelled'].includes(data.job.status)) {
                stopPolling();
                await fetchSources();
                await refreshView();
            }
        } catch (error) {
            sourceError.value = error.message || String(error);
        }
    };

    const startSourceImport = async (source = sourceSelected.value) => {
        sourceLoading.value = true;
        sourceError.value = '';
        try {
            const payload = sourcePayloadFor(source);
            if (!payload.source_id && !payload.path && !payload.endpoint && !payload.repository) {
                throw new Error('Choose a pack, GitHub repository, or enter an API URL first');
            }
            const res = await apiFetch('api/sources/import', {
                method: 'POST',
                body: JSON.stringify(payload),
            });
            const data = await res.json();
            if (!data?.success || !data.job) throw new Error(data?.error || 'Import failed to start');
            sourceJob.value = data.job;
            stopPolling();
            pollInterval = setInterval(pollSourceJob, 1000);
            await pollSourceJob();
        } catch (error) {
            sourceError.value = error.message || String(error);
        } finally {
            sourceLoading.value = false;
        }
    };

    const syncSource = async (source) => {
        sourceSelected.value = source;
        sourceUploadedPath.value = '';
        if (source?.discovered) {
            resetSourceCategoryMap();
            sourceForm.review = !!sourceDefaults.value.review;
            sourceForm.scope_mode = 'public';
            sourceForm.origin_target = '';
            sourceForm.assign_character = false;
            sourceForm.character = '';
            await startSourceImport(source);
            return;
        }
        sourceLoading.value = true;
        sourceError.value = '';
        try {
            const res = await apiFetch('api/sources/sync', {
                method: 'POST',
                body: JSON.stringify({ source_id: source?.source_id || '' }),
            });
            const data = await res.json();
            if (!data?.success || !data.job) throw new Error(data?.error || 'Sync failed to start');
            sourceJob.value = data.job;
            stopPolling();
            pollInterval = setInterval(pollSourceJob, 1000);
            await pollSourceJob();
        } catch (error) {
            sourceError.value = error.message || String(error);
        } finally {
            sourceLoading.value = false;
        }
    };

    const cancelSourceJob = async () => {
        if (!sourceJob.value?.job_id) return;
        await apiFetch('api/sources/jobs/cancel', {
            method: 'POST',
            body: JSON.stringify({ job_id: sourceJob.value.job_id }),
        });
        await pollSourceJob();
    };

    const forgetSource = async (source) => {
        if (!source?.source_id || source.discovered) return;
        const message = t(
            'pages.dashboard.sources.forget_confirm',
            'Forget this source? Imported images will remain in your library.'
        );
        if (!await showConfirm(message)) return;
        const res = await apiFetch('api/sources/delete', {
            method: 'POST',
            body: JSON.stringify({ source_id: source.source_id }),
        });
        const data = await res.json();
        if (data?.success) await fetchSources();
        else sourceError.value = data?.error || 'Failed to forget source';
    };

    return {
        handleSourceFile,
        inspectExternalApi,
        inspectGitHubSource,
        sourcePayloadFor,
        inspectSource,
        pollSourceJob,
        startSourceImport,
        syncSource,
        cancelSourceJob,
        forgetSource,
        stopPolling,
    };
}
