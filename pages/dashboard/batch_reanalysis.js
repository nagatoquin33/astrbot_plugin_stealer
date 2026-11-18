const TOKEN_WARNING_THRESHOLD = 20;

export function createBatchReanalysis({
    ref, computed, apiFetch, selectedImages, showConfirm, showAlert, refreshView, loadThumbnail, t,
    getMissingScope,
}) {
    const job = ref(null);
    const open = ref(false);
    const error = ref('');
    const submitting = ref(false);
    const busy = computed(() => ['running', 'stopping'].includes(job.value?.status));
    let timer = null;
    let stopped = false;
    let polling = null;
    const reviewHash = ref('');
    const reviewResult = ref(null);
    const reviewImage = ref('');
    const reviewLoading = ref(false);
    const readyResults = computed(() => (job.value?.results || []).filter((item) => item.status === 'ready'));
    let reviewSequence = 0;

    const schedulePoll = (delay) => {
        clearTimeout(timer);
        if (!stopped) timer = setTimeout(poll, delay);
    };

    const requestData = async (path, options) => {
        const response = await apiFetch(path, options);
        const data = await response.json();
        if (!data?.success) {
            if (data?.task) job.value = data.task;
            throw new Error(data?.error || t('pages.dashboard.reanalysis.request_failed', 'Request failed.'));
        }
        return data;
    };

    const request = async (path, options) => (await requestData(path, options)).task || null;

    const selectResult = async (hash) => {
        if (stopped) return;
        const sequence = ++reviewSequence;
        reviewHash.value = hash;
        reviewResult.value = null;
        reviewImage.value = '';
        reviewLoading.value = Boolean(hash);
        if (!hash) return;
        try {
            const query = new URLSearchParams({ task_id: job.value.id, hash });
            const data = await requestData('api/images/reanalyze/result?' + query);
            if (stopped || sequence !== reviewSequence) return;
            reviewResult.value = data.result;
            // 图片加载失败不影响查看文字结果。
            const image = await loadThumbnail(hash).catch(() => null);
            if (!stopped && sequence === reviewSequence) reviewImage.value = image?.url || '';
        } catch (err) {
            if (!stopped && sequence === reviewSequence) error.value = err.message;
        } finally {
            if (sequence === reviewSequence) reviewLoading.value = false;
        }
    };

    const syncReview = async () => {
        if (stopped) return;
        if (!readyResults.value.some((item) => item.hash === reviewHash.value)) {
            await selectResult(readyResults.value[0]?.hash || '');
        }
    };

    const poll = () => {
        if (stopped) return Promise.resolve();
        if (polling) return polling;
        clearTimeout(timer);
        polling = (async () => {
            try {
                const previous = job.value;
                const current = await request('api/images/reanalyze/status');
                if (stopped) return;
                job.value = current;
                error.value = previous && !current
                    ? t('pages.dashboard.reanalysis.interrupted', 'The plugin restarted; this batch is no longer available.') : '';
                await syncReview();
            } catch (err) {
                if (!stopped) error.value = err.message;
            } finally {
                polling = null;
                if (busy.value) schedulePoll(2000);
            }
        })();
        return polling;
    };

    const buildConfirmMessage = (count, options = {}) => {
        const parts = [];
        if (options.scopeLabel) {
            parts.push(
                t(
                    'pages.dashboard.reanalysis.scope_notice',
                    '当前筛选范围：{scope}',
                ).replace('{scope}', options.scopeLabel),
            );
        }
        if (options.totalMissing && options.totalMissing > count) {
            parts.push(
                t(
                    'pages.dashboard.reanalysis.capped_notice',
                    '共筛出 {total} 张无语义图片，单批最多先处理前 {count} 张。',
                ).replace('{total}', options.totalMissing).replace('{count}', count),
            );
        }
        if (count > TOKEN_WARNING_THRESHOLD) {
            parts.push(
                t(
                    'pages.dashboard.reanalysis.token_warning',
                    '待分析图片过多（共 {count} 张），请注意 token 消耗，是否继续？',
                ).replace('{count}', count),
            );
        } else if (options.fillEmptyOnly) {
            parts.push(
                t(
                    'pages.dashboard.reanalysis.confirm_missing',
                    '为筛出的 {count} 张无语义图片进行视觉模型分析？\n分析后支持一键全部保存或逐张抽查。',
                ).replace('{count}', count),
            );
        } else {
            parts.push(
                t(
                    'pages.dashboard.reanalysis.confirm',
                    'Analyze {count} images? Results will wait for individual review; nothing is saved automatically.',
                ).replace('{count}', count),
            );
        }
        return parts.join('\n\n');
    };

    const start = async (hashes, options = {}) => {
        if (busy.value || submitting.value || !hashes.length) return;
        submitting.value = true;
        try {
            const confirmed = await showConfirm(buildConfirmMessage(hashes.length, options));
            if (!confirmed || stopped) return;
            // 等待已有状态查询结束，避免旧响应覆盖新任务。
            if (polling) await polling;
            if (stopped) return;
            clearTimeout(timer);
            const payload = options.fillEmptyOnly ? { hashes, fill_empty_only: true } : { hashes };
            job.value = await request('api/images/reanalyze', {
                method: 'POST', body: JSON.stringify(payload),
            });
            error.value = '';
            open.value = true;
            await syncReview();
            schedulePoll(1000);
        } catch (err) {
            error.value = err.message;
            open.value = true;
            await syncReview();
            if (busy.value) schedulePoll(2000);
        } finally {
            submitting.value = false;
        }
    };

    const startSelected = () => start(Array.from(selectedImages.value));

    const startMissing = async () => {
        if (submitting.value) return;
        if (busy.value || job.value?.pending) {
            open.value = true;
            return;
        }
        const scope = (typeof getMissingScope === 'function' ? getMissingScope() : null) || {
            params: {},
            label: '',
        };
        const query = new URLSearchParams();
        for (const [key, value] of Object.entries(scope.params || {})) {
            if (value !== undefined && value !== null && String(value) !== '') {
                query.set(key, String(value));
            }
        }
        const endpoint = query.toString()
            ? `api/images/reanalyze/missing?${query.toString()}`
            : 'api/images/reanalyze/missing';

        let missing;
        submitting.value = true;
        try {
            const data = await requestData(endpoint);
            missing = data.missing || {};
        } catch (err) {
            error.value = err.message;
            open.value = true;
            return;
        } finally {
            submitting.value = false;
        }
        const hashes = Array.isArray(missing.hashes) ? missing.hashes : [];
        if (!hashes.length) {
            const emptyMsg = scope.label
                ? t(
                    'pages.dashboard.reanalysis.no_missing_scoped',
                    '当前范围「{scope}」内所有表情均已包含完整语义元数据，无需补充分析。',
                ).replace('{scope}', scope.label)
                : t(
                    'pages.dashboard.reanalysis.no_missing',
                    '所有表情均已包含完整语义元数据，无需补充分析。',
                );
            showAlert?.(emptyMsg, 'success');
            return;
        }
        await start(hashes, {
            fillEmptyOnly: true,
            totalMissing: Number(missing.total_missing || hashes.length),
            scopeLabel: scope.label || '',
        });
    };

    const retryFailed = async () => {
        if (busy.value || submitting.value || !job.value?.failed) return;
        submitting.value = true;
        try {
            if (polling) await polling;
            job.value = await request('api/images/reanalyze/retry', {
                method: 'POST', body: JSON.stringify({ task_id: job.value.id }),
            });
            error.value = '';
            schedulePoll(1000);
        } catch (err) {
            error.value = err.message;
        } finally {
            submitting.value = false;
        }
    };

    const review = async (decision) => {
        if (submitting.value || reviewLoading.value || reviewResult.value?.status !== 'ready') return;
        const taskID = job.value.id;
        const hash = reviewResult.value.hash;
        submitting.value = true;
        try {
            if (polling) await polling;
            if (stopped || reviewHash.value !== hash || job.value?.id !== taskID) return;
            clearTimeout(timer);
            job.value = await request('api/images/reanalyze/review', {
                method: 'POST', body: JSON.stringify({ task_id: taskID, hash, decision }),
            });
            error.value = '';
            if (decision === 'save') await refreshView();
        } catch (err) {
            error.value = err.message;
        } finally {
            await syncReview();
            if (busy.value) schedulePoll(2000);
            submitting.value = false;
        }
    };

    const reviewAll = async (decision) => {
        if (submitting.value || !readyResults.value.length || !job.value?.id) return;
        const taskID = job.value.id;
        submitting.value = true;
        try {
            if (polling) await polling;
            if (stopped || job.value?.id !== taskID) return;
            clearTimeout(timer);
            job.value = await request('api/images/reanalyze/review', {
                method: 'POST', body: JSON.stringify({ task_id: taskID, decision, all: true }),
            });
            error.value = '';
            if (decision === 'save') await refreshView();
        } catch (err) {
            error.value = err.message;
        } finally {
            await syncReview();
            if (busy.value) schedulePoll(2000);
            submitting.value = false;
        }
    };

    const cancel = async () => {
        if (!busy.value || submitting.value) return;
        submitting.value = true;
        try {
            if (polling) await polling;
            if (!busy.value || stopped) return;
            job.value = await request('api/images/reanalyze/cancel', {
                method: 'POST', body: JSON.stringify({ task_id: job.value.id }),
            });
            error.value = '';
            schedulePoll(1000);
        } catch (err) {
            error.value = err.message;
        } finally {
            submitting.value = false;
        }
    };

    const restore = async () => {
        await poll();
        if (!stopped && (busy.value || job.value?.pending)) open.value = true;
    };
    const dispose = () => {
        stopped = true;
        clearTimeout(timer);
    };

    return {
        job, open, error, submitting, busy, startSelected, startMissing, retryFailed, cancel, restore, dispose,
        readyResults, reviewHash, reviewResult, reviewImage, reviewLoading, selectResult,
        saveResult: () => review('save'), discardResult: () => review('discard'),
        saveAllResults: () => reviewAll('save'), discardAllResults: () => reviewAll('discard'),
    };
}
