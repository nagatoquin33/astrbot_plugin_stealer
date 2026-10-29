export function createBatchReanalysis({ ref, computed, apiFetch, selectedImages, showConfirm, refreshView, loadThumbnail, t }) {
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

    const start = async (hashes) => {
        if (busy.value || submitting.value || !hashes.length) return;
        submitting.value = true;
        try {
            const confirmed = await showConfirm(t('pages.dashboard.reanalysis.confirm',
                'Analyze {count} images? Results will wait for individual review; nothing is saved automatically.').replace('{count}', hashes.length));
            if (!confirmed || stopped) return;
            // 等待已有状态查询结束，避免旧响应覆盖新任务。
            if (polling) await polling;
            if (stopped) return;
            clearTimeout(timer);
            job.value = await request('api/images/reanalyze', {
                method: 'POST', body: JSON.stringify({ hashes }),
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
        job, open, error, submitting, busy, startSelected, retryFailed, cancel, restore, dispose,
        readyResults, reviewHash, reviewResult, reviewImage, reviewLoading, selectResult,
        saveResult: () => review('save'), discardResult: () => review('discard'),
    };
}
