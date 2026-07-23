import { TEMPLATE } from './template.js';
const { createApp, ref, reactive, computed, onMounted, onUnmounted, nextTick } = Vue;

const PLACEHOLDER = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';

function createLRUCache(maxSize) {
    const cache = new Map();
    return {
        get(key) {
            if (!cache.has(key)) return null;
            const value = cache.get(key);
            cache.delete(key);
            cache.set(key, value);
            return value;
        },
        set(key, value) {
            if (cache.has(key)) cache.delete(key);
            else if (cache.size >= maxSize) {
                const firstKey = cache.keys().next().value;
                cache.delete(firstKey);
            }
            cache.set(key, value);
        },
        has(key) { return cache.has(key); },
        clear() { cache.clear(); }
    };
}

function hashToColor(hash) {
    if (!hash) return '#1e2230';
    const num = parseInt(hash.slice(0, 6), 16) || 0;
    const h = num % 360;
    const s = 20 + (num % 15);
    const l = 15 + (num % 10);
    return `hsl(${h}, ${s}%, ${l}%)`;
}

createApp({
    setup() {
        const activeSection = ref('library');
        const images = ref([]);
        const categories = ref([]);
        const stats = reactive({ total: 0, categories: 0, today: 0 });
        const loading = ref(true);
        const searchQuery = ref('');
        const selectedCategory = ref('');
        const sortBy = ref('newest');
        const currentPage = ref(1);
        const pageSize = ref(24);
        const total = ref(0);

        const pendingImages = ref([]);
        const pendingTotal = ref(0);
        const pendingCategories = ref([]);
        const pendingStats = reactive({ pending: 0, capacity: 200, paused: false });
        const pendingLoading = ref(false);
        const pendingSearchQuery = ref('');
        const pendingCategory = ref('');
        const pendingCurrentPage = ref(1);
        const pendingPageSize = ref(24);
        let pendingFetchLock = false;
        const bridge = window.AstrBotPluginPage;
        const localeVersion = ref(0);

        const getLocale = () => {
            const locale = String(bridge?.getLocale?.() || bridge?.getContext?.()?.locale || 'en-US').trim();
            return locale || 'en-US';
        };

        const resolveUiLocale = () => (getLocale().toLowerCase().startsWith('zh') ? 'zh-CN' : 'en-US');

        const getByPath = (source, key) => {
            if (!source || typeof source !== 'object' || !key) return undefined;
            return String(key).split('.').reduce((current, part) => {
                if (!current || typeof current !== 'object' || !(part in current)) return undefined;
                return current[part];
            }, source);
        };

        const t = (key, fallback) => {
            localeVersion.value;
            const locale = resolveUiLocale();
            const messages = bridge?.getI18n?.() || bridge?.getContext?.()?.i18n || {};
            const value = getByPath(messages?.[locale], key);
            if (value === undefined || value === null) return fallback;
            return typeof value === 'string' ? value : String(value);
        };

        const updateDocumentMeta = () => {
            document.documentElement.lang = getLocale();
            document.title = t('pages.dashboard.title', 'Sticker Dashboard');
        };

        const getHealthText = (status) => {
            if (status === 'ok') return t('pages.dashboard.health.ok', 'Healthy');
            if (status === 'slow') return t('pages.dashboard.health.slow', 'Slow');
            if (status === 'error') return t('pages.dashboard.health.error', 'Error');
            return t('pages.dashboard.health.checking', 'Checking');
        };

        const updatePageSize = () => {
            const w = window.innerWidth;
            const h = window.innerHeight;
            const gap = 12;

            let slotSize, sidebarWidth, mainPadding;
            if (w < 768) {
                slotSize = 120;
                sidebarWidth = 0;
                mainPadding = 24;
            } else {
                slotSize = 160;
                sidebarWidth = 180;
                mainPadding = 44;
            }

            const availableWidth = w - sidebarWidth - mainPadding;
            const perRow = Math.max(2, Math.floor((availableWidth + gap) / (slotSize + gap)));

            const headerHeight = 56;
            const toolbarHeight = 64;
            const paginationHeight = 60;
            const availableHeight = h - headerHeight - toolbarHeight - paginationHeight - 40;
            const rows = Math.max(2, Math.floor((availableHeight + gap) / (slotSize + gap)));

            pageSize.value = perRow * rows;
        };

        const thumbnailCache = createLRUCache(50);

        const previewOpen = ref(false);
        const previewItem = ref(null);
        const isEditing = ref(false);
        const editForm = reactive({ category: '', tags: '', scene: '', desc: '', scope_mode: 'public' });

        // 审核区编辑弹窗（issue #87）
        const pendingEditOpen = ref(false);
        const pendingEditId = ref(null);
        const pendingEditForm = reactive({
            hash: '',
            category: '',
            scope_mode: 'public',
            desc: '',
            tagsText: '',
            scenesText: '',
        });

        const isBatchMode = ref(false);
        const selectedImages = ref(new Set());
        const batchMoveOpen = ref(false);
        const batchTargetCategory = ref('');
        const batchScopeOpen = ref(false);
        const batchScopeMode = ref('public');

        const uploadOpen = ref(false);
        const uploading = ref(false);
        const uploadFile = ref(null);
        const uploadPreviewUrl = ref(null);
        const uploadError = ref(null);
        const confirmOpen = ref(false);
        const confirmMessage = ref('');
        let confirmResolve = null;
        const showConfirm = (msg) => new Promise((resolve) => {
            confirmMessage.value = msg;
            confirmOpen.value = true;
            confirmResolve = resolve;
        });
        const onConfirmYes = () => { confirmOpen.value = false; confirmResolve?.(true); };
        const onConfirmNo = () => { confirmOpen.value = false; confirmResolve?.(false); };

        const promptOpen = ref(false);
        const promptMessage = ref('');
        const promptValue = ref('');
        let promptResolve = null;
        const showPrompt = (msg, initialValue = '') => new Promise((resolve) => {
            promptMessage.value = msg;
            promptValue.value = initialValue;
            promptOpen.value = true;
            promptResolve = resolve;
        });
        const onPromptOk = () => {
            promptOpen.value = false;
            promptResolve?.(promptValue.value);
            promptResolve = null;
        };
        const onPromptCancel = () => {
            promptOpen.value = false;
            promptResolve?.(null);
            promptResolve = null;
        };
        const toastOpen = ref(false);
        const toastMessage = ref('');
        let toastTimer = null;
        const showAlert = (msg) => {
            toastMessage.value = msg;
            toastOpen.value = true;
            clearTimeout(toastTimer);
            toastTimer = setTimeout(() => { toastOpen.value = false; }, 3000);
        };
        const uploadForm = reactive({ emotion: '', tags: '', scene: '', desc: '' });
        const availableEmotions = ref([]);
        const analysisScenes = ref([]);

        const batchUploadOpen = ref(false);
        const batchUploading = ref(false);
        const batchFolderMode = ref(false);
        const batchDragActive = ref(false);
        const batchFiles = ref([]);
        const batchPreviews = ref([]);
        const batchUploadError = ref(null);
        const batchUploadForm = reactive({ emotion: '', autoAnalyze: false });
        const batchTaskId = ref(null);
        const batchTaskStatus = ref(null);
        const batchTaskTotal = ref(0);
        const batchTaskProcessed = ref(0);
        const batchTaskSuccess = ref(0);
        const batchTaskFailed = ref(0);
        let batchPollInterval = null;
        let imgObserver = null;

        const observeImages = () => {
            if (!imgObserver) return;
            document.querySelectorAll('.item-image[data-hash]').forEach((el) => {
                if (!el.dataset.observed) {
                    el.dataset.observed = 'true';
                    imgObserver.observe(el);
                }
            });
        };

        const parseSceneList = (rawText) => {
            if (!rawText) return [];
            const seen = new Set();
            return String(rawText)
                .split(/[，,、;；\n\t]+/)
                .map((item) => item.trim())
                .filter((item) => {
                    if (!item || seen.has(item)) return false;
                    seen.add(item);
                    return true;
                });
        };

        const toggleScene = (scene) => {
            const sceneList = parseSceneList(uploadForm.scene);
            if (sceneList.includes(scene)) {
                uploadForm.scene = sceneList.filter((item) => item !== scene).join(', ');
                return;
            }
            uploadForm.scene = [...sceneList, scene].join(', ');
        };

        const isSceneSelected = (scene) => parseSceneList(uploadForm.scene).includes(scene);

        const formatOriginTarget = (target) => {
            const raw = String(target || '').trim();
            if (!raw) return t('pages.dashboard.messages.origin_unset', 'Not recorded');
            if (raw.startsWith('group:')) return `${t('pages.dashboard.messages.origin_group', 'Group')} ${raw.slice(6)}`;
            if (raw.startsWith('user:')) return `${t('pages.dashboard.messages.origin_user', 'User')} ${raw.slice(5)}`;
            return raw;
        };

        const getScopeLabel = (scopeMode) => (
            String(scopeMode || 'public').toLowerCase() === 'local'
                ? t('pages.dashboard.scope.local', 'Local only')
                : t('pages.dashboard.scope.public', 'Public')
        );

        const normalizeCategories = (rawCategories) => {
            if (Array.isArray(rawCategories)) {
                return rawCategories
                    .map((cat) => {
                        if (cat && typeof cat === 'object') {
                            const key = String(cat.key || cat.name || '').trim();
                            return key ? {
                                key,
                                name: String(cat.name || key),
                                count: Number(cat.count || 0),
                            } : null;
                        }
                        const key = String(cat || '').trim();
                        return key ? { key, name: key, count: 0 } : null;
                    })
                    .filter(Boolean);
            }
            if (rawCategories && typeof rawCategories === 'object') {
                return Object.entries(rawCategories).map(([key, count]) => ({
                    key,
                    name: key,
                    count: Number(count || 0),
                }));
            }
            return [];
        };

        const emotionsOpen = ref(false);
        const newEmotion = reactive({ key: '', name: '', desc: '' });
        const addingEmotion = ref(false);
        const deletingEmotionKey = ref('');

        let searchTimeout = null;

        const isDarkTheme = ref(true);
        const theme = computed(() => isDarkTheme.value ? 'dark' : 'light');

        const imageDataUrls = reactive({});
        const originalDataUrls = reactive({});

        const loadImageData = async (hash) => {
            if (!hash) return;
            const cached = thumbnailCache.get(hash);
            if (cached) { imageDataUrls[hash] = cached; return; }
            if (imageDataUrls[hash]) return;
            try {
                const data = await bridge.apiGet('thumbnail', { hash, size: 300 });
                if (data && data.url) {
                    imageDataUrls[hash] = data.url;
                    thumbnailCache.set(hash, data.url);
                }
            } catch (e) {
                console.error('Failed to load thumbnail:', hash, e);
            }
        };

        const loadOriginalImage = async (hash) => {
            if (!hash || originalDataUrls[hash]) return;
            try {
                const data = await bridge.apiGet('image-data', { hash });
                if (data && data.url) {
                    originalDataUrls[hash] = data.url;
                }
            } catch (e) {
                console.error('Failed to load original image:', hash, e);
            }
        };

        const downloadImage = async (item) => {
            if (!item?.hash) return;
            const dataUrl = originalDataUrls[item.hash] || imageDataUrls[item.hash];
            if (!dataUrl) return;
            const a = document.createElement('a');
            a.href = dataUrl;
            a.download = (item.desc || item.hash) + '.png';
            a.click();
        };

        const fileToBase64 = (file) => new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });

        const apiFetch = async (url, options = {}) => {
            const urlStr = String(url).replace(/^\/?api\//, '');
            const [path, queryString] = urlStr.split('?');
            const endpoint = path.replace(/\/$/, '');

            const params = {};
            if (queryString) {
                const sp = new URLSearchParams(queryString);
                for (const [k, v] of sp) { params[k] = v; }
            }

            const method = (options.method || 'GET').toUpperCase();
            let body = options.body;

            try {
                let data;

                if (method === 'POST' || method === 'PUT' || method === 'DELETE') {
                    if (body instanceof FormData) {
                        const file = body.get('file');
                        if (file instanceof File) {
                            data = await bridge.upload(endpoint, file);
                        } else {
                            const json = {};
                            const fileEntries = [];
                            for (const [k, v] of body.entries()) {
                                if (v instanceof File) {
                                    fileEntries.push({ key: k, file: v });
                                } else {
                                    json[k] = v;
                                }
                            }
                            if (fileEntries.length > 0) {
                                json._files = await Promise.all(
                                    fileEntries.map(async (entry) => ({
                                        key: entry.key,
                                        name: entry.file.name,
                                        base64: await fileToBase64(entry.file),
                                    }))
                                );
                            }
                            data = await bridge.apiPost(endpoint, json);
                        }
                    } else {
                        if (typeof body === 'string') {
                            try { body = JSON.parse(body); } catch (e) { }
                        }
                        data = await bridge.apiPost(endpoint, body || {});
                    }
                } else {
                    data = await bridge.apiGet(endpoint, Object.keys(params).length ? params : undefined);
                }

                return {
                    ok: true,
                    status: 200,
                    json: async () => data,
                    text: async () => (typeof data === 'string' ? data : JSON.stringify(data)),
                };
            } catch (e) {
                return {
                    ok: false,
                    status: 500,
                    json: async () => { throw e; },
                    text: async () => e.message,
                };
            }
        };

        const fetchStats = async () => {
            try {
                const res = await apiFetch('api/stats');
                const data = await res.json();
                Object.assign(stats, data.stats || {});
            } catch (e) {
                console.error(e);
            }
        };

        const healthStatus = ref('unknown');
        const checkHealth = async () => {
            const start = performance.now();
            try {
                const res = await apiFetch('api/health');
                healthStatus.value = (performance.now() - start) < 200 ? 'ok' : 'slow';
            } catch (e) { healthStatus.value = 'error'; }
        };

        let isFetching = false;
        const fetchImages = async (page = 1) => {
            if (isFetching) return;
            isFetching = true;
            loading.value = true;
            try {
                const params = new URLSearchParams({
                    page: page.toString(),
                    size: pageSize.value.toString(),
                    q: searchQuery.value,
                    category: selectedCategory.value === '__favorite__' ? '' : selectedCategory.value,
                    sort: sortBy.value,
                });
                if (selectedCategory.value === '__favorite__') {
                    params.set('favorite_only', 'true');
                }
                const res = await apiFetch('api/images?' + params.toString());
                const data = await res.json();
                const nextImages = data.images || [];
                const nextTotal = Number(data.total || 0);
                const lastPage = Math.max(1, Math.ceil(nextTotal / pageSize.value));

                if (page > lastPage && nextTotal > 0) {
                    isFetching = false;
                    return await fetchImages(lastPage);
                }

                currentPage.value = page;
                images.value = nextImages;
                total.value = nextTotal;
                categories.value = normalizeCategories(data.categories);
                favoriteCount.value = Number(data.favorite_count || 0);
                const currentHashes = new Set(nextImages.map(img => img.hash));
                for (const hash of Object.keys(imageDataUrls)) {
                    if (!currentHashes.has(hash)) delete imageDataUrls[hash];
                }
                nextTick(() => observeImages());
                if (selectedImages.value.size > 0) {
                    const visibleHashes = new Set(nextImages.map((img) => img.hash));
                    selectedImages.value = new Set(
                        Array.from(selectedImages.value).filter((hash) => visibleHashes.has(hash))
                    );
                }
                return nextImages;
            } catch (e) {
                console.error(e);
                return [];
            } finally {
                loading.value = false;
                isFetching = false;
            }
        };

        const fetchEmotions = async () => {
            try {
                const res = await apiFetch('api/emotions');
                const data = await res.json();
                availableEmotions.value = data.emotions || [];
            } catch (e) {
                console.error(e);
            }
        };

        const fetchPendingStats = async () => {
            try {
                const res = await apiFetch('api/pending/stats');
                const data = await res.json();
                if (data.success) Object.assign(pendingStats, data.stats);
            } catch (e) { console.error(e); }
        };

        const fetchPendingImages = async (page = 1) => {
            if (pendingFetchLock) return;
            pendingFetchLock = true;
            pendingLoading.value = true;
            try {
                const params = new URLSearchParams({
                    page: page.toString(),
                    size: pendingPageSize.value.toString(),
                    q: pendingSearchQuery.value,
                    category: pendingCategory.value,
                });
                const res = await apiFetch('api/pending?' + params.toString());
                const data = await res.json();
                if (!data.success) { pendingImages.value = []; pendingTotal.value = 0; return; }
                const nextImages = data.images || [];
                const nextTotal = Number(data.total || 0);
                const lastPage = Math.max(1, Math.ceil(nextTotal / pendingPageSize.value));
                if (page > lastPage && nextTotal > 0) {
                    pendingFetchLock = false;
                    return await fetchPendingImages(lastPage);
                }
                pendingCurrentPage.value = page;
                pendingImages.value = nextImages;
                pendingTotal.value = nextTotal;
                pendingCategories.value = normalizeCategories(data.categories);
                nextImages.forEach(img => { if (img.hash) loadImageData(img.hash); });
            } catch (e) { console.error(e); }
            finally { pendingLoading.value = false; pendingFetchLock = false; }
        };

        const switchSection = (section) => {
            activeSection.value = section;
            if (section === 'pending') {
                fetchPendingStats();
                fetchPendingImages(1);
            } else {
                fetchImages(1);
            }
        };

        const pendingDebouncedSearch = () => {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => fetchPendingImages(1), 400);
        };

        const approvePending = async (id) => {
            try {
                const res = await apiFetch('api/pending/approve', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ id }),
                });
                const data = await res.json();
                if (data.success) {
                    showAlert(t('pages.dashboard.alerts.pending_approved', 'Approved {count} item(s).').replace('{count}', data.approved));
                    await fetchPendingImages(pendingCurrentPage.value);
                    await fetchPendingStats();
                } else {
                    showAlert(`${t('pages.dashboard.alerts.approve_failed', 'Approve failed')}: ${data.error || t('pages.dashboard.messages.unknown_error', 'Unknown error')}`);
                }
            } catch (e) { showAlert(`${t('pages.dashboard.alerts.approve_failed', 'Approve failed')}: ${e.message}`); }
        };

        // issue #87：审核区编辑
        const parseListField = (value) => {
            if (Array.isArray(value)) {
                return value.map((v) => String(v || '').trim()).filter(Boolean);
            }
            return String(value || '')
                .split(/[,，]/)
                .map((v) => v.trim())
                .filter(Boolean);
        };

        const openPendingEdit = async (item) => {
            if (!item || item.id == null) return;
            pendingEditId.value = item.id;
            pendingEditForm.hash = item.hash || '';
            pendingEditForm.category = item.category || '';
            pendingEditForm.scope_mode = item.scope_mode || 'public';
            pendingEditForm.desc = item.desc || '';
            pendingEditForm.tagsText = (item.tags || []).join(', ');
            pendingEditForm.scenesText = (item.scenes || []).join(', ');
            pendingEditOpen.value = true;
            if (item.hash && !imageDataUrls[item.hash]) {
                loadOriginalImage(item.hash);
            }
        };

        const closePendingEdit = () => {
            pendingEditOpen.value = false;
            pendingEditId.value = null;
        };

        const savePendingEdit = async (alsoApprove) => {
            if (!pendingEditId.value) return;
            try {
                const payload = {
                    id: pendingEditId.value,
                    category: pendingEditForm.category,
                    scope_mode: pendingEditForm.scope_mode,
                    desc: pendingEditForm.desc,
                    tags: parseListField(pendingEditForm.tagsText),
                    scenes: parseListField(pendingEditForm.scenesText),
                };
                const res = await apiFetch('api/pending/update', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                const data = await res.json();
                if (!data.success) {
                    showAlert(`${t('pages.dashboard.alerts.save_failed', 'Save failed')}: ${data.error || t('pages.dashboard.messages.unknown_error', 'Unknown error')}`);
                    return;
                }

                if (alsoApprove) {
                    await approvePending(pendingEditId.value);
                    closePendingEdit();
                    return;
                }

                // 仅保存：刷新当前页
                showAlert(t('pages.dashboard.alerts.pending_saved', 'Pending item saved.'));
                closePendingEdit();
                await fetchPendingImages(pendingCurrentPage.value);
                await fetchPendingStats();
            } catch (e) {
                showAlert(`${t('pages.dashboard.alerts.save_failed', 'Save failed')}: ${e.message}`);
            }
        };

        const rejectPending = async (id, blacklist = false) => {
            try {
                const res = await apiFetch('api/pending/reject', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ id, blacklist }),
                });
                const data = await res.json();
                if (data.success) {
                    const suffix = data.blacklisted ? ` ${t('pages.dashboard.alerts.and_blacklisted', '(blacklisted)')}` : '';
                    showAlert(t('pages.dashboard.alerts.pending_deleted', 'Deleted {count} item(s).').replace('{count}', data.deleted) + suffix);
                    await fetchPendingImages(pendingCurrentPage.value);
                    await fetchPendingStats();
                } else {
                    showAlert(`${t('pages.dashboard.alerts.delete_failed', 'Delete failed')}: ${data.error || t('pages.dashboard.messages.unknown_error', 'Unknown error')}`);
                }
            } catch (e) { showAlert(`${t('pages.dashboard.alerts.delete_failed', 'Delete failed')}: ${e.message}`); }
        };

        const approvePendingBatch = async () => {
            const ids = Array.from(pendingSelectedImages.value);
            if (!ids.length) { showAlert(t('pages.dashboard.alerts.select_pending_first', 'Select pending items first.')); return; }
            const confirmed = await showConfirm(
                t('pages.dashboard.confirm.pending_approve_batch', 'Approve {count} pending item(s)?').replace('{count}', ids.length)
            );
            if (!confirmed) return;
            try {
                const res = await apiFetch('api/pending/approve', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ ids }),
                });
                const data = await res.json();
                if (data.approved) {
                    const failed = data.errors?.length
                        ? ` ${t('pages.dashboard.alerts.failed_count', '{count} failed.').replace('{count}', data.errors.length)}`
                        : '';
                    showAlert(t('pages.dashboard.alerts.pending_approved', 'Approved {count} item(s).').replace('{count}', data.approved) + failed);
                }
                pendingSelectedImages.value = new Set();
                pendingBatchMode.value = false;
                await fetchPendingImages(pendingCurrentPage.value);
                await fetchPendingStats();
            } catch (e) { showAlert(`${t('pages.dashboard.alerts.batch_approve_failed', 'Batch approve failed')}: ${e.message}`); }
        };

        const rejectPendingBatch = async (blacklist = false) => {
            const ids = Array.from(pendingSelectedImages.value);
            if (!ids.length) { showAlert(t('pages.dashboard.alerts.select_pending_first', 'Select pending items first.')); return; }
            const key = blacklist
                ? 'pages.dashboard.confirm.pending_delete_blacklist_batch'
                : 'pages.dashboard.confirm.pending_delete_batch';
            const fallback = blacklist
                ? 'Delete and blacklist {count} pending item(s)?'
                : 'Delete {count} pending item(s)?';
            const confirmed = await showConfirm(t(key, fallback).replace('{count}', ids.length));
            if (!confirmed) return;
            try {
                const res = await apiFetch('api/pending/reject', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ ids, blacklist }),
                });
                const data = await res.json();
                if (data.success) {
                    const suffix = data.blacklisted
                        ? ` ${t('pages.dashboard.alerts.blacklisted_count', 'Blacklisted {count} item(s).').replace('{count}', data.blacklisted)}`
                        : '';
                    showAlert(t('pages.dashboard.alerts.pending_deleted', 'Deleted {count} item(s).').replace('{count}', data.deleted) + suffix);
                }
                pendingSelectedImages.value = new Set();
                pendingBatchMode.value = false;
                await fetchPendingImages(pendingCurrentPage.value);
                await fetchPendingStats();
            } catch (e) { showAlert(`${t('pages.dashboard.alerts.batch_delete_failed', 'Batch delete failed')}: ${e.message}`); }
        };

        const pendingBatchMode = ref(false);
        const pendingSelectedImages = ref(new Set());

        const togglePendingBatchMode = () => {
            pendingBatchMode.value = !pendingBatchMode.value;
            if (!pendingBatchMode.value) pendingSelectedImages.value = new Set();
        };

        const togglePendingSelection = (item) => {
            const s = new Set(pendingSelectedImages.value);
            s.has(item.id) ? s.delete(item.id) : s.add(item.id);
            pendingSelectedImages.value = s;
        };

        const allPendingSelected = Vue.computed(() => {
            const imgs = pendingImages.value;
            if (!imgs.length) return false;
            const selected = pendingSelectedImages.value;
            return imgs.every(img => selected.has(img.id));
        });

        const toggleSelectAllPending = () => {
            const currentIds = new Set(pendingImages.value.map(img => img.id));
            if (allPendingSelected.value) {
                pendingSelectedImages.value = new Set();
            } else {
                pendingSelectedImages.value = currentIds;
            }
        };

        const loadAll = async () => {
            await fetchStats();
            await fetchEmotions();
            await fetchImages(1);
        };

        const debouncedSearch = () => {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => fetchImages(1), 400);
        };

        const refreshView = async () => {
            await fetchImages(currentPage.value);
            await fetchStats();
        };

        const prevPage = () => currentPage.value > 1 && fetchImages(currentPage.value - 1);
        const nextPage = () => currentPage.value * pageSize.value < total.value && fetchImages(currentPage.value + 1);

        const openPreview = (img) => {
            previewItem.value = img;
            previewOpen.value = true;
            if (img?.hash) {
                loadOriginalImage(img.hash);
            }
        };

        const closePreview = () => {
            previewOpen.value = false;
            previewItem.value = null;
            isEditing.value = false;
            for (const hash of Object.keys(originalDataUrls)) {
                delete originalDataUrls[hash];
            }
        };

        const navigateImage = (direction) => {
            if (!previewItem.value) return;
            const idx = images.value.findIndex((i) => i.hash === previewItem.value.hash);
            const nextIdx = idx + direction;
            if (nextIdx >= 0 && nextIdx < images.value.length) {
                previewItem.value = images.value[nextIdx];
                loadOriginalImage(previewItem.value.hash);
            }
        };
        const prevImage = () => navigateImage(-1);
        const nextImage = () => navigateImage(1);

        const handleKeydown = (e) => {
            if (!previewOpen.value) return;
            if (isEditing.value) return;
            if (e.key === 'ArrowLeft') prevImage();
            if (e.key === 'ArrowRight') nextImage();
            if (e.key === 'Escape') closePreview();
        };

        const startEdit = () => {
            if (!previewItem.value) return;
            Object.assign(editForm, {
                category: previewItem.value.category,
                tags: (previewItem.value.tags || []).join(', '),
                scene: (previewItem.value.scenes || []).join('、'),
                desc: previewItem.value.desc,
                scope_mode: previewItem.value.scope_mode || 'public',
            });
            isEditing.value = true;
        };

        const cancelEdit = () => {
            isEditing.value = false;
        };

        const saveEdit = async () => {
            if (!previewItem.value) return;
            try {
                const res = await apiFetch('api/images/update', {
                    method: 'POST',
                    body: JSON.stringify({ ...editForm, hash: previewItem.value.hash }),
                });
                const data = await res.json();
                if (data.success) {
                    isEditing.value = false;
                    const refreshedImages = await fetchImages(currentPage.value);
                    const refreshedItem = refreshedImages.find((item) => item.hash === previewItem.value.hash);
                    if (refreshedItem) {
                        previewItem.value = refreshedItem;
                    } else {
                        previewItem.value.category = editForm.category;
                        previewItem.value.tags = editForm.tags.split(',').map((t) => t.trim()).filter((t) => t);
                        previewItem.value.scenes = parseSceneList(editForm.scene);
                        previewItem.value.desc = editForm.desc;
                        previewItem.value.scope_mode = editForm.scope_mode || 'public';
                    }
                    await fetchStats();
                } else {
                    showAlert(data.error || t('pages.dashboard.alerts.save_failed', 'Save failed.'));
                }
            } catch (e) {
                showAlert(`${t('pages.dashboard.alerts.save_failed', 'Save failed')}: ${e.message}`);
            }
        };

        const deleteImage = async (img, blacklist = false) => {
            const msg = blacklist
                ? t(
                    'pages.dashboard.confirm.delete_and_blacklist_image',
                    'Delete and blacklist this image?\nIt will no longer be auto-collected.'
                )
                : t(
                    'pages.dashboard.confirm.delete_image',
                    'Delete this image? This action cannot be undone.'
                );
            if (!await showConfirm(msg)) return;
            try {
                const res = await apiFetch('api/images/delete', {
                    method: 'POST',
                    body: JSON.stringify({ hash: img.hash, blacklist }),
                });
                if (res.ok) {
                    closePreview();
                    if (images.value.length === 1 && currentPage.value > 1) {
                        currentPage.value--;
                    }
                    refreshView();
                } else {
                    showAlert(t('pages.dashboard.alerts.delete_failed', 'Delete failed.'));
                }
            } catch (e) {
                showAlert(t('pages.dashboard.alerts.action_failed', 'Action failed.'));
            }
        };

        const toggleBatchMode = () => {
            isBatchMode.value = !isBatchMode.value;
            selectedImages.value = new Set();
        };

        const toggleSelection = (img) => {
            const next = new Set(selectedImages.value);
            if (next.has(img.hash)) {
                next.delete(img.hash);
            } else {
                next.add(img.hash);
            }
            selectedImages.value = next;
        };

        const selectAll = () => {
            selectedImages.value = selectedImages.value.size === images.value.length
                ? new Set()
                : new Set(images.value.map(i => i.hash));
        };

        const handleBatchDelete = async () => {
            if (selectedImages.value.size === 0) return;
            if (!await showConfirm(
                t('pages.dashboard.confirm.delete_selected_images', 'Delete {count} selected image(s)?')
                    .replace('{count}', selectedImages.value.size)
            )) return;

            try {
                const res = await apiFetch('api/images/batch-delete', {
                    method: 'POST',
                    body: JSON.stringify({ hashes: Array.from(selectedImages.value) }),
                });
                const data = await res.json();
                if (data.success) {
                    selectedImages.value = new Set();
                    refreshView();
                } else {
                    showAlert(data.error || t('pages.dashboard.alerts.delete_failed', 'Delete failed.'));
                }
            } catch (e) {
                showAlert(`${t('pages.dashboard.alerts.action_failed', 'Action failed')}: ${e.message}`);
            }
        };

        const openBatchMoveModal = () => {
            if (selectedImages.value.size === 0) return;
            batchTargetCategory.value = '';
            batchMoveOpen.value = true;
        };

        const closeBatchMoveModal = () => {
            batchMoveOpen.value = false;
        };

        const openBatchScopeModal = () => {
            if (selectedImages.value.size === 0) return;
            batchScopeMode.value = 'public';
            batchScopeOpen.value = true;
        };

        const closeBatchScopeModal = () => {
            batchScopeOpen.value = false;
        };

        const confirmBatchMove = async () => {
            if (!batchTargetCategory.value) return;
            try {
                const res = await apiFetch('api/images/batch-move', {
                    method: 'POST',
                    body: JSON.stringify({
                        hashes: Array.from(selectedImages.value),
                        category: batchTargetCategory.value,
                    }),
                });
                const data = await res.json();
                if (data.success) {
                    batchMoveOpen.value = false;
                    selectedImages.value = new Set();
                    isBatchMode.value = false;
                    refreshView();
                } else {
                    showAlert(data.error || t('pages.dashboard.alerts.move_failed', 'Move failed.'));
                }
            } catch (e) {
                showAlert(`${t('pages.dashboard.alerts.action_failed', 'Action failed')}: ${e.message}`);
            }
        };

        const confirmBatchScope = async () => {
            if (!batchScopeMode.value) return;
            try {
                const res = await apiFetch('api/images/batch-scope', {
                    method: 'POST',
                    body: JSON.stringify({
                        hashes: Array.from(selectedImages.value),
                        scope_mode: batchScopeMode.value,
                    }),
                });
                const data = await res.json();
                if (data.success) {
                    batchScopeOpen.value = false;
                    selectedImages.value = new Set();
                    isBatchMode.value = false;
                    await fetchImages(currentPage.value);
                    if (Number(data.skipped || 0) > 0) {
                        showAlert(
                            t(
                                'pages.dashboard.alerts.batch_scope_partial',
                                'Updated {count} image(s). {skipped} skipped because origin group info is missing.'
                            )
                                .replace('{count}', data.count || 0)
                                .replace('{skipped}', data.skipped)
                        );
                    }
                } else {
                    showAlert(data.error || t('pages.dashboard.alerts.scope_set_failed', 'Scope update failed.'));
                }
            } catch (e) {
                showAlert(`${t('pages.dashboard.alerts.action_failed', 'Action failed')}: ${e.message}`);
            }
        };

        const toggleScope = async (img, scopeMode) => {
            if (!img) return;
            try {
                const res = await apiFetch('api/images/update', {
                    method: 'POST',
                    body: JSON.stringify({ hash: img.hash, scope_mode: scopeMode }),
                });
                const data = await res.json();
                if (data.success) {
                    if (previewItem.value && previewItem.value.hash === img.hash) {
                        previewItem.value.scope_mode = scopeMode;
                    }
                    await fetchImages(currentPage.value);
                } else if (data.error === 'Origin target missing') {
                    showAlert(t('pages.dashboard.alerts.scope_origin_missing', 'This image is missing origin group info and cannot be set to local.'));
                } else {
                    showAlert(data.error || t('pages.dashboard.alerts.scope_update_failed', 'Scope update failed.'));
                }
            } catch (e) {
                showAlert(`${t('pages.dashboard.alerts.action_failed', 'Action failed')}: ${e.message}`);
            }
        };

        const favoriteCount = ref(0);

        const toggleFavorite = async (img) => {
            if (!img?.hash) return;
            const newValue = !img.is_favorite;
            try {
                const res = await apiFetch('api/images/update', {
                    method: 'POST',
                    body: JSON.stringify({ hash: img.hash, is_favorite: newValue }),
                });
                const data = await res.json();
                if (data.success) {
                    img.is_favorite = newValue;
                    favoriteCount.value += newValue ? 1 : -1;
                    if (selectedCategory.value === '__favorite__' && !newValue) {
                        await fetchImages(currentPage.value);
                    }
                } else { showAlert(data.error || t('pages.dashboard.alerts.action_failed', 'Action failed.')); }
            } catch (e) { showAlert(`${t('pages.dashboard.alerts.favorite_failed', 'Favorite update failed')}: ${e.message}`); }
        };

        const batchSetFavorite = async (favorite) => {
            if (selectedImages.value.size === 0) return;
            try {
                const res = await apiFetch('api/images/batch-favorite', {
                    method: 'POST',
                    body: JSON.stringify({ hashes: Array.from(selectedImages.value), favorite }),
                });
                const data = await res.json();
                if (data.success) {
                    selectedImages.value = new Set();
                    isBatchMode.value = false;
                    await fetchImages(currentPage.value);
                    showAlert(
                        t(
                            favorite ? 'pages.dashboard.alerts.batch_favorite_added' : 'pages.dashboard.alerts.batch_favorite_removed',
                            favorite ? 'Favorited {count} image(s).' : 'Removed favorites from {count} image(s).'
                        ).replace('{count}', data.count || 0)
                    );
                } else { showAlert(data.error || t('pages.dashboard.alerts.batch_action_failed', 'Batch action failed.')); }
            } catch (e) { showAlert(`${t('pages.dashboard.alerts.batch_action_failed', 'Batch action failed')}: ${e.message}`); }
        };

        const runStorageCleanup = async () => {
            try {
                const scanRes = await apiFetch('api/storage/scan');
                const scan = await scanRes.json();
                if (!scan.success) {
                    showAlert(scan.error || t('pages.dashboard.alerts.storage_scan_failed', 'Storage scan failed.'));
                    return;
                }
                const totalCount =
                    Number(scan.stale_index?.count || 0) +
                    Number(scan.orphan_files?.count || 0) +
                    Number(scan.thumb_cache?.count || 0) +
                    Number(scan.temp_files?.count || 0);
                if (totalCount <= 0) {
                    showAlert(t('pages.dashboard.alerts.storage_nothing_to_clean', 'No storage items need cleanup.'));
                    return;
                }
                const ok = await showConfirm(
                    t(
                        'pages.dashboard.confirm.storage_cleanup',
                        'Found {count} cleanable item(s). This will remove stale indexes, orphan files, thumbnail cache, and temp files. Continue?'
                    ).replace('{count}', totalCount)
                );
                if (!ok) return;
                const cleanRes = await apiFetch('api/storage/cleanup', {
                    method: 'POST',
                    body: JSON.stringify({ strategy: 'balanced' }),
                });
                const clean = await cleanRes.json();
                if (!clean.success) {
                    showAlert(clean.error || t('pages.dashboard.alerts.storage_cleanup_failed', 'Storage cleanup failed.'));
                    return;
                }
                const removed = clean.removed || {};
                const removedCount = Object.values(removed).reduce((sum, value) => sum + Number(value || 0), 0);
                await fetchImages(currentPage.value);
                await fetchStats();
                showAlert(t('pages.dashboard.alerts.storage_cleanup_done', 'Storage cleanup completed. Processed {count} item(s).').replace('{count}', removedCount));
            } catch (e) {
                showAlert(`${t('pages.dashboard.alerts.storage_cleanup_failed', 'Storage cleanup failed')}: ${e.message}`);
            }
        };

        const repairSelectedScope = async () => {
            if (selectedImages.value.size === 0) return;
            const originTarget = await showPrompt(
                t(
                    'pages.dashboard.prompt.origin_scope',
                    'Enter the origin scope, for example group:123456 or user:123456.'
                )
            );
            if (!originTarget || !originTarget.trim()) return;
            try {
                const res = await apiFetch('api/images/scope-repair', {
                    method: 'POST',
                    body: JSON.stringify({
                        hashes: Array.from(selectedImages.value),
                        origin_target: originTarget.trim(),
                        scope_mode: 'local',
                        only_missing: false,
                    }),
                });
                const data = await res.json();
                if (data.success) {
                    selectedImages.value = new Set();
                    isBatchMode.value = false;
                    await fetchImages(currentPage.value);
                    showAlert(t('pages.dashboard.alerts.scope_repaired', 'Repaired origin scope for {count} image(s).').replace('{count}', data.count || 0));
                } else {
                    showAlert(data.error || t('pages.dashboard.alerts.scope_repair_failed', 'Origin scope repair failed.'));
                }
            } catch (e) {
                showAlert(`${t('pages.dashboard.alerts.scope_repair_failed', 'Origin scope repair failed')}: ${e.message}`);
            }
        };

        const openUploadModal = () => {
            uploadOpen.value = true;
            uploadFile.value = null;
            uploadPreviewUrl.value = null;
            uploadError.value = null;
            Object.assign(uploadForm, {
                emotion: selectedCategory.value || '',
                tags: '',
                scene: '',
                desc: '',
            });
            analysisScenes.value = [];
            fetchEmotions();
        };

        const closeUploadModal = () => {
            if (uploadPreviewUrl.value) URL.revokeObjectURL(uploadPreviewUrl.value);
            uploadOpen.value = false;
            analysisScenes.value = [];
        };

        const openBatchUploadModal = () => {
            batchUploadOpen.value = true;
            batchFiles.value = [];
            batchPreviews.value = [];
            batchUploadError.value = null;
            batchDragActive.value = false;
            batchTaskId.value = null;
            batchTaskStatus.value = null;
            Object.assign(batchUploadForm, {
                emotion: selectedCategory.value || '',
                autoAnalyze: false,
            });
            fetchEmotions();
        };

        const closeBatchUploadModal = () => {
            batchUploadOpen.value = false;
            batchDragActive.value = false;
            if (batchPollInterval) {
                clearInterval(batchPollInterval);
                batchPollInterval = null;
            }
        };

        const resetBatchInput = (inputEl) => {
            if (inputEl) inputEl.value = '';
        };

        const normalizeImageFiles = (fileList) => Array.from(fileList || []).filter((file) =>
            file && String(file.type || '').startsWith('image/')
        );

        const setBatchFiles = (files) => {
            batchPreviews.value.forEach((url) => URL.revokeObjectURL(url));
            batchFiles.value = files;
            batchPreviews.value = files.map((file) => URL.createObjectURL(file));
        };

        const clearBatchFiles = () => {
            setBatchFiles([]);
            resetBatchInput(batchFileInput.value);
            resetBatchInput(batchFolderInput.value);
        };

        const batchFileInput = ref(null);
        const batchFolderInput = ref(null);
        const openNativeFilePicker = (inputEl) => {
            if (!inputEl) return;
            resetBatchInput(inputEl);
            if (typeof inputEl.showPicker === 'function') {
                try {
                    inputEl.showPicker();
                    return;
                } catch (e) {
                    console.warn('showPicker failed, falling back to click():', e);
                }
            }
            inputEl.click();
        };
        const triggerBatchFileInput = () => {
            const el = batchFolderMode.value ? batchFolderInput.value : batchFileInput.value;
            openNativeFilePicker(el);
        };

        const handleBatchFileSelect = (e) => {
            const files = normalizeImageFiles(e.target?.files);
            if (files.length === 0) return;
            batchUploadError.value = null;
            setBatchFiles(files);
            resetBatchInput(e.target);
        };

        const batchAreaContainsDragTarget = (event) => {
            const currentTarget = event.currentTarget;
            const relatedTarget = event.relatedTarget;
            return Boolean(currentTarget && relatedTarget && currentTarget.contains(relatedTarget));
        };

        const onBatchDragEnter = (event) => {
            event.preventDefault();
            batchDragActive.value = true;
        };

        const onBatchDragOver = (event) => {
            event.preventDefault();
            if (event.dataTransfer) {
                event.dataTransfer.dropEffect = 'copy';
            }
            batchDragActive.value = true;
        };

        const onBatchDragLeave = (event) => {
            if (batchAreaContainsDragTarget(event)) return;
            batchDragActive.value = false;
        };

        const onBatchDrop = (event) => {
            event.preventDefault();
            batchDragActive.value = false;
            const files = normalizeImageFiles(event.dataTransfer?.files);
            if (files.length === 0) {
                batchUploadError.value = t('pages.dashboard.alerts.no_images_dropped', 'No image files were dropped.');
                return;
            }
            batchUploadError.value = null;
            setBatchFiles(files);
        };

        const formatBatchSize = () => {
            const totalSize = batchFiles.value.reduce((sum, f) => sum + f.size, 0);
            if (totalSize < 1024) return totalSize + ' B';
            if (totalSize < 1024 * 1024) return (totalSize / 1024).toFixed(1) + ' KB';
            return (totalSize / (1024 * 1024)).toFixed(1) + ' MB';
        };

        const submitBatchUpload = async () => {
            if (batchFiles.value.length === 0) return;
            if (!batchUploadForm.emotion && !batchUploadForm.autoAnalyze) {
                batchUploadError.value = t('pages.dashboard.alerts.select_category_or_auto', 'Select a category or enable auto analyze.');
                return;
            }
            batchUploading.value = true;
            batchUploadError.value = null;
            try {
                const formData = new FormData();
                for (const file of batchFiles.value) {
                    formData.append('files', file);
                }
                if (batchUploadForm.emotion) {
                    formData.append('category', batchUploadForm.emotion);
                }
                formData.append('auto_analyze', String(batchUploadForm.autoAnalyze));

                const res = await apiFetch('api/images/batch-upload', { method: 'POST', body: formData });
                const data = await res.json();
                if (data.success) {
                    batchTaskId.value = data.task_id;
                    batchTaskTotal.value = data.total;
                    batchTaskProcessed.value = 0;
                    batchTaskSuccess.value = 0;
                    batchTaskFailed.value = 0;
                    startBatchStatusPoll();
                } else {
                    batchUploadError.value = data.error || t('pages.dashboard.alerts.upload_failed', 'Upload failed.');
                }
            } catch (e) {
                batchUploadError.value = t('pages.dashboard.alerts.upload_error', 'Upload error.');
            } finally {
                batchUploading.value = false;
            }
        };

        const startBatchStatusPoll = () => {
            if (batchPollInterval) clearInterval(batchPollInterval);
            batchPollInterval = setInterval(async () => {
                if (!batchTaskId.value) return;
                try {
                    const res = await apiFetch('api/images/batch-upload-status?task_id=' + batchTaskId.value);
                    const data = await res.json();
                    if (data.success) {
                        batchTaskStatus.value = data.status;
                        batchTaskProcessed.value = data.processed;
                        batchTaskSuccess.value = Number(data.success_count || 0);
                        batchTaskFailed.value = Number(data.failed_count || 0);
                        if (data.status === 'completed' || data.status === 'failed') {
                            clearInterval(batchPollInterval);
                            batchPollInterval = null;
                            if (data.status === 'completed') {
                                fetchImages(1);
                                fetchStats();
                            } else {
                                batchUploadError.value = data.error || t('pages.dashboard.alerts.batch_import_failed', 'Batch import failed.');
                            }
                        }
                    }
                } catch (e) {
                    console.error('Batch status poll error:', e);
                }
            }, 1000);
        };

        const resetBatchUpload = () => {
            batchTaskId.value = null;
            batchTaskStatus.value = null;
            batchDragActive.value = false;
            clearBatchFiles();
            batchUploadError.value = null;
        };

        const handleFileSelect = (e) => {
            const file = e.target.files[0];
            if (file && file.type.startsWith('image/')) {
                if (uploadPreviewUrl.value) URL.revokeObjectURL(uploadPreviewUrl.value);
                uploadFile.value = file;
                uploadPreviewUrl.value = URL.createObjectURL(file);
                uploadError.value = null;
                uploadForm.scene = '';
                analysisScenes.value = [];
            }
        };

        const submitUpload = async () => {
            if (!uploadFile.value) return;
            uploading.value = true;
            try {
                const base64Data = await fileToBase64(uploadFile.value);
                const uploadRes = await apiFetch('api/images/upload', {
                    method: 'POST',
                    body: JSON.stringify({
                        base64: base64Data,
                        filename: uploadFile.value.name,
                        category: uploadForm.emotion,
                        tags: uploadForm.tags,
                        scene: uploadForm.scene,
                        desc: uploadForm.desc,
                    }),
                });
                const uploadData = await uploadRes.json();
                if (!uploadData.success || !uploadData.hash) {
                    uploadError.value = uploadData.error || t('pages.dashboard.alerts.upload_failed', 'Upload failed.');
                    return;
                }
                closeUploadModal();
                fetchImages(1);
                fetchStats();
            } catch (e) {
                uploadError.value = t('pages.dashboard.alerts.upload_error', 'Upload error.');
            } finally {
                uploading.value = false;
            }
        };

        const useImageAnalyzer = () => {
            const isAnalyzing = ref(false);

            const analyze = async (file) => {
                if (!file) {
                    throw new Error(t('pages.dashboard.alerts.select_image_first', 'Select an image first.'));
                }

                isAnalyzing.value = true;
                console.log('[Analyzer] Start analyzing image:', file.name);

                try {
                    const base64Data = await fileToBase64(file);
                    const res = await apiFetch('api/analyze', {
                        method: 'POST',
                        body: JSON.stringify({ base64: base64Data }),
                    });
                    const data = await res.json();

                    if (!data.success) {
                        throw new Error(data.error || t('pages.dashboard.alerts.analyze_failed', 'Analyze failed.'));
                    }

                    console.log('[Analyzer] Analyze success:', data);
                    return data;
                } catch (e) {
                    console.error('[Analyzer] Analyze failed:', e);
                    throw e;
                } finally {
                    isAnalyzing.value = false;
                }
            };

            const applyToForm = (data, form, categories = []) => {
                const result = { filled: false, fields: [] };

                if (data.category) {
                    const exists = categories.some(e => e.key === data.category);
                    if (exists) {
                        form.emotion = data.category;
                        result.fields.push('category');
                    } else if (categories.length > 0) {
                        console.warn('[Analyzer] Category missing, using default:', data.category);
                        form.emotion = categories[0].key;
                        result.fields.push('category');
                    }
                }

                if (data.tags && data.tags.length > 0) {
                    const existingTags = form.tags ? form.tags.split(',').map(t => t.trim()).filter(t => t) : [];
                    const newTags = data.tags.filter(t => !existingTags.includes(t));
                    if (newTags.length > 0) {
                        form.tags = [...existingTags, ...newTags].join(', ');
                        result.fields.push('tags');
                    }
                }

                if (Array.isArray(data.scenes) && data.scenes.length > 0) {
                    form.scene = parseSceneList(data.scenes.join(', ')).join(', ');
                    result.fields.push('scenes');
                }

                if (data.description && !form.desc) {
                    form.desc = data.description;
                    result.fields.push('desc');
                }

                result.filled = result.fields.length > 0;
                console.log('[Analyzer] Form fill result:', result);
                return result;
            };

            return {
                isAnalyzing,
                analyze,
                applyToForm,
            };
        };

        const imageAnalyzer = useImageAnalyzer();
        const analyzing = imageAnalyzer.isAnalyzing;

        const analyzeImage = async () => {
            uploadError.value = null;

            try {
                const data = await imageAnalyzer.analyze(uploadFile.value);
                analysisScenes.value = Array.isArray(data.scenes) ? data.scenes : [];
                const result = imageAnalyzer.applyToForm(data, uploadForm, availableEmotions.value);

                if (!result.filled) {
                    uploadError.value = t('pages.dashboard.alerts.no_valid_info', 'No valid information recognized.');
                }
            } catch (e) {
                uploadError.value = e.message || t('pages.dashboard.alerts.analyze_failed', 'Analyze failed.');
            }
        };

        const openEmotionsModal = () => {
            emotionsOpen.value = true;
            fetchEmotions();
        };

        const closeEmotionsModal = () => {
            emotionsOpen.value = false;
        };

        const addEmotion = async () => {
            const key = String(newEmotion.key || '').trim();
            if (!key) return;
            addingEmotion.value = true;
            try {
                const newCat = {
                    key,
                    name: String(newEmotion.name || '').trim(),
                    desc: String(newEmotion.desc || '').trim(),
                };
                const currentList = [...availableEmotions.value];
                const existingIdx = currentList.findIndex((c) => c.key === newCat.key);
                if (existingIdx >= 0) {
                    if (!await showConfirm(
                        t('pages.dashboard.confirm.update_existing_category', 'Category {key} already exists. Update it?')
                            .replace('{key}', newCat.key)
                    )) {
                        addingEmotion.value = false;
                        return;
                    }
                    currentList[existingIdx] = newCat;
                } else {
                    currentList.push(newCat);
                }

                const res = await apiFetch('api/categories', {
                    method: 'POST',
                    body: JSON.stringify({ categories: currentList }),
                });
                const data = await res.json();

                if (data.success) {
                    await fetchEmotions();
                    await fetchImages(1);
                    newEmotion.key = '';
                    newEmotion.name = '';
                    newEmotion.desc = '';
                } else {
                    showAlert(data.error || t('pages.dashboard.alerts.add_failed', 'Add failed.'));
                }
            } catch (e) {
                showAlert(`${t('pages.dashboard.alerts.action_failed', 'Action failed')}: ${e.message}`);
            } finally {
                addingEmotion.value = false;
            }
        };

        const deleteEmotion = async (cat) => {
            if (!cat?.key) return;
            if (!await showConfirm(
                t(
                    'pages.dashboard.confirm.delete_category',
                    'Delete category {key}? Images in this category will be deleted permanently.'
                ).replace('{key}', cat.key)
            ))
                return;
            deletingEmotionKey.value = cat.key;
            try {
                const res = await apiFetch('api/categories/delete', {
                    method: 'POST',
                    body: JSON.stringify({ key: cat.key }),
                });
                const data = await res.json().catch(() => ({}));
                if (res.ok && data.success) {
                    if (selectedCategory.value === cat.key) selectedCategory.value = '';
                    if (editForm.category === cat.key) editForm.category = '';
                    if (previewItem.value && previewItem.value.category === cat.key)
                        previewItem.value.category = 'unknown';
                    fetchEmotions();
                    refreshView();
                } else {
                    showAlert(data.error || t('pages.dashboard.alerts.delete_failed', 'Delete failed.'));
                }
            } catch (e) {
                showAlert(`${t('pages.dashboard.alerts.action_failed', 'Action failed')}: ${e.message}`);
            } finally {
                deletingEmotionKey.value = '';
            }
        };

        const formatDate = (timestamp) => {
            if (!timestamp) return t('pages.dashboard.messages.unknown', 'Unknown');
            const date = new Date(timestamp * 1000);
            return date.toLocaleString(resolveUiLocale(), {
                year: 'numeric',
                month: '2-digit',
                day: '2-digit',
                hour: '2-digit',
                minute: '2-digit',
            });
        };

        const applyTheme = () => {
            document.documentElement.setAttribute('data-theme', theme.value);
        };

        const syncThemeFromContext = (context = null) => {
            const nextContext = context || bridge?.getContext?.() || {};
            isDarkTheme.value = Boolean(nextContext?.isDark);
            applyTheme();
            localeVersion.value += 1;
            updateDocumentMeta();
        };

        let resizeTimer = null;
        const handleResize = () => {
            clearTimeout(resizeTimer);
            resizeTimer = setTimeout(() => { updatePageSize(); fetchImages(1); }, 300);
        };
        onMounted(() => {
            updateDocumentMeta();
            syncThemeFromContext();
            bridge?.onContext?.((context) => {
                syncThemeFromContext(context);
            });
            updatePageSize();
            window.addEventListener('keydown', handleKeydown);
            window.addEventListener('resize', handleResize);
            imgObserver = new IntersectionObserver((entries) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting) {
                        const hash = entry.target.dataset.hash;
                        if (hash) loadImageData(hash);
                        imgObserver.unobserve(entry.target);
                        entry.target.dataset.observed = '';
                    }
                });
            }, { rootMargin: '200px' });
            checkHealth();
            loadAll();
        });

        onUnmounted(() => {
            window.removeEventListener('keydown', handleKeydown);
            window.removeEventListener('resize', handleResize);
            if (imgObserver) imgObserver.disconnect();
            clearTimeout(resizeTimer);
            clearTimeout(searchTimeout);
        });

        return {
            activeSection,
            switchSection,
            images,
            categories,
            stats,
            loading,
            searchQuery,
            selectedCategory,
            sortBy,
            currentPage,
            pageSize,
            total,
            pendingImages,
            pendingTotal,
            pendingCategories,
            pendingStats,
            pendingLoading,
            pendingSearchQuery,
            pendingCategory,
            pendingCurrentPage,
            pendingPageSize,
            pendingBatchMode,
            pendingSelectedImages,
            fetchPendingImages,
            fetchPendingStats,
            pendingDebouncedSearch,
            approvePending,
            rejectPending,
            approvePendingBatch,
            rejectPendingBatch,
            togglePendingBatchMode,
            togglePendingSelection,
            allPendingSelected,
            toggleSelectAllPending,

            // issue #87：审核区编辑
            pendingEditOpen,
            pendingEditId,
            pendingEditForm,
            openPendingEdit,
            closePendingEdit,
            savePendingEdit,
            parseListField,

            previewOpen,
            previewItem,
            isEditing,
            editForm,
            openPreview,
            closePreview,
            prevImage,
            nextImage,
            startEdit,
            cancelEdit,
            saveEdit,

            isBatchMode,
            selectedImages,
            batchMoveOpen,
            batchTargetCategory,
            batchScopeOpen,
            batchScopeMode,
            toggleBatchMode,
            toggleSelection,
            selectAll,
            runStorageCleanup,
            handleBatchDelete,
            openBatchMoveModal,
            closeBatchMoveModal,
            confirmBatchMove,
            openBatchScopeModal,
            closeBatchScopeModal,
            confirmBatchScope,
            repairSelectedScope,

            uploadOpen,
            uploading,
            uploadFile,
            uploadPreviewUrl,
            uploadError,
            uploadForm,
            availableEmotions,
            analysisScenes,
            isSceneSelected,
            toggleScene,
            openUploadModal,
            closeUploadModal,
            handleFileSelect,
            submitUpload,

            analyzing,
            analyzeImage,

            batchUploadOpen,
            batchUploading,
            batchFolderMode,
            batchDragActive,
            batchFiles,
            batchPreviews,
            batchUploadError,
            batchUploadForm,
            batchTaskId,
            batchTaskStatus,
            batchTaskTotal,
            batchTaskProcessed,
            batchTaskSuccess,
            batchTaskFailed,
            openBatchUploadModal,
            closeBatchUploadModal,
            batchFileInput,
            batchFolderInput,
            triggerBatchFileInput,
            clearBatchFiles,
            handleBatchFileSelect,
            onBatchDragEnter,
            onBatchDragOver,
            onBatchDragLeave,
            onBatchDrop,
            formatBatchSize,
            submitBatchUpload,
            resetBatchUpload,

            emotionsOpen,
            newEmotion,
            addingEmotion,
            deletingEmotionKey,
            openEmotionsModal,
            closeEmotionsModal,
            addEmotion,
            deleteEmotion,

            fetchImages,
            debouncedSearch,
            deleteImage,
            toggleScope,
            prevPage,
            nextPage,
            refreshView,
            formatDate,
            formatOriginTarget,
            getScopeLabel,
            PLACEHOLDER,
            imageDataUrls,
            originalDataUrls,
            downloadImage,

            favoriteCount,
            toggleFavorite,
            batchSetFavorite,
            healthStatus,
            hashToColor,
            localeVersion,
            t,
            getHealthText,

            confirmOpen,
            confirmMessage,
            onConfirmYes,
            onConfirmNo,
            promptOpen,
            promptMessage,
            promptValue,
            onPromptOk,
            onPromptCancel,
            toastOpen,
            toastMessage,
        };
    },
    template: TEMPLATE,
}).mount('#app');
