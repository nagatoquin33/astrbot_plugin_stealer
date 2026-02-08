const { createApp, ref, reactive, onMounted } = Vue;

createApp({
    setup() {
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

        const previewOpen = ref(false);
        const previewItem = ref(null);
        const isEditing = ref(false);
        const editForm = reactive({ category: '', tags: '', desc: '' });

        const isBatchMode = ref(false);
        const selectedImages = ref(new Set());
        const batchMoveOpen = ref(false);
        const batchTargetCategory = ref('');

        const uploadOpen = ref(false);
        const uploading = ref(false);
        const uploadFile = ref(null);
        const uploadPreviewUrl = ref(null);
        const uploadError = ref(null);
        const uploadForm = reactive({ emotion: '', tags: '', desc: '' });
        const availableEmotions = ref([]);

        const newEmotion = reactive({ key: '', name: '', desc: '' });
        const addingEmotion = ref(false);
        const deletingEmotionKey = ref('');

        const emotionsOpen = ref(false);

        const searchTimeout = ref(null);
        const isAuthed = ref(false);
        const authRequired = ref(false);
        const authChecking = ref(true);
        const loginToken = ref('');
        const loginError = ref('');
        const showPassword = ref(false);
        const sessionTimeout = ref(3600);

        const apiFetch = (url, options = {}) => {
            const headers = new Headers(options.headers || {});
            return fetch(url, { ...options, headers, credentials: 'same-origin' })
                .then((res) => {
                    if (res.status === 401) {
                        isAuthed.value = false;
                        loginError.value = '会话已过期，请重新登录';
                    }
                    return res;
                })
                .catch(() => fetch(url, { ...options, headers, credentials: 'same-origin' }));
        };

        // Fetch Methods
        const fetchStats = async () => {
            try {
                const res = await apiFetch('api/stats');
                const data = await res.json();
                Object.assign(stats, data.stats || {});
            } catch (e) {
                console.error(e);
            }
        };

        const fetchImages = async (page = 1) => {
            loading.value = true;
            currentPage.value = page;
            try {
                const params = new URLSearchParams({
                    page,
                    size: pageSize.value,
                    q: searchQuery.value,
                    category: selectedCategory.value,
                    sort: sortBy.value,
                });
                const res = await apiFetch(`api/images?${params}`);
                const data = await res.json();
                images.value = data.images || [];
                total.value = data.total || 0;
                categories.value = data.categories || [];
            } catch (e) {
                console.error(e);
            } finally {
                loading.value = false;
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

        const loadAll = async () => {
            await fetchStats();
            await fetchEmotions();
            await fetchImages(1);
        };

        const initAuth = async () => {
            authChecking.value = true;
            loginError.value = '';
            try {
                const res = await fetch('auth/info');
                const data = await res.json();
                authRequired.value = !!(data && data.requires_auth);
                if (data && data.session_timeout) sessionTimeout.value = Number(data.session_timeout) || 3600;

                if (!authRequired.value) {
                    isAuthed.value = true;
                    authChecking.value = false;
                    await loadAll();
                    return;
                }

                const health = await apiFetch('api/health');
                if (health.ok) {
                    isAuthed.value = true;
                    authChecking.value = false;
                    await loadAll();
                    return;
                }
            } catch (e) {
                authRequired.value = true;
            } finally {
                authChecking.value = false;
            }

            isAuthed.value = false;
            loginToken.value = '';
            showPassword.value = false;
        };

        const submitLogin = async () => {
            loginError.value = '';
            const password = (loginToken.value || '').trim();
            if (!password) {
                loginError.value = '请输入密码';
                return;
            }

            try {
                const res = await apiFetch('auth/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ password }),
                });
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) {
                    loginError.value = data && data.error ? data.error : '登录失败';
                    return;
                }
            } catch (e) {
                loginError.value = '登录失败';
                return;
            }

            isAuthed.value = true;
            showPassword.value = false;
            await loadAll();
        };

        const logout = () => {
            loginToken.value = '';
            isAuthed.value = false;
            loginError.value = '';
            showPassword.value = false;
            apiFetch('auth/logout', { method: 'POST' }).catch(() => {});
        };

        const debouncedSearch = () => {
            clearTimeout(searchTimeout.value);
            searchTimeout.value = setTimeout(() => fetchImages(1), 400);
        };

        const deleteImage = async (img, blacklist = false) => {
            const msg = blacklist
                ? '确定要销毁并永久拉黑这份战利品吗？\n拉黑后将不再自动收集此相同的图片。'
                : '确定要丢弃这份战利品吗？此操作无法撤销。';
            if (!confirm(msg)) return;
            try {
                const url = blacklist ? `api/images/${img.hash}?blacklist=true` : `api/images/${img.hash}`;
                const res = await apiFetch(url, { method: 'DELETE' });
                if (res.ok) {
                    if (previewOpen.value) closePreview();
                    fetchImages(currentPage.value);
                    fetchStats();
                } else alert('删除失败');
            } catch (e) {
                alert('出错');
            }
        };

        // Pagination
        const prevPage = () => currentPage.value > 1 && fetchImages(currentPage.value - 1);
        const nextPage = () =>
            currentPage.value * pageSize.value < total.value && fetchImages(currentPage.value + 1);

        // Modal Control
        const openPreview = (img) => {
            previewItem.value = img;
            previewOpen.value = true;
        };
        const closePreview = () => {
            previewOpen.value = false;
            previewItem.value = null;
        };

        // Quick Switch Logic
        const getCurrentIndex = () => images.value.findIndex((i) => i.hash === previewItem.value?.hash);

        const prevImage = () => {
            if (!previewItem.value) return;
            const idx = getCurrentIndex();
            if (idx > 0) previewItem.value = images.value[idx - 1];
        };

        const nextImage = () => {
            if (!previewItem.value) return;
            const idx = getCurrentIndex();
            if (idx < images.value.length - 1) previewItem.value = images.value[idx + 1];
        };

        const handleKeydown = (e) => {
            if (!previewOpen.value) return;
            if (e.key === 'ArrowLeft') prevImage();
            if (e.key === 'ArrowRight') nextImage();
            if (e.key === 'Escape') closePreview();
        };

        const openUploadModal = () => {
            uploadOpen.value = true;
            uploadFile.value = null;
            uploadPreviewUrl.value = null;
            uploadError.value = null;
            Object.assign(uploadForm, { emotion: '', tags: '', desc: '' });
            fetchEmotions();
        };
        const closeUploadModal = () => (uploadOpen.value = false);

        const openEmotionsModal = () => {
            emotionsOpen.value = true;
            fetchEmotions();
        };
        const closeEmotionsModal = () => (emotionsOpen.value = false);

        // Batch Logic
        const toggleBatchMode = () => {
            isBatchMode.value = !isBatchMode.value;
            selectedImages.value.clear();
        };

        const toggleSelection = (img) => {
            if (selectedImages.value.has(img.hash)) {
                selectedImages.value.delete(img.hash);
            } else {
                selectedImages.value.add(img.hash);
            }
        };

        const selectAll = () => {
            if (selectedImages.value.size === images.value.length) {
                selectedImages.value.clear();
            } else {
                images.value.forEach((img) => selectedImages.value.add(img.hash));
            }
        };

        const handleBatchDelete = async () => {
            if (selectedImages.value.size === 0) return;
            if (!confirm(`确定要删除选中的 ${selectedImages.value.size} 张图片吗？`)) return;

            try {
                const res = await apiFetch('api/images/batch/delete', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ hashes: Array.from(selectedImages.value) }),
                });
                const data = await res.json();
                if (data.success) {
                    selectedImages.value.clear();
                    fetchImages(currentPage.value);
                    fetchStats();
                } else alert(data.error || '删除失败');
            } catch (e) {
                alert('操作出错: ' + e.message);
            }
        };

        const openBatchMoveModal = () => {
            if (selectedImages.value.size === 0) return;
            batchTargetCategory.value = '';
            batchMoveOpen.value = true;
            fetchEmotions();
        };
        const closeBatchMoveModal = () => (batchMoveOpen.value = false);

        const confirmBatchMove = async () => {
            if (!batchTargetCategory.value) return;
            try {
                const res = await apiFetch('api/images/batch/move', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        hashes: Array.from(selectedImages.value),
                        category: batchTargetCategory.value,
                    }),
                });
                const data = await res.json();
                if (data.success) {
                    batchMoveOpen.value = false;
                    selectedImages.value.clear();
                    isBatchMode.value = false;
                    fetchImages(currentPage.value);
                    fetchStats();
                } else alert(data.error || '移动失败');
            } catch (e) {
                alert('操作出错: ' + e.message);
            }
        };

        // Edit Logic
        const startEdit = () => {
            if (!previewItem.value) return;
            Object.assign(editForm, {
                category: previewItem.value.category,
                tags: previewItem.value.tags.join(', '),
                desc: previewItem.value.desc,
            });
            isEditing.value = true;
            fetchEmotions(); // ensure categories are loaded
        };

        const cancelEdit = () => {
            isEditing.value = false;
        };

        const saveEdit = async () => {
            if (!previewItem.value) return;
            try {
                const res = await apiFetch(`api/images/${previewItem.value.hash}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(editForm),
                });
                const data = await res.json();
                if (data.success) {
                    isEditing.value = false;
                    // Manually update local state to avoid flicker, though list refresh will happen
                    previewItem.value.category = editForm.category;
                    previewItem.value.tags = editForm.tags
                        .split(',')
                        .map((t) => t.trim())
                        .filter((t) => t);
                    previewItem.value.desc = editForm.desc;

                    fetchImages(currentPage.value);
                } else alert(data.error || '保存失败');
            } catch (e) {
                alert('保存出错: ' + e.message);
            }
        };

        const addEmotion = async () => {
            if (!newEmotion.key) return;
            addingEmotion.value = true;
            try {
                const newCat = { ...newEmotion };
                // Add to current list to send full list
                const currentList = [...availableEmotions.value];
                // Check if exists
                const existingIdx = currentList.findIndex((c) => c.key === newCat.key);
                if (existingIdx >= 0) {
                    if (!confirm(`分类 ${newCat.key} 已存在，确定要更新吗？`)) {
                        addingEmotion.value = false;
                        return;
                    }
                    currentList[existingIdx] = newCat;
                } else {
                    currentList.push(newCat);
                }

                const res = await apiFetch('api/categories', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ categories: currentList }),
                });
                const data = await res.json();

                if (data.success) {
                    fetchEmotions();
                    // Reset form
                    newEmotion.key = '';
                    newEmotion.name = '';
                    newEmotion.desc = '';
                } else {
                    alert(data.error || '添加失败');
                }
            } catch (e) {
                alert('操作出错: ' + e.message);
            } finally {
                addingEmotion.value = false;
            }
        };

        const deleteEmotion = async (cat) => {
            if (!cat?.key) return;
            if (!confirm(`确定要删除分类 ${cat.key} 吗？该分类下的图片会被直接删除且无法恢复。`))
                return;
            deletingEmotionKey.value = cat.key;
            try {
                const res = await apiFetch(`api/categories/${encodeURIComponent(cat.key)}`, {
                    method: 'DELETE',
                });
                const data = await res.json().catch(() => ({}));
                if (res.ok && data.success) {
                    if (selectedCategory.value === cat.key) selectedCategory.value = '';
                    if (editForm.category === cat.key) editForm.category = '';
                    if (previewItem.value && previewItem.value.category === cat.key)
                        previewItem.value.category = 'unknown';
                    fetchEmotions();
                    fetchImages(currentPage.value);
                    fetchStats();
                } else {
                    alert(data.error || '删除失败');
                }
            } catch (e) {
                alert('操作出错: ' + e.message);
            } finally {
                deletingEmotionKey.value = '';
            }
        };

        // Upload Logic
        const handleFileSelect = (e) => {
            const file = e.target.files[0];
            if (file && file.type.startsWith('image/')) {
                uploadFile.value = file;
                uploadPreviewUrl.value = URL.createObjectURL(file);
                uploadError.value = null;
            }
        };

        const submitUpload = async () => {
            if (!uploadFile.value) return;
            uploading.value = true;
            try {
                const formData = new FormData();
                formData.append('file', uploadFile.value);
                formData.append('emotion', uploadForm.emotion);
                formData.append('tags', uploadForm.tags);
                formData.append('desc', uploadForm.desc);

                const res = await apiFetch('api/images/upload', { method: 'POST', body: formData });
                const data = await res.json();
                if (data.success) {
                    closeUploadModal();
                    fetchImages(1);
                    fetchStats();
                } else {
                    uploadError.value = data.error || '上传失败';
                }
            } catch (e) {
                uploadError.value = '上传出错';
            } finally {
                uploading.value = false;
            }
        };

        onMounted(() => {
            initAuth();
            window.addEventListener('keydown', handleKeydown);
        });

        return {
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
            previewOpen,
            previewItem,
            openPreview,
            closePreview,
            prevImage,
            nextImage,
            uploadOpen,
            uploading,
            uploadFile,
            uploadPreviewUrl,
            uploadError,
            uploadForm,
            availableEmotions,
            openUploadModal,
            closeUploadModal,
            handleFileSelect,
            submitUpload,
            debouncedSearch,
            deleteImage,
            prevPage,
            nextPage,
            emotionsOpen,
            openEmotionsModal,
            closeEmotionsModal,
            fetchImages,
            newEmotion,
            addingEmotion,
            addEmotion,
            deletingEmotionKey,
            deleteEmotion,
            isEditing,
            editForm,
            startEdit,
            cancelEdit,
            saveEdit,
            isBatchMode,
            selectedImages,
            batchMoveOpen,
            batchTargetCategory,
            toggleBatchMode,
            toggleSelection,
            selectAll,
            handleBatchDelete,
            openBatchMoveModal,
            closeBatchMoveModal,
            confirmBatchMove,
            isAuthed,
            authRequired,
            authChecking,
            loginToken,
            loginError,
            showPassword,
            sessionTimeout,
            submitLogin,
            logout,
        };
    },
}).mount('#app');
