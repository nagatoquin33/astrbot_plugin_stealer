const { createApp, ref, reactive, onMounted } = Vue;

createApp({
    setup() {
        // State
        const images = ref([]);
        const categories = ref([]);
        const stats = reactive({ total: 0, categories: 0, today: 0 });
        const loading = ref(true);
        const searchQuery = ref('');
        const selectedCategory = ref('');
        const sortBy = ref('newest');
        const currentPage = ref(1);
        const pageSize = ref(30);
        const total = ref(0);

        // Preview Modal
        const previewOpen = ref(false);
        const previewItem = ref(null);
        const isEditing = ref(false);
        const editForm = reactive({ category: '', tags: '', scene: '', desc: '', scope_mode: 'public' });

        // Batch Mode
        const isBatchMode = ref(false);
        const selectedImages = ref(new Set());
        const batchMoveOpen = ref(false);
        const batchTargetCategory = ref('');
        const batchScopeOpen = ref(false);
        const batchScopeMode = ref('public');

        // Upload Modal
        const uploadOpen = ref(false);
        const uploading = ref(false);
        const uploadFile = ref(null);
        const uploadPreviewUrl = ref(null);
        const uploadError = ref(null);
        const uploadForm = reactive({ emotion: '', tags: '', scene: '', desc: '' });
        const availableEmotions = ref([]);
        const analysisScenes = ref([]);


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
                uploadForm.scene = sceneList.filter((item) => item !== scene).join('、');
                return;
            }
            uploadForm.scene = [...sceneList, scene].join('、');
        };

        const isSceneSelected = (scene) => parseSceneList(uploadForm.scene).includes(scene);

        const formatOriginTarget = (target) => {
            const raw = String(target || '').trim();
            if (!raw) return '未记录';
            if (raw.startsWith('group:')) return `群 ${raw.slice(6)}`;
            if (raw.startsWith('user:')) return `用户 ${raw.slice(5)}`;
            return raw;
        };

        const getScopeLabel = (scopeMode) => (
            String(scopeMode || 'public').toLowerCase() === 'local' ? '本群限定' : '公共'
        );

        // Upload Helpers

        // Category Management
        const emotionsOpen = ref(false);
        const newEmotion = reactive({ key: '', name: '', desc: '' });
        const addingEmotion = ref(false);
        const deletingEmotionKey = ref('');

        // Auth
        const isAuthed = ref(false);
        const authRequired = ref(false);
        const authChecking = ref(true);
        const loginToken = ref('');
        const loginError = ref('');
        const showPassword = ref(false);
        const sessionTimeout = ref(3600);

        const searchTimeout = ref(null);

        // Theme
        const isDarkTheme = ref(true);
        const theme = ref('dark');

        // API Helper
        const normalizeUrl = (url) => {
            const raw = String(url || '').trim();
            if (!raw) return '/';
            if (/^(?:[a-z]+:)?\/\//i.test(raw)) return raw;
            if (raw.startsWith('/')) return raw;
            return `/${raw.replace(/^\.?\//, '')}`;
        };

        const apiFetch = (url, options = {}) => {
            const requestUrl = normalizeUrl(url);
            const headers = new Headers(options.headers || {});
            return fetch(requestUrl, { ...options, headers, credentials: 'same-origin' })
                .then((res) => {
                    if (res.status === 401) {
                        isAuthed.value = false;
                        loginError.value = '会话已过期，请重新登录';
                    }
                    return res;
                })
                .catch(() =>
                    fetch(requestUrl, { ...options, headers, credentials: 'same-origin' })
                );
        };

        // Data Fetching
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
                    page: page.toString(),
                    size: pageSize.value.toString(),
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

        // Auth
        const initAuth = async () => {
            authChecking.value = true;
            loginError.value = '';
            try {
                const res = await fetch(normalizeUrl('auth/info'));
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

        // Search
        const debouncedSearch = () => {
            clearTimeout(searchTimeout.value);
            searchTimeout.value = setTimeout(() => fetchImages(1), 400);
        };

        // Pagination
        const prevPage = () => currentPage.value > 1 && fetchImages(currentPage.value - 1);
        const nextPage = () => currentPage.value * pageSize.value < total.value && fetchImages(currentPage.value + 1);

        // Preview Modal
        const openPreview = (img) => {
            previewItem.value = img;
            previewOpen.value = true;
        };

        const closePreview = () => {
            previewOpen.value = false;
            previewItem.value = null;
            isEditing.value = false;
        };

        const prevImage = () => {
            if (!previewItem.value) return;
            const idx = images.value.findIndex((i) => i.hash === previewItem.value.hash);
            if (idx > 0) previewItem.value = images.value[idx - 1];
        };

        const nextImage = () => {
            if (!previewItem.value) return;
            const idx = images.value.findIndex((i) => i.hash === previewItem.value.hash);
            if (idx < images.value.length - 1) previewItem.value = images.value[idx + 1];
        };

        const handleKeydown = (e) => {
            if (!previewOpen.value) return;
            if (e.key === 'ArrowLeft') prevImage();
            if (e.key === 'ArrowRight') nextImage();
            if (e.key === 'Escape') closePreview();
        };

        // Edit
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
                const res = await apiFetch(`api/images/${previewItem.value.hash}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(editForm),
                });
                const data = await res.json();
                if (data.success) {
                    isEditing.value = false;
                    previewItem.value.category = editForm.category;
                    previewItem.value.tags = editForm.tags.split(',').map((t) => t.trim()).filter((t) => t);
                    previewItem.value.scenes = parseSceneList(editForm.scene);
                    previewItem.value.desc = editForm.desc;
                    previewItem.value.scope_mode = editForm.scope_mode || 'public';
                    fetchImages(currentPage.value);
                } else {
                    alert(data.error || '保存失败');
                }
            } catch (e) {
                alert('保存出错: ' + e.message);
            }
        };

        // Delete
        const deleteImage = async (img, blacklist = false) => {
            const msg = blacklist
                ? '确定要删除并拉黑这张图片吗？\n拉黑后将不再自动收集此图片。'
                : '确定要删除这张图片吗？此操作无法撤销。';
            if (!confirm(msg)) return;
            try {
                const url = blacklist ? `api/images/${img.hash}?blacklist=true` : `api/images/${img.hash}`;
                const res = await apiFetch(url, { method: 'DELETE' });
                if (res.ok) {
                    closePreview();
                    fetchImages(currentPage.value);
                    fetchStats();
                } else {
                    alert('删除失败');
                }
            } catch (e) {
                alert('操作失败');
            }
        };

        // Batch Operations
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
                } else {
                    alert(data.error || '删除失败');
                }
            } catch (e) {
                alert('操作失败: ' + e.message);
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
                } else {
                    alert(data.error || '转移失败');
                }
            } catch (e) {
                alert('操作失败: ' + e.message);
            }
        };

        const confirmBatchScope = async () => {
            if (!batchScopeMode.value) return;
            try {
                const res = await apiFetch('api/images/batch/scope', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        hashes: Array.from(selectedImages.value),
                        scope_mode: batchScopeMode.value,
                    }),
                });
                const data = await res.json();
                if (data.success) {
                    batchScopeOpen.value = false;
                    selectedImages.value.clear();
                    isBatchMode.value = false;
                    await fetchImages(currentPage.value);
                    if (Number(data.skipped || 0) > 0) {
                        alert(`已更新 ${data.count || 0} 张，另有 ${data.skipped} 张缺少来源群信息，无法设为 local。`);
                    }
                } else {
                    alert(data.error || '作用域设置失败');
                }
            } catch (e) {
                alert('操作失败: ' + e.message);
            }
        };

        const toggleScope = async (img, scopeMode) => {
            if (!img) return;
            try {
                const res = await apiFetch(`api/images/${img.hash}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ scope_mode: scopeMode }),
                });
                const data = await res.json();
                if (data.success) {
                    if (previewItem.value && previewItem.value.hash === img.hash) {
                        previewItem.value.scope_mode = scopeMode;
                    }
                    await fetchImages(currentPage.value);
                } else if (data.error === 'Origin target missing') {
                    alert('该图片缺少来源群信息，无法设置为 local。');
                } else {
                    alert(data.error || '作用域更新失败');
                }
            } catch (e) {
                alert('操作失败: ' + e.message);
            }
        };

        // Upload
        const openUploadModal = () => {
            uploadOpen.value = true;
            uploadFile.value = null;
            uploadPreviewUrl.value = null;
            uploadError.value = null;
            Object.assign(uploadForm, { emotion: '', tags: '', scene: '', desc: '' });
            analysisScenes.value = [];
            fetchEmotions();
        };

        const closeUploadModal = () => {
            uploadOpen.value = false;
            analysisScenes.value = [];
        };

        const handleFileSelect = (e) => {
            const file = e.target.files[0];
            if (file && file.type.startsWith('image/')) {
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
                const formData = new FormData();
                formData.append('file', uploadFile.value);
                formData.append('emotion', uploadForm.emotion);
                formData.append('tags', uploadForm.tags);
                formData.append('scene', uploadForm.scene);
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

        /**
         * 图片智能分析功能 - 独立模块
         * 
         * 功能：调用 VLM 分析图片内容，自动提取分类、标签和描述
         * 使用场景：WebUI 上传表情包时自动填充元数据
         * 
         * 使用示例：
         * const analyzer = useImageAnalyzer();
         * const data = await analyzer.analyze(file);
         * analyzer.applyToForm(data, form, categories);
         */
        const useImageAnalyzer = () => {
            const isAnalyzing = ref(false);
            const lastAnalysisResult = ref(null);
            
            /**
             * 分析图片
             * @param {File} file - 图片文件
             * @returns {Promise<{category, tags, scenes, description}>} 分析结果
             */
            const analyze = async (file) => {
                if (!file) {
                    throw new Error('请先选择图片');
                }
                
                isAnalyzing.value = true;
                console.log('[Analyzer] 开始分析图片:', file.name);
                
                try {
                    const formData = new FormData();
                    formData.append('file', file);
                    
                    const res = await apiFetch('api/analyze', { method: 'POST', body: formData });
                    
                    if (res.status === 401) {
                        throw new Error('登录已过期，请重新登录');
                    }
                    
                    if (!res.ok) {
                        const errorText = await res.text();
                        throw new Error(`服务器错误 (${res.status}): ${errorText}`);
                    }
                    
                    const data = await res.json();
                    
                    if (!data.success) {
                        throw new Error(data.error || '分析失败');
                    }
                    
                    lastAnalysisResult.value = data;
                    console.log('[Analyzer] 分析成功:', data);
                    return data;
                } catch (e) {
                    console.error('[Analyzer] 分析失败:', e);
                    throw e;
                } finally {
                    isAnalyzing.value = false;
                }
            };
            
            /**
             * 将分析结果应用到表单
             * @param {Object} data - 分析结果
             * @param {Object} form - 表单对象
             * @param {Array} categories - 可用分类列表
             * @returns {Object} 填充结果统计
             */
            const applyToForm = (data, form, categories = []) => {
                const result = { filled: false, fields: [] };
                
                // 填充分类
                if (data.category) {
                    const exists = categories.some(e => e.key === data.category);
                    if (exists) {
                        form.emotion = data.category;
                        result.fields.push('category');
                    } else if (categories.length > 0) {
                        // 如果返回的分类不存在，使用第一个可用分类
                        console.warn('[Analyzer] 分类不存在，使用默认:', data.category);
                        form.emotion = categories[0].key;
                        result.fields.push('category');
                    }
                }
                
                // 填充标签（智能合并，避免重复）
                if (data.tags && data.tags.length > 0) {
                    const existingTags = form.tags ? form.tags.split(',').map(t => t.trim()).filter(t => t) : [];
                    const newTags = data.tags.filter(t => !existingTags.includes(t));
                    if (newTags.length > 0) {
                        form.tags = [...existingTags, ...newTags].join(', ');
                        result.fields.push('tags');
                    }
                }
                
                // 填充场景
                if (Array.isArray(data.scenes) && data.scenes.length > 0) {
                    form.scene = parseSceneList(data.scenes.join('、')).join('、');
                    result.fields.push('scenes');
                }

                // 填充描述（仅在空时填充）
                if (data.description && !form.desc) {
                    form.desc = data.description;
                    result.fields.push('desc');
                }
                
                result.filled = result.fields.length > 0;
                console.log('[Analyzer] 表单填充结果:', result);
                return result;
            };
            
            return {
                isAnalyzing,
                lastAnalysisResult,
                analyze,
                applyToForm,
            };
        };
        
        // 创建分析器实例
        const imageAnalyzer = useImageAnalyzer();
        const analyzing = imageAnalyzer.isAnalyzing;
        
        // 包装函数 - 用于上传模态框
        const analyzeImage = async () => {
            uploadError.value = null;
            
            try {
                const data = await imageAnalyzer.analyze(uploadFile.value);
                analysisScenes.value = Array.isArray(data.scenes) ? data.scenes : [];
                const result = imageAnalyzer.applyToForm(data, uploadForm, availableEmotions.value);
                
                if (!result.filled) {
                    uploadError.value = '未能识别有效信息';
                }
            } catch (e) {
                uploadError.value = e.message || '分析失败';
            }
        };

        // Category Management
        const openEmotionsModal = () => {
            emotionsOpen.value = true;
            fetchEmotions();
        };

        const closeEmotionsModal = () => {
            emotionsOpen.value = false;
        };

        const addEmotion = async () => {
            if (!newEmotion.key) return;
            addingEmotion.value = true;
            try {
                const newCat = { ...newEmotion };
                const currentList = [...availableEmotions.value];
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
                    newEmotion.key = '';
                    newEmotion.name = '';
                    newEmotion.desc = '';
                } else {
                    alert(data.error || '添加失败');
                }
            } catch (e) {
                alert('操作失败: ' + e.message);
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
                alert('操作失败: ' + e.message);
            } finally {
                deletingEmotionKey.value = '';
            }
        };

        // Utility
        const formatDate = (timestamp) => {
            if (!timestamp) return '未知';
            const date = new Date(timestamp * 1000);
            return date.toLocaleDateString('zh-CN', {
                year: 'numeric',
                month: '2-digit',
                day: '2-digit',
                hour: '2-digit',
                minute: '2-digit',
            });
        };

        // Theme functions
        const initTheme = () => {
            const savedTheme = localStorage.getItem('theme');
            if (savedTheme) {
                theme.value = savedTheme;
                isDarkTheme.value = savedTheme === 'dark';
            } else {
                // Check system preference
                const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
                theme.value = prefersDark ? 'dark' : 'light';
                isDarkTheme.value = prefersDark;
            }
            applyTheme();
        };

        const applyTheme = () => {
            document.documentElement.setAttribute('data-theme', theme.value);
            localStorage.setItem('theme', theme.value);
        };

        const toggleTheme = () => {
            // Create flash effect
            const flash = document.createElement('div');
            flash.className = 'theme-flash active';
            document.body.appendChild(flash);
            
            // Toggle theme
            isDarkTheme.value = !isDarkTheme.value;
            theme.value = isDarkTheme.value ? 'dark' : 'light';
            applyTheme();
            
            // Remove flash element after animation
            setTimeout(() => {
                flash.remove();
            }, 600);
        };

        onMounted(() => {
            initAuth();
            initTheme();
            window.addEventListener('keydown', handleKeydown);
        });

        return {
            // State
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
            
            // Preview
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
            
            // Batch
            isBatchMode,
            selectedImages,
            batchMoveOpen,
            batchTargetCategory,
            batchScopeOpen,
            batchScopeMode,
            toggleBatchMode,
            toggleSelection,
            selectAll,
            handleBatchDelete,
            openBatchMoveModal,
            closeBatchMoveModal,
            confirmBatchMove,
            openBatchScopeModal,
            closeBatchScopeModal,
            confirmBatchScope,
            
            // Upload
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
            
            // Auto Analysis (独立功能)
            analyzing,
            analyzeImage,
            
            // Category
            emotionsOpen,
            newEmotion,
            addingEmotion,
            deletingEmotionKey,
            openEmotionsModal,
            closeEmotionsModal,
            addEmotion,
            deleteEmotion,
            
            // Auth
            isAuthed,
            authRequired,
            authChecking,
            loginToken,
            loginError,
            showPassword,
            sessionTimeout,
            submitLogin,
            logout,

            // Theme
            isDarkTheme,
            theme,
            toggleTheme,

            // Actions
            fetchImages,
            debouncedSearch,
            deleteImage,
            toggleScope,
            prevPage,
            nextPage,
            formatDate,
            formatOriginTarget,
            getScopeLabel,
        };
    },
}).mount('#app');
