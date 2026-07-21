export const TEMPLATE = `
<header class="codex-header">
    <div class="header-title">
        <div class="header-icon">
            <svg style="width:28px;height:28px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"
                    d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
            </svg>
        </div>
        <div class="header-text">
            <h1>{{ t('pages.dashboard.header.brand', 'Henry\\'s Spoils') }}</h1>
            <p>{{ t('pages.dashboard.header.subtitle', 'Sticker Manager') }}</p>
        </div>
    </div>

    <div class="stats-bar">
        <div class="stat-item">
            <span class="stat-value">{{ stats.total || 0 }}</span>
            <span class="stat-label">{{ t('pages.dashboard.stats.total', 'Total') }}</span>
        </div>
        <div class="stat-item">
            <span class="stat-value">{{ stats.categories || 0 }}</span>
            <span class="stat-label">{{ t('pages.dashboard.stats.categories', 'Categories') }}</span>
        </div>
        <div class="stat-item">
            <span class="stat-value">{{ stats.today || 0 }}</span>
            <span class="stat-label">{{ t('pages.dashboard.stats.today', 'Today') }}</span>
        </div>
    </div>

    <div class="header-right">
        <div class="health-indicator" :class="healthStatus">
            <span class="health-dot"></span>
            <span class="health-text">{{ getHealthText(healthStatus) }}</span>
        </div>
    </div>
</header>

<div class="main-container">
    <aside class="sidebar">
        <div class="section-switcher">
            <div class="section-tab" :class="{ active: activeSection === 'pending' }" @click="switchSection('pending')">
                <svg class="section-tab-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"
                        d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
                </svg>
                <span class="section-tab-label">{{ t('pages.dashboard.sections.pending', 'Pending') }}</span>
                <span v-if="pendingStats.pending > 0" class="section-badge">{{ pendingStats.pending }}</span>
            </div>
            <div class="section-tab" :class="{ active: activeSection === 'library' }" @click="switchSection('library')">
                <svg class="section-tab-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"
                        d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                </svg>
                <span class="section-tab-label">{{ t('pages.dashboard.sections.library', 'Library') }}</span>
            </div>
        </div>

        <template v-if="activeSection === 'library'">
            <div class="sidebar-divider"></div>
            <div class="sidebar-title">{{ t('pages.dashboard.categories.title', 'Categories') }}</div>
            <div class="category-list">
                <div class="category-item favorite-category" :class="{ active: selectedCategory === '__favorite__' }"
                    @click="selectedCategory = '__favorite__'; fetchImages(1)">
                    <span class="category-icon">⭐</span>
                    <span class="category-name">{{ t('pages.dashboard.categories.favorites', 'Favorites') }}</span>
                    <span class="category-count">{{ favoriteCount }}</span>
                </div>
                <div class="category-item" :class="{ active: selectedCategory === '' }"
                    @click="selectedCategory = ''; fetchImages(1)">
                    <span class="category-name">{{ t('pages.dashboard.categories.all', 'All') }}</span>
                    <span class="category-count">{{ stats.total || 0 }}</span>
                </div>
                <div v-for="cat in categories" :key="cat.key" class="category-item"
                    :class="{ active: selectedCategory === cat.key }"
                    @click="selectedCategory = cat.key; fetchImages(1)">
                    <span class="category-name">{{ cat.name }}</span>
                    <span class="category-count">{{ cat.count }}</span>
                </div>
            </div>
        </template>

        <template v-if="activeSection === 'pending'">
            <div class="sidebar-divider"></div>
            <div class="pending-sidebar-stats">
                <div class="capacity-header">
                    <span class="capacity-label">{{ t('pages.dashboard.pending.pool', 'Pending Pool') }}</span>
                    <span class="capacity-count">{{ pendingStats.pending }}</span>
                </div>
                <div class="pending-capacity-bar">
                    <div class="capacity-fill"
                        :style="{ width: Math.min(100, pendingStats.pending / pendingStats.capacity * 100) + '%' }"
                        :class="{ full: pendingStats.paused }"></div>
                </div>
                <div class="capacity-sub">
                    <span>{{ t('pages.dashboard.pending.capacity', 'Capacity') }} {{ pendingStats.capacity }}</span>
                    <span v-if="pendingStats.paused" class="capacity-paused">{{ t('pages.dashboard.pending.paused', 'Paused') }}</span>
                </div>
            </div>
            <div class="sidebar-title">{{ t('pages.dashboard.pending.category_filter', 'Category Filter') }}</div>
            <div class="category-list">
                <div class="category-item" :class="{ active: pendingCategory === '' }"
                    @click="pendingCategory = ''; fetchPendingImages(1)">
                    <span class="category-name">{{ t('pages.dashboard.categories.all', 'All') }}</span>
                    <span class="category-count">{{ pendingTotal }}</span>
                </div>
                <div v-for="cat in pendingCategories" :key="cat.key" class="category-item"
                    :class="{ active: pendingCategory === cat.key }"
                    @click="pendingCategory = cat.key; fetchPendingImages(1)">
                    <span class="category-name">{{ cat.name }}</span>
                    <span class="category-count">{{ cat.count }}</span>
                </div>
            </div>
        </template>
    </aside>

    <main class="inventory-panel">
        <div class="modal-panel-corner-bl"></div>
        <div class="modal-panel-corner-br"></div>

        <template v-if="activeSection === 'library'">
            <div class="inventory-toolbar">
                <div class="toolbar-search">
                    <svg style="width:16px;height:16px;position:absolute;left:12px;top:50%;transform:translateY(-50%);color:var(--text-muted)"
                        fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                            d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                    <input v-model="searchQuery" @input="debouncedSearch"
                        :placeholder="t('pages.dashboard.search.library', 'Search stickers...')">
                </div>

                <div class="toolbar-actions">
                    <div class="toolbar-group mobile-category-select">
                        <select v-model="selectedCategory" @change="fetchImages(1)" class="codex-input">
                            <option value="">{{ t('pages.dashboard.categories.all', 'All') }}</option>
                            <option value="__favorite__">⭐ {{ t('pages.dashboard.categories.favorites', 'Favorites') }}</option>
                            <option v-for="cat in categories" :key="cat.key" :value="cat.key">{{ cat.name }}</option>
                        </select>
                    </div>

                    <div class="toolbar-group">
                        <select v-model="sortBy" @change="fetchImages(1)" class="codex-input toolbar-sort-select">
                            <option value="newest">{{ t('pages.dashboard.sort.newest', 'Newest') }}</option>
                            <option value="oldest">{{ t('pages.dashboard.sort.oldest', 'Oldest') }}</option>
                            <option value="most_used">{{ t('pages.dashboard.sort.most_used', 'Most Used') }}</option>
                            <option value="last_used">{{ t('pages.dashboard.sort.last_used', 'Last Used') }}</option>
                        </select>
                    </div>

                    <div class="toolbar-group">
                        <button @click="toggleBatchMode" class="codex-btn" :class="{ primary: isBatchMode }">
                            <svg style="width:16px;height:16px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                    d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                            </svg>
                            {{ isBatchMode ? t('pages.dashboard.actions.done', 'Done') : t('pages.dashboard.actions.batch', 'Batch') }}
                        </button>

                        <button @click="openEmotionsModal" class="codex-btn">
                            <svg style="width:16px;height:16px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                    d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
                            </svg>
                            {{ t('pages.dashboard.actions.categories', 'Categories') }}
                        </button>
                    </div>

                    <div class="toolbar-group">
                        <button @click="openUploadModal" class="codex-btn primary">
                            <svg style="width:16px;height:16px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                    d="M12 4v16m8-8H4" />
                            </svg>
                            {{ t('pages.dashboard.actions.add', 'Add') }}
                        </button>

                        <button @click="openBatchUploadModal" class="codex-btn">
                            <svg style="width:16px;height:16px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                    d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                            </svg>
                            {{ t('pages.dashboard.actions.batch_import', 'Batch Import') }}
                        </button>
                        <button @click="runStorageCleanup" class="codex-btn">
                            <svg style="width:16px;height:16px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                    d="M3 6h18M8 6V4h8v2m-6 4v7m4-7v7M6 6l1 14h10l1-14" />
                            </svg>
                            {{ t('pages.dashboard.actions.storage_cleanup', 'Storage Cleanup') }}
                        </button>
                    </div>
                </div>
            </div>

            <div v-if="loading" class="skeleton-grid">
                <div v-for="n in pageSize" :key="n" class="skeleton-card">
                    <div class="skeleton-image"></div>
                    <div class="skeleton-text"></div>
                </div>
            </div>

            <div v-else-if="images.length === 0" class="empty-state">
                <svg style="width:64px;height:64px;opacity:0.3;margin-bottom:16px" fill="none" stroke="currentColor"
                    viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"
                        d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                </svg>
                <p style="font-family:'Cinzel',serif;font-size:1.125rem">{{ t('pages.dashboard.empty.library_title', 'No stickers yet') }}</p>
                <p style="font-size:0.875rem;margin-top:8px;color:var(--text-muted)">{{ t('pages.dashboard.empty.library_hint', 'Click "Add" to upload a new sticker.') }}</p>
            </div>

            <div v-else class="inventory-grid">
                <div v-for="img in images" :key="img.hash" class="item-slot"
                    :class="{ selected: selectedImages.has(img.hash) }"
                    @click="isBatchMode ? toggleSelection(img) : openPreview(img)">
                    <div v-if="isBatchMode" class="batch-indicator">
                        <svg v-if="selectedImages.has(img.hash)" style="width:12px;height:12px" fill="none"
                            stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 13l4 4L19 7" />
                        </svg>
                    </div>

                    <button class="favorite-btn" :class="{ active: img.is_favorite }" @click.stop="toggleFavorite(img)"
                        :title="img.is_favorite ? t('pages.dashboard.actions.unfavorite', 'Remove favorite') : t('pages.dashboard.actions.favorite', 'Favorite')">
                        <svg viewBox="0 0 24 24">
                            <path
                                d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
                        </svg>
                    </button>

                    <div class="item-image" :data-hash="img.hash">
                        <div v-if="!imageDataUrls[img.hash]" class="image-placeholder"
                            :style="{ backgroundColor: hashToColor(img.hash) }"></div>
                        <img v-else :src="imageDataUrls[img.hash]" loading="lazy" :alt="img.desc" class="fade-in">
                    </div>

                    <div class="item-info">
                        <div class="item-category">{{ img.category }}</div>
                        <div class="item-meta-row">
                            <span class="scope-pill" :class="img.scope_mode === 'local' ? 'local' : 'public'">{{
                                getScopeLabel(img.scope_mode) }}</span>
                        </div>
                    </div>
                </div>
            </div>

            <div v-if="total > pageSize" class="pagination-bar">
                <button @click="prevPage" :disabled="currentPage === 1" class="codex-btn"
                    :class="{ disabled: currentPage === 1 }">
                    <svg style="width:16px;height:16px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
                    </svg>
                    {{ t('pages.dashboard.pagination.prev', 'Previous') }}
                </button>

                <span class="page-info">{{ currentPage }} / {{ Math.ceil(total / pageSize) }}</span>

                <button @click="nextPage" :disabled="currentPage * pageSize >= total" class="codex-btn"
                    :class="{ disabled: currentPage * pageSize >= total }">
                    {{ t('pages.dashboard.pagination.next', 'Next') }}
                    <svg style="width:16px;height:16px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
                    </svg>
                </button>
            </div>
        </template>

        <template v-if="activeSection === 'pending'">
            <div class="pending-progress">
                <div class="progress-track">
                    <div class="progress-fill"
                        :style="{ width: Math.min(100, pendingStats.pending / pendingStats.capacity * 100) + '%' }"
                        :class="{ full: pendingStats.paused }"></div>
                </div>
                <div class="progress-info">
                    <span>{{ t('pages.dashboard.pending.progress', 'Pending') }} {{ pendingStats.pending }} / {{ pendingStats.capacity }}</span>
                    <span v-if="pendingStats.paused" class="progress-paused-label">{{ t('pages.dashboard.pending.paused_hint', 'Stealing is paused and will resume after review.') }}</span>
                </div>
            </div>

            <div class="inventory-toolbar">
                <div class="toolbar-search">
                    <svg style="width:16px;height:16px;position:absolute;left:12px;top:50%;transform:translateY(-50%);color:var(--text-muted)"
                        fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                            d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                    <input v-model="pendingSearchQuery" @input="pendingDebouncedSearch"
                        :placeholder="t('pages.dashboard.search.pending', 'Search pending stickers...')">
                </div>

                <div class="toolbar-actions">
                    <div class="toolbar-group">
                        <button @click="togglePendingBatchMode" class="codex-btn"
                            :class="{ primary: pendingBatchMode }">
                            <svg style="width:16px;height:16px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                    d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                            </svg>
                            {{ pendingBatchMode ? t('pages.dashboard.actions.done', 'Done') : t('pages.dashboard.actions.batch', 'Batch') }}
                        </button>
                    </div>

                    <div v-if="pendingBatchMode" class="toolbar-group pending-batch-actions">
                        <button @click="approvePendingBatch" class="codex-btn approve-batch-btn">✅ {{ t('pages.dashboard.actions.approve_all', 'Approve All') }}</button>
                        <button @click="rejectPendingBatch(false)" class="codex-btn reject-batch-btn">🗑 {{ t('pages.dashboard.actions.delete_all', 'Delete All') }}</button>
                        <button @click="rejectPendingBatch(true)" class="codex-btn reject-bl-batch-btn">🚫 {{ t('pages.dashboard.actions.delete_blacklist', 'Delete + Blacklist') }}</button>
                    </div>
                </div>
            </div>

            <div v-if="pendingLoading" class="skeleton-grid">
                <div v-for="n in pendingPageSize" :key="n" class="skeleton-card">
                    <div class="skeleton-image"></div>
                    <div class="skeleton-text"></div>
                </div>
            </div>

            <div v-else-if="pendingImages.length === 0" class="empty-state">
                <svg style="width:64px;height:64px;opacity:0.3;margin-bottom:16px" fill="none" stroke="currentColor"
                    viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M5 13l4 4L19 7" />
                </svg>
                <p style="font-family:'Cinzel',serif;font-size:1.125rem">{{ t('pages.dashboard.empty.pending_title', 'No pending stickers') }}</p>
                <p style="font-size:0.875rem;margin-top:8px;color:var(--text-muted)">{{ t('pages.dashboard.empty.pending_hint', 'Newly stolen stickers will wait here for review.') }}</p>
            </div>

            <div v-else class="pending-grid">
                <div v-for="item in pendingImages" :key="item.id" class="pending-card"
                    :class="{ selected: pendingSelectedImages.has(item.id) }"
                    @click="pendingBatchMode ? togglePendingSelection(item) : null">
                    <div v-if="pendingBatchMode" class="batch-indicator">
                        <svg v-if="pendingSelectedImages.has(item.id)" style="width:12px;height:12px" fill="none"
                            stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 13l4 4L19 7" />
                        </svg>
                    </div>

                    <div class="pending-image">
                        <div v-if="!imageDataUrls[item.hash]" class="image-placeholder"
                            :style="{ backgroundColor: hashToColor(item.hash) }"></div>
                        <img v-else :src="imageDataUrls[item.hash]" loading="lazy" :alt="item.desc" class="fade-in">
                    </div>

                    <div class="pending-info">
                        <div class="pending-meta">
                            <span class="pending-category-badge">{{ item.category }}</span>
                            <span v-if="item.scope_mode === 'local'" class="scope-pill local">{{ t('pages.dashboard.scope.local_short', 'Local') }}</span>
                            <span class="pending-source">{{ item.source === 'auto' ? '🤖' : '👤' }}</span>
                        </div>
                        <div class="pending-desc">{{ item.desc || t('pages.dashboard.messages.no_description', 'No description') }}</div>
                        <div class="pending-tags" v-if="(item.tags || []).length">
                            <span v-for="tag in item.tags" :key="tag" class="tag pending-tag">{{ tag }}</span>
                        </div>
                        <div class="pending-actions" v-if="!pendingBatchMode">
                            <button @click.stop="approvePending(item.id)" class="pending-btn approve-btn"
                                :title="t('pages.dashboard.actions.approve', 'Approve')">
                                <svg style="width:14px;height:14px" fill="none" stroke="currentColor"
                                    viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5"
                                        d="M5 13l4 4L19 7" />
                                </svg>
                                {{ t('pages.dashboard.actions.approve', 'Approve') }}
                            </button>
                            <button @click.stop="openPendingEdit(item)" class="pending-btn edit-btn"
                                :title="t('pages.dashboard.actions.edit_approve', 'Edit & approve')">
                                <svg style="width:14px;height:14px" fill="none" stroke="currentColor"
                                    viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                        d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                                </svg>
                                {{ t('pages.dashboard.actions.edit', 'Edit') }}
                            </button>
                            <button @click.stop="rejectPending(item.id)" class="pending-btn reject-btn"
                                :title="t('pages.dashboard.actions.delete', 'Delete')">
                                <svg style="width:14px;height:14px" fill="none" stroke="currentColor"
                                    viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                        d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                </svg>
                                {{ t('pages.dashboard.actions.delete', 'Delete') }}
                            </button>
                            <button @click.stop="rejectPending(item.id, true)" class="pending-btn reject-bl-btn"
                                :title="t('pages.dashboard.actions.blacklist', 'Blacklist')">
                                <svg style="width:14px;height:14px" fill="none" stroke="currentColor"
                                    viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                        d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" />
                                </svg>
                                {{ t('pages.dashboard.actions.blacklist', 'Blacklist') }}
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <div v-if="pendingTotal > pendingPageSize" class="pagination-bar">
                <button @click="pendingCurrentPage > 1 && fetchPendingImages(pendingCurrentPage - 1)"
                    :disabled="pendingCurrentPage === 1" class="codex-btn"
                    :class="{ disabled: pendingCurrentPage === 1 }">
                    <svg style="width:16px;height:16px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
                    </svg>
                    {{ t('pages.dashboard.pagination.prev', 'Previous') }}
                </button>

                <span class="page-info">{{ pendingCurrentPage }} / {{ Math.ceil(pendingTotal / pendingPageSize) }}</span>

                <button
                    @click="pendingCurrentPage * pendingPageSize < pendingTotal && fetchPendingImages(pendingCurrentPage + 1)"
                    :disabled="pendingCurrentPage * pendingPageSize >= pendingTotal" class="codex-btn"
                    :class="{ disabled: pendingCurrentPage * pendingPageSize >= pendingTotal }">
                    {{ t('pages.dashboard.pagination.next', 'Next') }}
                    <svg style="width:16px;height:16px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
                    </svg>
                </button>
            </div>
        </template>
    </main>
</div>

<div v-if="previewOpen" class="modal-overlay" @click.self="closePreview">
    <div class="modal-panel">
        <div class="modal-panel-corner-bl"></div>
        <div class="modal-panel-corner-br"></div>

        <div class="modal-header">
            <h2>{{ isEditing ? t('pages.dashboard.modal.edit', 'Edit') : t('pages.dashboard.modal.details', 'Details') }}</h2>
            <button @click="closePreview" class="modal-close">
                <svg style="width:20px;height:20px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
            </button>
        </div>

        <div class="modal-content">
            <div v-if="!isEditing" class="item-detail">
                <div class="item-preview">
                    <button v-if="images.length > 1" @click.stop="prevImage" class="nav-btn left">
                        <svg style="width:24px;height:24px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
                        </svg>
                    </button>

                    <img :src="originalDataUrls[previewItem?.hash] || imageDataUrls[previewItem?.hash] || PLACEHOLDER"
                        :alt="previewItem?.desc">

                    <button v-if="images.length > 1" @click.stop="nextImage" class="nav-btn right">
                        <svg style="width:24px;height:24px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
                        </svg>
                    </button>
                </div>

                <div class="item-stats">
                    <div class="stat-row">
                        <span class="stat-name">{{ t('pages.dashboard.fields.category', 'Category') }}</span>
                        <span class="stat-value">{{ previewItem?.category }}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-name">{{ t('pages.dashboard.fields.scope', 'Scope') }}</span>
                        <span class="stat-value">
                            <span class="scope-pill"
                                :class="previewItem?.scope_mode === 'local' ? 'local' : 'public'">{{
                                getScopeLabel(previewItem?.scope_mode) }}</span>
                        </span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-name">{{ t('pages.dashboard.fields.use_count', 'Use Count') }}</span>
                        <span class="stat-value">{{ previewItem?.use_count || 0 }} {{ t('pages.dashboard.units.times', 'times') }}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-name">{{ t('pages.dashboard.fields.last_used', 'Last Used') }}</span>
                        <span class="stat-value">{{ previewItem?.last_used_at ? formatDate(previewItem.last_used_at) : t('pages.dashboard.messages.never_used', 'Never used') }}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-name">{{ t('pages.dashboard.fields.favorite', 'Favorite') }}</span>
                        <button class="favorite-toggle-btn" :class="{ active: previewItem?.is_favorite }"
                            @click="toggleFavorite(previewItem)">
                            {{ previewItem?.is_favorite ? t('pages.dashboard.messages.favorited', 'Favorited') : t('pages.dashboard.messages.not_favorited', 'Not favorited') }}
                        </button>
                    </div>
                    <div class="stat-row">
                        <span class="stat-name">{{ t('pages.dashboard.fields.origin', 'Origin') }}</span>
                        <span class="stat-value">{{ formatOriginTarget(previewItem?.origin_target) }}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-name">{{ t('pages.dashboard.fields.description', 'Description') }}</span>
                    </div>
                    <div style="padding:12px;background:rgba(0,0,0,0.3);margin-bottom:12px;border-left:3px solid var(--gold-dim)">
                        <p style="margin:0;color:var(--text-main);font-style:italic">
                            {{ previewItem?.desc || t('pages.dashboard.messages.no_description', 'No description') }}
                        </p>
                    </div>
                    <div class="stat-row">
                        <span class="stat-name">{{ t('pages.dashboard.fields.tags', 'Tags') }}</span>
                    </div>
                    <div class="item-tags" style="margin-bottom:12px">
                        <span v-for="tag in (previewItem?.tags || [])" :key="tag" class="tag">
                            {{ tag }}
                        </span>
                        <span v-if="!(previewItem?.tags || []).length"
                            style="font-size:0.85rem;color:var(--text-muted)">{{ t('pages.dashboard.messages.no_tags', 'No tags') }}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-name">{{ t('pages.dashboard.fields.scenes', 'Scenes') }}</span>
                    </div>
                    <div class="item-tags" style="margin-bottom:12px">
                        <span v-for="scene in (previewItem?.scenes || [])" :key="scene" class="tag scene-tag">
                            {{ scene }}
                        </span>
                        <span v-if="!(previewItem?.scenes || []).length"
                            style="font-size:0.85rem;color:var(--text-muted)">{{ t('pages.dashboard.messages.no_scenes', 'No scenes') }}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-name">{{ t('pages.dashboard.fields.created_at', 'Added At') }}</span>
                        <span class="stat-value">{{ formatDate(previewItem?.created_at) }}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-name">ID</span>
                        <span class="stat-value" style="font-size:0.75rem;word-break:break-all">{{ previewItem?.hash?.slice(0, 16) }}...</span>
                    </div>
                </div>
            </div>

            <div v-else style="padding:24px;width:100%">
                <div style="max-width:500px;margin:0 auto">
                    <div style="margin-bottom:20px">
                        <label
                            style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--text-muted);display:block;margin-bottom:8px">{{ t('pages.dashboard.fields.category', 'Category') }}</label>
                        <select v-model="editForm.category" class="codex-input">
                            <option v-for="cat in categories" :key="cat.key" :value="cat.key">{{ cat.name }}</option>
                        </select>
                    </div>

                    <div style="margin-bottom:20px">
                        <label
                            style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--text-muted);display:block;margin-bottom:8px">{{ t('pages.dashboard.fields.scope', 'Scope') }}</label>
                        <select v-model="editForm.scope_mode" class="codex-input">
                            <option value="public">public / {{ t('pages.dashboard.scope.public', 'Public') }}</option>
                            <option value="local">local / {{ t('pages.dashboard.scope.local', 'Local only') }}</option>
                        </select>
                        <div class="form-hint">{{ t('pages.dashboard.fields.origin', 'Origin') }}: {{ formatOriginTarget(previewItem?.origin_target) }}</div>
                    </div>

                    <div style="margin-bottom:20px">
                        <label
                            style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--text-muted);display:block;margin-bottom:8px">{{ t('pages.dashboard.fields.description', 'Description') }}</label>
                        <textarea v-model="editForm.desc" class="codex-input" rows="3"></textarea>
                    </div>

                    <div style="margin-bottom:20px">
                        <label
                            style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--text-muted);display:block;margin-bottom:8px">{{ t('pages.dashboard.fields.scenes', 'Scenes') }} ({{ t('pages.dashboard.messages.scene_separator_hint', 'comma separated') }})</label>
                        <input v-model="editForm.scene" type="text" class="codex-input"
                            :placeholder="t('pages.dashboard.placeholders.edit_scene', 'Example: celebration, happy')">
                    </div>

                    <div style="margin-bottom:20px">
                        <label
                            style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--text-muted);display:block;margin-bottom:8px">{{ t('pages.dashboard.fields.tags', 'Tags') }} ({{ t('pages.dashboard.messages.tag_separator_hint', 'comma separated') }})</label>
                        <input v-model="editForm.tags" type="text" class="codex-input"
                            :placeholder="t('pages.dashboard.placeholders.edit_tags', 'Example: cute, funny, rare')">
                    </div>
                </div>
            </div>
        </div>

        <div class="modal-actions">
            <template v-if="!isEditing">
                <a href="#" @click.prevent="downloadImage(previewItem)" class="codex-btn" style="flex:1">
                    <svg style="width:16px;height:16px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                            d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                    </svg>
                    {{ t('pages.dashboard.actions.download', 'Download') }}
                </a>
                <button @click="startEdit" class="codex-btn" style="flex:1">
                    <svg style="width:16px;height:16px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                            d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                    </svg>
                    {{ t('pages.dashboard.actions.edit', 'Edit') }}
                </button>
                <button @click="toggleScope(previewItem, previewItem?.scope_mode === 'local' ? 'public' : 'local')"
                    class="codex-btn" style="flex:1">
                    {{ previewItem?.scope_mode === 'local' ? t('pages.dashboard.actions.unset_local', 'Unset Local') : t('pages.dashboard.actions.set_local', 'Set Local') }}
                </button>
                <button @click="deleteImage(previewItem)" class="codex-btn danger" style="flex:1">
                    <svg style="width:16px;height:16px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                            d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                    {{ t('pages.dashboard.actions.delete', 'Delete') }}
                </button>
                <button @click="deleteImage(previewItem, true)" class="codex-btn danger" style="flex:1"
                    :title="t('pages.dashboard.actions.blacklist', 'Blacklist')">
                    <svg style="width:16px;height:16px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                            d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" />
                    </svg>
                    {{ t('pages.dashboard.actions.blacklist', 'Blacklist') }}
                </button>
            </template>
            <template v-else>
                <button @click="cancelEdit" class="codex-btn" style="flex:1">{{ t('pages.dashboard.actions.cancel', 'Cancel') }}</button>
                <button @click="saveEdit" class="codex-btn primary" style="flex:1">{{ t('pages.dashboard.actions.save', 'Save') }}</button>
            </template>
        </div>
    </div>
</div>

<div v-if="uploadOpen" class="modal-overlay" @click.self="closeUploadModal">
    <div class="modal-panel" style="max-width:600px">
        <div class="modal-panel-corner-bl"></div>
        <div class="modal-panel-corner-br"></div>

        <div class="modal-header">
            <h2>{{ t('pages.dashboard.modal.add_sticker', 'Add Sticker') }}</h2>
            <button @click="closeUploadModal" class="modal-close">
                <svg style="width:20px;height:20px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
            </button>
        </div>

        <form @submit.prevent="submitUpload" style="padding:24px">
            <div class="upload-area" @click="$refs.fileInput.click()">
                <input ref="fileInput" type="file" accept="image/*" @change="handleFileSelect" style="display:none">

                <div v-if="uploadPreviewUrl" class="upload-preview-row">
                    <img :src="uploadPreviewUrl" class="upload-preview">
                    <div class="upload-preview-info">
                        <p class="upload-preview-name">{{ uploadFile?.name }}</p>
                        <p class="upload-preview-size">{{ (uploadFile?.size / 1024).toFixed(1) }} KB</p>
                    </div>
                </div>

                <div v-else>
                    <svg style="width:48px;height:48px;margin:0 auto 16px auto;color:var(--gold-dim);opacity:0.5;display:block"
                        fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"
                            d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    <p style="margin:0;color:var(--text-muted);font-family:'Cinzel',serif;text-align:center">{{ t('pages.dashboard.upload.click_to_upload', 'Click to upload an image') }}</p>
                </div>
            </div>

            <div style="margin-top:20px">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                    <label
                        style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--text-muted)">{{ t('pages.dashboard.fields.category', 'Category') }} *</label>
                    <button v-if="uploadFile" type="button" @click.prevent="analyzeImage"
                        :disabled="analyzing || !uploadFile" class="codex-btn"
                        style="font-size:0.7rem;padding:6px 12px;min-height:auto">
                        <svg v-if="!analyzing" style="width:14px;height:14px" fill="none" stroke="currentColor"
                            viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                d="M13 10V3L4 14h7v7l9-11h-7z" />
                        </svg>
                        <svg v-else style="width:14px;height:14px;animation:spin 1s linear infinite" fill="none"
                            viewBox="0 0 24 24">
                            <circle style="opacity:0.25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                            <path style="opacity:0.75" fill="currentColor"
                                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        <span v-if="analyzing">{{ t('pages.dashboard.actions.analyzing', 'Analyzing...') }}</span>
                        <span v-else>{{ t('pages.dashboard.actions.auto_analyze', 'Auto Analyze') }}</span>
                    </button>
                </div>
                <select v-model="uploadForm.emotion" class="codex-input" required>
                    <option value="">{{ t('pages.dashboard.placeholders.select_category', 'Select a category...') }}</option>
                    <option v-for="emo in availableEmotions" :key="emo.key" :value="emo.key">{{ emo.name || emo.key }}</option>
                </select>
            </div>

            <div v-if="analysisScenes.length" class="analysis-result" style="margin-top:16px">
                <div class="analysis-result-head">
                    <div class="analysis-result-title">{{ t('pages.dashboard.analysis.scenes_title', 'Detected scenes') }}</div>
                    <div class="analysis-result-subtitle">{{ t('pages.dashboard.analysis.scenes_hint', 'Click a tag to add or remove it from the scene field.') }}</div>
                </div>
                <div class="item-tags" style="margin-top:10px">
                    <button v-for="scene in analysisScenes" :key="scene" type="button"
                        class="tag scene-tag scene-tag-btn" :class="{ active: isSceneSelected(scene) }"
                        @click="toggleScene(scene)">
                        {{ scene }}
                    </button>
                </div>
            </div>

            <div style="margin-top:16px">
                <label
                    style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--text-muted);display:block;margin-bottom:8px">{{ t('pages.dashboard.fields.scenes', 'Scenes') }}</label>
                <input v-model="uploadForm.scene" type="text" class="codex-input"
                    :placeholder="t('pages.dashboard.placeholders.upload_scene', 'Example: office, chat window, late night')">
                <p style="margin:8px 0 0;font-size:0.8rem;color:var(--text-muted)">{{ t('pages.dashboard.messages.scene_input_hint', 'You can separate scenes with commas or semicolons.') }}</p>
            </div>

            <div style="margin-top:16px">
                <label
                    style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--text-muted);display:block;margin-bottom:8px">{{ t('pages.dashboard.fields.tags', 'Tags') }}</label>
                <input v-model="uploadForm.tags" type="text" class="codex-input"
                    :placeholder="t('pages.dashboard.placeholders.upload_tags', 'Example: cute, funny')">
            </div>

            <div style="margin-top:16px">
                <label
                    style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--text-muted);display:block;margin-bottom:8px">{{ t('pages.dashboard.fields.description', 'Description') }}</label>
                <textarea v-model="uploadForm.desc" class="codex-input" rows="2"
                    :placeholder="t('pages.dashboard.placeholders.upload_desc', 'Describe this sticker...')"></textarea>
            </div>

            <div v-if="uploadError"
                style="color:#ef4444;font-size:0.875rem;border:1px solid rgba(239,68,68,0.3);background:rgba(239,68,68,0.1);padding:12px;margin-top:16px">
                {{ uploadError }}
            </div>

            <div class="modal-footer-actions">
                <button type="button" @click="closeUploadModal" class="codex-btn" style="flex:1">{{ t('pages.dashboard.actions.cancel', 'Cancel') }}</button>
                <button type="submit" :disabled="uploading || !uploadFile" class="codex-btn primary" style="flex:1">
                    <span v-if="uploading">{{ t('pages.dashboard.actions.uploading', 'Uploading...') }}</span>
                    <span v-else>{{ t('pages.dashboard.actions.confirm_add', 'Confirm Add') }}</span>
                </button>
            </div>
        </form>
    </div>
</div>

<div v-if="batchUploadOpen" class="modal-overlay" @click.self="closeBatchUploadModal">
    <div class="modal-panel" style="max-width:700px">
        <div class="modal-panel-corner-bl"></div>
        <div class="modal-panel-corner-br"></div>

        <div class="modal-header">
            <h2>{{ t('pages.dashboard.modal.batch_import', 'Batch Import Stickers') }}</h2>
            <button @click="closeBatchUploadModal" class="modal-close">
                <svg style="width:20px;height:20px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
            </button>
        </div>

        <form @submit.prevent="submitBatchUpload" style="padding:24px">
            <div v-if="!batchTaskId">
                <div class="upload-area batch-upload-area" :class="{ 'is-drag-active': batchDragActive }"
                    @click="triggerBatchFileInput"
                    @dragenter="onBatchDragEnter"
                    @dragover="onBatchDragOver"
                    @dragleave="onBatchDragLeave"
                    @drop="onBatchDrop"
                    style="min-height:150px">
                    <input v-if="!batchFolderMode" ref="batchFileInput" type="file" accept="image/*" multiple
                        @change="handleBatchFileSelect" class="native-file-input">
                    <input v-else ref="batchFolderInput" type="file" accept="image/*" webkitdirectory
                        @change="handleBatchFileSelect" class="native-file-input">

                    <div style="display:flex;align-items:center;gap:6px;margin-bottom:6px" @click.stop>
                        <label
                            style="font-size:12px;color:var(--text-muted);cursor:pointer;display:flex;align-items:center;gap:4px">
                            <input type="checkbox" v-model="batchFolderMode" style="accent-color:var(--gold-primary)">
                            {{ t('pages.dashboard.batch.include_subfolders', 'Include subfolders') }}
                        </label>
                    </div>

                    <div v-if="batchFiles.length">
                        <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px">
                            <svg style="width:32px;height:32px;color:var(--gold-primary)" fill="none"
                                stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"
                                    d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                            </svg>
                            <div>
                                <p style="margin:0 0 4px 0;color:var(--gold-primary);font-family:'Cinzel',serif">{{ t('pages.dashboard.batch.selected_count', 'Selected {count} image(s)').replace('{count}', batchFiles.length) }}</p>
                                <p style="margin:0;color:var(--text-muted);font-size:0.85rem">{{ formatBatchSize() }}</p>
                            </div>
                        </div>
                        <div class="batch-file-list">
                            <div v-for="(file, idx) in batchFiles.slice(0, 8)" :key="idx" class="batch-file-item">
                                <img v-if="batchPreviews[idx]" :src="batchPreviews[idx]" class="batch-file-thumb">
                                <span class="batch-file-name">{{ file.name }}</span>
                            </div>
                            <div v-if="batchFiles.length > 8" class="batch-file-more">
                                {{ t('pages.dashboard.batch.more_count', '{count} more...').replace('{count}', batchFiles.length - 8) }}
                            </div>
                        </div>
                        <button type="button" @click.stop="clearBatchFiles" class="codex-btn"
                            style="margin-top:12px;font-size:0.8rem;padding:6px 12px">
                            {{ t('pages.dashboard.actions.clear_selection', 'Clear Selection') }}
                        </button>
                    </div>

                    <div v-else>
                        <svg style="width:48px;height:48px;margin:0 auto 16px auto;color:var(--gold-dim);opacity:0.5;display:block"
                            fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"
                                d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                        </svg>
                        <p style="margin:0;color:var(--text-muted);font-family:'Cinzel',serif;text-align:center">{{ t('pages.dashboard.batch.drag_upload', 'Click or drag to upload multiple images') }}</p>
                        <p style="margin:8px 0 0;color:var(--text-muted);font-size:0.85rem;text-align:center">{{ t('pages.dashboard.batch.supported_formats', 'Supports PNG, JPG, GIF, WEBP, BMP') }}</p>
                    </div>
                </div>

                <div style="margin-top:20px">
                    <label
                        style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--text-muted);display:block;margin-bottom:8px">{{ t('pages.dashboard.batch.default_category', 'Default Category') }} *</label>
                    <select v-model="batchUploadForm.emotion" class="codex-input"
                        :disabled="batchUploadForm.autoAnalyze" required>
                        <option value="">{{ t('pages.dashboard.placeholders.select_category', 'Select a category...') }}</option>
                        <option v-for="emo in availableEmotions" :key="emo.key" :value="emo.key">{{ emo.name || emo.key }}</option>
                    </select>
                    <p style="margin:8px 0 0;font-size:0.8rem;color:var(--text-muted)">{{ t('pages.dashboard.batch.default_category_hint', 'Images will be saved into this category unless auto analyze is enabled.') }}</p>
                </div>

                <div style="margin-top:16px">
                    <label style="display:flex;align-items:center;gap:8px;cursor:pointer">
                        <input type="checkbox" v-model="batchUploadForm.autoAnalyze" class="codex-checkbox"
                            :disabled="batchUploadForm.emotion !== ''">
                        <span style="font-size:0.85rem;color:var(--text-main)">{{ t('pages.dashboard.batch.auto_analyze', 'Auto analyze each image and classify automatically') }}</span>
                    </label>
                    <p v-if="batchUploadForm.emotion !== ''"
                        style="margin:4px 0 0 24px;font-size:0.75rem;color:var(--gold-dim)">{{ t('pages.dashboard.batch.auto_analyze_disabled_hint', 'Clear the selected category before enabling auto analyze.') }}</p>
                    <p v-if="batchUploadForm.autoAnalyze"
                        style="margin:8px 0 0 24px;font-size:0.75rem;color:#f59e0b;padding:8px;background:rgba(245,158,11,0.1);border-radius:4px">
                        {{ t('pages.dashboard.batch.auto_analyze_warning', 'Auto analyze will call the VLM service concurrently. Make sure your API supports concurrency or upload in smaller batches.') }}
                    </p>
                </div>

                <div v-if="batchUploadError"
                    style="color:#ef4444;font-size:0.875rem;border:1px solid rgba(239,68,68,0.3);background:rgba(239,68,68,0.1);padding:12px;margin-top:16px">
                    {{ batchUploadError }}
                </div>

                <div class="modal-footer-actions">
                    <button type="button" @click="closeBatchUploadModal" class="codex-btn" style="flex:1">{{ t('pages.dashboard.actions.cancel', 'Cancel') }}</button>
                    <button type="submit" :disabled="batchUploading || batchFiles.length === 0"
                        class="codex-btn primary" style="flex:1">
                        <span v-if="batchUploading">{{ t('pages.dashboard.actions.uploading', 'Uploading...') }}</span>
                        <span v-else>{{ t('pages.dashboard.batch.start_import', 'Start Import ({count})').replace('{count}', batchFiles.length) }}</span>
                    </button>
                </div>
            </div>

            <div v-else style="padding:24px">
                <div style="text-align:center;margin-bottom:24px">
                    <div v-if="batchTaskStatus === 'processing'" class="batch-spinner">
                        <svg style="width:48px;height:48px;animation:spin 1s linear infinite;color:var(--gold-primary)"
                            fill="none" viewBox="0 0 24 24">
                            <circle style="opacity:0.25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                            <path style="opacity:0.75" fill="currentColor"
                                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                    </div>
                    <div v-else-if="batchTaskStatus === 'completed'" style="color:#22c55e">
                        <svg style="width:48px;height:48px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
                        </svg>
                    </div>
                    <div v-else-if="batchTaskStatus === 'failed'" style="color:#ef4444">
                        <svg style="width:48px;height:48px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </div>

                    <h3 style="margin:16px 0 8px;font-size:1.2rem;color:var(--text-main)">
                        <span v-if="batchTaskStatus === 'processing'">{{ t('pages.dashboard.batch.processing', 'Processing...') }}</span>
                        <span v-else-if="batchTaskStatus === 'completed'">{{ t('pages.dashboard.batch.completed', 'Import complete') }}</span>
                        <span v-else-if="batchTaskStatus === 'failed'">{{ t('pages.dashboard.batch.failed', 'Import failed') }}</span>
                    </h3>

                    <p style="margin:0;color:var(--text-muted);font-size:0.9rem">
                        {{ batchTaskProcessed }} / {{ batchTaskTotal }}
                        <span v-if="batchTaskSuccess > 0" style="color:#22c55e">({{ batchTaskSuccess }} {{ t('pages.dashboard.batch.success', 'success') }})</span>
                        <span v-if="batchTaskFailed > 0" style="color:#ef4444">({{ batchTaskFailed }} {{ t('pages.dashboard.batch.failed_count', 'failed') }})</span>
                    </p>
                </div>

                <div v-if="batchTaskStatus === 'processing'" style="margin-bottom:16px">
                    <div class="progress-bar">
                        <div class="progress-fill"
                            :style="{ width: (batchTaskProcessed / batchTaskTotal * 100) + '%' }"></div>
                    </div>
                </div>

                <div v-if="batchUploadError && batchTaskStatus === 'failed'"
                    style="color:#ef4444;font-size:0.875rem;text-align:center;margin-bottom:16px">
                    {{ batchUploadError }}
                </div>

                <div v-if="batchTaskStatus === 'completed'" style="display:flex;gap:12px">
                    <button type="button" @click="resetBatchUpload" class="codex-btn" style="flex:1">{{ t('pages.dashboard.batch.continue_import', 'Continue Import') }}</button>
                    <button type="button" @click="closeBatchUploadModal" class="codex-btn primary"
                        style="flex:1">{{ t('pages.dashboard.actions.done', 'Done') }}</button>
                </div>
                <div v-else-if="batchTaskStatus === 'failed'" style="display:flex;gap:12px">
                    <button type="button" @click="resetBatchUpload" class="codex-btn" style="flex:1">{{ t('pages.dashboard.actions.retry', 'Retry') }}</button>
                    <button type="button" @click="closeBatchUploadModal" class="codex-btn" style="flex:1">{{ t('pages.dashboard.actions.close', 'Close') }}</button>
                </div>
            </div>
        </form>
    </div>
</div>

<div v-if="emotionsOpen" class="modal-overlay" @click.self="closeEmotionsModal">
    <div class="modal-panel" style="max-width:700px">
        <div class="modal-panel-corner-bl"></div>
        <div class="modal-panel-corner-br"></div>

        <div class="modal-header">
            <h2>{{ t('pages.dashboard.modal.category_manager', 'Category Manager') }}</h2>
            <button @click="closeEmotionsModal" class="modal-close">
                <svg style="width:20px;height:20px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
            </button>
        </div>

        <div style="padding:24px">
            <div style="background:var(--bg-main);padding:16px;margin-bottom:20px;border:1px solid var(--gold-dark)">
                <h3 style="margin:0 0 16px 0;font-size:0.9rem;color:var(--gold-primary);font-family:'Cinzel',serif">{{ t('pages.dashboard.categories.add_new', 'Add Category') }}</h3>
                <div style="display:grid;grid-template-columns:1fr 1fr 1fr auto;gap:12px">
                    <input v-model="newEmotion.key" :placeholder="t('pages.dashboard.placeholders.category_key', 'Key (e.g. happy)')" class="codex-input">
                    <input v-model="newEmotion.name" :placeholder="t('pages.dashboard.placeholders.category_name', 'Name (e.g. Happy)')" class="codex-input">
                    <input v-model="newEmotion.desc" :placeholder="t('pages.dashboard.placeholders.category_desc', 'Description (optional)')" class="codex-input">
                    <button @click="addEmotion" :disabled="!newEmotion.key || addingEmotion" class="codex-btn primary">
                        {{ addingEmotion ? '...' : t('pages.dashboard.actions.add', 'Add') }}
                    </button>
                </div>
            </div>

            <div style="display:flex;flex-direction:column;gap:8px;max-height:400px;overflow-y:auto">
                <div v-for="cat in availableEmotions" :key="cat.key"
                    style="display:flex;align-items:center;justify-content:space-between;padding:16px;background:var(--bg-slot);border:1px solid var(--gold-dark)">
                    <div>
                        <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px">
                            <span style="font-family:'Cinzel',serif;color:var(--gold-primary);font-size:1.1rem">{{ cat.name || cat.key }}</span>
                            <span
                                style="font-size:0.75rem;color:var(--text-muted);background:var(--bg-main);padding:2px 8px;border:1px solid var(--gold-dark)">{{ cat.key }}</span>
                        </div>
                        <p style="margin:0;color:var(--text-muted);font-size:0.85rem;font-style:italic">{{ cat.desc || t('pages.dashboard.messages.no_description', 'No description') }}</p>
                    </div>
                    <button @click="deleteEmotion(cat)" :disabled="deletingEmotionKey === cat.key"
                        class="codex-btn danger">
                        {{ deletingEmotionKey === cat.key ? '...' : t('pages.dashboard.actions.delete', 'Delete') }}
                    </button>
                </div>

                <div v-if="availableEmotions.length === 0" class="empty-state" style="padding:40px">
                    <p>{{ t('pages.dashboard.empty.no_categories', 'No categories yet') }}</p>
                </div>
            </div>
        </div>
    </div>
</div>

<div v-if="batchMoveOpen" class="modal-overlay" @click.self="closeBatchMoveModal">
    <div class="modal-panel" style="max-width:400px">
        <div class="modal-panel-corner-bl"></div>
        <div class="modal-panel-corner-br"></div>

        <div class="modal-header">
            <h2>{{ t('pages.dashboard.modal.batch_move', 'Batch Move') }}</h2>
        </div>

        <div style="padding:24px">
            <p style="margin:0 0 16px 0;color:var(--text-muted)">{{ t('pages.dashboard.batch.selected_images', 'Selected {count} image(s)').replace('{count}', selectedImages.size) }}</p>

            <label
                style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--text-muted);display:block;margin-bottom:8px">{{ t('pages.dashboard.fields.target_category', 'Target Category') }}</label>
            <select v-model="batchTargetCategory" class="codex-input" style="margin-bottom:20px">
                <option value="">{{ t('pages.dashboard.placeholders.select', 'Select...') }}</option>
                <option v-for="cat in categories" :key="cat.key" :value="cat.key">{{ cat.name }}</option>
            </select>

            <div style="display:flex;gap:12px">
                <button @click="closeBatchMoveModal" class="codex-btn" style="flex:1">{{ t('pages.dashboard.actions.cancel', 'Cancel') }}</button>
                <button @click="confirmBatchMove" :disabled="!batchTargetCategory" class="codex-btn primary"
                    style="flex:1">{{ t('pages.dashboard.actions.confirm_move', 'Confirm Move') }}</button>
            </div>
        </div>
    </div>
</div>

<div v-if="batchScopeOpen" class="modal-overlay" @click.self="closeBatchScopeModal">
    <div class="modal-panel" style="max-width:400px">
        <div class="modal-panel-corner-bl"></div>
        <div class="modal-panel-corner-br"></div>

        <div class="modal-header">
            <h2>{{ t('pages.dashboard.modal.batch_scope', 'Batch Scope') }}</h2>
        </div>

        <div style="padding:24px">
            <p style="margin:0 0 16px 0;color:var(--text-muted)">{{ t('pages.dashboard.batch.selected_images', 'Selected {count} image(s)').replace('{count}', selectedImages.size) }}</p>

            <label
                style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--text-muted);display:block;margin-bottom:8px">{{ t('pages.dashboard.fields.target_scope', 'Target Scope') }}</label>
            <select v-model="batchScopeMode" class="codex-input" style="margin-bottom:20px">
                <option value="public">public / {{ t('pages.dashboard.scope.public', 'Public') }}</option>
                <option value="local">local / {{ t('pages.dashboard.scope.local', 'Local only') }}</option>
            </select>
            <div class="form-hint">{{ t('pages.dashboard.batch.scope_hint', 'Images missing origin group info will be skipped when setting local scope.') }}</div>

            <div style="display:flex;gap:12px;margin-top:20px">
                <button @click="closeBatchScopeModal" class="codex-btn" style="flex:1">{{ t('pages.dashboard.actions.cancel', 'Cancel') }}</button>
                <button @click="confirmBatchScope" class="codex-btn primary" style="flex:1">{{ t('pages.dashboard.actions.confirm_set', 'Confirm Set') }}</button>
            </div>
        </div>
    </div>
</div>

<div v-if="isBatchMode && selectedImages.size > 0" class="batch-bar">
    <span style="font-family:'Cinzel',serif;color:var(--gold-bright);font-size:1rem">{{ t('pages.dashboard.batch.selected_short', 'Selected {count}').replace('{count}', selectedImages.size) }}</span>
    <div style="width:1px;height:24px;background:var(--gold-dark)"></div>
    <button @click="selectAll" class="codex-btn" style="font-size:0.8rem;padding:8px 16px">{{ t('pages.dashboard.actions.select_all', 'Select All') }}</button>
    <button @click="openBatchMoveModal" class="codex-btn" style="font-size:0.8rem;padding:8px 16px">{{ t('pages.dashboard.actions.move', 'Move') }}</button>
    <button @click="handleBatchDelete" class="codex-btn danger" style="font-size:0.8rem;padding:8px 16px">{{ t('pages.dashboard.actions.delete', 'Delete') }}</button>
    <button @click="openBatchScopeModal" class="codex-btn" style="font-size:0.8rem;padding:8px 16px">{{ t('pages.dashboard.fields.scope', 'Scope') }}</button>
    <button @click="repairSelectedScope" class="codex-btn" style="font-size:0.8rem;padding:8px 16px">{{ t('pages.dashboard.actions.repair_origin', 'Repair Origin') }}</button>
    <button @click="batchSetFavorite(true)" class="codex-btn" style="font-size:0.8rem;padding:8px 16px">
        <svg style="width:14px;height:14px" fill="currentColor" viewBox="0 0 24 24">
            <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
        </svg>
        {{ t('pages.dashboard.actions.favorite', 'Favorite') }}
    </button>
    <button @click="batchSetFavorite(false)" class="codex-btn" style="font-size:0.8rem;padding:8px 16px">
        {{ t('pages.dashboard.actions.unfavorite', 'Remove Favorite') }}
    </button>
    <div style="width:1px;height:24px;background:var(--gold-dark)"></div>
    <button @click="toggleBatchMode" class="codex-btn icon-btn" style="width:32px;height:32px">
        <svg style="width:16px;height:16px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
        </svg>
    </button>
</div>

<div v-if="toastOpen" class="toast-notification" @click="toastOpen = false">
    {{ toastMessage }}
</div>

<div v-if="confirmOpen" class="modal-overlay" @click.self="onConfirmNo">
    <div class="modal-panel" style="max-width:400px">
        <div class="modal-header">
            <h2>{{ t('pages.dashboard.modal.confirm', 'Confirm Action') }}</h2>
        </div>
        <div style="padding:24px">
            <p style="margin:0 0 24px;color:var(--text-main);font-size:1rem">{{ confirmMessage }}</p>
            <div style="display:flex;gap:12px">
                <button @click="onConfirmNo" class="codex-btn" style="flex:1">{{ t('pages.dashboard.actions.cancel', 'Cancel') }}</button>
                <button @click="onConfirmYes" class="codex-btn danger" style="flex:1">{{ t('pages.dashboard.actions.confirm', 'Confirm') }}</button>
            </div>
        </div>
    </div>
</div>

<!-- 审核区编辑弹窗（issue #87） -->
<div v-if="pendingEditOpen" class="modal-overlay" @click.self="closePendingEdit">
    <div class="modal-panel">
        <div class="modal-panel-corner-bl"></div>
        <div class="modal-panel-corner-br"></div>

        <div class="modal-header">
            <h2>{{ t('pages.dashboard.modal.edit_pending', 'Edit Pending Sticker') }}</h2>
            <button @click="closePendingEdit" class="modal-close">
                <svg style="width:20px;height:20px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                        d="M6 18L18 6M6 6l12 12" />
                </svg>
            </button>
        </div>

        <div class="modal-content">
            <div style="padding:24px;width:100%">
                <div style="max-width:520px;margin:0 auto">
                    <div class="pending-edit-preview"
                        style="display:flex;gap:16px;align-items:center;margin-bottom:20px;padding:12px;background:rgba(0,0,0,0.25);border-radius:6px">
                        <img v-if="pendingEditForm.hash && imageDataUrls[pendingEditForm.hash]"
                            :src="imageDataUrls[pendingEditForm.hash]"
                            style="width:96px;height:96px;object-fit:contain;border-radius:4px;background:#000">
                        <div v-else
                            style="width:96px;height:96px;border-radius:4px;background:#000;display:flex;align-items:center;justify-content:center;color:var(--text-muted);font-size:0.75rem">
                            {{ t('pages.dashboard.messages.no_preview', 'No preview') }}
                        </div>
                        <div style="flex:1;min-width:0">
                            <div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--text-muted)">
                                {{ t('pages.dashboard.labels.hash', 'Hash') }}</div>
                            <div style="font-size:0.85rem;word-break:break-all;color:var(--text-main)">
                                {{ pendingEditForm.hash || '-' }}
                            </div>
                        </div>
                    </div>

                    <div style="margin-bottom:16px">
                        <label
                            style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--text-muted);display:block;margin-bottom:6px">
                            {{ t('pages.dashboard.fields.category', 'Category') }}
                        </label>
                        <select v-model="pendingEditForm.category" class="codex-input">
                            <option v-for="cat in categories" :key="cat.key" :value="cat.key">{{ cat.name }}</option>
                        </select>
                    </div>

                    <div style="margin-bottom:16px">
                        <label
                            style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--text-muted);display:block;margin-bottom:6px">
                            {{ t('pages.dashboard.fields.scope', 'Scope') }}
                        </label>
                        <select v-model="pendingEditForm.scope_mode" class="codex-input">
                            <option value="public">{{ t('pages.dashboard.scope.public', 'Public') }}</option>
                            <option value="local">{{ t('pages.dashboard.scope.local', 'Local only') }}</option>
                        </select>
                    </div>

                    <div style="margin-bottom:16px">
                        <label
                            style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--text-muted);display:block;margin-bottom:6px">
                            {{ t('pages.dashboard.fields.description', 'Description') }}
                        </label>
                        <textarea v-model="pendingEditForm.desc" class="codex-input" rows="3"></textarea>
                    </div>

                    <div style="margin-bottom:16px">
                        <label
                            style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--text-muted);display:block;margin-bottom:6px">
                            {{ t('pages.dashboard.fields.tags', 'Tags') }}
                            ({{ t('pages.dashboard.messages.tag_separator_hint', 'comma separated') }})
                        </label>
                        <input v-model="pendingEditForm.tagsText" type="text" class="codex-input">
                    </div>

                    <div style="margin-bottom:8px">
                        <label
                            style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--text-muted);display:block;margin-bottom:6px">
                            {{ t('pages.dashboard.fields.scenes', 'Scenes') }}
                            ({{ t('pages.dashboard.messages.scene_separator_hint', 'comma separated') }})
                        </label>
                        <input v-model="pendingEditForm.scenesText" type="text" class="codex-input">
                    </div>
                </div>
            </div>
        </div>

        <div class="modal-actions">
            <button @click="closePendingEdit" class="codex-btn" style="flex:1">
                {{ t('pages.dashboard.actions.cancel', 'Cancel') }}
            </button>
            <button @click="savePendingEdit(false)" class="codex-btn" style="flex:1">
                {{ t('pages.dashboard.actions.save_only', 'Save') }}
            </button>
            <button @click="savePendingEdit(true)" class="codex-btn primary" style="flex:1">
                {{ t('pages.dashboard.actions.save_and_approve', 'Save & Approve') }}
            </button>
        </div>
    </div>
</div>

<div v-if="promptOpen" class="modal-overlay" @click.self="onPromptCancel">
    <div class="modal-panel" style="max-width:420px">
        <div class="modal-header">
            <h2>{{ t('pages.dashboard.modal.input', 'Input') }}</h2>
        </div>
        <div style="padding:24px">
            <p style="margin:0 0 16px;color:var(--text-main);font-size:1rem">{{ promptMessage }}</p>
            <input v-model="promptValue" type="text" class="codex-input" @keyup.enter="onPromptOk">
            <div style="display:flex;gap:12px;margin-top:20px">
                <button @click="onPromptCancel" class="codex-btn" style="flex:1">{{ t('pages.dashboard.actions.cancel', 'Cancel') }}</button>
                <button @click="onPromptOk" class="codex-btn primary" style="flex:1">{{ t('pages.dashboard.actions.confirm', 'Confirm') }}</button>
            </div>
        </div>
    </div>
</div>`;
