export const EMOTION_LABELS_TEMPLATE = `
<span class="emotion-labels" :class="{ compact }" :title="keys.map(label).join(' · ')">
    <span v-for="(key, index) in keys" :key="key" class="emotion-chip" :class="{ primary: index === 0 }">{{ label(key) }}</span>
</span>`;

export const TEMPLATE = `
<template v-if="isBiliTheme">
<div class="page-wrap" :class="{ 'is-scrolled': biliHeaderScrolled }">
    <svg class="icon-sprite" aria-hidden="true" xmlns="http://www.w3.org/2000/svg">
        <symbol id="i-search" viewBox="0 0 24 24"><circle cx="11" cy="11" r="7"/><path d="M16.2 16.2 21 21"/></symbol>
        <symbol id="i-upload" viewBox="0 0 24 24"><path d="M12 16V4"/><path d="m6.5 9.5 5.5-5.5 5.5 5.5"/><path d="M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"/></symbol>
        <symbol id="i-flame" viewBox="0 0 24 24"><path d="M12 3c3.2 3.5 6.5 6.3 6.5 10.2A6.5 6.5 0 0 1 12 21a6.5 6.5 0 0 1-6.5-6.3C5.5 11.4 7 9.2 9 7c.2 1.7 1 3 2.3 3.7C11.7 8.5 11.2 6 12 3Z"/></symbol>
        <symbol id="i-grid" viewBox="0 0 24 24"><rect x="4" y="4" width="6.5" height="6.5" rx="1.6"/><rect x="13.5" y="4" width="6.5" height="6.5" rx="1.6"/><rect x="4" y="13.5" width="6.5" height="6.5" rx="1.6"/><rect x="13.5" y="13.5" width="6.5" height="6.5" rx="1.6"/></symbol>
        <symbol id="i-chevron" viewBox="0 0 24 24"><path d="m6 9 6 6 6-6"/></symbol>
        <symbol id="i-arrow-left" viewBox="0 0 24 24"><path d="m14 6-6 6 6 6"/></symbol>
        <symbol id="i-arrow-right" viewBox="0 0 24 24"><path d="m10 6 6 6-6 6"/></symbol>
        <symbol id="i-star" viewBox="0 0 24 24"><path d="m12 3.6 2.5 5.2 5.7.8-4.1 4 1 5.7-5.1-2.7-5.1 2.7 1-5.7-4.1-4 5.7-.8L12 3.6Z"/></symbol>
        <symbol id="i-heart" viewBox="0 0 24 24"><path d="M12 20s-7.5-4.4-7.5-9.4A4.1 4.1 0 0 1 12 7.4a4.1 4.1 0 0 1 7.5 3.2C19.5 15.6 12 20 12 20Z"/></symbol>
        <symbol id="i-trash" viewBox="0 0 24 24"><path d="M4 7h16"/><path d="M9 7V4.8h6V7"/><path d="M6.5 7 7.4 20h9.2L17.5 7"/><path d="M10 11v5M14 11v5"/></symbol>
        <symbol id="i-close" viewBox="0 0 24 24"><path d="M6 6l12 12M18 6 6 18"/></symbol>
        <symbol id="i-refresh" viewBox="0 0 24 24"><path d="M20 11a8 8 0 1 0-2.3 6.3"/><path d="M20 4v7h-7"/></symbol>
        <symbol id="i-message" viewBox="0 0 24 24"><path d="M4 5.5A1.5 1.5 0 0 1 5.5 4h13A1.5 1.5 0 0 1 20 5.5v9a1.5 1.5 0 0 1-1.5 1.5H9l-4.2 3.2A.6.6 0 0 1 4 18.7Z"/></symbol>
        <symbol id="i-history" viewBox="0 0 24 24"><circle cx="12" cy="12" r="8"/><path d="M12 7.5V12l3 2"/></symbol>
        <symbol id="i-create" viewBox="0 0 24 24"><rect x="4" y="5" width="16" height="14" rx="2"/><path d="M8 13.5 13.5 8l2.5 2.5L10.5 16H8Z"/></symbol>
        <symbol id="i-play" viewBox="0 0 24 24"><circle cx="12" cy="12" r="8.5"/><path d="M10.4 9.2v5.6l4.6-2.8Z"/></symbol>
        <symbol id="i-check" viewBox="0 0 24 24"><path d="m5 12.5 4.2 4.2L19 7"/></symbol>
        <symbol id="i-image" viewBox="0 0 24 24"><rect x="4" y="5" width="16" height="14" rx="2"/><circle cx="9" cy="10" r="1.5"/><path d="m5.5 17 4.2-5 3.2 3.4 2.2-2.2 3.4 3.8"/></symbol>
        <symbol id="i-folder" viewBox="0 0 24 24"><path d="M4 7.5A1.5 1.5 0 0 1 5.5 6H10l2 2.5h6.5A1.5 1.5 0 0 1 20 10v7.5a1.5 1.5 0 0 1-1.5 1.5h-13A1.5 1.5 0 0 1 4 17.5Z"/></symbol>
        <symbol id="i-shield" viewBox="0 0 24 24"><path d="M12 3.5 19 6v5.2c0 4.6-3 7.4-7 9.3-4-1.9-7-4.7-7-9.3V6Z"/><path d="m9 12 2.2 2.2L15.5 10"/></symbol>
        <symbol id="i-sort" viewBox="0 0 24 24"><path d="M7 4v16M7 20l-3-3M7 20l3-3"/><path d="M17 20V4M17 4l-3 3M17 4l3 3"/></symbol>
    </svg>

    <header class="site-header" :class="{ 'is-stuck': biliHeaderScrolled || activeSection !== 'library' }">
        <div class="header-inner">
            <div class="header-left">
                <a class="brand" href="#" @click.prevent="switchSection('library')" aria-label="表情包小偷首页">
                    <img class="brand-mark" src="./logo.png" alt="表情包小偷">
                    <span class="brand-text">
                        <span class="brand-name">表情包小偷</span>
                        <span class="brand-sub">MEME STEALER</span>
                    </span>
                </a>
                <nav class="main-nav" aria-label="主导航">
                    <a class="nav-item" :class="{ 'is-active': activeSection === 'library' }" href="#" @click.prevent="switchSection('library')">首页</a>
                    <a class="nav-item" href="#" @click.prevent="goBiliLibrary('')">表情库</a>
                    <a class="nav-item" :class="{ 'is-active': activeSection === 'pending' }" href="#" @click.prevent="switchSection('pending')">审核区<i class="nav-badge" v-if="pendingStats.pending">{{ pendingStats.pending }}</i></a>
                    <a class="nav-item" href="#" @click.prevent="scrollBiliTo('bili-stats')">数据</a>
                    <a class="nav-item" href="#" @click.prevent="scrollBiliTo('bili-about')">关于</a>
                    <div class="nav-more-wrap">
                        <button class="nav-item nav-more" type="button" @click.stop="biliMoreOpen = !biliMoreOpen">更多<svg class="ico ico-chevron"><use href="#i-chevron"/></svg></button>
                        <div v-if="biliMoreOpen" class="nav-more-menu" @click.stop>
                            <button type="button" @click="openSourceModal(); biliMoreOpen = false">外部源</button>
                            <button type="button" @click="openEmotionsModal(); biliMoreOpen = false">分类管理</button>
                            <button type="button" @click="openCharactersModal(); biliMoreOpen = false">角色管理</button>
                            <button type="button" @click="runStorageCleanup(); biliMoreOpen = false">存储清理</button>
                            <button type="button" @click="showAlert('插件设置入口：AstrBot 插件配置'); biliMoreOpen = false">插件设置</button>
                        </div>
                    </div>
                </nav>
            </div>

            <div class="header-center">
                <form class="search" role="search" @submit.prevent="submitBiliSearch">
                    <input class="search-input" v-model="searchQuery" @input="debouncedSearch"
                        @focus="biliSearchOpen = true" type="search"
                        placeholder="搜索表情包、标签、场景…" autocomplete="off" aria-label="搜索表情包">
                    <button class="search-btn" type="submit" aria-label="搜索"><svg class="ico"><use href="#i-search"/></svg></button>
                    <div v-if="biliSearchOpen" class="search-suggest" @click.stop>
                        <div class="search-suggest-head">
                            <span>热门搜索</span>
                            <button type="button" class="search-suggest-close" aria-label="关闭搜索建议"
                                @click="biliSearchOpen = false"><svg class="ico"><use href="#i-close"/></svg></button>
                        </div>
                        <div class="search-suggest-chips">
                            <button v-for="cat in biliHotKeywords" :key="'kw-' + cat.key" type="button"
                                class="suggest-chip" @click="quickBiliSearch(cat)">{{ cat.name }}</button>
                        </div>
                        <div class="search-suggest-foot">Enter 搜索 · 共 {{ total }} 个表情包</div>
                    </div>
                </form>
            </div>

            <div class="header-right">
                <a class="icon-entry" :class="{ 'is-active': selectedLibrary === 'favorites' }" href="#" @click.prevent="goBiliFavorites()"><svg class="ico"><use href="#i-star"/></svg><span>收藏</span></a>
                <a class="icon-entry" :class="{ 'is-active': biliPanelOpen && biliPanelType === 'history' }" href="#" @click.prevent="openBiliHistory"><svg class="ico"><use href="#i-history"/></svg><span>历史</span></a>
                <a class="icon-entry" :class="{ 'is-active': biliPanelOpen && biliPanelType === 'studio' }" href="#" @click.prevent="openBiliStudio"><svg class="ico"><use href="#i-create"/></svg><span>创作中心</span></a>
                <button class="btn-upload" type="button" @click="openUploadModal"><svg class="ico"><use href="#i-upload"/></svg>投稿</button>
                <div class="theme-picker">
                    <button type="button" class="icon-entry" @click.stop="themePickerOpen = !themePickerOpen"
                        aria-label="主题" title="主题">
                        <svg class="ico" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"
                                d="M7 21a4.000 4.000 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4.000 4.000 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" />
                        </svg>
                        <span>主题</span>
                    </button>
                    <div v-if="themePickerOpen" class="theme-popover" @click.stop>
                        <p class="theme-save-hint">{{ t('pages.dashboard.themes.save_hint', 'Your pick is saved as the default theme.') }}</p>
                        <div class="theme-group-label">{{ t('pages.dashboard.themes.group_original', 'Original') }}</div>
                        <button v-for="opt in originalThemeOptions" :key="opt.value" type="button" class="theme-option"
                            :class="{ active: themeMode === opt.value }" @click="setThemeMode(opt.value); themePickerOpen = false">
                            <span v-if="opt.swatch" class="theme-swatch"
                                :style="{ background: 'linear-gradient(135deg, ' + opt.swatch.split(',')[0] + ' 50%, ' + opt.swatch.split(',')[1] + ' 50%)' }"></span>
                            <span v-else class="theme-swatch" style="background: conic-gradient(#161b2a 50%, #faf8f3 50%)"></span>
                            {{ t('pages.dashboard.themes.' + opt.key, opt.fallback) }}
                            <span v-if="themeMode === opt.value" class="theme-default-tag">{{ t('pages.dashboard.themes.saved_default', 'Default') }}</span>
                        </button>
                        <div class="theme-group-label">{{ t('pages.dashboard.themes.group_game', 'Game inventory') }}</div>
                        <button v-for="opt in gameThemeOptions" :key="opt.value" type="button" class="theme-option"
                            :class="{ active: themeMode === opt.value }" @click="setThemeMode(opt.value); themePickerOpen = false">
                            <span class="theme-swatch"
                                :style="{ background: 'linear-gradient(135deg, ' + opt.swatch.split(',')[0] + ' 50%, ' + opt.swatch.split(',')[1] + ' 50%)' }"></span>
                            {{ t('pages.dashboard.themes.' + opt.key, opt.fallback) }}
                            <span v-if="themeMode === opt.value" class="theme-default-tag">{{ t('pages.dashboard.themes.saved_default', 'Default') }}</span>
                        </button>
                        <div class="theme-group-label">{{ t('pages.dashboard.themes.group_special', 'Featured') }}</div>
                        <button v-for="opt in specialThemeOptions" :key="opt.value" type="button" class="theme-option"
                            :class="{ active: themeMode === opt.value }" @click="setThemeMode(opt.value); themePickerOpen = false">
                            <span class="theme-swatch"
                                :style="{ background: 'linear-gradient(135deg, ' + opt.swatch.split(',').join(', ') + ')' }"></span>
                            {{ t('pages.dashboard.themes.' + opt.key, opt.fallback) }}
                            <span v-if="themeMode === opt.value" class="theme-default-tag">{{ t('pages.dashboard.themes.saved_default', 'Default') }}</span>
                        </button>
                    </div>
                </div>
                <button class="avatar" type="button" aria-label="个人中心" @click="showAlert('个人中心')"><span>偷</span></button>
            </div>
        </div>
    </header>

    <section v-if="activeSection === 'library'" class="banner" aria-label="精选专题">
        <div class="banner-track">
            <article class="banner-slide" :class="{ 'is-active': biliBannerIndex === 0 }" data-theme="pink">
                <div class="banner-deco" aria-hidden="true"><span class="blob b1"></span><span class="blob b2"></span><span class="banner-grid"></span></div>
                <div class="banner-stickers" aria-hidden="true">
                    <span class="sticker s1">😂</span><span class="sticker s2">🥺</span>
                    <span class="sticker s3">😼</span><span class="sticker s4">😭</span>
                    <span class="sticker s5">🫠</span>
                </div>
                <div class="banner-copy">
                    <h1 class="banner-title">全网表情包，一网打尽</h1>
                    <div class="banner-actions">
                        <a class="btn btn-white" href="#" @click.prevent="goBiliLibrary('')">逛逛表情库</a>
                    </div>
                </div>
            </article>
            <article class="banner-slide" :class="{ 'is-active': biliBannerIndex === 1 }" data-theme="blue">
                <div class="banner-deco" aria-hidden="true"><span class="blob b1"></span><span class="blob b2"></span><span class="banner-grid"></span></div>
                <div class="banner-stickers" aria-hidden="true">
                    <span class="sticker s1">🤯</span><span class="sticker s2">😱</span>
                    <span class="sticker s3">🤔</span><span class="sticker s4">😳</span>
                    <span class="sticker s5">✨</span>
                </div>
                <div class="banner-copy">
                    <h1 class="banner-title">看一眼，就懂是什么情绪</h1>
                    <div class="banner-actions">
                        <a class="btn btn-white" href="#" @click.prevent="goBiliLibrary('')">看 AI 起的名字</a>
                    </div>
                </div>
            </article>
            <article class="banner-slide" :class="{ 'is-active': biliBannerIndex === 2 }" data-theme="orange">
                <div class="banner-deco" aria-hidden="true"><span class="blob b1"></span><span class="blob b2"></span><span class="banner-grid"></span></div>
                <div class="banner-stickers" aria-hidden="true">
                    <span class="sticker s1">😍</span><span class="sticker s2">🙏</span>
                    <span class="sticker s3">😌</span><span class="sticker s4">🫡</span>
                    <span class="sticker s5">💖</span>
                </div>
                <div class="banner-copy">
                    <h1 class="banner-title">喜欢的表情，永远留在库里</h1>
                    <div class="banner-actions">
                        <a class="btn btn-white" href="#" @click.prevent="goBiliFavorites()">收藏夹</a>
                    </div>
                </div>
            </article>
        </div>
        <button class="banner-refresh" type="button" title="换一换" @click="nextBiliBanner">
            <span>换一换</span><svg class="ico"><use href="#i-refresh"/></svg>
        </button>
        <div class="banner-dots" role="tablist" aria-label="专题切换">
            <button v-for="n in 3" :key="'bd' + n" type="button" class="banner-dot"
                :class="{ 'is-active': biliBannerIndex === n - 1 }" @click="biliBannerIndex = n - 1"></button>
        </div>
    </section>

    <nav v-if="activeSection === 'library'" class="channel-bar" aria-label="表情分类">
        <div class="channel-icons">
            <button class="channel-icon" type="button"
                :class="{ 'is-active': selectedLibrary === '' && selectedCategory === '' && sortBy !== 'most_used' }"
                @click="goBiliAll()">
                <span class="icon-bg icon-bg--all"><svg class="ico"><use href="#i-grid"/></svg></span>
                <span class="channel-icon-text">全部</span>
            </button>
            <button class="channel-icon" type="button" :class="{ 'is-active': sortBy === 'most_used' }"
                @click="goBiliHot()">
                <span class="icon-bg icon-bg--hot"><svg class="ico"><use href="#i-flame"/></svg></span>
                <span class="channel-icon-text">热门</span>
            </button>
        </div>
        <div class="channel-links">
            <button v-for="cat in categories" :key="'ch-' + cat.key" type="button" class="channel-link"
                :class="{ 'is-active': selectedCategory === cat.key }" @click="selectLibraryCategory(cat.key)">
                {{ cat.name }}
            </button>
        </div>
        <div class="channel-side">
            <a class="channel-side-link" href="#" @click.prevent="switchSection('pending')"><svg class="ico"><use href="#i-shield"/></svg>审核区</a>
            <a class="channel-side-link" href="#" @click.prevent="openSourceModal"><svg class="ico"><use href="#i-folder"/></svg>外部源</a>
            <a class="channel-side-link" href="#" @click.prevent="runStorageCleanup"><svg class="ico"><use href="#i-trash"/></svg>存储清理</a>
        </div>
    </nav>

    <main class="page" :class="{ 'page--offset': activeSection !== 'library' }">
        <template v-if="activeSection === 'library'">
        <section class="hero-row" aria-label="今日精选" v-if="biliHeroSlides.length">
            <div class="hero-carousel">
                <div class="hero-slides">
                    <a v-for="(slide, i) in biliHeroSlides" :key="'hero-' + slide.hash" class="hero-slide"
                        :data-hash="slide.hash"
                        :class="{ 'is-active': biliHeroIndex === i }" href="#" @click.prevent="openPreview(slide)">
                        <img v-if="imageDataUrls[slide.hash]" class="hero-img" :src="imageDataUrls[slide.hash]" alt="" :loading="i === 0 ? 'eager' : 'lazy'" decoding="async">
                        <div class="hero-shade"></div>
                        <div class="hero-copy">
                            <span class="hero-badge">{{ getCategoryName(slide.category) }}</span>
                            <h3 class="hero-title">{{ slide.desc || slide.overlay_text || getCategoryName(slide.category) }}</h3>
                            <p v-if="(slide.tags || []).length" class="hero-desc">{{ slide.tags.join(' · ') }}</p>
                            <span class="hero-stats">{{ slide.use_count || 0 }} 次使用 · {{ formatDate(slide.created_at) }}</span>
                        </div>
                    </a>
                </div>
                <button class="hero-arrow hero-arrow--prev" type="button" aria-label="上一张" @click="prevBiliHero"><svg class="ico"><use href="#i-arrow-left"/></svg></button>
                <button class="hero-arrow hero-arrow--next" type="button" aria-label="下一张" @click="nextBiliHero"><svg class="ico"><use href="#i-arrow-right"/></svg></button>
                <div class="hero-dots">
                    <button v-for="(slide, i) in biliHeroSlides" :key="'hd-' + slide.hash" type="button" class="hero-dot"
                        :class="{ 'is-active': biliHeroIndex === i }" @click="biliHeroIndex = i"></button>
                </div>
            </div>
            <div class="hero-grid" aria-label="今日热门表情">
                <article class="meme-card" v-for="img in biliHot" :key="'hot-' + img.hash">
                    <div class="meme-cover-wrap" :data-hash="img.hash" role="button" tabindex="0"
                        @click="openPreview(img)" @keydown.enter.prevent="openPreview(img)">
                        <img v-if="imageDataUrls[img.hash]" class="meme-cover-bg" :src="imageDataUrls[img.hash]" alt="" aria-hidden="true" loading="lazy" decoding="async">
                        <img v-if="imageDataUrls[img.hash]" class="meme-cover" :src="imageDataUrls[img.hash]" :alt="img.desc" loading="lazy" decoding="async">
                        <span class="meme-badge">{{ getCategoryName(img.category) }}</span>
                        <div class="meme-stats">
                            <div class="meme-stats-left">
                                <span class="meme-stat"><svg class="ico"><use href="#i-play"/></svg><b>{{ img.use_count || 0 }}</b></span>
                            </div>
                            <span class="meme-format">{{ (img.format || 'img').toUpperCase() }}</span>
                        </div>
                        <button class="meme-star" type="button" :class="{ 'is-fav': img.is_favorite }"
                            aria-label="收藏" @click.stop="toggleFavorite(img)">
                            <svg class="ico"><use href="#i-star"/></svg>
                        </button>
                    </div>
                    <div class="meme-info">
                        <h3 class="meme-title"><a href="#" @click.prevent="openPreview(img)">{{ img.desc || img.overlay_text || getCategoryName(img.category) }}</a></h3>
                        <div class="meme-meta">
                            <span class="meme-avatar" :style="catAccent(img.category)">{{ (getCategoryName(img.category) || '偷').slice(0, 1) }}</span>
                            <span class="meme-author">{{ getCategoryName(img.category) }}</span>
                            <span class="meme-date">{{ formatDate(img.created_at) }}</span>
                        </div>
                    </div>
                </article>
            </div>
        </section>

        <div class="content-grid">
            <section class="feed" id="bili-library">
                <header class="feed-head">
                    <div class="feed-head-main">
                        <h2 class="feed-title">{{ libraryTitle(selectedLibrary) }}<span class="feed-total">{{ total }}</span></h2>
                        <p v-if="selectedCategory" class="feed-sub">{{ getCategoryName(selectedCategory) }}</p>
                    </div>
                    <div class="feed-tabs" role="tablist" aria-label="表情包排序">
                        <button class="feed-tab" :class="{ 'is-active': biliTab === 'recommend' }" type="button" @click="setBiliTab('recommend')">推荐</button>
                        <button class="feed-tab" :class="{ 'is-active': biliTab === 'new' }" type="button" @click="setBiliTab('new')">最新</button>
                        <button class="feed-tab" :class="{ 'is-active': biliTab === 'used' }" type="button" @click="setBiliTab('used')">最常用</button>
                        <button class="feed-tab" :class="{ 'is-active': biliTab === 'fav' }" type="button" @click="setBiliTab('fav')">收藏</button>
                    </div>
                </header>

                <div class="feed-toolbar">
                    <div class="filter-chips">
                        <button class="chip" :class="{ 'is-active': selectedLibrary === '' }" type="button" @click="goBiliLibrary('')">全部</button>
                        <button class="chip" :class="{ 'is-active': selectedLibrary === 'favorites' }" type="button" @click="goBiliFavorites()">收藏</button>
                        <button class="chip" :class="{ 'is-active': selectedLibrary === 'characters' }" type="button" @click="goBiliLibrary('characters')">角色</button>
                        <button class="chip" :class="{ 'is-active': selectedLibrary === 'general' }" type="button" @click="goBiliLibrary('general')">通用</button>
                    </div>
                    <div class="feed-tools">
                        <label class="select-wrap" title="排序">
                            <svg class="ico"><use href="#i-sort"/></svg>
                            <select class="select" v-model="sortBy" @change="fetchImages(1)" aria-label="排序方式">
                                <option value="newest">最新入库</option>
                                <option value="oldest">最旧入库</option>
                                <option value="most_used">使用最多</option>
                                <option value="least_used">使用最少</option>
                                <option value="last_used">最近使用</option>
                            </select>
                            <svg class="ico ico-chevron"><use href="#i-chevron"/></svg>
                        </label>
                        <button class="btn btn-outline" type="button" @click="toggleBatchMode"><svg class="ico"><use href="#i-check"/></svg>批量</button>
                        <button class="btn btn-outline" type="button" @click="openUploadModal"><svg class="ico"><use href="#i-upload"/></svg>上传</button>
                    </div>
                </div>

                <div v-if="loading" class="meme-grid">
                    <div class="meme-card skeleton" v-for="n in 8" :key="'sk' + n">
                        <div class="sk-cover sk-shimmer"></div>
                        <div class="sk-line sk-line--title sk-shimmer"></div>
                        <div class="sk-line sk-line--title short sk-shimmer"></div>
                        <div class="sk-meta"><span class="sk-dot sk-shimmer"></span><span class="sk-line sk-line--meta sk-shimmer"></span></div>
                    </div>
                </div>

                <div v-else-if="images.length" class="meme-grid" :class="{ 'is-batch': isBatchMode }">
                    <article class="meme-card" v-for="img in images" :key="img.hash">
                        <div class="meme-cover-wrap" :data-hash="img.hash" role="button" tabindex="0"
                            @click="isBatchMode ? toggleSelection(img) : openPreview(img)"
                            @keydown.enter.prevent="isBatchMode ? toggleSelection(img) : openPreview(img)">
                            <img v-if="imageDataUrls[img.hash]" class="meme-cover-bg" :src="imageDataUrls[img.hash]" alt="" aria-hidden="true" loading="lazy" decoding="async">
                            <img v-if="imageDataUrls[img.hash]" class="meme-cover" :src="imageDataUrls[img.hash]" :alt="img.desc" loading="lazy" decoding="async">
                            <span class="meme-badge">{{ getCategoryName(img.category) }}</span>
                            <div class="meme-mask">
                                <div class="meme-mask-actions">
                                    <button class="mask-btn" type="button" aria-label="收藏" @click.stop="toggleFavorite(img)"><svg class="ico"><use href="#i-star"/></svg></button>
                                    <button class="mask-btn" type="button" aria-label="详情" @click.stop="openPreview(img)"><svg class="ico"><use href="#i-search"/></svg></button>
                                    <button class="mask-btn" type="button" aria-label="删除" @click.stop="deleteImage(img)"><svg class="ico"><use href="#i-trash"/></svg></button>
                                </div>
                            </div>
                            <div class="meme-stats">
                                <div class="meme-stats-left">
                                    <span class="meme-stat"><svg class="ico"><use href="#i-play"/></svg><b>{{ img.use_count || 0 }}</b></span>
                                </div>
                                <span class="meme-format">{{ (img.format || 'img').toUpperCase() }}</span>
                            </div>
                            <label class="meme-check" v-if="isBatchMode" @click.stop>
                                <input type="checkbox" :checked="selectedImages.has(img.hash)" @change="toggleSelection(img)"><i></i>
                            </label>
                            <button class="meme-star" type="button" :class="{ 'is-fav': img.is_favorite }"
                                aria-label="收藏" @click.stop="toggleFavorite(img)">
                                <svg class="ico"><use href="#i-star"/></svg>
                            </button>
                        </div>
                        <div class="meme-info">
                            <h3 class="meme-title"><a href="#" @click.prevent="openPreview(img)">{{ img.desc || img.overlay_text || getCategoryName(img.category) }}</a></h3>
                            <div class="meme-meta">
                                <span class="meme-avatar" :style="catAccent(img.category)">{{ (getCategoryName(img.category) || '偷').slice(0, 1) }}</span>
                                <span class="meme-author">{{ getCategoryName(img.category) }}</span>
                                <span class="meme-date">{{ formatDate(img.created_at) }}</span>
                            </div>
                        </div>
                    </article>
                </div>

                <div v-else class="empty">
                    <div class="empty-emoji">🫥</div>
                    <h3>没有找到匹配的表情包</h3>
                    <p>换个关键词，或者 <button class="link-btn" type="button" @click="goBiliAll">清空筛选</button> 看看全部。</p>
                </div>

                <nav class="pager" v-if="total > pageSize" aria-label="分页">
                    <button class="page-btn" type="button" :disabled="currentPage === 1" @click="prevPage" aria-label="上一页"><svg class="ico"><use href="#i-arrow-left"/></svg></button>
                    <button v-for="n in biliPages" :key="'pg' + n" type="button" class="page-btn"
                        :class="{ 'is-active': n === currentPage }" @click="goBiliPage(n)">{{ n }}</button>
                    <button class="page-btn" type="button" :disabled="currentPage * pageSize >= total" @click="nextPage" aria-label="下一页"><svg class="ico"><use href="#i-arrow-right"/></svg></button>
                </nav>
            </section>

            <aside class="rail" aria-label="图库信息">
                <section class="floor-card rank-card">
                    <div class="floor-head">
                        <h3 class="floor-title">本周使用榜</h3>
                        <span class="floor-more">TOP 5</span>
                    </div>
                    <ol class="rank-list">
                        <li class="rank-item" v-for="(img, i) in biliRank" :key="'rank-' + img.hash" :data-hash="img.hash" @click="openPreview(img)">
                            <span class="rank-num">{{ i + 1 }}</span>
                            <img v-if="imageDataUrls[img.hash]" class="rank-cover" :src="imageDataUrls[img.hash]" alt="" loading="lazy" decoding="async">
                            <span v-else class="rank-cover" :style="{ backgroundColor: hashToColor(img.hash) }"></span>
                            <div class="rank-info">
                                <p class="rank-title">{{ img.desc || img.overlay_text || getCategoryName(img.category) }}</p>
                                <span class="rank-uses">{{ img.use_count || 0 }} 次使用</span>
                            </div>
                        </li>
                    </ol>
                </section>

                <section class="floor-card">
                    <div class="floor-head">
                        <h3 class="floor-title">最近收录</h3>
                        <a class="floor-more" href="#" @click.prevent="setBiliTab('new')">更多</a>
                    </div>
                    <ul class="recent-list">
                        <li class="recent-item" v-for="img in biliRecent" :key="'recent-' + img.hash" :data-hash="img.hash" @click="openPreview(img)">
                            <img v-if="imageDataUrls[img.hash]" class="recent-cover" :src="imageDataUrls[img.hash]" alt="" loading="lazy" decoding="async">
                            <span v-else class="recent-cover" :style="{ backgroundColor: hashToColor(img.hash) }"></span>
                            <div class="recent-info">
                                <p class="recent-title">{{ img.desc || getCategoryName(img.category) }}</p>
                                <span class="recent-time">{{ formatDate(img.created_at) }}</span>
                            </div>
                        </li>
                    </ul>
                </section>

                <section class="floor-card status-card" id="bili-stats">
                    <div class="floor-head">
                        <h3 class="floor-title">图库状态</h3>
                        <span class="health"><i class="health-dot"></i>{{ getHealthText(healthStatus) }}</span>
                    </div>
                    <div class="stat-grid">
                        <div class="stat"><b>{{ stats.total || 0 }}</b><span>表情包</span></div>
                        <div class="stat"><b>{{ stats.categories || 0 }}</b><span>分类</span></div>
                        <div class="stat"><b>{{ libraries.favorites || 0 }}</b><span>收藏</span></div>
                        <div class="stat"><b>{{ pendingStats.pending || 0 }}</b><span>待审核</span></div>
                    </div>
                    <div class="meter">
                        <div class="meter-head"><span>自动淘汰计数</span><b>{{ libraries.automatic || 0 }} / {{ automaticLimit }}</b></div>
                        <div class="meter-track"><i class="meter-fill" :style="{ width: Math.min(100, (libraries.automatic || 0) / (automaticLimit || 1) * 100) + '%' }"></i></div>
                    </div>
                    <div class="meter">
                        <div class="meter-head"><span>待审核池</span><b>{{ pendingStats.pending || 0 }} / {{ pendingStats.capacity || 0 }}</b></div>
                        <div class="meter-track"><i class="meter-fill meter-fill--blue" :style="{ width: Math.min(100, (pendingStats.pending || 0) / (pendingStats.capacity || 1) * 100) + '%' }"></i></div>
                    </div>
                    <div class="status-links">
                        <button class="link-btn" type="button" @click="switchSection('pending')">去审核</button>
                        <button class="link-btn" type="button" @click="openEmotionsModal">分类管理</button>
                        <button class="link-btn" type="button" @click="showAlert('插件设置入口：AstrBot 插件配置')">插件设置</button>
                    </div>
                </section>

                <section class="floor-card about-card" id="bili-about">
                    <div class="about-mark">偷</div>
                    <div>
                        <h3 class="floor-title">表情包小偷</h3>
                        <p class="about-text">AstrBot 表情包收集 / 分类 / 检索插件。</p>
                    </div>
                </section>
            </aside>
        </div>
        </template>
        <template v-else>
        <header class="review-head" id="bili-pending">
            <div class="review-head-main">
                <h1 class="review-title">审核区 <span class="feed-total">{{ pendingTotal }}</span></h1>
                <p class="review-sub">
                    <span class="loc-metric"><span class="loc-label">待审核</span><span class="loc-num">{{ pendingStats.pending }}</span><span class="loc-cap">/ {{ pendingStats.capacity }}</span></span>
                    <span class="loc-metric"><span class="loc-label">排队</span><span class="loc-num">{{ pendingStats.queued }}</span></span>
                    <span class="loc-metric"><span class="loc-label">识图中</span><span class="loc-num">{{ pendingStats.processing }}</span></span>
                    <span class="loc-metric" :class="{ 'loc-metric--warn': pendingStats.failed > 0 }"><span class="loc-label">失败</span><span class="loc-num">{{ pendingStats.failed }}</span></span>
                </p>
            </div>
            <div class="review-head-side">
                <div class="capacity-row">
                    <span class="capacity-name">待审核容量</span>
                    <span class="capacity-num">{{ pendingStats.pending }}<i>/ {{ pendingStats.capacity }}</i></span>
                </div>
                <div class="progress-track">
                    <div class="progress-fill" :class="{ full: pendingStats.paused }"
                        :style="{ width: Math.min(100, pendingStats.pending / (pendingStats.capacity || 1) * 100) + '%' }"></div>
                </div>
                <div v-if="pendingStats.paused" class="paused-banner">
                    <svg class="ico" aria-hidden="true" viewBox="0 0 24 24"><path d="M9 5v14M15 5v14"/></svg>
                    <span>偷图已暂停，审核后自动恢复</span>
                </div>
            </div>
        </header>

        <div class="review-chips">
            <button type="button" class="feed-chip" :class="{ 'is-active': pendingCategory === '' }"
                @click="selectPendingCategory('')">
                全部<span class="feed-chip-count">{{ pendingCategoryTotal }}</span>
            </button>
            <button v-for="cat in pendingCategories" :key="'pc-' + cat.key" type="button" class="feed-chip"
                :class="{ 'is-active': pendingCategory === cat.key }" :style="catAccent(cat.key)"
                @click="selectPendingCategory(cat.key)">
                <span class="cat-dot"></span>{{ cat.name }}<span class="feed-chip-count">{{ cat.count }}</span>
            </button>
        </div>

        <div class="review-toolbar">
            <div class="review-search">
                <svg class="ico"><use href="#i-search"/></svg>
                <input v-model="pendingSearchQuery" @input="pendingDebouncedSearch" type="search"
                    placeholder="搜索待审核表情…" aria-label="搜索待审核表情">
            </div>
            <div class="review-tools">
                <button type="button" class="btn btn-outline" @click="togglePendingBatchMode">{{ pendingBatchMode ? '完成' : '批量' }}</button>
                <template v-if="pendingBatchMode">
                    <button type="button" class="btn btn-outline" @click="toggleSelectAllPending">{{ allPendingSelected ? '取消全选' : '全选本页' }}</button>
                    <button type="button" class="btn btn-primary" @click="approvePendingBatch">通过所选</button>
                    <button type="button" class="btn btn-outline danger" @click="rejectPendingBatch(false)">删除所选</button>
                    <button type="button" class="btn btn-outline danger" @click="rejectPendingBatch(true)">删除并拉黑</button>
                </template>
            </div>
        </div>

        <div v-if="pendingLoading" class="review-grid">
            <div v-for="n in 6" :key="'prsk' + n" class="review-card skeleton">
                <div class="sk-cover sk-shimmer"></div>
                <div class="review-body">
                    <div class="sk-line sk-line--title sk-shimmer"></div>
                    <div class="sk-line sk-line--title short sk-shimmer"></div>
                </div>
            </div>
        </div>
        <div v-else-if="pendingImages.length === 0" class="empty">
            <div class="empty-emoji">🔍</div>
            <h3>没有待审核的表情</h3>
            <p>偷到的新表情会先出现在这里，审核后才会进入表情库。</p>
        </div>
        <div v-else class="review-grid">
            <article v-for="item in pendingImages" :key="item.id" class="review-card"
                :class="{ selected: pendingSelectedImages.has(item.id) }"
                @click="pendingBatchMode ? togglePendingSelection(item) : openPendingPreview(item)">
                <div class="review-cover" :data-hash="item.hash">
                    <label v-if="pendingBatchMode" class="meme-check" @click.stop>
                        <input type="checkbox" :checked="pendingSelectedImages.has(item.id)"
                            @change="togglePendingSelection(item)"><i></i>
                    </label>
                    <img v-if="imageDataUrls[item.hash]" :src="imageDataUrls[item.hash]" alt=""
                        loading="lazy" decoding="async">
                    <span v-else :style="{ backgroundColor: hashToColor(item.hash) }"></span>
                </div>
                <div class="review-body">
                    <span class="meme-badge">{{ item.category ? getCategoryName(item.category) : '待识图' }}</span>
                    <h3 class="review-card-title">{{ item.desc || '待识图' }}</h3>
                    <p class="review-summary">{{ pendingStatusLabel(item) }}<template v-if="item.analysis_error"> · {{ item.analysis_error }}</template></p>
                    <div v-if="(item.tags || []).length" class="review-tags">
                        <span v-for="tag in item.tags.slice(0, 4)" :key="tag" class="tag">{{ tag }}</span>
                    </div>
                    <div v-if="!pendingBatchMode" class="review-actions">
                        <button type="button" class="review-btn approve" :disabled="pendingApproving.has(item.id)"
                            @click.stop="approvePending(item.id)">通过</button>
                        <button type="button" class="review-btn" @click.stop="openPendingEdit(item)">修改</button>
                        <span class="review-danger-group">
                            <button type="button" class="review-btn danger" @click.stop="rejectPending(item.id)">删除</button>
                            <button type="button" class="review-btn danger" @click.stop="rejectPending(item.id, true)">拉黑</button>
                        </span>
                    </div>
                    <div class="review-foot">
                        <span>{{ item.source === 'auto' ? '自动偷取' : '手动收录' }}</span>
                        <span>{{ formatDate(item.created_at) }}</span>
                    </div>
                </div>
            </article>
        </div>

        <nav class="pager" v-if="pendingTotal > pendingPageSize" aria-label="审核区分页">
            <button class="page-btn" type="button" :disabled="pendingCurrentPage === 1"
                @click="fetchPendingImages(pendingCurrentPage - 1)" aria-label="上一页"><svg class="ico"><use href="#i-arrow-left"/></svg></button>
            <button v-for="n in pendingPages" :key="'ppg' + n" type="button" class="page-btn"
                :class="{ 'is-active': n === pendingCurrentPage }" @click="fetchPendingImages(n)">{{ n }}</button>
            <button class="page-btn" type="button" :disabled="pendingCurrentPage * pendingPageSize >= pendingTotal"
                @click="fetchPendingImages(pendingCurrentPage + 1)" aria-label="下一页"><svg class="ico"><use href="#i-arrow-right"/></svg></button>
        </nav>
        </template>
    </main>

    <footer class="site-footer">
        <div class="footer-inner">
            <div class="footer-brand">
                <span class="footer-logo">表情包小偷</span>
                <p>偷走每一张好笑的脸，在合适的时候还给你。</p>
            </div>
            <div class="footer-links">
                <div class="footer-col">
                    <h4>产品</h4>
                    <a href="#" @click.prevent="goBiliLibrary('')">表情库</a>
                    <a href="#" @click.prevent="switchSection('pending')">审核区</a>
                    <a href="#" @click.prevent="goBiliFavorites()">收藏夹</a>
                    <a href="#" @click.prevent="scrollBiliTo('bili-stats')">数据面板</a>
                    <a href="#" @click.prevent="runStorageCleanup">容量清理</a>
                </div>
                <div class="footer-col">
                    <h4>视图</h4>
                    <a href="#" @click.prevent="setThemeMode('dark')">切换暗色主题</a>
                    <a href="#" @click.prevent="setThemeMode('light')">切换亮色主题</a>
                    <a href="#" @click.prevent="setThemeMode('auto')">跟随宿主</a>
                </div>
            </div>
        </div>
        <div class="footer-bottom">
            <p>表情包小偷 · WebUI</p>
        </div>
    </footer>

    <div v-if="biliPanelOpen" class="modal-overlay" @click.self="closeBiliPanel">
        <div class="modal-panel bili-panel" role="dialog" aria-modal="true"
            :aria-label="biliPanelType === 'history' ? '最近使用' : '创作中心'">
            <div class="modal-header">
                <h2>{{ biliPanelType === 'history' ? '最近使用' : '创作中心' }}</h2>
                <button type="button" class="modal-close" aria-label="关闭" @click="closeBiliPanel">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                </button>
            </div>
            <div class="modal-content bili-panel-content">
                <template v-if="biliPanelType === 'history'">
                    <div v-if="biliHistoryLoading" class="bili-panel-empty">加载中…</div>
                    <div v-else-if="!biliHistory.length" class="bili-panel-empty">还没有使用记录，先去群里发几张表情吧。</div>
                    <div v-else class="bili-history-grid">
                        <button v-for="img in biliHistory" :key="'hist-' + img.hash" type="button"
                            class="bili-history-item" @click="openBiliHistoryItem(img)">
                            <span class="bili-history-cover">
                                <img v-if="imageDataUrls[img.hash]" :src="imageDataUrls[img.hash]" alt="" loading="lazy" decoding="async">
                            </span>
                            <span class="bili-history-title">{{ img.desc || img.overlay_text || getCategoryName(img.category) }}</span>
                            <span class="bili-history-meta">{{ formatDate(img.last_used_at) }} · 用过 {{ img.use_count || 0 }} 次</span>
                        </button>
                    </div>
                </template>
                <template v-else>
                    <div class="bili-studio-grid">
                        <button type="button" class="studio-card studio-card--primary"
                            @click="openUploadModal(); closeBiliPanel()">
                            <svg class="ico"><use href="#i-upload"/></svg>
                            <b>投稿表情</b><span>上传图片，VLM 自动分类入库</span>
                        </button>
                        <button type="button" class="studio-card" @click="openBatchUploadModal(); closeBiliPanel()">
                            <svg class="ico"><use href="#i-folder"/></svg>
                            <b>批量导入</b><span>一次导入多张，自动排队识图</span>
                        </button>
                        <button type="button" class="studio-card" @click="openEmotionsModal(); closeBiliPanel()">
                            <svg class="ico"><use href="#i-grid"/></svg>
                            <b>分类管理</b><span>新增、改名、调整情绪分类</span>
                        </button>
                        <button type="button" class="studio-card" @click="openCharactersModal(); closeBiliPanel()">
                            <svg class="ico"><use href="#i-star"/></svg>
                            <b>角色管理</b><span>按角色归档表情库</span>
                        </button>
                        <button type="button" class="studio-card" @click="openSourceModal(); closeBiliPanel()">
                            <svg class="ico"><use href="#i-folder"/></svg>
                            <b>外部源</b><span>导入表情包合集与远程图库</span>
                        </button>
                        <button type="button" class="studio-card" @click="runStorageCleanup(); closeBiliPanel()">
                            <svg class="ico"><use href="#i-trash"/></svg>
                            <b>存储清理</b><span>清理临时文件与孤儿图片</span>
                        </button>
                    </div>
                </template>
            </div>
        </div>
    </div>

    <nav class="mobile-tabbar" aria-label="底部导航">
        <button type="button" class="tabbar-item" :class="{ active: activeSection === 'library' }"
            @click="switchSection('library')">
            <svg class="ico"><use href="#i-grid"/></svg><span>首页</span>
        </button>
        <button type="button" class="tabbar-item" :class="{ active: activeSection === 'pending' }"
            @click="switchSection('pending')">
            <svg class="ico"><use href="#i-shield"/></svg><span>审核</span>
            <i v-if="pendingStats.pending" class="tabbar-badge">{{ pendingStats.pending }}</i>
        </button>
        <button type="button" class="tabbar-item tabbar-publish" @click="openUploadModal">
            <span class="tabbar-publish-icon"><svg class="ico"><use href="#i-upload"/></svg></span><span>投稿</span>
        </button>
        <button type="button" class="tabbar-item" @click="goBiliFavorites()">
            <svg class="ico"><use href="#i-star"/></svg><span>收藏</span>
        </button>
        <button type="button" class="tabbar-item" @click="showAlert('个人中心')">
            <span class="tabbar-avatar">偷</span><span>我的</span>
        </button>
    </nav>

    <button v-if="biliShowBackTop" class="back-top" type="button" aria-label="回到顶部" @click="scrollBiliTop">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="m6 14 6-6 6 6" />
        </svg>
    </button>
</div>
</template>
<template v-else>
<div class="fo-chassis" aria-hidden="true">
    <div class="fo-bezel fo-bezel-l">
        <div class="fo-thumbwheel"></div>
        <div class="fo-screw"></div>
        <div class="fo-screw"></div>
    </div>
    <div class="fo-bezel fo-bezel-r">
        <div class="fo-radio-knob"></div>
        <div class="fo-screw"></div>
        <div class="fo-screw"></div>
    </div>
    <div class="fo-bezel fo-bezel-b"></div>
</div>
<header class="codex-header">
    <div class="fo-boot" aria-hidden="true">
        <span>ROBCO INDUSTRIES UNIFIED OPERATING SYSTEM</span>
        <span>COPYRIGHT 2075-2077 ROBCO INDUSTRIES</span>
    </div>
    <div class="header-left">
    <div class="header-title">
        <button class="mobile-menu-button" type="button" @click="sidebarOpen = true"
            :aria-label="t('pages.dashboard.actions.open_navigation', 'Open navigation')"
            aria-controls="dashboard-sidebar" :aria-expanded="sidebarOpen">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16" />
            </svg>
        </button>
        <button class="header-sidebar-toggle" type="button" @click="toggleSidebarCollapsed"
            aria-controls="dashboard-sidebar" :aria-expanded="!sidebarCollapsed"
            :aria-label="sidebarCollapsed ? t('pages.dashboard.actions.expand_sidebar', '展开侧栏') : t('pages.dashboard.actions.collapse_sidebar', '折叠侧栏')"
            :title="sidebarCollapsed ? t('pages.dashboard.actions.expand_sidebar', '展开侧栏') : t('pages.dashboard.actions.collapse_sidebar', '折叠侧栏')">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true">
                <rect x="2.5" y="3.5" width="19" height="17" rx="2.5" stroke-width="1.7" />
                <path d="M9 3.5v17" stroke-width="1.7" />
                <path v-if="sidebarCollapsed" d="m14 9 3 3-3 3" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.7" />
                <path v-else d="m17 9-3 3 3 3" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.7" />
            </svg>
        </button>
        <div class="fo-vaultboy fo-vaultboy-header" role="img" aria-label="Vault Boy"></div>
        <div class="header-text">
            <h1>{{ t('pages.dashboard.header.brand', 'Henry\\'s Spoils') }}</h1>
            <p>{{ t('pages.dashboard.header.subtitle', 'Sticker Manager') }}</p>
        </div>
        <div class="fo-title" aria-hidden="true">
            <span class="fo-title-os">PIP-BOY 3000 MK IV</span>
            <span class="fo-title-sub">VAULT-TEC  //  HOLOTAPE ARCHIVE</span>
        </div>
    </div>

    <nav class="header-nav" aria-label="Sections">
        <button type="button" class="section-tab" :class="{ active: activeSection === 'library' }" @click="switchSection('library')">
            <svg class="section-tab-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"
                    d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
            </svg>
            <span class="section-tab-label">{{ t('pages.dashboard.sections.library', 'Library') }}</span>
        </button>
        <button type="button" class="section-tab" :class="{ active: activeSection === 'pending' }" @click="switchSection('pending')">
            <svg class="section-tab-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"
                    d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
            </svg>
            <span class="section-tab-label">{{ t('pages.dashboard.sections.pending', 'Pending') }}</span>
            <span v-if="pendingStats.pending > 0" class="section-badge">{{ pendingStats.pending }}</span>
        </button>
    </nav>
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
        <div class="theme-picker">
            <button type="button" class="theme-menu-btn"
                @click.stop="themePickerOpen = !themePickerOpen"
                :aria-label="t('pages.dashboard.themes.title', 'Theme')"
                :title="t('pages.dashboard.themes.title', 'Theme')">
                <svg fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"
                        d="M7 21a4.000 4.000 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4.000 4.000 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" />
                </svg>
            </button>
            <div v-if="themePickerOpen" class="theme-popover" @click.stop>
                <p class="theme-save-hint">{{ t('pages.dashboard.themes.save_hint', 'Your pick is saved as the default theme.') }}</p>
                <div class="theme-group-label">{{ t('pages.dashboard.themes.group_original', 'Original') }}</div>
                <button v-for="opt in originalThemeOptions" :key="opt.value" type="button" class="theme-option"
                    :class="{ active: themeMode === opt.value }" @click="setThemeMode(opt.value); themePickerOpen = false">
                    <span v-if="opt.swatch" class="theme-swatch"
                        :style="{ background: 'linear-gradient(135deg, ' + opt.swatch.split(',')[0] + ' 50%, ' + opt.swatch.split(',')[1] + ' 50%)' }"></span>
                    <span v-else class="theme-swatch" style="background: conic-gradient(#161b2a 50%, #faf8f3 50%)"></span>
                    {{ t('pages.dashboard.themes.' + opt.key, opt.fallback) }}
                    <span v-if="themeMode === opt.value" class="theme-default-tag">{{ t('pages.dashboard.themes.saved_default', 'Default') }}</span>
                </button>
                <div class="theme-group-label">{{ t('pages.dashboard.themes.group_game', 'Game inventory') }}</div>
                <button v-for="opt in gameThemeOptions" :key="opt.value" type="button" class="theme-option"
                    :class="{ active: themeMode === opt.value }" @click="setThemeMode(opt.value); themePickerOpen = false">
                    <span class="theme-swatch"
                        :style="{ background: 'linear-gradient(135deg, ' + opt.swatch.split(',')[0] + ' 50%, ' + opt.swatch.split(',')[1] + ' 50%)' }"></span>
                    {{ t('pages.dashboard.themes.' + opt.key, opt.fallback) }}
                    <span v-if="themeMode === opt.value" class="theme-default-tag">{{ t('pages.dashboard.themes.saved_default', 'Default') }}</span>
                </button>
                <div class="theme-group-label">{{ t('pages.dashboard.themes.group_special', 'Featured') }}</div>
                <button v-for="opt in specialThemeOptions" :key="opt.value" type="button" class="theme-option"
                    :class="{ active: themeMode === opt.value }" @click="setThemeMode(opt.value); themePickerOpen = false">
                    <span class="theme-swatch"
                        :style="{ background: 'linear-gradient(135deg, ' + opt.swatch.split(',').join(', ') + ')' }"></span>
                    {{ t('pages.dashboard.themes.' + opt.key, opt.fallback) }}
                    <span v-if="themeMode === opt.value" class="theme-default-tag">{{ t('pages.dashboard.themes.saved_default', 'Default') }}</span>
                </button>
            </div>
        </div>
    </div>
    <nav class="fo-pip-tabs" aria-label="Pip-Boy">
        <span class="fo-pip-tab is-chrome">STAT</span>
        <button type="button" class="fo-pip-tab" :class="{ active: activeSection === 'library' }" @click="switchSection('library')">INV</button>
        <button type="button" class="fo-pip-tab" :class="{ active: activeSection === 'pending' }" @click="switchSection('pending')">DATA</button>
        <span class="fo-pip-tab is-chrome">MAP</span>
        <span class="fo-pip-tab is-chrome">RADIO</span>
    </nav>
</header>

<div class="main-container" :class="{ 'sidebar-collapsed': sidebarCollapsed }">
    <div v-if="sidebarOpen" class="mobile-sidebar-backdrop" @click="sidebarOpen = false"></div>
    <aside id="dashboard-sidebar" class="sidebar" :class="{ 'is-open': sidebarOpen, 'is-collapsed': sidebarCollapsed }">
        <div class="section-switcher">
            <button type="button" class="section-tab" :class="{ active: activeSection === 'pending' }" @click="switchSection('pending')">
                <svg class="section-tab-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"
                        d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
                </svg>
                <span class="section-tab-label">{{ t('pages.dashboard.sections.pending', 'Pending') }}</span>
                <span v-if="pendingStats.pending > 0" class="section-badge">{{ pendingStats.pending }}</span>
            </button>
            <button type="button" class="section-tab" :class="{ active: activeSection === 'library' }" @click="switchSection('library')">
                <svg class="section-tab-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"
                        d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                </svg>
                <span class="section-tab-label">{{ t('pages.dashboard.sections.library', 'Library') }}</span>
            </button>
        </div>

        <template v-if="activeSection === 'library'">
            <div class="sidebar-divider"></div>
            <div class="sidebar-title">{{ t('pages.dashboard.layout.libraries', '表情库') }}</div>
            <nav class="library-navigation" :aria-label="t('pages.dashboard.layout.libraries', '表情库')">
                <button v-for="key in ['', 'general', 'favorites', 'characters']" :key="key" type="button"
                    class="library-nav-item" :class="{ active: selectedLibrary === key }"
                    :aria-current="selectedLibrary === key ? 'page' : undefined"
                    :aria-label="libraryTitle(key) + ' ' + (key ? libraries[key] : libraries.general + libraries.favorites + libraries.characters)"
                    :title="libraryTitle(key)" @click="selectLibrary(key)">
                    <span class="library-nav-mark" aria-hidden="true">{{ key === 'favorites' ? '☆' : key === 'characters' ? '♙' : key === 'general' ? '▦' : '◫' }}</span>
                    <span class="library-nav-label">{{ libraryTitle(key) }}</span>
                    <span class="library-nav-count">{{ key ? libraries[key] : libraries.general + libraries.favorites + libraries.characters }}</span>
                </button>
            </nav>
            <div class="sidebar-divider"></div>
            <div class="sidebar-title">{{ t('pages.dashboard.layout.emotions', '当前库 · 情绪分类') }}</div>
            <div class="category-list">
                <button type="button" class="category-item" :class="{ active: selectedCategory === '' }" @click="selectLibraryCategory('')">
                    <span class="category-name">{{ t('pages.dashboard.layout.all_emotions', '全部情绪') }}</span>
                    <span class="category-count">{{ libraryCategoryTotal }}</span>
                </button>
                <button v-for="cat in categories" :key="cat.key" type="button" class="category-item"
                    :class="{ active: selectedCategory === cat.key }" :style="catAccent(cat.key)"
                    @click="selectLibraryCategory(cat.key)">
                    <span class="cat-dot"></span>
                    <span class="category-name">{{ cat.name }}</span>
                    <span class="category-count">{{ cat.count }}</span>
                </button>
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
                <button type="button" class="category-item" :class="{ active: pendingCategory === '' }"
                    @click="selectPendingCategory('')">
                    <span class="category-name">{{ t('pages.dashboard.categories.all', 'All') }}</span>
                    <span class="category-count">{{ pendingCategoryTotal }}</span>
                </button>
                <button v-for="cat in pendingCategories" :key="cat.key" type="button" class="category-item"
                    :class="{ active: pendingCategory === cat.key }" :style="catAccent(cat.key)"
                    @click="selectPendingCategory(cat.key)">
                    <span class="cat-dot"></span>
                    <span class="category-name">{{ cat.name }}</span>
                    <span class="category-count">{{ cat.count }}</span>
                </button>
            </div>
        </template>
    </aside>

    <main class="inventory-panel">
        <div class="modal-panel-corner-bl"></div>
        <div class="modal-panel-corner-br"></div>

        <template v-if="activeSection === 'library'">
            <header class="library-heading">
                <div class="library-heading-main">
                    <span class="library-eyebrow">{{ t('pages.dashboard.layout.collection', '表情收藏册') }}</span>
                    <h2>{{ libraryTitle(selectedLibrary) }} <span class="library-result-count">{{ total }}</span></h2>
                    <div class="library-location">
                        <span>{{ selectedCharacter ? characterLabel(selectedCharacter) : t('pages.dashboard.layout.all_characters', '全部角色') }}</span>
                        <span aria-hidden="true">/</span>
                        <span>{{ selectedCategory ? getCategoryName(selectedCategory) : t('pages.dashboard.layout.all_emotions', '全部情绪') }}</span>
                    </div>
                </div>
                <details class="library-policy">
                    <summary>{{ t('pages.dashboard.libraries.quota', '自动淘汰计数') }} <strong>{{ libraries.automatic }} / {{ automaticLimit }}</strong></summary>
                    <p>{{ t('pages.dashboard.libraries.hint', '仅通用库普通表情参与自动淘汰，收藏和角色库另行管理。') }}</p>
                </details>
            </header>
            <details class="usage-guide" :open="stats.total === 0">
                <summary>{{ t('pages.dashboard.guide.title', '开始使用 · 让表情接上你的话') }}</summary>
                <ol>
                    <li><strong>{{ t('pages.dashboard.guide.collect', '准备图库') }}</strong><span>{{ t('pages.dashboard.guide.collect_hint', '上传图片并分析，或在插件配置中开启收集、选择视觉模型。') }}</span></li>
                    <li><strong>{{ t('pages.dashboard.guide.review', '确认内容') }}</strong><span>{{ t('pages.dashboard.guide.review_hint', '在待审核区通过图片；检查图上文字和“适合回应”的句子，再分配角色。') }}</span></li>
                    <li><strong>{{ t('pages.dashboard.guide.chat', '正常聊天') }}</strong><span>{{ t('pages.dashboard.guide.chat_hint', 'Bot 在聊天中结合语境主动用表情表达情绪、接梗和回应，选择贴合语境的图片。') }}</span></li>
                </ol>
                <p>{{ t('pages.dashboard.guide.pending_hint', '待审核图片不会参与发送。图片已入库但仍选不准时，可预览重新分析，再确认写回。') }}</p>
                <div class="usage-guide-actions">
                    <button type="button" class="codex-btn primary" @click="openUploadModal">{{ t('pages.dashboard.guide.upload', '上传并分析') }}</button>
                    <button type="button" class="codex-btn" @click="switchSection('pending')">{{ t('pages.dashboard.guide.pending', '去审核') }}</button>
                </div>
                <div v-if="setupInfo" class="usage-guide-status" aria-live="polite">
                    <span>{{ t('pages.dashboard.guide.auto', '自动发送') }} · {{ setupInfo.auto_send_enabled ? t('pages.dashboard.guide.on', '开启') : t('pages.dashboard.guide.off', '关闭') }}</span>
                    <span>{{ t('pages.dashboard.guide.context', '语境理解') }} · {{ setupInfo.context_enabled ? t('pages.dashboard.guide.on', '开启') : t('pages.dashboard.guide.off', '关闭') }}</span>
                    <span>{{ t('pages.dashboard.guide.embedding', '语义检索') }} · {{ setupInfo.embedding_enabled ? t('pages.dashboard.guide.on', '开启') : t('pages.dashboard.guide.off', '关闭') }}</span>
                </div>
            </details>
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

                <div class="toolbar-controls">
                    <div class="toolbar-group mobile-category-select">
                        <select v-model="selectedCategory" @change="fetchImages(1)" class="codex-input">
                            <option value="">{{ t('pages.dashboard.categories.all', 'All') }}</option>
                            <option v-for="cat in categories" :key="cat.key" :value="cat.key">{{ cat.name }}</option>
                        </select>
                    </div>

                    <div class="toolbar-group">
                        <select v-model="sortBy" @change="fetchImages(1)" class="codex-input toolbar-sort-select">
                            <option value="newest">{{ t('pages.dashboard.sort.newest', 'Newest') }}</option>
                            <option value="oldest">{{ t('pages.dashboard.sort.oldest', 'Oldest') }}</option>
                            <option value="least_used">{{ t('pages.dashboard.sort.least_used', '使用最少') }}</option>
                            <option value="most_used">{{ t('pages.dashboard.sort.most_used', 'Most Used') }}</option>
                            <option value="last_used">{{ t('pages.dashboard.sort.last_used', 'Last Used') }}</option>
                        </select>
                    </div>
                    <div class="toolbar-group">
                        <div class="view-toggle-group">
                            <button type="button" class="view-toggle-btn" :class="{ active: viewMode === 'grid' }"
                                @click="setViewMode('grid')"
                                :aria-label="t('pages.dashboard.view.grid', 'Grid view')"
                                :title="t('pages.dashboard.view.grid', 'Grid view')">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                        d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
                                </svg>
                            </button>
                            <button type="button" class="view-toggle-btn" :class="{ active: viewMode === 'list' }"
                                @click="setViewMode('list')"
                                :aria-label="t('pages.dashboard.view.list', 'List view')"
                                :title="t('pages.dashboard.view.list', 'List view')">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                        d="M4 6h16M4 10h16M4 14h16M4 18h16" />
                                </svg>
                            </button>
                        </div>
                    </div>
                </div>

                <div class="toolbar-actions">
                    <div class="toolbar-group">
                        <button @click="toggleBatchMode" class="codex-btn" :class="{ primary: isBatchMode }">
                            <svg style="width:16px;height:16px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                    d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                            </svg>
                            {{ isBatchMode ? t('pages.dashboard.actions.done', 'Done') : t('pages.dashboard.actions.batch', 'Batch') }}
                        </button>

                        <button v-if="reanalysisJob" @click="reanalysisOpen = true" class="codex-btn">
                            {{ t('pages.dashboard.reanalysis.progress', 'Analysis progress') }} · {{ reanalysisJob.processed }}/{{ reanalysisJob.total }}
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
                        <button @click="openSourceModal" class="codex-btn source-open-btn">
                            <svg style="width:16px;height:16px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                    d="M8 12h8m-4-4v8M5 19a4 4 0 010-8 7 7 0 0113.8 1.6A3.5 3.5 0 0118.5 19H5z" />
                            </svg>
                            {{ t('pages.dashboard.actions.sources', 'Sources') }}
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

            <div v-if="selectedLibrary === 'characters'" class="character-filter-bar">
                <span class="character-filter-label">{{ t('pages.dashboard.characters.title', '角色') }}</span>
                <button type="button" class="character-chip" :class="{ active: !selectedCharacter }"
                    @click="selectLibrary('characters')">
                    {{ t('pages.dashboard.characters.all', '全部角色') }}
                </button>
                <button v-for="item in characters" :key="item.key" type="button" class="character-chip"
                    :class="{ active: selectedCharacter === item.key }"
                    @click="selectLibraryCharacter(item.key)">
                    {{ item.name }}
                    <span class="character-chip-count">{{ item.count || 0 }}</span>
                </button>
                <button type="button" @click="openCharactersModal" class="codex-btn character-manage-btn">
                    {{ t('pages.dashboard.actions.characters', '角色管理') }}
                </button>
            </div>

            <div v-if="loading" class="skeleton-grid">
                <div v-for="n in pageSize" :key="n" class="skeleton-card">
                    <div class="skeleton-image"></div>
                    <div class="skeleton-text"></div>
                </div>
            </div>

            <div v-else-if="images.length === 0" class="empty-state">
                <div class="fo-vaultboy fo-vaultboy-empty" role="img" aria-label="Vault Boy"></div>
                <svg class="empty-state-icon" style="width:64px;height:64px;opacity:0.3;margin-bottom:16px" fill="none" stroke="currentColor"
                    viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"
                        d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                </svg>
                <p style="font-family:var(--font-display);font-size:var(--fs-xl)">{{ t('pages.dashboard.empty.library_title', 'No stickers yet') }}</p>
                <p style="font-size:var(--fs-md);margin-top:8px;color:var(--text-muted)">{{ t('pages.dashboard.empty.library_hint', 'Click "Add" to upload a new sticker.') }}</p>
            </div>

            <div v-else class="inventory-grid library-grid" :class="{ 'list-mode': viewMode === 'list' }">
                <div v-for="(img, idx) in images" :key="img.hash" class="item-slot"
                    :class="{ selected: selectedImages.has(img.hash) }"
                    :style="{ animationDelay: Math.min(idx * 16, 200) + 'ms' }"
                    role="button" tabindex="0"
                    :aria-label="img.desc || getCategoryName(img.category)"
                    @mouseenter="onItemSlotEnter($event)"
                    @contextmenu.prevent.stop="openContextMenu($event, img)"
                    @click="isBatchMode ? toggleSelection(img) : openPreview(img)"
                    @keydown.enter.prevent="isBatchMode ? toggleSelection(img) : openPreview(img)"
                    @keydown.space.prevent="isBatchMode ? toggleSelection(img) : openPreview(img)">
                    <div v-if="isBatchMode" class="batch-indicator">
                        <svg v-if="selectedImages.has(img.hash)" style="width:12px;height:12px" fill="none"
                            stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 13l4 4L19 7" />
                        </svg>
                    </div>

                    <button class="item-more-btn" type="button" @click.stop="openContextMenu($event, img)"
                        @contextmenu.prevent.stop="openContextMenu($event, img)"
                        :aria-label="t('pages.dashboard.actions.more', '更多操作')"
                        :title="t('pages.dashboard.actions.more', '更多操作')">
                        <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                            <circle cx="12" cy="5" r="1.8" />
                            <circle cx="12" cy="12" r="1.8" />
                            <circle cx="12" cy="19" r="1.8" />
                        </svg>
                    </button>

                    <button @keydown.enter.stop @keydown.space.stop class="favorite-btn" :class="{ active: img.is_favorite }" @click.stop="toggleFavorite(img)"
                        :aria-label="img.is_favorite ? t('pages.dashboard.actions.unfavorite', 'Remove favorite') : t('pages.dashboard.actions.favorite', 'Favorite')"
                        :title="img.is_favorite ? t('pages.dashboard.actions.unfavorite', 'Remove favorite') : t('pages.dashboard.actions.favorite', 'Favorite')">
                        <svg viewBox="0 0 24 24">
                            <path
                                d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
                        </svg>
                    </button>

                    <div class="item-image" :data-hash="img.hash">
                        <div v-if="!imageDataUrls[img.hash]" class="image-placeholder"
                            :style="{ backgroundColor: hashToColor(img.hash) }"></div>
                        <img v-else :src="imageDataUrls[img.hash]" loading="lazy" decoding="async"
                            :alt="img.desc" class="fade-in">
                    </div>

                    <div class="item-info">
                        <div class="list-main">
                            <div class="item-category">
                                <emotion-labels :item="img" :label="getCategoryName" :compact="true" />
                                <span v-if="img.character" class="item-character-tag">{{ characterLabel(img.character) }}</span>
                            </div>
                            <div v-if="viewMode === 'list'" class="list-desc">{{ img.desc || t('pages.dashboard.messages.no_description', 'No description') }}</div>
                            <div v-if="viewMode === 'list' && (img.tags || []).length" class="list-tags">
                                <span v-for="tag in img.tags.slice(0, 4)" :key="tag" class="tag list-tag">{{ tag }}</span>
                            </div>
                        </div>
                        <div v-if="viewMode === 'list'" class="list-side">
                            <span class="list-use">{{ t('pages.dashboard.fields.use_count', 'Use Count') }} {{ img.use_count || 0 }}</span>
                            <span class="list-date">{{ formatDate(img.created_at) }}</span>
                            <span class="scope-pill" :class="img.scope_mode === 'local' ? 'local' : 'public'">{{
                                getScopeLabel(img.scope_mode) }}</span>
                        </div>
                        <div v-else class="item-meta-row">
                            <span class="card-usage">{{ t('pages.dashboard.fields.use_count', '使用次数') }} {{ img.use_count || 0 }}</span>
                            <span class="scope-pill" :class="img.scope_mode === 'local' ? 'local' : 'public'">{{ getScopeLabel(img.scope_mode) }}</span>
                        </div>
                        <div v-if="viewMode !== 'list'" class="card-bottom-row">
                            <span :title="retentionLabel(img)">{{ retentionLabel(img) }}</span>
                            <time :title="t('pages.dashboard.fields.created_at', '入库时间')">{{ formatDate(img.created_at) }}</time>
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
            <header class="library-heading pending-heading">
                <div class="library-heading-main">
                    <span class="library-eyebrow">{{ t('pages.dashboard.pending.pool', '待审核池') }}</span>
                    <h2>{{ t('pages.dashboard.sections.pending', '审核区') }} <span class="library-result-count">{{ pendingTotal }}</span></h2>
                    <div class="library-location">
                        <span class="loc-metric"><span class="loc-label">{{ t('pages.dashboard.pending.progress', 'Pending') }}</span><span class="loc-num">{{ pendingStats.pending }}</span><span class="loc-cap">/ {{ pendingStats.capacity }}</span></span>
                        <span class="loc-metric"><span class="loc-label">{{ t('pages.dashboard.pending.queued_count', 'Queued') }}</span><span class="loc-num">{{ pendingStats.queued }}</span></span>
                        <span class="loc-metric"><span class="loc-label">{{ t('pages.dashboard.pending.processing_count', 'Analyzing') }}</span><span class="loc-num">{{ pendingStats.processing }}</span></span>
                        <span class="loc-metric" :class="{ 'loc-metric--warn': (pendingStats.failed || 0) > 0 }"><span class="loc-label">{{ t('pages.dashboard.pending.failed_count', 'Failed') }}</span><span class="loc-num">{{ pendingStats.failed }}</span></span>
                    </div>
                </div>
                <div class="pending-heading-side">
                    <div class="capacity-row">
                        <span class="capacity-name">{{ t('pages.dashboard.pending.capacity', 'Capacity') }}</span>
                        <span class="capacity-num">{{ pendingStats.pending }}<i>/ {{ pendingStats.capacity }}</i></span>
                    </div>
                    <div class="progress-track">
                        <div class="progress-fill"
                            :style="{ width: Math.min(100, pendingStats.pending / pendingStats.capacity * 100) + '%' }"
                            :class="{ full: pendingStats.paused }"></div>
                    </div>
                    <div v-if="pendingStats.paused" class="paused-banner">
                        <svg style="width:13px;height:13px;flex:none" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5v14M15 5v14" />
                        </svg>
                        <span>{{ t('pages.dashboard.pending.paused_hint', 'Stealing is paused and will resume after review.') }}</span>
                    </div>
                </div>
            </header>

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
                        <button @click="toggleSelectAllPending" class="codex-btn select-all-btn">
                            {{ allPendingSelected ? '☐ ' + t('pages.dashboard.actions.deselect_all', 'Deselect All') : '☑ ' + t('pages.dashboard.actions.select_all', 'Select All') }}
                        </button>
                        <button @click="approvePendingBatch" class="codex-btn approve-batch-btn">✅ {{ t('pages.dashboard.actions.approve_all', 'Approve All') }}</button>
                        <button @click="rejectPendingBatch(false)" class="codex-btn reject-batch-btn">🗑 {{ t('pages.dashboard.actions.delete_all', 'Delete All') }}</button>
                        <button @click="rejectPendingBatch(true)" class="codex-btn reject-bl-batch-btn">🚫 {{ t('pages.dashboard.actions.delete_blacklist', 'Delete + Blacklist') }}</button>
                    </div>
                </div>
            </div>

            <div v-if="!pendingLoading && pendingImages.length" class="kbd-hints">
                <span><kbd>←</kbd><kbd>→</kbd>{{ t('pages.dashboard.kbd.navigate', 'Navigate') }}</span>
                <span><kbd>Enter</kbd>{{ t('pages.dashboard.kbd.preview', 'Preview') }}</span>
                <span><kbd>A</kbd>{{ t('pages.dashboard.actions.approve', 'Approve') }}</span>
                <span><kbd>R</kbd>{{ t('pages.dashboard.actions.delete', 'Delete') }}</span>
                <span><kbd>B</kbd>{{ t('pages.dashboard.actions.blacklist', 'Blacklist') }}</span>
                <span><kbd>E</kbd>{{ t('pages.dashboard.actions.edit_approve', 'Edit & approve') }}</span>
                <span><kbd>Esc</kbd>{{ t('pages.dashboard.kbd.clear_focus', 'Clear focus') }}</span>
            </div>

            <div v-if="pendingLoading" class="skeleton-grid pending-skeleton">
                <div v-for="n in pendingPageSize" :key="n" class="skeleton-card">
                    <div class="skeleton-image"></div>
                    <div class="skeleton-text"></div>
                </div>
            </div>

            <div v-else-if="pendingImages.length === 0" class="empty-state pending-empty">
                <div class="fo-vaultboy fo-vaultboy-empty" role="img" aria-label="Vault Boy"></div>
                <div class="empty-illustration" aria-hidden="true">
                    <svg class="empty-state-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M5 13l4 4L19 7" />
                    </svg>
                </div>
                <p class="empty-title">{{ t('pages.dashboard.empty.pending_title', 'No pending stickers') }}</p>
                <p class="empty-hint">{{ t('pages.dashboard.empty.pending_hint', 'Newly stolen stickers will wait here for review.') }}</p>
            </div>

            <div v-else class="pending-grid">
                <div v-for="(item, idx) in pendingImages" :key="item.id" class="pending-card"
                    :class="{ selected: pendingSelectedImages.has(item.id), 'kbd-focused': focusedPendingId === item.id }"
                    :data-pending-id="item.id"
                    :style="{ animationDelay: Math.min(idx * 16, 200) + 'ms' }"
                    @click="pendingBatchMode ? togglePendingSelection(item) : openPendingPreview(item)">
                    <div v-if="pendingBatchMode" class="batch-indicator">
                        <svg v-if="pendingSelectedImages.has(item.id)" style="width:12px;height:12px" fill="none"
                            stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 13l4 4L19 7" />
                        </svg>
                    </div>

                    <div class="pending-image" :data-hash="item.hash">
                        <div v-if="!imageDataUrls[item.hash]" class="image-placeholder"
                            :style="{ backgroundColor: hashToColor(item.hash) }"></div>
                        <img v-else :src="imageDataUrls[item.hash]" loading="lazy" decoding="async"
                            :alt="item.desc" class="fade-in">
                    </div>

                    <div class="pending-info">
                        <div class="pending-meta">
                            <emotion-labels :item="item" :label="getCategoryName" :compact="true" />
                            <span v-if="item.character" class="item-character-tag">{{ characterLabel(item.character) }}</span>
                            <span v-if="item.scope_mode === 'local'" class="scope-pill local">{{ t('pages.dashboard.scope.local_short', 'Local') }}</span>
                            <span class="pending-source">{{ item.source === 'auto' ? '🤖' : '👤' }}</span>
                        </div>
                        <div class="pending-summary">
                            <div v-if="item.desc" class="pending-desc">{{ item.desc }}</div>
                            <div class="pending-status" :title="pendingStatusLabel(item)">{{ pendingStatusLabel(item) }}</div>
                            <div v-if="item.analysis_error" class="pending-error" :title="item.analysis_error">{{ item.analysis_error }}</div>
                        </div>
                        <div class="pending-tags" v-if="(item.tags || []).length">
                            <span v-for="tag in item.tags" :key="tag" class="tag pending-tag">{{ tag }}</span>
                        </div>
                        <div class="pending-actions" v-if="!pendingBatchMode">
                            <button type="button" @click.stop="approvePending(item.id)"
                                :disabled="pendingApproving.has(item.id) || ['queued', 'processing'].includes(item.review_status)"
                                class="pending-btn approve-btn"
                                :aria-label="t('pages.dashboard.actions.approve', 'Approve')"
                                :data-tooltip="t('pages.dashboard.actions.approve', 'Approve')">
                                <svg aria-hidden="true" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5"
                                        d="M5 13l4 4L19 7" />
                                </svg>
                            </button>
                            <button type="button" @click.stop="openPendingEdit(item)"
                                :disabled="['queued', 'processing'].includes(item.review_status)"
                                class="pending-btn edit-btn"
                                :aria-label="t('pages.dashboard.actions.edit_approve', 'Edit & approve')"
                                :data-tooltip="t('pages.dashboard.actions.edit_approve', 'Edit & approve')">
                                <svg aria-hidden="true" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                        d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                                </svg>
                            </button>
                            <button type="button" @click.stop="rejectPending(item.id)"
                                :disabled="item.review_status === 'processing'"
                                class="pending-btn reject-btn"
                                :aria-label="t('pages.dashboard.actions.delete', 'Delete')"
                                :data-tooltip="t('pages.dashboard.actions.delete', 'Delete')">
                                <svg aria-hidden="true" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                        d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                </svg>
                            </button>
                            <button type="button" @click.stop="rejectPending(item.id, true)"
                                :disabled="item.review_status === 'processing'"
                                class="pending-btn reject-bl-btn"
                                :aria-label="t('pages.dashboard.actions.blacklist', 'Blacklist')"
                                :data-tooltip="t('pages.dashboard.actions.blacklist', 'Blacklist')">
                                <svg aria-hidden="true" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                        d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" />
                                </svg>
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
</template>

<div v-if="contextMenu.open && contextMenu.img" class="item-context-menu" role="menu"
    :style="{ left: contextMenu.x + 'px', top: contextMenu.y + 'px' }" @click.stop @contextmenu.prevent>
    <div class="context-menu-head">
        <span>{{ getCategoryName(contextMenu.img.category) }}</span>
        <small>{{ contextMenu.img.use_count || 0 }}×</small>
    </div>
    <button type="button" role="menuitem" @click="runContextAction('preview')">{{ t('pages.dashboard.context_menu.preview', '查看详情') }}</button>
    <button type="button" role="menuitem" @click="runContextAction('edit')">{{ t('pages.dashboard.actions.edit', '编辑') }}</button>
    <button type="button" role="menuitem" @click="runContextAction('favorite')">
        {{ contextMenu.img.is_favorite ? t('pages.dashboard.actions.unfavorite', '取消收藏') : t('pages.dashboard.actions.favorite', '收藏') }}
    </button>
    <button type="button" role="menuitem" @click="runContextAction('select')">
        {{ selectedImages.has(contextMenu.img.hash) ? t('pages.dashboard.context_menu.unselect', '取消选择') : t('pages.dashboard.context_menu.select', '加入批量选择') }}
    </button>
    <button type="button" role="menuitem" @click="runContextAction('download')">{{ t('pages.dashboard.actions.download', '下载') }}</button>
    <div class="context-menu-separator"></div>
    <button type="button" role="menuitem" class="danger" @click="runContextAction('delete')">{{ t('pages.dashboard.actions.delete', '删除') }}</button>
    <button type="button" role="menuitem" class="danger" @click="runContextAction('blacklist')">{{ t('pages.dashboard.actions.delete_blacklist', '删除并拉黑') }}</button>
</div>

<div class="fo-vaultboy fo-vaultboy-mascot" role="img" aria-label="Vault Boy"></div>
<footer class="fo-hud" aria-hidden="true">
    <div class="fo-hud-group">
        <span class="fo-hud-k">HP</span>
        <div class="fo-hud-bar"><i :style="{ width: hudHpPct }"></i></div>
        <span class="fo-hud-v">{{ stats.total || 0 }}</span>
    </div>
    <div class="fo-hud-group">
        <span class="fo-hud-k">AP</span>
        <div class="fo-hud-bar"><i :style="{ width: hudApPct }"></i></div>
        <span class="fo-hud-v">{{ pendingStats.pending || 0 }}</span>
    </div>
    <div class="fo-hud-group">
        <span class="fo-hud-k">LVL</span>
        <div class="fo-hud-bar fo-hud-xp"><i :style="{ width: hudXpPct }"></i></div>
        <span class="fo-hud-v">{{ stats.today || 0 }}</span>
    </div>
</footer>

<div v-if="previewOpen" class="modal-overlay" @click.self="closePreview">
    <div class="modal-panel" role="dialog" aria-modal="true"
        :aria-label="previewSource === 'pending' ? t('pages.dashboard.modal.review_pending', 'Review Sticker') : (isEditing ? t('pages.dashboard.modal.edit', 'Edit') : t('pages.dashboard.modal.details', 'Details'))">
        <div class="modal-panel-corner-bl"></div>
        <div class="modal-panel-corner-br"></div>

        <div class="modal-header">
            <h2>{{ previewSource === 'pending' ? t('pages.dashboard.modal.review_pending', 'Review Sticker') : (isEditing ? t('pages.dashboard.modal.edit', 'Edit') : t('pages.dashboard.modal.details', 'Details')) }}</h2>
            <button @click="closePreview" class="modal-close" :aria-label="t('pages.dashboard.actions.close', 'Close')">
                <svg style="width:20px;height:20px" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
            </button>
        </div>

        <div class="modal-content">
            <div class="preview-stack">
                <div class="item-detail">
                <div class="item-preview">
                    <button v-if="previewListLength > 1 && !isEditing" @click.stop="prevImage" class="nav-btn left"
                        :aria-label="t('pages.dashboard.pagination.prev', 'Previous')">
                        <svg style="width:24px;height:24px" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
                        </svg>
                    </button>

                    <img :key="previewItem?.hash" class="fade-in"
                        :src="originalDataUrls[previewItem?.hash] || imageDataUrls[previewItem?.hash] || PLACEHOLDER"
                        :alt="previewItem?.desc"
                        :class="{ zoomable: previewZoom === 1, zoomed: previewZoom > 1, panning: isPanning }"
                        :style="{ transform: previewTransform }"
                        decoding="async"
                        @wheel.prevent="onPreviewWheel"
                        @mousedown.prevent="startPan"
                        @dblclick.prevent="toggleZoom">
                    <div v-if="previewLoading" class="preview-loading">{{ t('pages.dashboard.messages.loading_original', 'Loading full image…') }}</div>

                    <button v-if="previewListLength > 1 && !isEditing" @click.stop="nextImage" class="nav-btn right"
                        :aria-label="t('pages.dashboard.pagination.next', 'Next')">
                        <svg style="width:24px;height:24px" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
                        </svg>
                    </button>

                    <div v-if="previewZoom > 1" class="zoom-indicator">{{ Math.round(previewZoom * 100) }}%</div>
                    <span v-if="previewListLength > 1 && !isEditing" class="preview-counter">{{ (previewSource === 'pending' ? pendingImages : images).indexOf(previewItem) + 1 }} / {{ previewListLength }}</span>
                </div>

                <div v-if="!isEditing" class="item-stats">
                    <section class="stat-group">
                        <h3 class="stat-group-title">{{ t('pages.dashboard.detail.group_recognition', '图像识别') }}</h3>
                        <div class="stat-row">
                            <span class="stat-name">{{ t('pages.dashboard.fields.emotions', 'Emotions') }}</span>
                            <button v-if="previewSource !== 'pending'" type="button" class="emotion-edit-btn editable-detail" @click="startEdit('category')" :title="t('pages.dashboard.actions.edit', 'Edit')"><emotion-labels :item="previewItem" :label="getCategoryName" /></button>
                            <emotion-labels v-else :item="previewItem" :label="getCategoryName" />
                        </div>
                        <div class="stat-block">
                            <span class="stat-name">{{ t('pages.dashboard.fields.description', 'Description') }}</span>
                            <div class="desc-quote" :class="{ 'editable-detail': previewSource !== 'pending' }" :role="previewSource !== 'pending' ? 'button' : undefined" :tabindex="previewSource !== 'pending' ? 0 : undefined" @click="previewSource !== 'pending' && startEdit('desc')" @keydown.enter.prevent="previewSource !== 'pending' && startEdit('desc')">
                                <p style="margin:0;color:var(--text-main)">
                                    {{ previewItem?.desc || t('pages.dashboard.messages.no_description', 'No description') }}
                                </p>
                            </div>
                        </div>
                        <div class="stat-row">
                            <span class="stat-name">{{ t('pages.dashboard.fields.overlay_text', '图上文字') }}</span>
                            <button v-if="previewSource !== 'pending'" type="button" class="detail-edit-link" @click="startEdit('overlay')">{{ previewItem?.overlay_text || t('pages.dashboard.messages.no_overlay_text', '无') }}</button>
                            <span v-else class="stat-value">{{ previewItem?.overlay_text || t('pages.dashboard.messages.no_overlay_text', '无') }}</span>
                        </div>
                    </section>

                    <section class="stat-group">
                        <h3 class="stat-group-title">{{ t('pages.dashboard.detail.group_tags', '标签与场景') }}</h3>
                        <div class="stat-block">
                            <span class="stat-name">{{ t('pages.dashboard.fields.tags', 'Tags') }}</span>
                            <div class="item-tags" :class="{ 'editable-detail': previewSource !== 'pending' }" :role="previewSource !== 'pending' ? 'button' : undefined" :tabindex="previewSource !== 'pending' ? 0 : undefined" @click="previewSource !== 'pending' && startEdit('tags')" @keydown.enter.prevent="previewSource !== 'pending' && startEdit('tags')">
                                <span v-for="tag in (previewItem?.tags || [])" :key="tag" class="tag">
                                    {{ tag }}
                                </span>
                                <span v-if="!(previewItem?.tags || []).length" class="stat-empty">{{ t('pages.dashboard.messages.no_tags', 'No tags') }}</span>
                            </div>
                        </div>
                        <div class="stat-block">
                            <span class="stat-name">{{ t('pages.dashboard.fields.scenes', 'Scenes') }}</span>
                            <div class="item-tags" :class="{ 'editable-detail': previewSource !== 'pending' }" :role="previewSource !== 'pending' ? 'button' : undefined" :tabindex="previewSource !== 'pending' ? 0 : undefined" @click="previewSource !== 'pending' && startEdit('scene')" @keydown.enter.prevent="previewSource !== 'pending' && startEdit('scene')">
                                <span v-for="scene in (previewItem?.scenes || [])" :key="scene" class="tag scene-tag">
                                    {{ scene }}
                                </span>
                                <span v-if="!(previewItem?.scenes || []).length" class="stat-empty">{{ t('pages.dashboard.messages.no_scenes', 'No scenes') }}</span>
                            </div>
                        </div>
                    </section>

                    <section class="stat-group">
                        <h3 class="stat-group-title">{{ t('pages.dashboard.detail.group_usage', '使用与归属') }}</h3>
                        <div class="stat-row">
                            <span class="stat-name">{{ t('pages.dashboard.fields.scope', 'Scope') }}</span>
                            <span class="stat-value">
                                <span class="scope-pill" :class="[previewItem?.scope_mode === 'local' ? 'local' : 'public', { 'editable-detail': previewSource !== 'pending' }]" :role="previewSource !== 'pending' ? 'button' : undefined" :tabindex="previewSource !== 'pending' ? 0 : undefined" @click="previewSource !== 'pending' && startEdit('scope')" @keydown.enter.prevent="previewSource !== 'pending' && startEdit('scope')">{{
                                    getScopeLabel(previewItem?.scope_mode) }}</span>
                            </span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-name">{{ t('pages.dashboard.fields.character', '角色') }}</span>
                            <button v-if="previewSource !== 'pending'" type="button" class="detail-edit-link" @click="startEdit('character')">{{ characterLabel(previewItem?.character) }}</button>
                            <span v-else class="stat-value">{{ characterLabel(previewItem?.character) }}</span>
                        </div>
                        <div v-if="previewSource !== 'pending'" class="stat-row">
                            <span class="stat-name">{{ t('pages.dashboard.fields.favorite', 'Favorite') }}</span>
                            <button class="favorite-toggle-btn" :class="{ active: previewItem?.is_favorite }"
                                @click="toggleFavorite(previewItem)">
                                <svg style="width:11px;height:11px" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round" viewBox="0 0 24 24" aria-hidden="true">
                                    <path d="m12 3.6 2.5 5.2 5.7.8-4.1 4 1 5.7-5.1-2.7-5.1 2.7 1-5.7-4.1-4 5.7-.8L12 3.6Z" />
                                </svg>
                                {{ previewItem?.is_favorite ? t('pages.dashboard.messages.favorited', 'Favorited') : t('pages.dashboard.messages.not_favorited', 'Not favorited') }}
                            </button>
                        </div>
                        <div v-if="previewSource !== 'pending'" class="stat-row">
                            <span class="stat-name">{{ t('pages.dashboard.fields.use_count', 'Use Count') }}</span>
                            <span class="stat-value">{{ previewItem?.use_count || 0 }} {{ t('pages.dashboard.units.times', 'times') }}</span>
                        </div>
                        <div v-if="previewSource !== 'pending'" class="stat-row">
                            <span class="stat-name">{{ t('pages.dashboard.fields.last_used', 'Last Used') }}</span>
                            <span class="stat-value">{{ previewItem?.last_used_at ? formatDate(previewItem.last_used_at) : t('pages.dashboard.messages.never_used', 'Never used') }}</span>
                        </div>
                    </section>

                    <section class="stat-group">
                        <h3 class="stat-group-title">{{ t('pages.dashboard.detail.group_file', '文件与来源') }}</h3>
                        <div class="stat-row">
                            <span class="stat-name">{{ t('pages.dashboard.fields.origin', 'Origin') }}</span>
                            <span class="stat-value">{{ formatItemOrigin(previewItem) }}</span>
                        </div>
                        <div v-if="previewItem?.width || previewItem?.format || previewItem?.bytes" class="stat-row">
                            <span class="stat-name">{{ t('pages.dashboard.fields.image_meta', 'Image') }}</span>
                            <span class="stat-value">{{ [previewItem?.width && previewItem?.height ? previewItem.width + '×' + previewItem.height : '', previewItem?.format ? String(previewItem.format).toUpperCase() : '', formatBytes(previewItem?.bytes)].filter(Boolean).join(' · ') }}</span>
                        </div>
                        <div v-if="previewItem?.add_method || previewItem?.reviewed_at" class="stat-row">
                            <span class="stat-name">{{ t('pages.dashboard.fields.added', 'Added') }}</span>
                            <span class="stat-value">{{ formatAddMethod(previewItem?.add_method) }}{{ previewItem?.reviewed_at ? ' · ' + t('pages.dashboard.fields.reviewed_at', 'Reviewed') + ' ' + formatDate(previewItem.reviewed_at) : '' }}</span>
                        </div>
                        <div v-if="previewItem?.source_url" class="stat-row">
                            <span class="stat-name">{{ t('pages.dashboard.fields.source', 'Source') }}</span>
                            <span class="stat-value" style="word-break:break-all">{{ previewItem.source_url }}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-name">{{ t('pages.dashboard.fields.created_at', 'Added At') }}</span>
                            <span class="stat-value">{{ formatDate(previewItem?.created_at) }}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-name">{{ t('pages.dashboard.fields.id', '编号') }}</span>
                            <span class="stat-value" style="font-size:var(--fs-xs);word-break:break-all">{{ previewItem?.hash?.slice(0, 16) }}...</span>
                        </div>
                    </section>

                    <div v-if="previewSource !== 'pending'" class="vlm-reanalysis-panel">
                        <div class="vlm-reanalysis-toolbar">
                            <div>
                                <div class="vlm-reanalysis-title">{{ t('pages.dashboard.analysis.reanalysis_title', 'VLM re-analysis') }}</div>
                                <div class="vlm-reanalysis-hint">{{ t('pages.dashboard.analysis.reanalysis_hint', 'Run the vision model again and compare the stored labels before applying them.') }}</div>
                            </div>
                            <button @click="reanalyzePreview" class="codex-btn" :disabled="vlmReanalysisLoading || vlmReanalysisApplying">
                                <span v-if="vlmReanalysisLoading">{{ t('pages.dashboard.actions.analyzing', 'Analyzing...') }}</span>
                                <span v-else>{{ t('pages.dashboard.actions.reanalyze', 'Re-analyze') }}</span>
                            </button>
                        </div>

                        <div v-if="vlmReanalysisError" class="vlm-reanalysis-error">{{ vlmReanalysisError }}</div>

                        <div v-if="vlmReanalysisResult" class="vlm-compare-grid">
                            <section v-for="side in ['before', 'after']" :key="side" class="vlm-compare-card" :class="{ proposed: side === 'after' }">
                                <div class="vlm-compare-label">{{ side === 'before' ? 'A · ' + t('pages.dashboard.analysis.current', 'Current') : 'B · ' + t('pages.dashboard.analysis.new_result', 'New VLM result') }}</div>
                                <dl>
                                    <div><dt>{{ t('pages.dashboard.fields.emotions', 'Emotions') }}</dt><dd><emotion-labels :item="vlmReanalysisResult[side]" :label="getCategoryName" /></dd></div>
                                    <div v-for="field in ['description', 'overlay_text', 'tags', 'scenes']" :key="field">
                                        <dt>{{ t('pages.dashboard.fields.' + field, field) }}</dt>
                                        <dd>{{ formatAnalysisValue(vlmReanalysisResult[side][field]) }}</dd>
                                    </div>
                                </dl>
                            </section>
                        </div>

                        <div v-if="vlmReanalysisResult" class="vlm-reanalysis-actions">
                            <button @click="applyReanalysis" class="codex-btn primary" :disabled="vlmReanalysisApplying">
                                <span v-if="vlmReanalysisApplying">{{ t('pages.dashboard.actions.saving', 'Applying...') }}</span>
                                <span v-else>{{ t('pages.dashboard.actions.apply_vlm', 'Apply B result') }}</span>
                            </button>
                            <span class="vlm-reanalysis-preserve">{{ t('pages.dashboard.analysis.reanalysis_preserve', 'Character, scope, favorites and usage history stay unchanged.') }}</span>
                        </div>
                    </div>
                </div>
                <div v-else class="item-stats inline-edit-fields">
                    <div style="margin-bottom:20px">
                        <label
                            class="form-label">{{ t('pages.dashboard.fields.category', 'Category') }}</label>
                        <select v-model="editForm.category" data-edit-field="category" class="codex-input">
                            <option v-for="cat in categories" :key="cat.key" :value="cat.key">{{ cat.name }}</option>
                        </select>
                    </div>

                    <div style="margin-bottom:20px">
                        <label class="form-label">{{ t('pages.dashboard.fields.secondary_emotions', 'Secondary emotions') }}</label>
                        <div class="item-tags editable-labels">
                            <span v-for="key in editForm.emotions.filter((key) => key !== editForm.category)" :key="key" class="tag editable-label">
                                {{ getCategoryName(key) }}
                                <button type="button" @click="removeEditEmotion(key)" :aria-label="t('pages.dashboard.actions.remove', 'Remove') + ' ' + getCategoryName(key)">×</button>
                            </span>
                        </div>
                        <div class="label-add-row">
                            <select v-model="newEditEmotion" class="codex-input" :aria-label="t('pages.dashboard.fields.secondary_emotions', 'Secondary emotions')">
                                <option value="">{{ t('pages.dashboard.placeholders.select', 'Select...') }}</option>
                                <option v-for="cat in categories.filter((item) => item.key !== editForm.category && !editForm.emotions.includes(item.key))" :key="cat.key" :value="cat.key">{{ cat.name }}</option>
                            </select>
                            <button type="button" class="codex-btn" :disabled="!newEditEmotion" :aria-label="t('pages.dashboard.actions.add', 'Add') + ' ' + t('pages.dashboard.fields.secondary_emotions', 'Secondary emotions')" @click="addEditEmotion">+</button>
                        </div>
                    </div>

                    <div style="margin-bottom:20px">
                        <label
                            class="form-label">{{ t('pages.dashboard.fields.scope', 'Scope') }}</label>
                        <select v-model="editForm.scope_mode" data-edit-field="scope" class="codex-input">
                            <option value="public">{{ t('pages.dashboard.scope.public', 'Public') }}</option>
                            <option value="local">{{ t('pages.dashboard.scope.local', 'Local only') }}</option>
                        </select>
                        <div class="form-hint">{{ t('pages.dashboard.fields.origin', 'Origin') }}: {{ formatItemOrigin(previewItem) }}</div>
                    </div>

                    <div style="margin-bottom:20px">
                        <label
                            class="form-label">{{ t('pages.dashboard.fields.description', 'Description') }}</label>
                        <textarea v-model="editForm.desc" data-edit-field="desc" class="codex-input" rows="3"></textarea>
                    </div>

                    <div style="margin-bottom:20px">
                        <label class="form-label">{{ t('pages.dashboard.fields.character', '角色') }}</label>
                        <select v-model="editForm.character" data-edit-field="character" class="codex-input">
                            <option value="">{{ t('pages.dashboard.characters.unassigned', '通用表情库') }}</option>
                            <option v-for="item in characters" :key="item.key" :value="item.key">{{ item.name }}</option>
                        </select>
                    </div>

                    <div style="margin-bottom:20px">
                        <label class="form-label">{{ t('pages.dashboard.fields.overlay_text', '图上文字') }}</label>
                        <input v-model="editForm.overlay_text" data-edit-field="overlay" type="text" class="codex-input"
                            :placeholder="t('pages.dashboard.placeholders.overlay_text', '图上印的字')">
                    </div>

                    <div style="margin-bottom:20px">
                        <label
                            class="form-label">{{ t('pages.dashboard.fields.scenes', 'Scenes') }} ({{ t('pages.dashboard.messages.scene_separator_hint', 'comma separated') }})</label>
                        <div class="item-tags editable-labels">
                            <span v-for="item in parseSceneList(editForm.scene)" :key="item" class="tag editable-label">
                                {{ item }}<button type="button" @click="removeEditLabel('scene', item)" :aria-label="t('pages.dashboard.actions.remove', 'Remove') + ' ' + item">×</button>
                            </span>
                        </div>
                        <div class="label-add-row">
                            <input v-model="newEditScene" data-edit-field="scene" type="text" class="codex-input" :aria-label="t('pages.dashboard.fields.scenes', 'scenes')" @keydown.enter.prevent="addEditLabel('scene')">
                            <button type="button" class="codex-btn" :disabled="!newEditScene.trim()" :aria-label="t('pages.dashboard.actions.add', 'Add') + ' ' + t('pages.dashboard.fields.scenes', 'Scenes')" @click="addEditLabel('scene')">+</button>
                        </div>
                    </div>

                    <div style="margin-bottom:20px">
                        <label
                            class="form-label">{{ t('pages.dashboard.fields.tags', 'Tags') }} ({{ t('pages.dashboard.messages.tag_separator_hint', 'comma separated') }})</label>
                        <div class="item-tags editable-labels">
                            <span v-for="item in parseListField(editForm.tags)" :key="item" class="tag editable-label">
                                {{ item }}<button type="button" @click="removeEditLabel('tags', item)" :aria-label="t('pages.dashboard.actions.remove', 'Remove') + ' ' + item">×</button>
                            </span>
                        </div>
                        <div class="label-add-row">
                            <input v-model="newEditTag" data-edit-field="tags" type="text" class="codex-input" :aria-label="t('pages.dashboard.fields.tags', 'tags')" @keydown.enter.prevent="addEditLabel('tags')">
                            <button type="button" class="codex-btn" :disabled="!newEditTag.trim()" :aria-label="t('pages.dashboard.actions.add', 'Add') + ' ' + t('pages.dashboard.fields.tags', 'Tags')" @click="addEditLabel('tags')">+</button>
                        </div>
                    </div>
                </div>
                </div>
            </div>


        </div>

        <div class="modal-actions">
            <template v-if="previewSource === 'pending'">
                <button @click="rejectPreviewPending(true)" class="codex-btn danger" style="flex:1"
                    :title="t('pages.dashboard.actions.blacklist', 'Blacklist')">
                    <svg style="width:16px;height:16px" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                            d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" />
                    </svg>
                    {{ t('pages.dashboard.actions.blacklist', 'Blacklist') }}
                </button>
                <button @click="rejectPreviewPending(false)" class="codex-btn danger" style="flex:1">
                    <svg style="width:16px;height:16px" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                            d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                    {{ t('pages.dashboard.actions.delete', 'Delete') }}
                </button>
                <button @click="editPreviewPending" class="codex-btn" style="flex:1">
                    <svg style="width:16px;height:16px" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                            d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                    </svg>
                    {{ t('pages.dashboard.actions.edit', 'Edit') }}
                </button>
                <button @click="approvePreviewPending" class="codex-btn primary" style="flex:1">
                    <svg style="width:16px;height:16px" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M5 13l4 4L19 7" />
                    </svg>
                    {{ t('pages.dashboard.actions.approve', 'Approve') }}
                </button>
            </template>
            <template v-else-if="!isEditing">
                <a href="#" @click.prevent="downloadImage(previewItem)" class="codex-btn" style="flex:1">
                    <svg style="width:16px;height:16px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                            d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                    </svg>
                    {{ t('pages.dashboard.actions.download', 'Download') }}
                </a>
                <button @click="startEdit" class="codex-btn primary" style="flex:1">
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
                <span class="preview-actions-sep" aria-hidden="true"></span>
                <button @click="deleteImage(previewItem)" class="codex-btn danger danger-quiet" style="flex:1">
                    <svg style="width:16px;height:16px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                            d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                    {{ t('pages.dashboard.actions.delete', 'Delete') }}
                </button>
                <button @click="deleteImage(previewItem, true)" class="codex-btn danger danger-quiet" style="flex:1"
                    :title="t('pages.dashboard.actions.blacklist', 'Blacklist')">
                    <svg style="width:16px;height:16px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                            d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" />
                    </svg>
                    {{ t('pages.dashboard.actions.blacklist', 'Blacklist') }}
                </button>
            </template>
            <template v-else>
                <button @click="cancelEdit" :disabled="savingEdit" class="codex-btn" style="flex:1">{{ t('pages.dashboard.actions.cancel', 'Cancel') }}</button>
                <button @click="saveEdit" :disabled="savingEdit" class="codex-btn primary" style="flex:1">{{ t('pages.dashboard.actions.save', 'Save') }}</button>
            </template>
        </div>
    </div>
</div>

<div v-if="uploadOpen" class="modal-overlay" @click.self="closeUploadModal">
    <div class="modal-panel modal-md" role="dialog" aria-modal="true" :aria-label="t('pages.dashboard.modal.add_sticker', 'Add Sticker')">
        <div class="modal-panel-corner-bl"></div>
        <div class="modal-panel-corner-br"></div>

        <div class="modal-header">
            <h2>{{ t('pages.dashboard.modal.add_sticker', 'Add Sticker') }}</h2>
            <button @click="closeUploadModal" class="modal-close" :aria-label="t('pages.dashboard.actions.close', 'Close')">
                <svg style="width:20px;height:20px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
            </button>
        </div>

        <form @submit.prevent="submitUpload" class="modal-form">
            <div class="modal-form-body">
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
                    <p style="margin:0;color:var(--text-muted);font-family:var(--font-display);text-align:center">{{ t('pages.dashboard.upload.click_to_upload', 'Click to upload an image') }}</p>
                </div>
            </div>

            <div class="mt-20">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                    <label
                        class="form-label inline">{{ t('pages.dashboard.fields.category', 'Category') }} *</label>
                    <button v-if="uploadFile" type="button" @click.prevent="analyzeImage"
                        :disabled="analyzing || !uploadFile" class="codex-btn"
                        style="font-size:var(--fs-2xs);padding:6px 12px;min-height:auto">
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

            <div v-if="analysisScenes.length" class="analysis-result mt-16">
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

            <div class="mt-16">
                <label
                    class="form-label">{{ t('pages.dashboard.fields.scenes', 'Scenes') }}</label>
                <input v-model="uploadForm.scene" type="text" class="codex-input"
                    :placeholder="t('pages.dashboard.placeholders.upload_scene', 'Example: office, chat window, late night')">
                <p class="hint-text" style="margin:8px 0 0">{{ t('pages.dashboard.messages.scene_input_hint', 'You can separate scenes with commas or semicolons.') }}</p>
            </div>

            <div class="mt-16">
                <label
                    class="form-label">{{ t('pages.dashboard.fields.tags', 'Tags') }}</label>
                <input v-model="uploadForm.tags" type="text" class="codex-input"
                    :placeholder="t('pages.dashboard.placeholders.upload_tags', 'Example: cute, funny')">
            </div>

            <div class="mt-16">
                <label
                    class="form-label">{{ t('pages.dashboard.fields.description', 'Description') }}</label>
                <textarea v-model="uploadForm.desc" class="codex-input" rows="2"
                    :placeholder="t('pages.dashboard.placeholders.upload_desc', 'Describe this sticker...')"></textarea>
            </div>

            <div class="mt-16">
                <label class="form-label">{{ t('pages.dashboard.fields.overlay_text', '图上文字') }}</label>
                <input v-model="uploadForm.overlay_text" type="text" class="codex-input"
                    :placeholder="t('pages.dashboard.placeholders.overlay_text', '图上印的字')">
            </div>

            <div class="mt-16">
                <label class="form-label">{{ t('pages.dashboard.fields.character', '角色') }}</label>
                <select v-model="uploadForm.character" class="codex-input">
                    <option value="">{{ t('pages.dashboard.characters.unassigned', '通用表情库') }}</option>
                    <option v-for="item in characters" :key="item.key" :value="item.key">{{ item.name }}</option>
                </select>
            </div>

            <div v-if="uploadError" class="error-banner">
                {{ uploadError }}
            </div>
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
    <div class="modal-panel modal-lg" role="dialog" aria-modal="true" :aria-label="t('pages.dashboard.modal.batch_import', 'Batch Import Stickers')">
        <div class="modal-panel-corner-bl"></div>
        <div class="modal-panel-corner-br"></div>

        <div class="modal-header">
            <h2>{{ t('pages.dashboard.modal.batch_import', 'Batch Import Stickers') }}</h2>
            <button @click="closeBatchUploadModal" class="modal-close" :aria-label="t('pages.dashboard.actions.close', 'Close')">
                <svg style="width:20px;height:20px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
            </button>
        </div>

        <form @submit.prevent="submitBatchUpload" class="modal-pad">
            <div v-if="!batchTaskId">
                <div class="upload-area batch-upload-area" :class="{ 'is-drag-active': batchDragActive }"
                    @click="triggerBatchFileInput"
                    @dragenter="onBatchDragEnter"
                    @dragover="onBatchDragOver"
                    @dragleave="onBatchDragLeave"
                    @drop="onBatchDrop"
                    class="roomy">
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
                                <p style="margin:0 0 4px 0;color:var(--gold-primary);font-family:var(--font-display)">{{ t('pages.dashboard.batch.selected_count', 'Selected {count} image(s)').replace('{count}', batchFiles.length) }}</p>
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
                        <p style="margin:0;color:var(--text-muted);font-family:var(--font-display);text-align:center">{{ t('pages.dashboard.batch.drag_upload', 'Click or drag to upload multiple images') }}</p>
                        <p style="margin:8px 0 0;color:var(--text-muted);font-size:0.85rem;text-align:center">{{ t('pages.dashboard.batch.supported_formats', 'Supports PNG, JPG, GIF, WEBP, BMP') }}</p>
                    </div>
                </div>

                <div class="mt-20">
                    <label
                        class="form-label">{{ t('pages.dashboard.batch.default_category', 'Default Category') }} *</label>
                    <select v-model="batchUploadForm.emotion" class="codex-input"
                        :disabled="batchUploadForm.autoAnalyze" required>
                        <option value="">{{ t('pages.dashboard.placeholders.select_category', 'Select a category...') }}</option>
                        <option v-for="emo in availableEmotions" :key="emo.key" :value="emo.key">{{ emo.name || emo.key }}</option>
                    </select>
                    <p class="hint-text" style="margin:8px 0 0">{{ t('pages.dashboard.batch.default_category_hint', 'Images will be saved into this category unless auto analyze is enabled.') }}</p>
                </div>

                <div class="mt-16">
                    <label class="form-label">{{ t('pages.dashboard.fields.character', '角色') }}</label>
                    <select v-model="batchUploadForm.character" class="codex-input">
                        <option value="">{{ t('pages.dashboard.characters.unassigned', '通用表情库') }}</option>
                        <option v-for="item in characters" :key="item.key" :value="item.key">{{ item.name }}</option>
                    </select>
                    <p class="hint-text" style="margin:8px 0 0">{{ t('pages.dashboard.batch.character_hint', '这批图会打上该角色标记。可稍后在图库里再改。') }}</p>
                </div>

                <div class="mt-16">
                    <label style="display:flex;align-items:center;gap:8px;cursor:pointer">
                        <input type="checkbox" v-model="batchUploadForm.autoAnalyze" class="codex-checkbox"
                            :disabled="batchUploadForm.emotion !== ''">
                        <span style="font-size:0.85rem;color:var(--text-main)">{{ t('pages.dashboard.batch.auto_analyze', 'Auto analyze each image and classify automatically') }}</span>
                    </label>
                    <p v-if="batchUploadForm.emotion !== ''"
                        style="margin:4px 0 0 24px;font-size:0.75rem;color:var(--gold-dim)">{{ t('pages.dashboard.batch.auto_analyze_disabled_hint', 'Clear the selected category before enabling auto analyze.') }}</p>
                    <p v-if="batchUploadForm.autoAnalyze"
                        style="margin:8px 0 0 24px;font-size:0.75rem;color:#f59e0b;padding:8px;background:rgba(245,158,11,0.1);border-radius:4px">
                        {{ t('pages.dashboard.batch.auto_analyze_warning', 'Images are analyzed one at a time using the configured model request interval.') }}
                    </p>
                </div>

                <div v-if="batchUploadError" class="error-banner">
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

            <div v-else class="modal-pad">
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

<div v-if="sourceOpen" class="modal-overlay" @click.self="closeSourceModal">
    <div class="modal-panel modal-lg source-modal" role="dialog" aria-modal="true" :aria-label="t('pages.dashboard.sources.title', 'External Sources')">
        <div class="modal-panel-corner-bl"></div>
        <div class="modal-panel-corner-br"></div>
        <div class="modal-header">
            <div>
                <h2>{{ t('pages.dashboard.sources.title', 'External Sources') }}</h2>
                <p class="source-header-sub">Meme Manager · AstrBot Meme Pack · GitHub Repo · HTTPS JSON API</p>
            </div>
            <button @click="closeSourceModal" class="modal-close" :aria-label="t('pages.dashboard.actions.close', 'Close')">
                <svg style="width:20px;height:20px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
            </button>
        </div>

        <div class="modal-pad source-layout">
            <div class="source-safety-note">
                <strong>{{ t('pages.dashboard.sources.copy_title', 'Safe local copy') }}</strong>
                <span>{{ t('pages.dashboard.sources.copy_hint', 'Imported images are validated, deduplicated, and copied into this plugin. Original plugin files stay untouched.') }}</span>
            </div>

            <div class="source-input-grid">
                <label class="source-input-card">
                    <span class="source-card-kicker">PACK</span>
                    <strong>{{ t('pages.dashboard.sources.pack', 'Import resource pack') }}</strong>
                    <span>{{ sourceFile?.name || t('pages.dashboard.sources.pack_hint', 'ZIP / .meme-pack export') }}</span>
                    <input type="file" accept=".zip,.meme-pack,application/zip" @change="handleSourceFile">
                </label>
                <div class="source-input-card">
                    <span class="source-card-kicker">GITHUB</span>
                    <strong>{{ t('pages.dashboard.sources.github', 'GitHub meme repository') }}</strong>
                    <div class="source-api-row">
                        <input v-model="sourceForm.github" type="text" class="codex-input"
                            placeholder="https://github.com/owner/meme-pack or owner/repo">
                        <button type="button" class="codex-btn" @click="inspectGitHubSource"
                            :disabled="sourceLoading || !sourceForm.github">
                            {{ t('pages.dashboard.sources.preflight', 'Preflight') }}
                        </button>
                    </div>
                </div>
                <div class="source-input-card">
                    <span class="source-card-kicker">API</span>
                    <strong>{{ t('pages.dashboard.sources.api', 'External JSON catalog') }}</strong>
                    <div class="source-api-row">
                        <input v-model="sourceForm.endpoint" type="url" class="codex-input"
                            placeholder="https://example.com/memes.json">
                        <button type="button" class="codex-btn" @click="inspectExternalApi"
                            :disabled="sourceLoading || !sourceForm.endpoint">
                            {{ t('pages.dashboard.sources.preflight', 'Preflight') }}
                        </button>
                    </div>
                </div>
            </div>

            <div class="source-format-note">
                <strong>{{ t('pages.dashboard.sources.format_title', 'Accepted pack format') }}</strong>
                <span>{{ t('pages.dashboard.sources.format_hint', 'Preferred: AstrBot Meme Pack v2 export or a repository containing manifest.json + memes/; semantic_metadata.json is supported. A ZIP with supported images is also accepted. previews/ are ignored.') }}</span>
            </div>

            <div v-if="sourceError" class="error-banner">{{ sourceError }}</div>
            <div v-if="sourceLoading" class="source-loading-line">
                <span class="batch-spinner"></span>
                {{ t('pages.dashboard.sources.checking', 'Checking source...') }}
            </div>

            <section v-if="sourceInspection" class="source-inspection">
                <div class="source-inspection-head">
                    <div>
                        <span class="source-card-kicker">PREFLIGHT OK</span>
                        <h3>{{ sourceInspection.name }}</h3>
                    </div>
                    <div class="source-metrics">
                        <span><b>{{ sourceInspection.item_count }}</b> images</span>
                        <span><b>{{ formatBytes(sourceInspection.total_bytes) || '—' }}</b></span>
                        <span><b>{{ sourceInspection.categories?.length || 0 }}</b> categories</span>
                    </div>
                </div>
                <div v-if="sourceInspection.manifest?.license || sourceInspection.manifest?.attribution || sourceInspection.manifest?.author"
                    class="source-license-note">
                    <span v-if="sourceInspection.manifest?.license">{{ t('pages.dashboard.sources.license', 'License') }}: {{ sourceInspection.manifest.license }}</span>
                    <span v-if="sourceInspection.manifest?.attribution || sourceInspection.manifest?.author">{{ t('pages.dashboard.sources.attribution', 'Attribution') }}: {{ sourceInspection.manifest.attribution || sourceInspection.manifest.author }}</span>
                </div>

                <div v-if="sourceInspection.capacity?.would_exceed_limit" class="source-warning">
                    {{ t('pages.dashboard.sources.capacity_note', 'This pack is larger than the native collection limit. External copies are protected and use a separate retention class.') }}
                </div>
                <div v-if="sourceInspection.warnings?.length" class="source-warning">
                    <div v-for="warning in sourceInspection.warnings" :key="warning">{{ warning }}</div>
                </div>

                <div v-if="sourceInspection.categories?.length" class="source-map-grid">
                    <label v-for="sourceCategory in sourceInspection.categories" :key="sourceCategory">
                        <span>{{ sourceCategory }}</span>
                        <select v-model="sourceCategoryMap[sourceCategory]" class="codex-input">
                            <option value="">{{ t('pages.dashboard.sources.auto_map', 'Auto map') }}</option>
                            <option v-for="cat in categories" :key="cat.key" :value="cat.key">{{ cat.name }}</option>
                        </select>
                    </label>
                </div>

                <div class="source-options">
                    <label>
                        <input type="checkbox" v-model="sourceForm.review"
                            :disabled="sourceDefaults.review_forced">
                        {{ t('pages.dashboard.sources.review', 'Send imports to Pending first') }}
                    </label>
                    <label>
                        {{ t('pages.dashboard.sources.scope', 'Scope') }}
                        <select v-model="sourceForm.scope_mode" class="codex-input">
                            <option value="public">{{ t('pages.dashboard.scope.public', 'Public') }}</option>
                            <option value="local">{{ t('pages.dashboard.scope.local', 'Local only') }}</option>
                        </select>
                    </label>
                    <input v-if="sourceForm.scope_mode === 'local'" v-model="sourceForm.origin_target"
                        class="codex-input source-origin-input" placeholder="group:123456 / user:123456">
                    <label class="source-character-toggle">
                        <input type="checkbox" v-model="sourceForm.assign_character">
                        {{ t('pages.dashboard.sources.assign_character', 'Assign a character') }}
                    </label>
                    <input v-if="sourceForm.assign_character" v-model="sourceForm.character"
                        list="source-character-options" class="codex-input source-character-input"
                        :placeholder="t('pages.dashboard.sources.character_placeholder', 'Existing or new character key')">
                    <span v-if="sourceForm.assign_character" class="source-character-hint">
                        {{ t('pages.dashboard.sources.character_hint', 'Existing keys are reused; a new key is created during import.') }}
                    </span>
                    <datalist id="source-character-options">
                        <option v-for="item in characters" :key="item.key" :value="item.key">{{ item.name }}</option>
                    </datalist>
                    <button type="button" class="codex-btn primary" @click="startSourceImport()"
                        :disabled="sourceLoading || sourceJob?.status === 'queued' || sourceJob?.status === 'running'">
                        {{ t('pages.dashboard.sources.import', 'Import Source') }}
                    </button>
                </div>
            </section>

            <section v-if="sourceJob" class="source-job" :class="'is-' + sourceJob.status">
                <div class="source-job-head">
                    <strong>{{ sourceJob.source_name || t('pages.dashboard.sources.import_job', 'Import job') }}</strong>
                    <span>{{ sourceJob.status }}</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" :style="{ width: (sourceJob.total ? sourceJob.processed / sourceJob.total * 100 : 0) + '%' }"></div>
                </div>
                <div class="source-job-stats">
                    <span>{{ sourceJob.processed }} / {{ sourceJob.total }}</span>
                    <span class="source-ok">+{{ sourceJob.imported }} imported</span>
                    <span>{{ sourceJob.duplicates }} duplicates</span>
                    <span v-if="sourceJob.pending">{{ sourceJob.pending }} pending</span>
                    <span v-if="sourceJob.failed" class="source-failed">{{ sourceJob.failed }} failed</span>
                    <button v-if="sourceJob.status === 'queued' || sourceJob.status === 'running'"
                        type="button" class="codex-btn" @click="cancelSourceJob">Cancel</button>
                </div>
                <div v-if="sourceJob.error" class="error-banner">{{ sourceJob.error }}</div>
                <div v-if="sourceJob.errors?.length" class="source-job-errors">
                    <div v-for="error in sourceJob.errors" :key="error">{{ error }}</div>
                </div>
            </section>

            <section class="source-registry">
                <div class="source-section-head">
                    <div>
                        <span class="source-card-kicker">REGISTRY</span>
                        <h3>{{ t('pages.dashboard.sources.available', 'Available sources') }}</h3>
                    </div>
                    <button type="button" class="codex-btn" @click="fetchSources" :disabled="sourceLoading">↻</button>
                </div>
                <div v-if="!sourceList.length && !sourceLoading" class="source-empty">
                    {{ t('pages.dashboard.sources.empty', 'No registered sources. Same-instance Meme Manager packs appear here automatically.') }}
                </div>
                <div v-for="source in sourceList" :key="source.source_id" class="source-row">
                    <div class="source-row-main">
                        <strong>{{ source.name }}</strong>
                        <span>{{ source.source_type }} · {{ source.item_count || 0 }} items</span>
                        <small>{{ source.endpoint }}</small>
                    </div>
                    <span class="source-status" :class="'is-' + source.status">{{ source.status }}</span>
                    <div class="source-row-actions">
                        <button type="button" class="codex-btn" @click="inspectSource(source)">
                            {{ t('pages.dashboard.sources.preflight', 'Preflight') }}
                        </button>
                        <button type="button" class="codex-btn primary" @click="syncSource(source)">
                            {{ source.discovered ? t('pages.dashboard.sources.import', 'Import') : t('pages.dashboard.sources.sync', 'Sync') }}
                        </button>
                        <button v-if="!source.discovered" type="button" class="codex-btn danger"
                            @click="forgetSource(source)">{{ t('pages.dashboard.sources.forget', 'Forget') }}</button>
                    </div>
                </div>
            </section>
        </div>
    </div>
</div>

<div v-if="emotionsOpen" class="modal-overlay" @click.self="closeEmotionsModal">
    <div class="modal-panel modal-lg" role="dialog" aria-modal="true" :aria-label="t('pages.dashboard.modal.category_manager', 'Category Manager')">
        <div class="modal-panel-corner-bl"></div>
        <div class="modal-panel-corner-br"></div>

        <div class="modal-header">
            <h2>{{ t('pages.dashboard.modal.category_manager', 'Category Manager') }}</h2>
            <button @click="closeEmotionsModal" class="modal-close" :aria-label="t('pages.dashboard.actions.close', 'Close')">
                <svg style="width:20px;height:20px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
            </button>
        </div>

        <div class="modal-pad category-manager">
            <section class="category-add-panel">
                <h3 class="category-add-title">{{ t('pages.dashboard.categories.add_new', 'Add Category') }}</h3>
                <div class="category-add-grid">
                    <input v-model="newEmotion.key" maxlength="48" autocapitalize="none" spellcheck="false"
                        pattern="[a-z][a-z0-9_-]{0,47}" :placeholder="t('pages.dashboard.placeholders.category_key', 'Key (e.g. happy)')" class="codex-input">
                    <input v-model="newEmotion.name" maxlength="40" :placeholder="t('pages.dashboard.placeholders.category_name', 'Name (e.g. Happy)')" class="codex-input">
                    <input v-model="newEmotion.desc" maxlength="200" :placeholder="t('pages.dashboard.placeholders.category_desc', 'Description (optional)')" class="codex-input">
                    <button @click="addEmotion" :disabled="!newEmotion.key || addingEmotion" class="codex-btn primary">
                        {{ addingEmotion ? '...' : t('pages.dashboard.actions.add', 'Add') }}
                    </button>
                </div>
                <p class="hint-text category-add-hint">{{ t('pages.dashboard.categories.key_hint', 'Key 以英文字母开头，仅使用小写字母、数字、_、-，最长48字符。显示名称可以使用中文。') }}</p>
            </section>

            <div class="category-manager-list">
                <div v-for="cat in availableEmotions" :key="cat.key" class="category-manager-row">
                    <div class="category-manager-main">
                        <div class="category-manager-name">
                            <span class="category-manager-label">{{ cat.name || cat.key }}</span>
                            <span class="category-manager-key">{{ cat.key }}</span>
                        </div>
                        <p class="category-manager-desc">{{ cat.desc || t('pages.dashboard.messages.no_description', 'No description') }}</p>
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

<div v-if="charactersOpen" class="modal-overlay" @click.self="closeCharactersModal">
    <div class="modal-panel modal-lg" role="dialog" aria-modal="true" :aria-label="t('pages.dashboard.modal.character_manager', '角色管理')">
        <div class="modal-panel-corner-bl"></div>
        <div class="modal-panel-corner-br"></div>

        <div class="modal-header">
            <h2>{{ t('pages.dashboard.modal.character_manager', '角色管理') }}</h2>
            <button @click="closeCharactersModal" class="modal-close" :aria-label="t('pages.dashboard.actions.close', 'Close')">
                <svg style="width:20px;height:20px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
            </button>
        </div>
        <div class="modal-pad">
            <p class="hint-text" style="margin:0 0 16px">{{ t('pages.dashboard.characters.hint', '角色由你在管理面板手工归档，VLM 不会识别角色。先创建角色，再点卡片编辑或用批量分配。') }}</p>
            <div style="background:var(--bg-main);padding:16px;margin-bottom:20px;border:1px solid var(--gold-dark)">
                <h3 style="margin:0 0 16px 0;font-size:0.9rem;color:var(--gold-primary);font-family:var(--font-display)">{{ t('pages.dashboard.actions.characters', '角色管理') }}</h3>
                <div class="character-create-row">
                    <input v-model="newCharacter.key" class="codex-input"
                        :placeholder="t('pages.dashboard.placeholders.character_key', '标识（如 neurosama）')">
                    <input v-model="newCharacter.name" class="codex-input"
                        :placeholder="t('pages.dashboard.placeholders.character_name', '显示名（如 Neuro-sama）')">
                    <button @click="addCharacter" :disabled="addingCharacter || !newCharacter.key" class="codex-btn primary">
                        {{ addingCharacter ? '...' : t('pages.dashboard.actions.add', '添加') }}
                    </button>
                </div>
            </div>
            <p v-if="personaBindingError" role="alert" class="hint-text" style="color:var(--danger, #e36a72)">{{ personaBindingError }}</p>
            <label class="persona-isolation-control">
                <input type="checkbox" :checked="personaIsolation" :disabled="personaBindingBusy"
                    @change="setPersonaIsolation($event.target.checked, $event.target)">
                <span>{{ t('pages.dashboard.characters.persona_isolation', '启用人格表情库隔离') }}</span>
            </label>
            <p class="hint-text">{{ t('pages.dashboard.characters.persona_hint', '关联后，启用人格隔离时会使用该角色与通用表情。一个角色可以关联多个人格。') }}</p>
            <p v-if="!personaIds.length && !personaBindingError" class="hint-text">{{ t('pages.dashboard.characters.no_personas_available', '未找到人格，请先在 AstrBot 中创建。') }}</p>
            <div style="display:flex;flex-direction:column;gap:8px;max-height:400px;overflow-y:auto">
                <div v-for="item in characters" :key="item.key" class="character-list-row">
                    <div>
                        <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px">
                            <span style="font-family:var(--font-display);color:var(--gold-primary);font-size:1.1rem">{{ item.name || item.key }}</span>
                            <span style="font-size:0.75rem;color:var(--text-muted);background:var(--bg-main);padding:2px 8px;border:1px solid var(--gold-dark)">{{ item.key }}</span>
                        </div>
                        <p style="margin:0;color:var(--text-muted);font-size:0.85rem">{{ item.count || 0 }}</p>
                        <div class="persona-bindings">
                            <span v-for="id in characterPersonas(item.key)" :key="id" class="persona-binding-chip">
                                {{ id }}
                                <button type="button" :disabled="personaBindingBusy" @click="saveCharacterPersona(id, '')"
                                    :aria-label="t('pages.dashboard.characters.unlink_persona', '解除关联') + ' ' + id">×</button>
                            </span>
                            <span v-if="!characterPersonas(item.key).length" class="hint-text">{{ t('pages.dashboard.characters.no_persona', '未关联人格') }}</span>
                        </div>
                        <div class="persona-bind-controls">
                            <select :value="personaSelections[item.key] || ''" @change="personaSelections[item.key] = $event.target.value" class="codex-input" :disabled="personaBindingBusy || !personaIds.length"
                                :aria-label="t('pages.dashboard.characters.select_persona', '选择人格') + ' ' + (item.name || item.key)">
                                <option value="">{{ t('pages.dashboard.characters.select_persona', '选择人格') }}</option>
                                <option v-for="id in personaIds.filter((id) => !characterPersonas(item.key).includes(id))" :key="id" :value="id">{{ id }}{{ personaOwnerLabel(id) ? ' · ' + personaOwnerLabel(id) : '' }}</option>
                            </select>
                            <button type="button" class="codex-btn" :disabled="personaBindingBusy || !personaSelections[item.key]"
                                @click="saveCharacterPersona(personaSelections[item.key], item.key)">{{ t('pages.dashboard.characters.bind_persona', '关联') }}</button>
                        </div>
                    </div>
                    <button @click="deleteCharacter(item)" :disabled="deletingCharacterKey === item.key" class="codex-btn danger">
                        {{ deletingCharacterKey === item.key ? '...' : t('pages.dashboard.actions.delete', '删除') }}
                    </button>
                </div>
                <div v-if="characters.length === 0" class="empty-state" style="padding:40px">
                    <p>{{ t('pages.dashboard.empty.no_characters', '还没有角色。先创建一个，再到图库里分配。') }}</p>
                </div>
            </div>
        </div>
    </div>
</div>

<div v-if="batchMoveOpen" class="modal-overlay" @click.self="closeBatchMoveModal">
    <div class="modal-panel modal-narrow" role="dialog" aria-modal="true" :aria-label="t('pages.dashboard.modal.batch_move', 'Batch Move')">
        <div class="modal-panel-corner-bl"></div>
        <div class="modal-panel-corner-br"></div>

        <div class="modal-header">
            <h2>{{ t('pages.dashboard.modal.batch_move', 'Batch Move') }}</h2>
        </div>

        <div class="modal-pad">
            <p style="margin:0 0 16px 0;color:var(--text-muted)">{{ t('pages.dashboard.batch.selected_images', 'Selected {count} image(s)').replace('{count}', selectedImages.size) }}</p>

            <label
                class="form-label">{{ t('pages.dashboard.fields.target_category', 'Target Category') }}</label>
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

<div v-if="batchCharacterOpen" class="modal-overlay" @click.self="closeBatchCharacterModal">
    <div class="modal-panel modal-narrow" role="dialog" aria-modal="true" :aria-label="t('pages.dashboard.modal.batch_character', '批量分配角色')">
        <div class="modal-header">
            <h2>{{ t('pages.dashboard.modal.batch_character', '批量分配角色') }}</h2>
        </div>
        <div class="modal-pad">
            <label class="form-label">{{ t('pages.dashboard.fields.character', '角色') }}</label>
            <select v-model="batchTargetCharacter" class="codex-input" style="margin-bottom:20px">
                <option value="">{{ t('pages.dashboard.characters.unassigned', '通用表情库') }}</option>
                <option v-for="item in characters" :key="item.key" :value="item.key">{{ item.name }}</option>
            </select>
            <div style="display:flex;gap:12px">
                <button @click="closeBatchCharacterModal" class="codex-btn" style="flex:1">{{ t('pages.dashboard.actions.cancel', 'Cancel') }}</button>
                <button @click="confirmBatchCharacter" class="codex-btn primary" style="flex:1">{{ t('pages.dashboard.actions.save', 'Save') }}</button>
            </div>
        </div>
    </div>
</div>

<div v-if="batchScopeOpen" class="modal-overlay" @click.self="closeBatchScopeModal">
    <div class="modal-panel modal-narrow" role="dialog" aria-modal="true" :aria-label="t('pages.dashboard.modal.batch_scope', 'Batch Scope')">
        <div class="modal-panel-corner-bl"></div>
        <div class="modal-panel-corner-br"></div>

        <div class="modal-header">
            <h2>{{ t('pages.dashboard.modal.batch_scope', 'Batch Scope') }}</h2>
        </div>

        <div class="modal-pad">
            <p style="margin:0 0 16px 0;color:var(--text-muted)">{{ t('pages.dashboard.batch.selected_images', 'Selected {count} image(s)').replace('{count}', selectedImages.size) }}</p>

            <label
                class="form-label">{{ t('pages.dashboard.fields.target_scope', 'Target Scope') }}</label>
            <select v-model="batchScopeMode" class="codex-input" style="margin-bottom:20px">
                <option value="public">{{ t('pages.dashboard.scope.public', 'Public') }}</option>
                <option value="local">{{ t('pages.dashboard.scope.local', 'Local only') }}</option>
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
    <span style="font-family:var(--font-display);color:var(--gold-bright);font-size:1rem">{{ t('pages.dashboard.batch.selected_short', 'Selected {count}').replace('{count}', selectedImages.size) }}</span>
    <div style="width:1px;height:24px;background:var(--gold-dark)"></div>
    <button @click="selectAll" class="codex-btn" style="font-size:0.8rem;padding:8px 16px">{{ t('pages.dashboard.actions.select_all', 'Select All') }}</button>
    <button @click="startBatchReanalysis" :disabled="reanalysisBusy || reanalysisSubmitting" class="codex-btn primary" style="font-size:0.8rem;padding:8px 16px">
        {{ t('pages.dashboard.reanalysis.title', 'Batch re-analysis') }}
    </button>
    <button @click="openBatchMoveModal" class="codex-btn" style="font-size:0.8rem;padding:8px 16px">{{ t('pages.dashboard.actions.move', 'Move') }}</button>
    <button @click="openBatchCharacterModal" class="codex-btn" style="font-size:0.8rem;padding:8px 16px">{{ t('pages.dashboard.actions.assign_character', '分配角色') }}</button>
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

<div v-if="reanalysisOpen" class="modal-overlay" @click.self="reanalysisOpen = false">
    <section class="modal-panel modal-narrow reanalysis-panel" :class="{ 'reanalysis-review': reanalysisReady.length }" role="dialog" aria-modal="true" :aria-label="t('pages.dashboard.reanalysis.title', 'Batch re-analysis')">
        <div class="modal-header">
            <h2>{{ t('pages.dashboard.reanalysis.title', 'Batch re-analysis') }}</h2>
            <button type="button" class="modal-close" @click="reanalysisOpen = false" :aria-label="t('pages.dashboard.actions.close', 'Close')">×</button>
        </div>
        <div class="modal-pad">
            <p class="hint-text">{{ t('pages.dashboard.reanalysis.hint', 'Analysis results are saved only after you review and confirm each image. Categories, characters, favorites and scope are kept.') }}</p>
            <template v-if="reanalysisJob">
                <div class="reanalysis-status" aria-live="polite">
                    <strong>{{ t('pages.dashboard.reanalysis.' + reanalysisJob.status, reanalysisJob.status) }}</strong>
                    <span>{{ reanalysisJob.processed }} / {{ reanalysisJob.total }}</span>
                </div>
                <progress class="reanalysis-progress" :value="reanalysisJob.processed" :max="reanalysisJob.total" :aria-label="t('pages.dashboard.reanalysis.progress', 'Analysis progress')"></progress>
                <div class="reanalysis-counts">
                    <span>{{ t('pages.dashboard.reanalysis.analyzed', 'Analyzed') }} <b>{{ reanalysisJob.succeeded }}</b></span>
                    <span>{{ t('pages.dashboard.reanalysis.pending_review', 'Awaiting review') }} <b>{{ reanalysisJob.pending }}</b></span>
                    <span>{{ t('pages.dashboard.reanalysis.saved', 'Saved') }} <b>{{ reanalysisJob.saved }}</b></span>
                    <span>{{ t('pages.dashboard.reanalysis.failed_count', 'Failed') }} <b>{{ reanalysisJob.failed }}</b></span>
                    <span>{{ t('pages.dashboard.reanalysis.skipped', 'Skipped') }} <b>{{ reanalysisJob.skipped }}</b></span>
                </div>
                <p v-if="reanalysisJob.current" class="reanalysis-current">{{ reanalysisJob.current }}</p>
                <p class="hint-text">{{ t('pages.dashboard.reanalysis.background_hint', 'You can close this panel or page. Reload to see progress; restarting the plugin ends this batch.') }}</p>
                <div v-if="reanalysisReady.length" class="reanalysis-review-section">
                    <div class="reanalysis-review-picker">
                        <img v-if="reanalysisImage" :src="reanalysisImage" :alt="reanalysisResult?.name || ''" class="reanalysis-review-image">
                        <label>
                            <span>{{ t('pages.dashboard.reanalysis.choose_result', 'Choose an image to review') }}</span>
                            <select :value="reanalysisHash" @change="selectReanalysisResult($event.target.value)" :disabled="reanalysisSubmitting" class="codex-input">
                                <option v-for="item in reanalysisReady" :key="item.hash" :value="item.hash">{{ item.name }} · {{ item.hash.slice(0, 8) }}</option>
                            </select>
                        </label>
                    </div>
                    <p v-if="reanalysisLoading" class="hint-text" role="status">{{ t('pages.dashboard.reanalysis.loading_result', 'Loading comparison…') }}</p>
                    <div v-if="reanalysisResult" class="reanalysis-comparison">
                        <div class="reanalysis-comparison-head"><span></span><strong>{{ t('pages.dashboard.reanalysis.before', 'Before analysis') }}</strong><strong>{{ t('pages.dashboard.reanalysis.after', 'New result') }}</strong></div>
                        <div v-for="field in ['desc', 'tags', 'scenes', 'overlay_text', 'emotions']" :key="field" class="reanalysis-comparison-row">
                            <strong>{{ t('pages.dashboard.reanalysis.field_' + field, field) }}</strong>
                            <span>{{ formatAnalysisValue(reanalysisResult.before[field]) }}</span>
                            <span>{{ formatAnalysisValue(reanalysisResult.after[field]) }}</span>
                        </div>
                    </div>
                    <div class="reanalysis-actions">
                        <button @click="discardReanalysisResult" :disabled="reanalysisSubmitting || reanalysisLoading || !reanalysisResult" class="codex-btn">{{ t('pages.dashboard.reanalysis.discard_one', 'Keep original and continue') }}</button>
                        <button @click="saveReanalysisResult" :disabled="reanalysisSubmitting || reanalysisLoading || !reanalysisResult" class="codex-btn primary">{{ t('pages.dashboard.reanalysis.save_one', 'Save this image and continue') }}</button>
                    </div>
                </div>
                <p v-else class="hint-text">{{ t('pages.dashboard.reanalysis.no_pending', 'No results awaiting review.') }}</p>
                <details v-if="reanalysisJob.errors.length" class="reanalysis-errors">
                    <summary>{{ t('pages.dashboard.reanalysis.details', 'Failure and skip details') }} ({{ reanalysisJob.errors.length }})</summary>
                    <ul><li v-for="item in reanalysisJob.errors" :key="item.hash"><code>{{ item.hash.slice(0, 12) }}</code> · {{ item.error }}</li></ul>
                </details>
            </template>
            <p v-if="reanalysisError" class="error-banner" role="alert">{{ reanalysisError }}</p>
            <div class="reanalysis-actions">
                <button v-if="reanalysisBusy" @click="cancelBatchReanalysis" :disabled="reanalysisSubmitting || reanalysisJob.status === 'stopping'" class="codex-btn danger">{{ t('pages.dashboard.reanalysis.stop', 'Stop after current image') }}</button>
                <button v-else-if="reanalysisJob?.failed" @click="retryBatchReanalysis" :disabled="reanalysisSubmitting" class="codex-btn primary">{{ t('pages.dashboard.reanalysis.retry', 'Retry failed images') }}</button>
                <button @click="reanalysisOpen = false" class="codex-btn">{{ t('pages.dashboard.actions.close', 'Close') }}</button>
            </div>
        </div>
    </section>
</div>

<div v-if="toastOpen" class="toast-notification" :class="toastType" @click="toastOpen = false" style="white-space:pre-line">
    <svg v-if="toastType === 'success'" class="toast-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
    <svg v-else-if="toastType === 'error'" class="toast-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M12 8v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
    </svg>
    <svg v-else class="toast-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
    <span>{{ toastMessage }}</span>
</div>

<div v-if="confirmOpen" class="modal-overlay" @click.self="onConfirmNo">
    <div class="modal-panel modal-narrow" role="alertdialog" aria-modal="true" :aria-label="t('pages.dashboard.modal.confirm', 'Confirm Action')">
        <div class="modal-header">
            <h2>{{ t('pages.dashboard.modal.confirm', 'Confirm Action') }}</h2>
        </div>
        <div class="modal-pad">
            <p style="margin:0 0 24px;color:var(--text-main);font-size:1rem;white-space:pre-line">{{ confirmMessage }}</p>
            <div style="display:flex;gap:12px">
                <button @click="onConfirmNo" class="codex-btn" style="flex:1">{{ t('pages.dashboard.actions.cancel', 'Cancel') }}</button>
                <button @click="onConfirmYes" class="codex-btn danger" style="flex:1">{{ t('pages.dashboard.actions.confirm', 'Confirm') }}</button>
            </div>
        </div>
    </div>
</div>

<!-- 审核区图片元数据编辑弹窗 -->
<div v-if="pendingEditOpen" class="modal-overlay" @click.self="closePendingEdit">
    <div class="modal-panel" role="dialog" aria-modal="true" :aria-label="t('pages.dashboard.modal.edit_pending', 'Edit Pending Sticker')">
        <div class="modal-panel-corner-bl"></div>
        <div class="modal-panel-corner-br"></div>

        <div class="modal-header">
            <h2>{{ t('pages.dashboard.modal.edit_pending', 'Edit Pending Sticker') }}</h2>
            <button @click="closePendingEdit" class="modal-close" :aria-label="t('pages.dashboard.actions.close', 'Close')">
                <svg style="width:20px;height:20px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                        d="M6 18L18 6M6 6l12 12" />
                </svg>
            </button>
        </div>

        <div class="modal-content">
            <div class="modal-pad" style="width:100%">
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
                            class="form-label sm">
                            {{ t('pages.dashboard.fields.category', 'Category') }}
                        </label>
                        <select v-model="pendingEditForm.category" class="codex-input">
                            <option v-for="cat in categories" :key="cat.key" :value="cat.key">{{ cat.name }}</option>
                        </select>
                    </div>

                    <div style="margin-bottom:16px">
                        <label class="form-label sm">{{ t('pages.dashboard.fields.character', '角色') }}</label>
                        <select v-model="pendingEditForm.character" class="codex-input">
                            <option value="">{{ t('pages.dashboard.characters.unassigned', '通用表情库') }}</option>
                            <option v-for="item in characters" :key="item.key" :value="item.key">{{ item.name }}</option>
                        </select>
                    </div>

                    <div style="margin-bottom:16px">
                        <label class="form-label sm">{{ t('pages.dashboard.fields.overlay_text', '图上文字') }}</label>
                        <input v-model="pendingEditForm.overlay_text" type="text" class="codex-input"
                            :placeholder="t('pages.dashboard.placeholders.overlay_text', '图上印的字')">
                    </div>

                    <div style="margin-bottom:16px">
                        <label
                            class="form-label sm">
                            {{ t('pages.dashboard.fields.scope', 'Scope') }}
                        </label>
                        <select v-model="pendingEditForm.scope_mode" class="codex-input">
                            <option value="public">{{ t('pages.dashboard.scope.public', 'Public') }}</option>
                            <option value="local">{{ t('pages.dashboard.scope.local', 'Local only') }}</option>
                        </select>
                    </div>

                    <div style="margin-bottom:16px">
                        <label
                            class="form-label sm">
                            {{ t('pages.dashboard.fields.description', 'Description') }}
                        </label>
                        <textarea v-model="pendingEditForm.desc" class="codex-input" rows="3"></textarea>
                    </div>

                    <div style="margin-bottom:16px">
                        <label
                            class="form-label sm">
                            {{ t('pages.dashboard.fields.tags', 'Tags') }}
                            ({{ t('pages.dashboard.messages.tag_separator_hint', 'comma separated') }})
                        </label>
                        <input v-model="pendingEditForm.tagsText" type="text" class="codex-input">
                    </div>

                    <div style="margin-bottom:8px">
                        <label
                            class="form-label sm">
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
    <div class="modal-panel modal-narrow" role="dialog" aria-modal="true" :aria-label="t('pages.dashboard.modal.input', 'Input')">
        <div class="modal-header">
            <h2>{{ t('pages.dashboard.modal.input', 'Input') }}</h2>
        </div>
        <div class="modal-pad">
            <p style="margin:0 0 16px;color:var(--text-main);font-size:1rem">{{ promptMessage }}</p>
            <input v-model="promptValue" type="text" class="codex-input" @keyup.enter="onPromptOk">
            <div style="display:flex;gap:12px;margin-top:20px">
                <button @click="onPromptCancel" class="codex-btn" style="flex:1">{{ t('pages.dashboard.actions.cancel', 'Cancel') }}</button>
                <button @click="onPromptOk" class="codex-btn primary" style="flex:1">{{ t('pages.dashboard.actions.confirm', 'Confirm') }}</button>
            </div>
        </div>
    </div>
</div>`;
