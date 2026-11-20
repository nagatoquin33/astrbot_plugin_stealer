"""WebUI 预览服务器（tests/web/test_server.py）的冒烟测试。

验证 mock 桥接注入、静态资源、API 形状与关键交互流（审核/上传/删除），
确保预览服务器与 pages/dashboard 前端、plugin_api.py 的响应结构保持一致。
"""

import base64
import io
import json
import shutil
import subprocess

import pytest

from tests.web.test_server import DASHBOARD_DIR, PLUGIN_BASE, PreviewState, create_app

try:
    from aiohttp.test_utils import TestClient, TestServer
except ImportError as exc:  # pragma: no cover
    pytest.skip(f"aiohttp 不可用，跳过 WebUI 预览测试: {exc}", allow_module_level=True)


def _png_base64(color=(200, 60, 60)) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _client(state: PreviewState | None = None) -> TestClient:
    return TestClient(TestServer(create_app(state)))


@pytest.mark.asyncio
async def test_index_injects_bridge_and_i18n():
    async with _client() as client:
        resp = await client.get("/")
        assert resp.status == 200
        html = await resp.text()
        assert "window.AstrBotPluginPage" not in html  # 桥接走外链脚本而非内联
        assert '<script src="/mock/bridge.js"></script>' in html
        assert "__STEALER_I18N__" in html
        assert '"zh-CN"' in html and '"en-US"' in html
        assert "./app.js" in html
        # 快速项：prod 构建 + favicon
        assert "./vendor/vue.global.prod.js" in html
        assert 'rel="icon"' in html
        assert (DASHBOARD_DIR / "vendor" / "vue.global.prod.js").is_file()
        for css_name in ("app.css", "light.css", "dark.css", "minecraft.css", "fallout.css", "bili.css", "host_wallpaper.css"):
            assert f'href="./{css_name}"' in html
            assert (DASHBOARD_DIR / css_name).is_file()


@pytest.mark.asyncio
async def test_bridge_js_and_static_assets_served():
    async with _client() as client:
        bridge = await client.get("/mock/bridge.js")
        assert bridge.status == 200
        bridge_text = await bridge.text()
        for fn in ("apiGet", "apiPost", "upload", "getLocale", "getI18n", "_files"):
            assert fn in bridge_text

        for path, keyword in (
            ("/app.js", "createApp"),
            ("/template.js", "TEMPLATE"),
            ("/app.css", "{"),
            ("/light.css", '[data-theme="light"]'),
            ("/dark.css", '[data-theme="dark"]'),
            ("/minecraft.css", '[data-theme="minecraft"]'),
            ("/fallout.css", '[data-theme="fallout"]'),
            ("/bili.css", '[data-theme="bili"]'),
            ("/host_wallpaper.css", '[data-host-wallpaper="on"]'),
            ("/vendor/vue.global.prod.js", "Vue"),
        ):
            resp = await client.get(path)
            assert resp.status == 200, path
            assert await resp.text()

        logo = await client.get("/logo.png")
        assert logo.status == 200
        assert logo.content_type == "image/png"
        assert await logo.read()

        vaultboy = await client.get("/vaultboy.png")
        assert vaultboy.status == 200
        assert vaultboy.content_type == "image/png"
        assert await vaultboy.read()


def test_grid_does_not_prefetch_originals_on_hover():
    """验证列表展示缩略图，悬停时使用提示信息而不预取原图。"""
    template = (DASHBOARD_DIR / "template.js").read_text(encoding="utf-8")
    app_js = (DASHBOARD_DIR / "app.js").read_text(encoding="utf-8")
    assert '@mouseenter="loadOriginalImage' not in template
    assert "originalDataUrls[img.hash] || imageDataUrls[img.hash]" not in template
    assert "originalDataUrls[item.hash] || imageDataUrls[item.hash]" not in template
    assert "originalDataUrls[previewItem?.hash]" in template
    assert "if (img?.hash) loadOriginalImage(img.hash)" not in app_js
    assert "requestOriginalForPreview" in app_js


def test_library_cards_expose_context_menu_and_collapsible_sidebar():
    template = (DASHBOARD_DIR / "template.js").read_text(encoding="utf-8")
    app_js = (DASHBOARD_DIR / "app.js").read_text(encoding="utf-8")
    css = (DASHBOARD_DIR / "app.css").read_text(encoding="utf-8")
    assert '@contextmenu.prevent.stop="openContextMenu($event, img)"' in template
    assert 'role="menu"' in template and "runContextAction('blacklist')" in template
    assert "sidebarCollapsed" in template and "toggleSidebarCollapsed" in template
    assert "const openContextMenu" in app_js and "const runContextAction" in app_js
    assert ".item-context-menu" in css and ".sidebar.is-collapsed" in css


def test_preview_exposes_vlm_reanalysis_and_ab_apply_controls():
    template = (DASHBOARD_DIR / "template.js").read_text(encoding="utf-8")
    app_js = (DASHBOARD_DIR / "app.js").read_text(encoding="utf-8")
    css = (DASHBOARD_DIR / "app.css").read_text(encoding="utf-8")
    assert "reanalyzePreview" in template and "applyReanalysis" in template
    assert "vlm-compare-grid" in template
    assert "body: JSON.stringify({ hash })" in app_js
    assert "vlmReanalysisResult.value" in app_js
    assert "previewItem.value = {" in app_js
    base_rule = css.index(".vlm-reanalysis-panel {")
    responsive_rule = css.rindex("@media (max-width: 1100px)")
    assert responsive_rule > base_rule


def test_shared_emotion_component_handles_multiple_and_legacy_labels():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed to execute the frontend component")
    script = r"""
const fs = require('fs'), vm = require('vm');
let options;
const source = fs.readFileSync(process.argv[1], 'utf8').replace(/^import .*?;\r?\n/gm, '');
vm.runInNewContext(source, {
    Vue: { createApp(value) { options = value; return { mount() {} }; } },
    TEMPLATE: '', EMOTION_LABELS_TEMPLATE: '',
});
const items = [
    {category:'dumb', emotions:['dumb','sigh','tired','sigh']},
    {category:'happy'},
    {category:'dumb', emotions:'sigh，tired'},
    {emotions:['happy','sad']},
];
console.log(JSON.stringify(items.map(item => options.components.EmotionLabels.computed.keys.call({item}))));
"""
    result = subprocess.run([node, "-e", script, str(DASHBOARD_DIR / "app.js")], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == [
        ["dumb", "sigh", "tired"], ["happy"], ["dumb", "sigh", "tired"], ["happy", "sad"],
    ]


def test_edit_emotions_deduplicates_selected_primary():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed to execute frontend logic")
    script = r"""
const fs = require('fs'), vm = require('vm');
const source = fs.readFileSync(process.argv[1], 'utf8').replace(/^import .*?;\r?\n/gm, '');
vm.runInNewContext(source + `\nconsole.log(JSON.stringify(editedEmotionKeys({category:'sad', emotions:['happy','sad','tired']})));`, {
    Vue: { createApp() { return { mount() {} }; } }, TEMPLATE: '', EMOTION_LABELS_TEMPLATE: '', console
});
"""
    result = subprocess.run([node, "-e", script, str(DASHBOARD_DIR / "app.js")], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == ["sad", "happy", "tired"]


def test_inline_editor_saves_label_changes_and_pending_drafts():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed to execute frontend logic")
    script = r"""
const fs = require('fs'), vm = require('vm');
let options, payload;
const source = fs.readFileSync(process.argv[1], 'utf8').replace(/^import .*?;\r?\n/gm, '');
const ref = value => ({value});
const bridge = {
  apiPost: async (endpoint, body) => { if (endpoint === 'images/update') payload = body; return {success:true}; },
  apiGet: async () => ({success:true, images:[], categories:[], stats:{}}),
};
const context = {
  Vue: {createApp(value) { options = value; return {mount(){}}; }, ref, reactive: x => x,
        computed: fn => ({get value(){return fn();}}), onMounted(){}, onUnmounted(){}, nextTick(){}},
  window: {AstrBotPluginPage:bridge}, document:{querySelectorAll:()=>[]},
  TEMPLATE:'', EMOTION_LABELS_TEMPLATE:'', ImagePreviewClient:class {},
  createSourceActions:()=>({}), createBatchReanalysis:()=>({}),
  URLSearchParams, FormData, console, setTimeout, clearTimeout,
};
vm.runInNewContext(source, context);
(async () => {
  const ui=options.setup();
  ui.previewItem.value={hash:'h',category:'happy',emotions:['happy','sad'],tags:['old'],scenes:['old scene'],desc:'desc'};
  ui.startEdit();
  ui.removeEditEmotion('sad'); ui.newEditEmotion.value='tired';
  ui.removeEditLabel('tags','old'); ui.newEditTag.value='new tag';
  ui.removeEditLabel('scene','old scene'); ui.newEditScene.value='new scene';
  await ui.saveEdit();
  console.log(JSON.stringify(payload));
})().catch(error => { console.error(error); process.exitCode=1; });
"""
    result = subprocess.run([node, "-e", script, str(DASHBOARD_DIR / "app.js")], capture_output=True, text=True, check=True)
    payload = json.loads(result.stdout)
    assert payload["hash"] == "h"
    assert payload["emotions"] == ["happy", "tired"]
    assert payload["tags"] == "new tag" and payload["scene"] == "new scene"


def test_external_import_origin_shows_channel_in_preview():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed to execute the frontend source formatter")
    script = r"""
const { pathToFileURL } = require('url');
(async () => {
    const { formatItemOriginLabel } = await import(pathToFileURL(process.argv[1]).href);
    const labels = {
        'pages.dashboard.fields.add_method_external': 'External import',
        'pages.dashboard.fields.source_channel_pack': 'Meme pack',
        'pages.dashboard.fields.source_channel_github': 'GitHub',
        'pages.dashboard.fields.source_channel_http_json': 'JSON API',
    };
    const t = (key, fallback) => labels[key] || fallback;
    const formatTarget = (target) => target ? `Group ${target.slice(6)}` : 'Not recorded';
    const items = [
        { origin_target: 'group:123', source: 'auto' },
        { source: 'external:meme_pack', add_method: 'external_import' },
        { origin_target: 'group:123', source: 'external:github' },
        { source: 'external:http_json' },
        { add_method: 'external_import' },
        { source: 'manual' },
    ];
    console.log(JSON.stringify(items.map((item) => formatItemOriginLabel(item, formatTarget, t))));
})().catch((error) => { console.error(error); process.exit(1); });
"""
    result = subprocess.run(
        [node, "-e", script, str(DASHBOARD_DIR / "source_labels.js")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    assert json.loads(result.stdout) == [
        "Group 123",
        "External import · Meme pack",
        "Group 123 · External import · GitHub",
        "External import · JSON API",
        "External import",
        "Not recorded",
    ]
    template = (DASHBOARD_DIR / "template.js").read_text(encoding="utf-8")
    assert template.count("formatItemOrigin(previewItem)") == 2


def test_light_theme_character_filter_uses_readable_surface():
    """验证亮色主题的角色筛选栏使用对应的背景和边框颜色。"""
    css = (DASHBOARD_DIR / "light.css").read_text(encoding="utf-8")
    selector = '[data-theme="light"] .character-filter-bar'
    start = css.index(selector)
    rule = css[start:css.index("}", start)]
    assert "background: var(--bg-elevated)" in rule
    assert "border-bottom-color: var(--glass-border)" in rule


def test_dashboard_font_assets_are_pinned():
    css = "\n".join(
        (DASHBOARD_DIR / name).read_text(encoding="utf-8")
        for name in ("app.css", "light.css", "dark.css", "minecraft.css", "fallout.css", "bili.css")
    )
    assert "@latest" not in css

    active_count_selector = '[data-theme="light"] .character-chip.active .character-chip-count'
    count_start = css.index(active_count_selector)
    count_rule = css[count_start:css.index("}", count_start)]
    assert "color: currentColor" in count_rule


def test_fallout_theme_avoids_fullpage_compositing():
    """验证辐射 4 主题避免使用引发整页 GPU 重合成的混合模式与对比度滤镜。"""
    app_css = (DASHBOARD_DIR / "app.css").read_text(encoding="utf-8")
    fallout_css = (DASHBOARD_DIR / "fallout.css").read_text(encoding="utf-8")
    start = app_css.find('[data-theme="fallout"]')
    assert start != -1
    for sheet in (app_css[start:], fallout_css):
        assert "mix-blend-mode" not in sheet
        assert "filter: contrast" not in sheet
    assert 'html[data-theme="fallout"]::after' not in app_css
    assert 'html[data-theme="fallout"]::after' not in fallout_css


def test_night_and_day_theme_labels_and_preserved_keys():
    """验证黑夜/白天主题文案已统一，同时保留 dark/light 存储键值兼容。"""
    repo_root = DASHBOARD_DIR.parents[1]
    schema = json.loads((repo_root / "_conf_schema.json").read_text(encoding="utf-8"))
    theme_conf = schema["webui_theme"]
    assert theme_conf["options"][1:3] == ["dark", "light"]
    assert theme_conf["labels"][1:3] == ["黑夜", "白天"]

    zh = json.loads((repo_root / ".astrbot-plugin" / "i18n" / "zh-CN.json").read_text(encoding="utf-8"))
    en = json.loads((repo_root / ".astrbot-plugin" / "i18n" / "en-US.json").read_text(encoding="utf-8"))
    ru = json.loads((repo_root / ".astrbot-plugin" / "i18n" / "ru-RU.json").read_text(encoding="utf-8"))

    assert zh["pages"]["dashboard"]["themes"]["dark"] == "黑夜"
    assert zh["pages"]["dashboard"]["themes"]["light"] == "白天"
    assert en["pages"]["dashboard"]["themes"]["dark"] == "Night"
    assert en["pages"]["dashboard"]["themes"]["light"] == "Day"
    assert ru["pages"]["dashboard"]["themes"]["dark"] == "Ночь"
    assert ru["pages"]["dashboard"]["themes"]["light"] == "День"


def test_vue_templates_compile_without_errors():
    """使用真实 Vue 编译器校验全量多主题模板 AST，拦截 v-if/v-else-if/标签闭合语法回归。"""
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed to compile Vue templates")
    script = r"""
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import { pathToFileURL } from 'node:url';

const [vuePath, templatePath] = process.argv.slice(1);
const ctx = {
    window: {},
    document: { createElement: () => ({ children: [{ getAttribute: () => '' }], textContent: '' }) },
};
ctx.globalThis = ctx;
vm.runInNewContext(fs.readFileSync(vuePath, 'utf8'), ctx);
const { TEMPLATE, EMOTION_LABELS_TEMPLATE } = await import(pathToFileURL(templatePath));

for (const [name, tpl] of [['TEMPLATE', TEMPLATE], ['EMOTION_LABELS_TEMPLATE', EMOTION_LABELS_TEMPLATE]]) {
    const errors = [];
    const render = ctx.Vue.compile(tpl, {
        decodeEntities: (raw) => raw,
        onError: (err) => { errors.push(err.message || String(err)); },
    });
    assert.equal(errors.length, 0, `${name} compile errors: ${errors.join('; ')}`);
    assert.equal(typeof render, 'function');
}
// 反向自检：确认编译器在遇到非法 v-else 时确实报错而非静默放行
const negativeErrors = [];
ctx.Vue.compile('<div><span v-else></span></div>', {
    decodeEntities: (raw) => raw,
    onError: (err) => { negativeErrors.push(err); },
});
assert.equal(negativeErrors.length, 1);
"""
    result = subprocess.run(
        [
            node, "--input-type=module", "-e", script,
            str(DASHBOARD_DIR / "vendor" / "vue.global.prod.js"),
            str(DASHBOARD_DIR / "template.js"),
        ],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_dashboard_setup_category_edit_theme_routing_and_scoped_missing():
    """在真实 Vue 响应式上下文中运行 app.js，验证情绪标签折叠、分类原位编辑、主题路由与范围化语义补充。"""
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed to execute the frontend setup")
    script = r"""
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import { pathToFileURL } from 'node:url';

const [vuePath, dashboardDir] = process.argv.slice(1);
const vueCtx = { window: {}, document: { createElement: () => ({}) } };
vueCtx.globalThis = vueCtx;
vm.runInNewContext(fs.readFileSync(vuePath, 'utf8'), vueCtx);

const { TEMPLATE, EMOTION_LABELS_TEMPLATE } = await import(pathToFileURL(`${dashboardDir}/template.js`));
const { ImagePreviewClient } = await import(pathToFileURL(`${dashboardDir}/image_preview_client.js`));
const { createSourceActions } = await import(pathToFileURL(`${dashboardDir}/source_actions.js`));
const { createBatchReanalysis } = await import(pathToFileURL(`${dashboardDir}/batch_reanalysis.js`));
const { formatItemOriginLabel } = await import(pathToFileURL(`${dashboardDir}/source_labels.js`));

let appOptions = null;
const requests = [];
const Vue = {
    ...vueCtx.Vue,
    onMounted: () => {},
    onUnmounted: () => {},
    createApp(opts) {
        appOptions = opts;
        return { mount() {} };
    },
};

const docEl = { dataset: {}, setAttribute(k, v) { this[k] = v; }, getAttribute(k) { return this[k]; } };
const sandbox = {
    Vue,
    TEMPLATE,
    EMOTION_LABELS_TEMPLATE,
    ImagePreviewClient,
    createSourceActions,
    createBatchReanalysis,
    formatItemOriginLabel,
    URL,
    URLSearchParams,
    FormData: class { constructor() { this.items = []; } append(k, v) { this.items.push([k, v]); } },
    setTimeout: () => 1,
    clearTimeout: () => {},
    setInterval: () => 1,
    clearInterval: () => {},
    localStorage: { getItem: () => null, setItem: () => {}, removeItem: () => {} },
    matchMedia: () => ({ matches: true, addEventListener: () => {}, removeEventListener: () => {} }),
    document: {
        documentElement: docEl,
        querySelectorAll: () => [],
        addEventListener: () => {},
        removeEventListener: () => {},
    },
    window: {
        location: { search: '', pathname: '/', origin: 'http://127.0.0.1:8091' },
        AstrBotPluginPage: {
            apiGet: async (endpoint, params) => {
                requests.push({ endpoint, method: 'GET', params: params ? { ...params } : null });
                if (endpoint === 'images/reanalyze/missing') {
                    return { success: true, missing: { hashes: [], total_missing: 0, max_images: 500 } };
                }
                if (endpoint === 'emotions') {
                    return { success: true, emotions: [{ key: 'happy', name: '超开心', desc: '更新描述' }] };
                }
                if (endpoint === 'images') {
                    return {
                        success: true, total: 1, page: 1, size: 24, images: [],
                        categories: [{ key: 'happy', name: '超开心', count: 1 }],
                        characters: [{ key: 'mika', name: '美香', count: 1 }],
                        libraries: { general: 0, favorites: 0, characters: 1, automatic: 0 },
                    };
                }
                return { success: true };
            },
            apiPost: async (endpoint, body) => {
                requests.push({ endpoint, method: 'POST', body: body ? JSON.parse(JSON.stringify(body)) : null });
                return { success: true };
            },
            t: (_k, fb) => fb,
        },
        addEventListener: () => {},
        removeEventListener: () => {},
    },
    console,
};
sandbox.globalThis = sandbox;
const appSource = fs.readFileSync(`${dashboardDir}/app.js`, 'utf8').replace(/^import .*?;\r?\n/gm, '');
vm.runInNewContext(appSource, sandbox);

// 1. EmotionLabels 组件去重与紧凑折叠逻辑
const comp = appOptions.components.EmotionLabels;
const makeChipCtx = (item, compact) => {
    const c = { item, compact };
    c.keys = comp.computed.keys.call(c);
    c.visibleKeys = comp.computed.visibleKeys.call(c);
    c.overflowCount = comp.computed.overflowCount.call(c);
    return c;
};
const compactChips = makeChipCtx({ category: 'happy', emotions: ['happy', 'sad', 'angry', 'tired'] }, true);
assert.deepEqual([...compactChips.visibleKeys], ['happy', 'sad']);
assert.equal(compactChips.overflowCount, 2);
const fullChips = makeChipCtx({ category: 'happy', emotions: ['happy', 'sad', 'angry', 'tired'] }, false);
assert.deepEqual([...fullChips.visibleKeys], ['happy', 'sad', 'angry', 'tired']);
assert.equal(fullChips.overflowCount, 0);

// 2. 运行真实 setup() 状态机
const state = appOptions.setup();

// 主题切换路由互斥性（辐射 4 / 我的世界 / B站风 / 黑夜白天）
state.setThemeMode('fallout', false);
assert.equal(state.isFalloutTheme.value, true);
assert.equal(state.isMinecraftTheme.value, false);
assert.equal(state.isBiliTheme.value, false);

state.setThemeMode('minecraft', false);
assert.equal(state.isFalloutTheme.value, false);
assert.equal(state.isMinecraftTheme.value, true);

sandbox.window.parent = sandbox.window;
state.setThemeMode('auto', false);
assert.equal(docEl.dataset.hostWallpaper, 'off');
sandbox.window.parent = {};
state.setThemeMode('auto', false);
assert.equal(docEl.getAttribute('data-theme'), 'dark');
assert.equal(docEl.dataset.hostWallpaper, 'on');
assert.equal(state.THEME_OPTIONS.some(({ value }) => value === 'palette'), false);
assert.equal(state.isMinecraftTheme.value, false);

state.setThemeMode('light', false);
assert.equal(docEl.getAttribute('data-theme'), 'light');
assert.equal(docEl.dataset.hostTheme, 'dark');
assert.equal(docEl.dataset.hostWallpaper, 'on');
assert.equal(state.isFalloutTheme.value, false);
assert.equal(state.isMinecraftTheme.value, false);
assert.equal(state.isBiliTheme.value, false);

state.setThemeMode('dark', false);
assert.equal(docEl.dataset.hostWallpaper, 'on');
state.setThemeMode('bili', false);
assert.equal(docEl.dataset.hostWallpaper, 'off');
sandbox.window.parent = sandbox.window;
state.setThemeMode('light', false);
assert.equal(docEl.dataset.hostWallpaper, 'off');

// 分类原位编辑：不弹重复覆盖确认框，直接就地替换原 key 对应的名称与描述
state.availableEmotions.value = [
    { key: 'happy', name: '开心', desc: '旧描述' },
    { key: 'sad', name: '难过', desc: '' },
];
state.startEditEmotion({ key: 'happy', name: '开心', desc: '旧描述' });
assert.equal(state.editingEmotionKey.value, 'happy');
state.newEmotion.name = '超开心';
state.newEmotion.desc = '更新描述';
await state.addEmotion();
const editReq = requests.find((r) => r.endpoint === 'categories' && r.method === 'POST');
assert.ok(editReq);
assert.deepEqual(editReq.body, {
    categories: [
        { key: 'happy', name: '超开心', desc: '更新描述' },
        { key: 'sad', name: '难过', desc: '' },
    ],
});
assert.equal(state.editingEmotionKey.value, '');

// 切换子库/角色/情绪后，补充表情包语义请求自动携带完整当前筛选范围，且一键清空筛选生效
state.selectLibraryCharacter('mika');
state.selectLibraryCategory('happy');
assert.equal(state.selectedLibrary.value, 'characters');
assert.equal(state.hasActiveLibraryFilters.value, true);
await state.startMissingReanalysis();
const missingReq = requests.find((r) => r.endpoint === 'images/reanalyze/missing');
assert.ok(missingReq);
assert.deepEqual(missingReq.params, { library: 'characters', category: 'happy', character: 'mika' });

state.clearLibraryFilters();
assert.equal(state.selectedCategory.value, '');
assert.equal(state.selectedCharacter.value, '');
assert.equal(state.hasActiveLibraryFilters.value, false);
"""
    result = subprocess.run(
        [
            node, "--input-type=module", "-e", script,
            str(DASHBOARD_DIR / "vendor" / "vue.global.prod.js"),
            str(DASHBOARD_DIR).replace("\\", "/"),
        ],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_server_theme_overrides_host_query_after_prefs_load():
    """AstrBot 必带的 ?theme=dark/light 只代表宿主状态，不能覆盖页面偏好。"""
    app_js = (DASHBOARD_DIR / "app.js").read_text(encoding="utf-8")
    prefs_block = app_js[app_js.index("const loadDashboardPrefs"):]
    assert "!readStored(THEME_STORAGE_KEY)" not in prefs_block
    assert "hasQueryTheme" not in prefs_block
    assert "requestRevision === themePreferenceRevision" in prefs_block
    assert "setThemeMode(resolveThemeValue(data.theme), false)" in prefs_block
    assert "hostThemeFromQuery" in app_js
    assert "themeMode = ref(resolveThemeValue(readStored" in app_js


@pytest.mark.asyncio
async def test_prefs_persist_theme_view_and_sidebar():
    state = PreviewState(seed=False)
    async with _client(state) as client:
        before = await (await client.get(f"{PLUGIN_BASE}/prefs")).json()
        assert before["success"] is True
        assert before["theme"] == "auto"
        assert before["sidebar"] == "expanded"

        saved = await (await client.post(
            f"{PLUGIN_BASE}/prefs",
            json={"theme": "fallout", "view": "list", "sidebar": "collapsed"},
        )).json()
        assert saved == {
            "success": True,
            "theme": "fallout",
            "view": "list",
            "sidebar": "collapsed",
        }

        again = await (await client.get(f"{PLUGIN_BASE}/prefs")).json()
        assert again["theme"] == "fallout"
        assert again["view"] == "list"
        assert again["sidebar"] == "collapsed"

        ignored = await (await client.post(
            f"{PLUGIN_BASE}/prefs", json={"theme": "not-a-theme"}
        )).json()
        assert ignored["theme"] == "fallout"


@pytest.mark.asyncio
async def test_health_and_stats_shapes():
    async with _client() as client:
        health = await client.get(f"{PLUGIN_BASE}/health")
        assert (await health.json())["status"] == "ok"

        stats = (await (await client.get(f"{PLUGIN_BASE}/stats")).json())["stats"]
        seeded = PreviewState(seed=True)
        assert stats == {
            "total": len(seeded.library),
            "categories": len({m["category"] for m in seeded.library}),
            "today": stats["today"],
        }
        assert isinstance(stats["today"], int)


@pytest.mark.asyncio
async def test_images_list_pagination_and_categories():
    async with _client() as client:
        data = (await (await client.get(f"{PLUGIN_BASE}/images?page=1&size=5")).json())
        assert data["success"] is True
        assert len(data["images"]) <= 5
        assert data["total"] > 0
        assert {"key", "name", "count"} <= set(data["categories"][0])
        counts_sum = sum(c["count"] for c in data["categories"])
        assert counts_sum == data["total"]

        # 第二页不与第一页重叠（种子数据足够多）
        page2 = (await (await client.get(f"{PLUGIN_BASE}/images?page=2&size=5")).json())
        hashes1 = {i["hash"] for i in data["images"]}
        hashes2 = {i["hash"] for i in page2["images"]}
        assert hashes1.isdisjoint(hashes2)


@pytest.mark.asyncio
async def test_category_create_validation_and_persistence():
    state = PreviewState(seed=True)
    async with _client(state) as client:
        current = (await (await client.get(f"{PLUGIN_BASE}/emotions")).json())["emotions"]
        created = await client.post(
            f"{PLUGIN_BASE}/categories",
            json={"categories": current + [{"key": " Custom_Tag ", "name": "自定义", "desc": "测试"}]},
        )
        assert (await created.json())["success"] is True
        refreshed = (await (await client.get(f"{PLUGIN_BASE}/emotions")).json())["emotions"]
        assert refreshed[-1] == {"key": "custom_tag", "name": "自定义", "desc": "测试"}

        rejected = await client.post(
            f"{PLUGIN_BASE}/categories",
            json={"categories": refreshed + [{"key": "../escape", "name": "危险"}]},
        )
        assert (await rejected.json())["success"] is False
        assert state.emotions == refreshed


@pytest.mark.asyncio
async def test_images_search_filter_by_query():
    async with _client() as client:
        all_items = (await (await client.get(f"{PLUGIN_BASE}/images?page=1&size=100")).json())
        target = next(m for m in all_items["images"] if m["tags"])
        keyword = target["tags"][0]

        found = (await (await client.get(f"{PLUGIN_BASE}/images?page=1&size=100&q={keyword}")).json())
        assert found["total"] >= 1
        assert any(m["hash"] == target["hash"] for m in found["images"])

        by_cat = (await (await client.get(f"{PLUGIN_BASE}/images?category={target['category']}")).json())
        assert all(m["category"] == target["category"] for m in by_cat["images"])


@pytest.mark.asyncio
async def test_thumbnail_and_image_data_return_data_urls():
    async with _client() as client:
        listing = (await (await client.get(f"{PLUGIN_BASE}/images?page=1&size=1")).json())
        img_hash = listing["images"][0]["hash"]

        thumb = (await (await client.get(f"{PLUGIN_BASE}/thumbnail?hash={img_hash}&size=120")).json())
        assert thumb["success"] is True
        assert thumb["url"].startswith("data:image/png;base64,")

        original = (await (await client.get(f"{PLUGIN_BASE}/image-data?hash={img_hash}")).json())
        assert original["success"] is True
        assert original["url"].startswith("data:image/png;base64,")

        missing = (await (await client.get(f"{PLUGIN_BASE}/thumbnail?hash=nope")).json())
        assert missing["success"] is False


@pytest.mark.asyncio
async def test_pending_list_stats_and_approve_flow():
    state = PreviewState(seed=True)
    async with _client(state) as client:
        before_total = (await (await client.get(f"{PLUGIN_BASE}/stats")).json())["stats"]["total"]

        pending = (await (await client.get(f"{PLUGIN_BASE}/pending?page=1&size=10")).json())
        assert pending["success"] is True
        first_id = pending["images"][0]["id"]
        pending_before = len(state.pending)

        pstats = (await (await client.get(f"{PLUGIN_BASE}/pending/stats")).json())["stats"]
        assert pstats == {"pending": pending_before, "capacity": 200, "paused": False}

        approved = (await (await client.post(
            f"{PLUGIN_BASE}/pending/approve", json={"id": first_id}
        )).json())
        assert approved["success"] is True and approved["approved"] == 1

        after_total = (await (await client.get(f"{PLUGIN_BASE}/stats")).json())["stats"]["total"]
        assert after_total == before_total + 1
        assert len(state.pending) == pending_before - 1

        missing = (await (await client.post(
            f"{PLUGIN_BASE}/pending/approve", json={"id": 999999}
        )).json())
        assert missing["success"] is False


@pytest.mark.asyncio
async def test_pending_update_then_reject_with_blacklist():
    state = PreviewState(seed=True)
    async with _client(state) as client:
        pending = (await (await client.get(f"{PLUGIN_BASE}/pending?page=1&size=1")).json())
        row = pending["images"][0]
        pid = row["id"]

        updated = (await (await client.post(
            f"{PLUGIN_BASE}/pending/update",
            json={"id": pid, "desc": "改过的描述", "tags": ["新标签"], "scope_mode": "local"},
        )).json())
        assert updated["success"] is True
        assert updated["item"]["desc"] == "改过的描述"
        assert updated["item"]["tags"] == ["新标签"]

        bad_cat = (await (await client.post(
            f"{PLUGIN_BASE}/pending/update", json={"id": pid, "category": "nope"}
        )).json())
        assert bad_cat["success"] is False

        rejected = (await (await client.post(
            f"{PLUGIN_BASE}/pending/reject", json={"id": pid, "blacklist": True}
        )).json())
        assert rejected == {"success": True, "deleted": 1, "blacklisted": 1}
        assert row["hash"] in state.blacklist


@pytest.mark.asyncio
async def test_library_update_scope_validation_delete():
    async with _client() as client:
        listing = (await (await client.get(f"{PLUGIN_BASE}/images?page=1&size=50")).json())
        item = listing["images"][0]

        ok = (await (await client.post(
            f"{PLUGIN_BASE}/images/update",
            json={"hash": item["hash"], "desc": "更新后的描述", "is_favorite": True},
        )).json())
        assert ok["success"] is True

        refreshed = (await (await client.get(f"{PLUGIN_BASE}/images?page=1&size=50&q=更新后的描述")).json())
        hit = next(m for m in refreshed["images"] if m["hash"] == item["hash"])
        assert hit["is_favorite"] is True

        # 无 origin_target 时设为 local 应报错（与 plugin_api 行为一致）
        local_fail = (await (await client.post(
            f"{PLUGIN_BASE}/images/update", json={"hash": item["hash"], "scope_mode": "local"}
        )).json())
        assert local_fail == {"success": False, "error": "Origin target missing"}

        deleted = (await (await client.post(
            f"{PLUGIN_BASE}/images/delete", json={"hash": item["hash"]}
        )).json())
        assert deleted["success"] is True and deleted["count"] == 1

        gone = (await (await client.get(f"{PLUGIN_BASE}/images?q={item['hash']}&page=1&size=50")).json())
        assert gone["total"] == 0


@pytest.mark.asyncio
async def test_batch_operations_move_scope_favorite_delete():
    async with _client() as client:
        listing = (await (await client.get(f"{PLUGIN_BASE}/images?page=1&size=3")).json())
        hashes = [m["hash"] for m in listing["images"]][:2]

        moved = (await (await client.post(
            f"{PLUGIN_BASE}/images/batch-move", json={"hashes": hashes, "category": "troll"}
        )).json())
        assert moved["success"] is True and moved["count"] == 2

        scoped = (await (await client.post(
            f"{PLUGIN_BASE}/images/batch-scope", json={"hashes": hashes, "scope_mode": "public"}
        )).json())
        assert scoped["success"] is True and scoped["skipped"] == 0

        fav = (await (await client.post(
            f"{PLUGIN_BASE}/images/batch-favorite", json={"hashes": hashes, "favorite": True}
        )).json())
        assert fav["success"] is True and fav["count"] == 2

        deleted = (await (await client.post(
            f"{PLUGIN_BASE}/images/batch-delete", json={"hashes": hashes}
        )).json())
        assert deleted["success"] is True and deleted["count"] == 2


@pytest.mark.asyncio
async def test_upload_single_and_duplicate_detection():
    async with _client() as client:
        b64 = _png_base64()
        first = (await (await client.post(
            f"{PLUGIN_BASE}/images/upload",
            json={"base64": b64, "filename": "hello.png", "emotion": "shy", "tags": "测试,上传"},
        )).json())
        assert first["success"] is True
        assert first["image"]["category"] == "shy"

        duplicate = (await (await client.post(
            f"{PLUGIN_BASE}/images/upload",
            json={"base64": b64, "filename": "hello.png", "emotion": "shy"},
        )).json())
        assert duplicate["hash"] == first["hash"]

        empty = (await (await client.post(f"{PLUGIN_BASE}/images/upload", json={})).json())
        assert empty["success"] is False


@pytest.mark.asyncio
async def test_batch_upload_task_lifecycle():
    async with _client() as client:
        payload = {
            "_files": [
                {"key": "file", "name": "a.png", "base64": _png_base64((10, 180, 90))},
                {"key": "file", "name": "b.png", "base64": _png_base64((40, 80, 220))},
            ],
            "emotion": "love",
        }
        created = (await (await client.post(f"{PLUGIN_BASE}/images/batch-upload", json=payload)).json())
        assert created["success"] is True
        assert created["task_id"]

        status = (await (await client.get(
            f"{PLUGIN_BASE}/images/batch-upload-status?task_id={created['task_id']}"
        )).json())
        assert status["success"] is True
        assert status["status"] == "completed"
        assert status["processed"] == 2
        assert status["success_count"] == 2
        assert status["failed_count"] == 0

        unknown = (await (await client.get(
            f"{PLUGIN_BASE}/images/batch-upload-status?task_id=nope"
        )).json())
        assert unknown["success"] is False


@pytest.mark.asyncio
async def test_analyze_emotions_storage_scan_cleanup():
    async with _client() as client:
        analyzed = (await (await client.post(
            f"{PLUGIN_BASE}/analyze", json={"base64": _png_base64()}
        )).json())
        assert analyzed["success"] is True
        assert {"category", "tags", "desc"} <= set(analyzed)

        item = (await (await client.get(f"{PLUGIN_BASE}/images?page=1&size=1")).json())["images"][0]
        reanalyzed = (await (await client.post(
            f"{PLUGIN_BASE}/analyze", json={"hash": item["hash"]}
        )).json())
        assert reanalyzed["success"] is True
        applied = (await (await client.post(
            f"{PLUGIN_BASE}/images/update",
            json={
                "hash": item["hash"],
                "category": reanalyzed["category"],
                "tags": reanalyzed["tags"],
                "desc": reanalyzed["desc"],
                "scenes": reanalyzed["scenes"],
                "overlay_text": reanalyzed["overlay_text"],
                "emotions": reanalyzed["emotions"],
            },
        )).json())
        assert applied["success"] is True

        emotions = (await (await client.get(f"{PLUGIN_BASE}/emotions")).json())
        assert emotions["success"] is True and len(emotions["emotions"]) == 17

        scan = (await (await client.get(f"{PLUGIN_BASE}/storage/scan")).json())
        assert scan["success"] is True
        for section in ("stale_index", "orphan_files", "thumb_cache", "temp_files"):
            assert {"count", "bytes", "samples"} <= set(scan[section])

        cleanup = (await (await client.post(
            f"{PLUGIN_BASE}/storage/cleanup", json={"strategy": "balanced"}
        )).json())
        assert cleanup["success"] is True
        assert sum(int(v) for v in cleanup["removed"].values()) >= 0


@pytest.mark.asyncio
async def test_unknown_endpoint_returns_404_envelope():
    async with _client() as client:
        resp = await client.get(f"{PLUGIN_BASE}/definitely-not-a-route")
        assert resp.status == 404
        body = await resp.json()
        assert body["success"] is False
