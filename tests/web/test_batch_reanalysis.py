"""执行独立前端模块，验证选择快照、轮询串行化和失败重试。"""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_batch_reanalysis_controller():
    node = shutil.which("node")
    if not node:
        pytest.skip("需要 Node.js 执行前端模块")
    module = Path(__file__).resolve().parents[2] / "pages/dashboard/batch_reanalysis.js"
    script = r"""
import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
const { createBatchReanalysis } = await import(pathToFileURL(process.argv[1]));
const timers = new Map(); let timerID = 0;
globalThis.setTimeout = (fn) => { timers.set(++timerID, fn); return timerID; };
globalThis.clearTimeout = (id) => timers.delete(id);
let serverJob = null, hold = null, refreshed = 0, statusCalls = 0, posts = [];
const selected = { value: new Set(['a', 'b', 'c']) };
const controller = createBatchReanalysis({
    ref: (value) => ({ value }), computed: (fn) => ({ get value() { return fn(); } }),
    selectedImages: selected, t: (_key, fallback) => fallback,
    refreshView: async () => { refreshed++; },
    loadThumbnail: async () => ({ url: 'data:image/png;base64,test' }),
    showConfirm: async () => { selected.value = new Set(['changed']); return true; },
    apiFetch: async (url, options) => {
        if (url.includes('/result?')) {
            const hash = new URLSearchParams(url.split('?')[1]).get('hash');
            return { json: async () => ({ success: true, result: { hash, status: 'ready', before: {desc:'old'}, after: {desc:'new'} } }) };
        }
        if (url.endsWith('/status')) { statusCalls++; if (hold) await hold; }
        else {
            const payload = JSON.parse(options.body); posts.push({url, ...payload});
            if (url.endsWith('/review')) {
                const item = serverJob.results.find((item) => item.hash === payload.hash);
                item.status = payload.decision === 'save' ? 'saved' : 'discarded';
                serverJob.pending--;
                if (payload.decision === 'save') serverJob.saved++;
            } else if (url.endsWith('/retry')) {
                serverJob.status = 'running'; serverJob.failed = 0;
            } else if (url.endsWith('/cancel')) serverJob.status = 'stopping';
            else serverJob = { id: 'job1', status: 'running', errors: [], results: [], pending: 0, saved: 0, processed: 0, total: payload.hashes.length };
        }
        return { json: async () => ({ success: true, task: serverJob ? structuredClone(serverJob) : null }) };
    },
});
await controller.restore();
assert.equal(timers.size, 0);
await controller.startSelected();
assert.deepEqual(posts[0].hashes, ['a', 'b', 'c']);
let release; hold = new Promise((resolve) => { release = resolve; });
const tick = [...timers.values()][0];
const first = tick(), second = tick();
assert.equal(first, second);
assert.equal(statusCalls, 2);
serverJob = { ...serverJob, status: 'completed', failed: 1, pending: 2,
    results: ['a','b'].map(hash=>({hash,name:hash,status:'ready'})),
    errors: [{ hash: 'c', status: 'failed' }] };
release(); await first; hold = null;
assert.equal(refreshed, 0);
assert.equal(posts.length, 1);
assert.equal(controller.reviewResult.value.hash, 'a');
assert.equal(controller.reviewResult.value.before.desc, 'old');
await controller.retryFailed();
assert.equal(posts[1].task_id, 'job1');
assert.equal(controller.readyResults.value.length, 2);
await controller.saveResult();
assert.equal(posts[2].hash, 'a');
assert.equal(posts[2].decision, 'save');
assert.equal(controller.job.value.saved, 1);
assert.equal(controller.reviewResult.value.hash, 'b');
assert.equal(refreshed, 1);
await controller.discardResult();
assert.equal(posts[3].hash, 'b');
assert.equal(posts[3].decision, 'discard');
assert.equal(controller.job.value.pending, 0);
assert.equal(refreshed, 1);
await controller.cancel();
assert.equal(controller.job.value.status, 'stopping');

// 验证无语义筛选、超过 20 张 Token 消耗门控警告、以及一键全部保存
let confirmMessages = [];
const missingHashes = Array.from({ length: 25 }, (_, idx) => `m${idx}`);
const controller2 = createBatchReanalysis({
    ref: (value) => ({ value }), computed: (fn) => ({ get value() { return fn(); } }),
    selectedImages: { value: new Set() },
    t: (_key, fallback) => fallback,
    refreshView: async () => { refreshed++; },
    loadThumbnail: async () => ({ url: 'data:image/png;base64,test' }),
    showConfirm: async (msg) => { confirmMessages.push(msg); return true; },
    apiFetch: async (url, options) => {
        if (url.endsWith('/missing')) {
            return { json: async () => ({ success: true, missing: { hashes: missingHashes, total_missing: 25, max_images: 500 } }) };
        }
        if (url.includes('/result?')) {
            const hash = new URLSearchParams(url.split('?')[1]).get('hash');
            return { json: async () => ({ success: true, result: { hash, status: 'ready', before: {}, after: { desc: 'filled' } } }) };
        }
        const payload = options?.body ? JSON.parse(options.body) : {};
        posts.push({ url, ...payload });
        if (url.endsWith('/review') && payload.all) {
            serverJob.results.forEach((item) => { item.status = 'saved'; });
            serverJob.saved = serverJob.pending;
            serverJob.pending = 0;
        } else {
            serverJob = {
                id: 'job2', status: 'completed', errors: [], pending: missingHashes.length, saved: 0,
                processed: missingHashes.length, total: missingHashes.length,
                results: missingHashes.map((hash) => ({ hash, name: hash, status: 'ready' })),
            };
        }
        return { json: async () => ({ success: true, task: structuredClone(serverJob) }) };
    },
});
await controller2.startMissing();
assert.equal(confirmMessages.length, 1);
assert.match(confirmMessages[0], /请注意 token 消耗/);
assert.match(confirmMessages[0], /25/);
const lastStart = posts.find((item) => item.fill_empty_only === true);
assert.ok(lastStart);
assert.equal(lastStart.hashes.length, 25);
await controller2.saveAllResults();
const bulkSave = posts.find((item) => item.all === true);
assert.ok(bulkSave);
assert.equal(bulkSave.decision, 'save');
assert.equal(controller2.job.value.pending, 0);
assert.equal(controller2.job.value.saved, 25);
controller2.dispose();

// 验证按当前视图（子库 + 角色 + 分类）筛选无语义图片与范围确认提示
let scopedUrls = [], scopedAlerts = [], scopedConfirms = [];
const controller3 = createBatchReanalysis({
    ref: (value) => ({ value }), computed: (fn) => ({ get value() { return fn(); } }),
    selectedImages: { value: new Set() },
    t: (_key, fallback) => fallback,
    refreshView: async () => {},
    loadThumbnail: async () => null,
    showAlert: (msg) => { scopedAlerts.push(msg); },
    showConfirm: async (msg) => { scopedConfirms.push(msg); return false; },
    getMissingScope: () => ({
        params: { library: 'characters', character: 'mika', category: 'happy' },
        label: '角色表情库 · 美香 · 开心',
    }),
    apiFetch: async (url) => {
        scopedUrls.push(url);
        return {
            json: async () => ({
                success: true,
                missing: { hashes: ['h1', 'h2'], total_missing: 2, max_images: 500 },
            }),
        };
    },
});
await controller3.startMissing();
assert.equal(scopedUrls[0], 'api/images/reanalyze/missing?library=characters&character=mika&category=happy');
assert.equal(scopedConfirms.length, 1);
assert.match(scopedConfirms[0], /当前筛选范围：角色表情库 · 美香 · 开心/);
assert.match(scopedConfirms[0], /2 张/);
controller3.dispose();

// 验证外部源勾选入库自动语义化且无语义图片 > 20 张时弹出 token 警告门控
const { createSourceActions } = await import(pathToFileURL(process.argv[2]));
let sourceConfirms = [], importCalled = 0;
const sourceActions = createSourceActions({
    apiFetch: async () => { importCalled++; return { json: async () => ({ success: true, job: { job_id: 'j1', status: 'completed' } }) }; },
    applySourceInspection: () => {},
    resetSourceCategoryMap: () => {},
    fetchSources: async () => {},
    refreshView: async () => {},
    showConfirm: async (msg) => { sourceConfirms.push(msg); return false; },
    t: (_key, fallback) => fallback,
    sourceLoading: { value: false },
    sourceError: { value: '' },
    sourceFile: { value: null },
    sourceUploadedPath: { value: '/tmp/pack.zip' },
    sourceSelected: { value: null },
    sourceInspection: { value: { item_count: 30, missing_semantics_count: 24 } },
    sourceCategoryMap: {},
    sourceForm: { review: false, auto_analyze: true, scope_mode: 'public', origin_target: '', assign_character: false, character: '' },
    sourceDefaults: { value: { review: false } },
    sourceJob: { value: null },
});
await sourceActions.startSourceImport();
assert.equal(sourceConfirms.length, 1);
assert.match(sourceConfirms[0], /请注意 token 消耗/);
assert.match(sourceConfirms[0], /24/);
assert.equal(importCalled, 0);

controller.dispose();
assert.equal(timers.size, 0);
console.log('controller passed');
"""
    source_module = Path(__file__).resolve().parents[2] / "pages/dashboard/source_actions.js"
    result = subprocess.run([node, "--input-type=module", "-e", script, str(module), str(source_module)],
                            capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr
