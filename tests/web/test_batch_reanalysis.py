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
controller.dispose();
assert.equal(timers.size, 0);
console.log('controller passed');
"""
    result = subprocess.run([node, "--input-type=module", "-e", script, str(module)],
                            capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr
