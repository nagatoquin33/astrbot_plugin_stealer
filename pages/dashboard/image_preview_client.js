class LRUCache {
    constructor(limit) {
        this.limit = Math.max(1, Number(limit) || 1);
        this.values = new Map();
    }

    get(key) {
        if (!this.values.has(key)) return null;
        const value = this.values.get(key);
        this.values.delete(key);
        this.values.set(key, value);
        return value;
    }

    set(key, value) {
        if (this.values.has(key)) this.values.delete(key);
        else if (this.values.size >= this.limit) {
            this.values.delete(this.values.keys().next().value);
        }
        this.values.set(key, value);
    }

    has(key) {
        return this.values.has(key);
    }
}

export class ImagePreviewClient {
    constructor(apiGet, { maxConcurrent = 4 } = {}) {
        this.apiGet = apiGet;
        this.maxConcurrent = Math.max(1, Number(maxConcurrent) || 4);
        this.active = 0;
        this.queue = [];
        this.requests = new Map();
        this.thumbnailCache = new LRUCache(300);
        this.originalCache = new LRUCache(4);
    }

    loadThumbnail(hash) {
        return this._request({
            requestKey: `thumbnail:${hash}`,
            hash,
            cache: this.thumbnailCache,
            run: async () => {
                try {
                    return await this.apiGet('thumbnail', { hash, size: 300 });
                } catch (error) {
                    console.error('Failed to load thumbnail:', hash, error);
                    return null;
                }
            },
        });
    }

    loadOriginal(hash) {
        return this._request({
            requestKey: `original:${hash}`,
            hash,
            cache: this.originalCache,
            priority: true,
            run: async () => {
                try {
                    return await this.apiGet('image-data', { hash });
                } catch (error) {
                    console.error('Failed to load original image:', hash, error);
                    return null;
                }
            },
        });
    }

    _request({ requestKey, hash, cache, run, priority = false }) {
        const cached = cache.get(hash);
        if (cached) return Promise.resolve({ url: cached });

        const existing = this.requests.get(requestKey);
        if (existing) {
            if (priority && this.queue.includes(existing.job)) {
                this.queue.splice(this.queue.indexOf(existing.job), 1);
                this.queue.unshift(existing.job);
            }
            return existing.promise;
        }

        const job = { requestKey, hash, cache, run, resolve: null, reject: null };
        const promise = new Promise((resolve, reject) => {
            job.resolve = resolve;
            job.reject = reject;
        });
        this.requests.set(requestKey, { job, promise });
        if (priority) this.queue.unshift(job);
        else this.queue.push(job);
        this._pump();
        return promise;
    }

    _pump() {
        while (this.active < this.maxConcurrent && this.queue.length) {
            const job = this.queue.shift();
            this.active += 1;
            Promise.resolve()
                .then(() => job.run())
                .then((data) => {
                    if (data?.url) job.cache.set(job.hash, data.url);
                    job.resolve(data);
                })
                .catch((error) => job.reject(error))
                .finally(() => {
                    const current = this.requests.get(job.requestKey);
                    if (current?.job === job) this.requests.delete(job.requestKey);
                    this.active -= 1;
                    this._pump();
                });
        }
    }
}
