function externalChannelLabel(item, t) {
    const source = String(item?.source || '').trim().toLowerCase();
    const addMethod = String(item?.add_method || '').trim().toLowerCase();
    if (!source.startsWith('external:') && addMethod !== 'external_import') return '';

    const imported = t('pages.dashboard.fields.add_method_external', 'External import');
    const kind = source.startsWith('external:') ? source.slice('external:'.length) : '';
    let channel = '';
    if (['meme_pack', 'pack', 'meme_manager', 'meme-manager'].includes(kind)) {
        channel = t('pages.dashboard.fields.source_channel_pack', 'Meme pack');
    } else if (['github', 'github_repo', 'github-repo', 'repository'].includes(kind)) {
        channel = t('pages.dashboard.fields.source_channel_github', 'GitHub');
    } else if (['http_json', 'http', 'api'].includes(kind)) {
        channel = t('pages.dashboard.fields.source_channel_http_json', 'JSON API');
    }
    return channel ? `${imported} · ${channel}` : imported;
}

export function formatItemOriginLabel(item, formatOriginTarget, t) {
    const originTarget = String(item?.origin_target || '').trim();
    const imported = externalChannelLabel(item, t);
    if (originTarget && imported) return `${formatOriginTarget(originTarget)} · ${imported}`;
    if (originTarget) return formatOriginTarget(originTarget);
    return imported || formatOriginTarget('');
}
