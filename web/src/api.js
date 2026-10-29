// Shared job-server calls for the Studio and the Capture Coach.

// XHR (not fetch) for upload progress on multi-GB videos. Resolves with the JSON reply; rejects with
// err.down = true when the server can't be reached.
export function uploadFile(url, file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', url);
    xhr.upload.onprogress = (ev) => onProgress?.(ev.loaded);
    xhr.onload = () => {
      let res = {};
      try { res = JSON.parse(xhr.responseText || '{}'); } catch { /* proxy error page */ }
      xhr.status === 201 ? resolve(res) : reject(new Error(res.error || `upload failed (${xhr.status})`));
    };
    xhr.onerror = () => reject(Object.assign(new Error('server unreachable'), { down: true }));
    xhr.send(file);
  });
}

export async function postJson(url, body) {
  let r;
  try {
    r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  } catch { throw Object.assign(new Error('server unreachable'), { down: true }); }
  const res = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(res.error || `request failed (${r.status})`);
  return res;
}
