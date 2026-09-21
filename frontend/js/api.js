/* Thin API client: token storage + fetch wrapper. */
const API = (() => {
  const KEY = 'neeldrik.token', UKEY = 'neeldrik.user';
  let mem = { token: null, user: null };

  function store(k, v) {
    try { v === null ? localStorage.removeItem(k) : localStorage.setItem(k, v); }
    catch (e) { /* private mode — fall back to memory */ }
  }
  function read(k) {
    try { return localStorage.getItem(k); } catch (e) { return null; }
  }

  function setSession(token, user) {
    mem.token = token; mem.user = user;
    store(KEY, token); store(UKEY, JSON.stringify(user));
  }
  function clearSession() {
    mem = { token: null, user: null };
    store(KEY, null); store(UKEY, null);
  }
  function token() { return mem.token || read(KEY); }
  function user() {
    if (mem.user) return mem.user;
    try { return JSON.parse(read(UKEY) || 'null'); } catch (e) { return null; }
  }

  async function req(path, { method = 'GET', body, form } = {}) {
    const headers = {};
    const t = token();
    if (t) headers['Authorization'] = 'Bearer ' + t;
    let payload;
    if (form) { payload = form; }
    else if (body !== undefined) {
      headers['Content-Type'] = 'application/json';
      payload = JSON.stringify(body);
    }
    const res = await fetch(path, { method, headers, body: payload });
    let data = null;
    try { data = await res.json(); } catch (e) { /* non-JSON */ }
    if (!res.ok) {
      const err = new Error((data && data.error) || `Request failed (${res.status})`);
      err.status = res.status;
      throw err;
    }
    return data;
  }

  return {
    setSession, clearSession, token, user, req,
    health: () => req('/api/health'),
    model: () => req('/api/model'),
    register: (b) => req('/api/auth/register', { method: 'POST', body: b }),
    login: (b) => req('/api/auth/login', { method: 'POST', body: b }),
    me: () => req('/api/auth/me'),
    demo: () => req('/api/auth/demo', { method: 'POST' }),
    dashboard: () => req('/api/dashboard'),
    setAvatar: (dataUri) => req('/api/account/avatar',
                 { method: 'POST', body: { avatar: dataUri } }),
    detections: () => req('/api/detections'),
    detection: (id) => req('/api/detections/' + id),
    del: (id) => req('/api/detections/' + id + '/delete', { method: 'POST' }),
    purge: () => req('/api/detections/purge', { method: 'POST' }),
    samples: () => req('/api/samples'),
    drift: (b) => req('/api/drift', { method: 'POST', body: b }),
    driftVerify: () => req('/api/drift/verify'),
    /* opts: { sceneKm, windMs, lat, lon } - every field optional. The scene
       centre is what places the slick on a map; without it the result is
       returned with scene_center:null and is deliberately NOT mapped. */
    detect: (file, opts) => {
      const o = opts || {};
      const fd = new FormData();
      fd.append('file', file, file.name || 'upload.png');
      const put = (k, v) => {
        if (v !== null && v !== undefined && v !== '' && !Number.isNaN(v))
          fd.append(k, String(v));
      };
      put('scene_km', o.sceneKm);
      put('wind_ms', o.windMs);
      put('lat', o.lat);
      put('lon', o.lon);
      return req('/api/detect', { method: 'POST', form: fd });
    }
  };
})();
