/* NEELDRIK console: hash router + views. */
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const view = $('#view');
  const state = { last: null, samples: [], detections: [] };


  const esc = (s) => String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
  const fmtTime = (t) => new Date(t * 1000).toLocaleString();
  const pillClass = (v) => v === 'SPILL' ? 'spill' : v === 'LOOKALIKE' ? 'look'
    : v === 'REVIEW' ? 'review' : v === 'UNSUPPORTED' ? 'unsup' : 'clean';
  const pillText = (v) => v === 'SPILL' ? 'OIL SPILL'
    : v === 'LOOKALIKE' ? 'LOOK-ALIKE'
    : v === 'REVIEW' ? 'NEEDS REVIEW'
    : v === 'UNSUPPORTED' ? 'NOT ANALYSED' : 'CLEAN';

  const initials = (label) => (String(label).replace(/[^A-Za-z ]/g, '').trim()
    .split(/\s+/).slice(0, 2).map(w => w[0] || '').join('').toUpperCase() || 'AN');

  function paintAvatar(uri) {
    const av = $('#avatar'), im = $('#avatarimg');
    if (!av || !im) return;
    if (uri) { im.src = uri; im.classList.remove('hidden'); av.classList.add('hidden'); }
    else { im.classList.add('hidden'); av.classList.remove('hidden'); }
  }

  // ------------------------------------------------------------------ //
  const VIEWS = {
    ops: { title: 'Dashboard', render: () => window.renderOps({
             view, API, pillClass, pillText, go }) },
    dashboard: { title: 'Dashboard', render: renderDashboard },
    detect: { title: 'Detect spill', render: renderDetect },
    history: { title: 'History', render: renderHistory },
    drift: { title: 'Drift & origin', render: renderDrift },
    model: { title: 'Model', render: renderModel },
    account: { title: 'Account & privacy', render: renderAccount }
  };

  function go(name) {
    if (!VIEWS[name]) name = 'ops';
    $$('#nav a').forEach(a => a.classList.toggle('on', a.dataset.view === name));
    $('#vtitle').textContent = VIEWS[name].title;
    $('#crumb').textContent = 'NEELDRIK / ' + VIEWS[name].title;
    $('#vmeta').textContent = '';
    view.innerHTML = '<div class="empty"><span class="spinner"></span></div>';
    location.hash = name;
    Promise.resolve(VIEWS[name].render()).catch(e => {
      if (e.status === 401) { API.clearSession(); location.replace('/login'); return; }
      view.innerHTML = `<div class="card"><h3>Error</h3><p>${esc(e.message)}</p></div>`;
    });
  }

  $('#nav').addEventListener('click', e => {
    const a = e.target.closest('a[data-view]');
    if (a) { e.preventDefault(); go(a.dataset.view); }
  });
  $('#logout').onclick = () => { API.clearSession(); location.replace('/login'); };
  addEventListener('hashchange', () => go(location.hash.slice(1)));

  // ------------------------------------------------------------------ //
  async function renderDashboard() {
    const d = await API.dashboard();
    const s = d.stats;
    const by = s.by_verdict || {};
    view.innerHTML = `
      <div class="grid g4" style="margin-bottom:16px">
        <div class="card stat"><div class="n">${s.total}</div><div class="l">scenes analysed</div></div>
        <div class="card stat"><div class="n" style="color:#EE9C93">${by.SPILL || 0}</div><div class="l">spills confirmed</div></div>
        <div class="card stat"><div class="n" style="color:#E4C463">${by.LOOKALIKE || 0}</div><div class="l">look-alikes rejected</div></div>
        <div class="card stat"><div class="n">${(s.area_km2 || 0).toFixed(1)}</div><div class="l">km² affected (est.)</div></div>
      </div>
      <div class="grid split">
        <div class="card">
          <h3>Recent detections</h3>
          ${d.recent.length ? `<table><thead><tr><th>Scene</th><th>Verdict</th>
            <th>Confidence</th><th>Area km²</th><th>When</th></tr></thead><tbody>
            ${d.recent.map(r => `<tr data-open="${r.id}" style="cursor:pointer">
              <td class="mono">${esc(r.filename)}</td>
              <td><span class="pill ${pillClass(r.verdict)}">${pillText(r.verdict)}</span></td>
              <td class="mono">${((r.confidence || 0) * 100).toFixed(1)}%</td>
              <td class="mono">${(r.area_km2 || 0).toFixed(2)}</td>
              <td class="dim">${fmtTime(r.created)}</td></tr>`).join('')}
            </tbody></table>`
            : `<div class="empty">No scenes analysed yet.<br>
                 <button class="btn sm" id="first" style="margin-top:12px">Analyse your first scene</button></div>`}
        </div>
        <div>
          <div class="card" style="margin-bottom:16px">
            <h3>Detection model</h3>
            <div class="kv">
              <span class="k">Architecture</span><span class="v">U-Net · 5 class</span>
              <span class="k">Runtime</span><span class="v">${esc(d.model.runtime)}</span>
              <span class="k">Parameters</span><span class="v">${(d.model.parameters || 0).toLocaleString()}</span>
              ${d.model.real_holdout ? `
              <span class="k">Precision <i class="qmark" title="On held-out real Sentinel-1 scenes, split by acquisition.">?</i></span>
                <span class="v">${d.model.real_holdout.precision != null
                  ? (d.model.real_holdout.precision * 100).toFixed(1) + '%' : '—'}</span>
              <span class="k">Recall</span>
                <span class="v">${d.model.real_holdout.recall != null
                  ? (d.model.real_holdout.recall * 100).toFixed(1) + '%' : '—'}</span>
              <span class="k">IoU (oil), real SAR</span>
                <span class="v">${typeof d.model.real_holdout.iou.Oil === 'number'
                  ? d.model.real_holdout.iou.Oil.toFixed(3) : '—'}</span>` : `
              <span class="k">Spill accuracy</span><span class="v">${d.model.spill_accuracy != null ? (d.model.spill_accuracy * 100).toFixed(1) + '% (synthetic)' : '—'}</span>
              <span class="k">IoU (oil)</span><span class="v">${
                d.model.iou && typeof d.model.iou.Oil === 'number'
                  ? d.model.iou.Oil.toFixed(3) + ' (synthetic)' : '—'}</span>`}
              <span class="k">Storage</span><span class="v">${esc(d.storage)}</span>
            </div>
          </div>
          <div class="card">
            <h3>Pipeline</h3>
            <div style="font-size:13px;line-height:2;color:var(--dim)">
              Upload → validate → preprocess (Lee despeckle) → <b style="color:var(--water)">U-Net</b>
              → post-process → characterise → evidence → JSON → dashboard
            </div>
          </div>
        </div>
      </div>`;
    const f = $('#first'); if (f) f.onclick = () => go('detect');
    $$('#view tr[data-open]').forEach(tr => tr.onclick = () => openDetection(tr.dataset.open));
  }

  // ------------------------------------------------------------------ //
  async function renderDetect() {
    view.innerHTML = `
      <div class="grid split">
        <div>
          <div class="card" style="margin-bottom:16px">
            <h3>Upload a SAR scene</h3>
            <div class="drop" id="drop">
              <div style="font-size:15px;font-weight:600">Drop a Sentinel-1 image here</div>
              <div class="dim" style="font-size:13px;margin-top:5px">
                or click to browse · PNG, JPEG, GeoTIFF · max 25 MB</div>
              <input type="file" id="file" class="hidden" accept=".png,.jpg,.jpeg,.tif,.tiff,.bmp">
            </div>
            <div id="chosen" class="dim mono hidden" style="margin-top:10px;font-size:13px"></div>
            <div style="display:flex;gap:10px;align-items:flex-end;margin-top:14px">
              <div style="flex:1">
                <label for="km">Assumed scene width (km)</label>
                <input id="km" type="number" value="40" min="1" max="500" step="1">
              </div>
              <div style="flex:1">
                <label for="wind">Wind speed (m/s) <span class="dim">— optional</span></label>
                <input id="wind" type="number" placeholder="unknown" min="0" max="60" step="0.1">
              </div>
              <button class="btn" id="run" style="width:auto;padding:11px 22px" disabled>Analyse spill</button>
            </div>
            <div class="dimmer" style="font-size:12px;margin-top:8px">
              Scene width is used only for the area estimate when the raster carries
              no CRS. Wind is the strongest physical check available: below 3 m/s a
              glassy sea makes dark patches that mimic oil, and above 12 m/s a real
              slick breaks up. Leave it blank if you do not know it — the system
              will say so rather than assume.
            </div>
            <div class="msg" id="dmsg"></div>
          </div>
          <div class="card">
            <h3>Or try a labelled sample</h3>
            <div class="samples" id="samples"><span class="dim">loading…</span></div>
          </div>
        </div>
        <div class="card" id="rescard">
          <h3>Result</h3>
          <div class="empty">Upload a scene to run the detector.</div>
        </div>
      </div>`;

    const drop = $('#drop'), file = $('#file'), run = $('#run');
    let chosen = null;

    function pick(f) {
      chosen = f;
      $('#chosen').classList.remove('hidden');
      $('#chosen').textContent = `${f.name} · ${(f.size / 1024).toFixed(0)} KB`;
      run.disabled = false;
      $$('#samples button').forEach(b => b.classList.remove('on'));
    }
    drop.onclick = () => file.click();
    file.onchange = () => file.files[0] && pick(file.files[0]);
    ['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => {
      e.preventDefault(); drop.classList.add('over');
    }));
    ['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => {
      e.preventDefault(); drop.classList.remove('over');
    }));
    drop.addEventListener('drop', e => {
      const f = e.dataTransfer.files[0]; if (f) pick(f);
    });

    run.onclick = async () => {
      if (!chosen) return;
      run.disabled = true;
      const lab = run.textContent;
      run.innerHTML = '<span class="spinner"></span>';
      $('#dmsg').className = 'msg';
      try {
        const windRaw = $('#wind') ? $('#wind').value : '';
        const wind = windRaw === '' ? null : parseFloat(windRaw);
        const res = await API.detect(chosen, parseFloat($('#km').value) || null,
                                     Number.isFinite(wind) ? wind : null);
        state.last = res;
        showResult(res);
      } catch (e) {
        $('#dmsg').className = 'msg err';
        $('#dmsg').textContent = e.message;
      } finally {
        run.disabled = false; run.textContent = lab;
      }
    };

    const sm = await API.samples();
    state.samples = sm.items;
    $('#samples').innerHTML = sm.items.length ? sm.items.map(s => `
      <button data-url="${esc(s.url)}" data-name="${esc(s.file)}" title="${esc(s.name)}">
        <img src="${esc(s.url)}" alt="${esc(s.name)}">
        <span>${esc(s.kind)}</span>
      </button>`).join('')
      : '<span class="dim">No samples installed.</span>';
    $$('#samples button').forEach(b => b.onclick = async () => {
      $$('#samples button').forEach(x => x.classList.remove('on'));
      b.classList.add('on');
      const blob = await (await fetch(b.dataset.url)).blob();
      chosen = new File([blob], b.dataset.name, { type: 'image/png' });
      $('#chosen').classList.remove('hidden');
      $('#chosen').textContent = b.dataset.name + ' · sample';
      run.disabled = false;
    });
  }

  function showResult(r) {
    const cls = pillClass(r.verdict);

    // Out-of-domain input is refused rather than guessed at.
    if (r.analysed === false) {
      $('#rescard').innerHTML = `
        <h3>Result</h3>
        <div class="verdict unsup" style="margin-bottom:16px">
          <div>
            <div class="big">NOT ANALYSED</div>
            <div class="dim" style="font-size:13px">${esc(r.label)}</div>
          </div>
        </div>
        <div class="imgwrap" style="margin-bottom:16px">
          <img src="${r.scene_png}" alt="uploaded image">
        </div>
        <h4 style="margin:0 0 4px;color:var(--t-review)">Why no verdict was given</h4>
        <p style="font-size:14px;line-height:1.65;margin:0 0 12px">${esc(r.domain_reason)}</p>
        <p class="dim" style="font-size:13px;line-height:1.6;margin-top:0">
          Deliberately refusing input the model was not trained on is what stops
          false alarms. Upload a Sentinel-1 SAR scene, or use one of the labelled
          samples, to get a verdict.</p>`;
      return;
    }
    $('#rescard').innerHTML = `
      <h3>Result</h3>
      <div class="verdict ${cls}" style="margin-bottom:16px">
        <div>
          <div class="big">${pillText(r.verdict)}</div>
          <div class="dim" style="font-size:13px">${esc(r.label)}</div>
        </div>
        <div class="conf">
          <div class="n">${(r.confidence * 100).toFixed(1)}%</div>
          <div class="dim" style="font-size:11px;letter-spacing:.12em;text-transform:uppercase">AI confidence</div>
        </div>
      </div>
      ${(r.reasons && r.reasons.length) ? `
      <div class="why">
        <h4>Why this verdict</h4>
        <ul>${r.reasons.map(x => `<li>${esc(x)}</li>`).join('')}</ul>
      </div>` : ''}
      ${r.verdict === 'REVIEW' ? `
      <div class="why" style="border-color:rgba(214,132,42,.48)">
        <h4 style="color:#F0B173">Deferred, not decided</h4>
        <p style="margin:0;font-size:13px;line-height:1.65">
          Oil and look-alikes genuinely overlap in SAR — forcing this scene into
          spill or clean would manufacture an error. It is routed to an analyst
          instead, which is how CleanSeaNet operates: it issues alerts for
          verification, not verdicts.</p>
      </div>` : ''}

      <div class="imgtabs">
        <button class="on" data-img="overlay">Detection overlay</button>
        <button data-img="scene">Original scene</button>
      </div>
      <div class="imgwrap" style="margin-bottom:16px">
        <img id="resimg" src="${r.overlay_png}" alt="detection overlay">
      </div>

      <div class="kv" style="margin-bottom:16px">
        <span class="k">Candidate regions</span><span class="v">${r.candidate_count}</span>
        <span class="k">Estimated area</span><span class="v">${r.estimated_area_km2.toFixed(3)} km²${r.area_is_estimate ? ' *' : ''}</span>
        <span class="k">Oil coverage</span><span class="v">${(r.oil_area_frac * 100).toFixed(2)}%</span>
        <span class="k">P(oil) in mask</span><span class="v">${r.oil_evidence.toFixed(3)}</span>
        <span class="k">P(look-alike)</span><span class="v">${r.lookalike_evidence.toFixed(3)}</span>
        <span class="k">Georeferenced</span><span class="v">${r.georeferenced ? 'yes' : 'no'}</span>
        <span class="k">Inference</span><span class="v">${r.elapsed_ms} ms · ${esc(r.runtime)}</span>
      </div>
      ${r.area_is_estimate ? `<div class="dimmer" style="font-size:12px;margin-bottom:14px">
        * scene is not georeferenced — area derived from the assumed scene width.</div>` : ''}

      <h3>Evidence trail</h3>
      <ul class="evidence">
        ${r.evidence.map(e => `<li>
          <span class="tick ${e.ok ? 'y' : 'n'}">${e.ok ? '✓' : '–'}</span>
          <span><b>${esc(e.step)}</b><br><span class="d">${esc(e.detail)}</span></span></li>`).join('')}
      </ul>

      ${r.regions.length ? `<h3 style="margin-top:16px">Regions</h3>
        <table><thead><tr><th>#</th><th>Area %</th><th>P(oil)</th>
        <th>Elongation</th><th>Centroid</th></tr></thead><tbody>
        ${r.regions.map((g, i) => `<tr><td class="mono">${i + 1}</td>
          <td class="mono">${(g.area_frac * 100).toFixed(2)}</td>
          <td class="mono">${g.mean_oil_p.toFixed(3)}</td>
          <td class="mono">${g.elongation.toFixed(2)}</td>
          <td class="mono dim">${g.centroid[0].toFixed(2)}, ${g.centroid[1].toFixed(2)}</td>
        </tr>`).join('')}</tbody></table>` : ''}

      <div style="display:flex;gap:9px;margin-top:16px;flex-wrap:wrap">
        <button class="btn ghost sm" id="dl-json">Download JSON</button>
        <button class="btn ghost sm" id="dl-geo">Download GeoJSON</button>
        <button class="btn ghost sm" id="to-map">Open in map</button>
      </div>`;

    $$('#rescard .imgtabs button').forEach(b => b.onclick = () => {
      $$('#rescard .imgtabs button').forEach(x => x.classList.remove('on'));
      b.classList.add('on');
      $('#resimg').src = b.dataset.img === 'scene' ? r.scene_png : r.overlay_png;
    });
    const save = (obj, name) => {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(new Blob([JSON.stringify(obj, null, 2)],
        { type: 'application/json' }));
      a.download = name; a.click(); URL.revokeObjectURL(a.href);
    };
    $('#dl-json').onclick = () => {
      const c = Object.assign({}, r); delete c.overlay_png; delete c.scene_png;
      save(c, `neeldrik-${r.id}.json`);
    };
    $('#dl-geo').onclick = () => save(r.geojson, `neeldrik-${r.id}.geojson`);
    $('#to-map').onclick = () => go('map');
  }

  // ------------------------------------------------------------------ //
  async function renderHistory() {
    const d = await API.detections();
    state.detections = d.items;
    $('#vmeta').textContent = d.items.length + ' records';
    view.innerHTML = `<div class="card">
      ${d.items.length ? `<table><thead><tr><th>S.No.</th><th>Scene</th><th>Verdict</th>
        <th>Confidence</th><th>Area km²</th><th>Regions</th><th>When</th><th></th></tr></thead>
        <tbody>${d.items.map((r, i) => `<tr>
          <!-- Position in the list, not the database key. The key is never
               reused after a delete, so showing it leaves growing gaps
               (#78, #81, #84) that read as a counting bug. The real id is
               kept in the tooltip and still drives open/delete. -->
          <td class="mono dim" title="Record ID ${r.id}">${d.items.length - i}</td>
          <td class="mono" data-open="${r.id}" style="cursor:pointer">${esc(r.filename)}</td>
          <td><span class="pill ${pillClass(r.verdict)}">${pillText(r.verdict)}</span></td>
          <td class="mono">${((r.confidence || 0) * 100).toFixed(1)}%</td>
          <td class="mono">${(r.area_km2 || 0).toFixed(2)}</td>
          <td class="mono">${r.candidates}</td>
          <td class="dim">${fmtTime(r.created)}</td>
          <td><button class="btn danger sm" data-del="${r.id}">Delete</button></td>
        </tr>`).join('')}</tbody></table>`
        : '<div class="empty">No detections stored yet.</div>'}
      </div>
      <div id="detail" style="margin-top:16px"></div>`;
    $$('#view [data-open]').forEach(el => el.onclick = () => openDetection(el.dataset.open));
    $$('#view [data-del]').forEach(b => b.onclick = async () => {
      b.disabled = true;
      await API.del(b.dataset.del);
      renderHistory();
    });
  }

  async function openDetection(id) {
    const r = await API.detection(id);
    go('detect');
    setTimeout(() => { state.last = r; showResult(r); }, 60);
  }

  // ------------------------------------------------------------------ //
  async function renderDrift() {
    const last = state.last;
    view.innerHTML = `
      <div class="grid split">
        <div class="card">
          <h3>Back-track the slick to its origin</h3>
          <p class="dim" style="font-size:13.5px;margin-top:0">
            Surface oil moves with the current plus about 3% of the wind, deflected
            by Coriolis. Running that backwards in time gives the release point.</p>
          <div class="grid g2" style="gap:12px">
            <div><label for="d-lat">Slick latitude</label>
              <input id="d-lat" type="number" step="0.0001" value="18.9500"></div>
            <div><label for="d-lon">Slick longitude</label>
              <input id="d-lon" type="number" step="0.0001" value="72.3000"></div>
            <div><label for="d-ws">Wind speed (m/s)</label>
              <input id="d-ws" type="number" step="0.1" value="7"></div>
            <div><label for="d-wd">Wind FROM (deg)</label>
              <input id="d-wd" type="number" step="1" value="225"></div>
            <div><label for="d-cs">Current speed (m/s)</label>
              <input id="d-cs" type="number" step="0.01" value="0.25"></div>
            <div><label for="d-cd">Current TOWARDS (deg)</label>
              <input id="d-cd" type="number" step="1" value="45"></div>
            <div><label for="d-h">Elapsed times (hours)</label>
              <input id="d-h" type="text" value="6, 12, 24"></div>
            <div><label for="d-k">Diffusivity (m²/s)</label>
              <input id="d-k" type="number" step="0.5" value="5"></div>
          </div>
          <div style="display:flex;gap:9px;margin-top:14px;flex-wrap:wrap">
            <button class="btn" id="d-run" style="width:auto;padding:11px 22px">Back-track</button>
            <button class="btn ghost sm" id="d-verify">Verify the physics</button>
          </div>
          <div class="msg" id="d-msg"></div>
        </div>
        <div class="card" id="d-res">
          <h3>Estimated origin</h3>
          <div class="empty">Enter the slick position and the conditions, then back-track.</div>
        </div>
      </div>
      <div class="card" style="margin-top:16px">
        <h3>Back-track path</h3>
        <div id="d-map"></div>
      </div>`;

    if (last && last.geojson && last.geojson.features.length && last.georeferenced) {
      const c = last.geojson.features[0].geometry.coordinates[0][0];
      $('#d-lon').value = c[0].toFixed(4); $('#d-lat').value = c[1].toFixed(4);
    }

    $('#d-verify').onclick = async () => {
      const v = await API.driftVerify();
      $('#d-msg').className = 'msg ' + (v.passed ? 'ok' : 'err');
      $('#d-msg').textContent = v.passed
        ? `All ${v.checks} physics checks passed — still water, the 3% rule, Coriolis `
          + `direction, exact forward/backward reversibility, step-size independence, `
          + `√(2Kt) spreading, and recovery of a known release point.`
        : 'Failed: ' + v.failed_checks.join(', ');
    };

    $('#d-run').onclick = async () => {
      const b = $('#d-run'); b.disabled = true;
      const lab = b.textContent; b.innerHTML = '<span class="spinner"></span>';
      $('#d-msg').className = 'msg';
      try {
        const hours = $('#d-h').value.split(',').map(s => parseFloat(s.trim()))
          .filter(x => !isNaN(x));
        const r = await API.drift({
          lat: $('#d-lat').value, lon: $('#d-lon').value,
          wind_speed: $('#d-ws').value, wind_dir: $('#d-wd').value,
          cur_speed: $('#d-cs').value, cur_dir: $('#d-cd').value,
          diffusivity: $('#d-k').value, hours
        });
        state.drift = r;
        showDrift(r);
      } catch (e) {
        $('#d-msg').className = 'msg err'; $('#d-msg').textContent = e.message;
      } finally { b.disabled = false; b.textContent = lab; }
    };
  }

  function showDrift(r) {
    $('#d-res').innerHTML = `
      <h3>Estimated origin</h3>
      <table><thead><tr><th>Hours before</th><th>Latitude</th><th>Longitude</th>
        <th>Distance</th><th>± 90%</th></tr></thead><tbody>
      ${r.candidates.map(c => `<tr>
        <td class="mono">${c.hours_before.toFixed(1)} h</td>
        <td class="mono">${c.lat.toFixed(4)}</td>
        <td class="mono">${c.lon.toFixed(4)}</td>
        <td class="mono">${c.distance_from_slick_km.toFixed(2)} km</td>
        <td class="mono dim">${(c.uncertainty_m / 1000).toFixed(2)} km</td>
      </tr>`).join('')}</tbody></table>
      <p class="dim" style="font-size:12.5px;margin-top:12px;margin-bottom:0">
        Each row is where the slick would have been that many hours earlier.
        The ± column is the 90th percentile spread of the particle cloud, which
        grows as √(2Kt) — that is the search radius, not an error bar on the path.</p>
      <div style="display:flex;gap:9px;margin-top:14px">
        <button class="btn ghost sm" id="d-geo">Download GeoJSON</button>
      </div>`;
    $('#d-geo').onclick = () => {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(new Blob([JSON.stringify(r.geojson, null, 2)],
        { type: 'application/json' }));
      a.download = 'neeldrik-backtrack.geojson'; a.click(); URL.revokeObjectURL(a.href);
    };

    const box = $('#d-map');
    if (typeof L === 'undefined') {
      box.outerHTML = `<div class="map-fallback"><div>Map tiles need an internet
        connection.<br>The coordinates and GeoJSON above are still exact.</div></div>`;
      return;
    }
    box.innerHTML = ''; box.id = 'd-map';
    if (box._map) { box._map.remove(); }
    const map = L.map('d-map');
    box._map = map;
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
      { attribution: '© OpenStreetMap', maxZoom: 18 }).addTo(map);
    const line = L.polyline(r.trajectory.map(p => [p.lat, p.lon]),
      { color: '#C9A227', weight: 3, dashArray: '6 5' }).addTo(map);
    L.circleMarker([r.slick.lat, r.slick.lon], { radius: 8, color: '#C2372D',
      fillColor: '#C2372D', fillOpacity: .85 }).addTo(map)
      .bindPopup('<b>Slick detected here</b>');
    r.candidates.forEach(c => {
      L.circle([c.lat, c.lon], { radius: c.uncertainty_m, color: '#2E7F9E',
        weight: 1, fillOpacity: .12 }).addTo(map);
      L.circleMarker([c.lat, c.lon], { radius: 6, color: '#2E7F9E',
        fillColor: '#2E7F9E', fillOpacity: .9 }).addTo(map)
        .bindPopup(`<b>${c.hours_before} h before</b><br>${c.distance_from_slick_km} km `
                   + `from the slick<br>search radius ${(c.uncertainty_m/1000).toFixed(2)} km`);
    });
    map.fitBounds(line.getBounds(), { padding: [40, 40] });
  }

  // ------------------------------------------------------------------ //
  async function renderModel() {
    const m = await API.model();
    const iou = m.iou || {};
    const rh = m.real_holdout;
    const pct = (v) => v != null ? (v * 100).toFixed(1) + '%' : '—';

    /* Real-SAR scores lead; the synthetic ones are kept but labelled, because
       they run several points higher and an unlabelled number here is the
       one that ends up quoted. */
    const bars = (obj, only) => Object.entries(obj)
      .filter(([k, v]) => (!only || only.includes(k)) && typeof v === 'number')
      .map(([k, v]) => `
        <div style="margin-bottom:11px">
          <div style="display:flex;justify-content:space-between;font-size:13.5px">
            <span>${esc(k)}</span><span class="mono">${v.toFixed(3)}</span></div>
          <div class="bar"><i style="width:${Math.max(2, v * 100)}%;background:${
            k === 'Oil' ? '#C2372D' : k === 'Look-alike' ? '#C9A227' : '#2E7F9E'}"></i></div>
        </div>`).join('');

    view.innerHTML = `<div class="grid g2">
      <div class="card">
        <h3>Architecture</h3>
        <div class="kv">
          <span class="k">Network</span><span class="v">${esc(m.architecture)}</span>
          <span class="k">Runtime</span><span class="v">${esc(m.runtime)}</span>
          <span class="k">Input</span><span class="v">${esc(m.input)}</span>
          <span class="k">Parameters</span><span class="v">${(m.parameters || 0).toLocaleString()}</span>
          <span class="k">Training patches</span><span class="v">${m.trained_on || '—'}</span>
          <span class="k">Epochs</span><span class="v">${m.epochs || '—'}</span>
        </div>
      </div>

      ${rh ? `
      <div class="card">
        <h3>Real Sentinel-1 — held-out scenes</h3>
        <div class="kv">
          <span class="k">Unseen scenes</span><span class="v">${rh.scenes}</span>
          <span class="k">Patches</span><span class="v">${rh.patches}</span>
          <span class="k">Oil IoU</span><span class="v">${
            typeof rh.iou.Oil === 'number' ? rh.iou.Oil.toFixed(3) : '—'}</span>
          <span class="k">Precision</span><span class="v">${pct(rh.precision)}</span>
          <span class="k">Recall</span><span class="v">${pct(rh.recall)}</span>
          <span class="k">F1</span><span class="v">${pct(rh.f1)}</span>
        </div>
        ${bars(rh.iou, ['Sea', 'Oil'])}
        <p class="dimmer" style="font-size:11.5px;line-height:1.55;margin:4px 0 0">
          Split by acquisition, so no scene appears in both training and test.
          Look-alike, Ship and Land receive no pixel supervision in this
          dataset — their IoU is meaningless and is not shown.
          Decision thresholds were calibrated on a set that includes these
          patches, so treat this as a held-out split rather than an untouched
          test set.
        </p>
      </div>` : `
      <div class="card">
        <h3>Real Sentinel-1</h3>
        <div class="empty">No real-SAR evaluation yet. Convert a labelled
          dataset and re-run train.py.</div>
      </div>`}

      <div class="card" style="grid-column:1/-1">
        <h3>Synthetic scenes <span class="pill review">not a real-SAR claim</span></h3>
        <p class="dim" style="font-size:13px;margin-top:0">
          Generated test scenes, kept because they are the only source of
          supervision for Ship and Land. They score higher than real imagery
          and should not be quoted on their own.
          Spill accuracy ${pct(m.spill_accuracy)} · precision ${pct(m.precision)} ·
          recall ${pct(m.recall)} · F1 ${pct(m.f1)}
        </p>
        ${Object.keys(iou).length ? `<div class="grid g2">${bars(iou)}</div>`
          : '<div class="empty">Run train.py to populate metrics.</div>'}
      </div>
      <div class="card" style="grid-column:1/-1">
        <h3>Decision thresholds (calibrated on held-out data)</h3>
        <div class="kv" style="max-width:520px">
          ${Object.entries(m.decision || {}).map(([k, v]) =>
            `<span class="k">${esc(k)}</span><span class="v">${v}</span>`).join('')}
        </div>
        <p class="dim" style="font-size:13px;margin-top:14px;margin-bottom:0">
          A pixel is an oil candidate when P(Oil) exceeds <b>oil_thr</b>. Components
          smaller than <b>min_area_frac</b> of the scene are discarded as speckle noise.
          A spill is declared only when total oil coverage reaches
          <b>spill_area_frac</b> <i>and</i> oil evidence exceeds look-alike evidence by
          the <b>dominance</b> factor — this is the gate that rejects low-wind and
          biogenic look-alikes.</p>
      </div>
    </div>`;
  }

  // ------------------------------------------------------------------ //
  async function renderAccount() {
    const u = API.user() || {};
    const pic = u.avatar || localStorage.getItem('nd-avatar') || '';
    view.innerHTML = `
      <div class="card profilecard" style="margin-bottom:18px">
        <h3>Profile</h3>
        <div class="profilerow">
          <div class="profilepic" id="ppic">
            ${pic ? `<img src="${pic}" alt="Profile picture">`
                  : `<span>${esc(initials(u.name || u.email || 'AN'))}</span>`}
          </div>
          <div class="profilemeta">
            <div class="pname">${esc(u.name || '—')}</div>
            <div class="pmail dim">${esc(u.email || '')}</div>
            <div class="profilebtns">
              <!-- A real <label for> opens the picker with no JavaScript, so the
                   button cannot be dead even if a script further down fails. -->
              <label class="btn sm" for="picfile" id="pickpic" tabindex="0"
                     role="button">Upload picture</label>
              <button class="btn ghost sm" id="rmpic" ${pic ? '' : 'disabled'}>Remove</button>
              <input type="file" id="picfile" class="visually-hidden"
                     accept="image/png,image/jpeg,image/webp,image/gif,image/*">
            </div>
            <p class="dimmer" style="font-size:11.5px;margin:10px 0 0;line-height:1.55">
              The picture is resized to 256px in your browser before it is sent,
              so only a few kilobytes are stored on your account.
            </p>
            <div class="msg" id="pmsg"></div>
          </div>
        </div>
      </div>

      <div class="grid g2">
      <div class="card">
        <h3>Account</h3>
        <div class="kv">
          <span class="k">Name</span><span class="v">${esc(u.name)}</span>
          <span class="k">Email</span><span class="v">${esc(u.email)}</span>
          <span class="k">Organisation</span><span class="v">${esc(u.org || '—')}</span>
        </div>
      </div>
      <div class="card">
        <h3>Privacy controls</h3>
        <p class="dim" style="font-size:13.5px;margin-top:0">
          Your password is held only as a salted PBKDF2-HMAC-SHA256 hash
          (200,000 rounds). The session token is a signed JWT that expires after
          8 hours. Detections are visible only to this account.</p>
        <div style="display:flex;gap:9px;flex-wrap:wrap">
          <button class="btn danger sm" id="purge">Erase all my detections</button>
          <button class="btn ghost sm" id="out2">Sign out</button>
        </div>
        <div class="msg" id="amsg"></div>
      </div>
      </div>`;
    /* --- profile picture ------------------------------------------- */
    const pfile = $('#picfile'), ppic = $('#ppic');
    const pmsg = (cls, text) => { $('#pmsg').className = 'msg ' + cls;
                                  $('#pmsg').textContent = text; };

    /* A server started before this endpoint existed answers the POST with a
       bare 404, which reads as "upload is broken". Ask it what it can do
       first and say so plainly instead. */
    API.health().then(h => {
      /* A server old enough to lack the endpoint is also old enough to send no
         feature list at all, so a missing list counts as stale, not as fine. */
      const feats = (h && h.features) || [];
      if (feats.indexOf('avatar') === -1) throw 0;
    }).catch(() => {
      const lbl = $('#pickpic');
      if (!lbl) return;
      lbl.classList.add('disabled');
      lbl.setAttribute('for', '');
      pmsg('err', 'The backend running right now predates profile pictures. '
                + 'Stop it (Ctrl+C) and start it again, then reload this page.');
    });

    /* Keyboard users get the same affordance as a real button. */
    $('#pickpic').addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pfile.click(); }
    });

    pfile.onchange = () => {
      const f = pfile.files[0];
      if (!f) return;
      if (f.size > 30 * 1024 * 1024)
        return pmsg('err', 'That image is over 30 MB — pick a smaller one.');
      if (f.type && !/^image\//.test(f.type))
        return pmsg('err', 'Choose an image file (JPG, PNG or WebP).');
      pmsg('ok', 'Preparing picture…');
      const img = new Image();
      img.onload = () => {
        /* Resize in the browser: a phone photo is several megabytes and the
           server caps the upload, so shrinking here is what makes it work at
           all rather than failing with a size error. */
        const S = 256, c = document.createElement('canvas');
        c.width = c.height = S;
        const g = c.getContext('2d');
        const side = Math.min(img.width, img.height);
        g.drawImage(img, (img.width - side) / 2, (img.height - side) / 2,
                    side, side, 0, 0, S, S);
        const uri = c.toDataURL('image/jpeg', 0.82);
        API.setAvatar(uri).then(() => {
          u.avatar = uri;
          API.setSession(API.token(), Object.assign(API.user() || {}, { avatar: uri }));
          try { localStorage.setItem('nd-avatar', uri); } catch (e) {}
          ppic.innerHTML = '<img src="' + uri + '" alt="Profile picture">';
          $('#rmpic').disabled = false;
          paintAvatar(uri);
          pmsg('ok', 'Profile picture updated.');
        }).catch(e => pmsg('err', e.status === 404
            ? 'The backend running right now predates profile pictures. '
              + 'Stop it (Ctrl+C) and start it again, then reload this page.'
            : e.message));
        URL.revokeObjectURL(img.src);
        pfile.value = '';          // so re-picking the same file fires onchange
      };
      img.onerror = () => {
        URL.revokeObjectURL(img.src);
        pfile.value = '';
        pmsg('err', 'The browser could not decode that file. iPhone HEIC photos '
                  + 'are not supported — save it as JPG or PNG first.');
      };
      img.src = URL.createObjectURL(f);
    };
    $('#rmpic').onclick = () => {
      API.setAvatar(null).then(() => {
        u.avatar = null;
        API.setSession(API.token(), Object.assign(API.user() || {}, { avatar: null }));
        try { localStorage.removeItem('nd-avatar'); } catch (e) {}
        ppic.innerHTML = '<span>' + esc(initials(u.name || u.email || 'AN')) + '</span>';
        $('#rmpic').disabled = true;
        paintAvatar(null);
        pmsg('ok', 'Profile picture removed.');
      }).catch(e => pmsg('err', e.message));
    };

    $('#out2').onclick = () => { API.clearSession(); location.replace('/login'); };
    $('#purge').onclick = async () => {
      if (!confirm('Permanently delete every detection on this account?')) return;
      const r = await API.purge();
      $('#amsg').className = 'msg ok';
      $('#amsg').textContent = `Deleted ${r.deleted} record(s).`;
    };
  }

  // ------------------------------------------------------------------ //
  /* No token yet (someone just opened the public link)? Start a demo session
     so the dashboard is the first thing they see, not a sign-in form. */
  const session = API.token()
    ? Promise.resolve()
    : API.demo().then(d => API.setSession(d.token, d.user));

  session.then(() => API.me()).then(r => {
    const label = r.user.name || r.user.email;
    $('#who').textContent = label;
    const av = $('#avatar');
    if (av) av.textContent = initials(label);
    /* The server row is the source of truth: the JWT was issued before any
       picture existed, so a fresh browser would otherwise show initials
       forever. localStorage is only a cache to avoid a flash on reload. */
    API.setSession(API.token(), r.user);
    const uri = r.user.avatar || null;
    try {
      uri ? localStorage.setItem('nd-avatar', uri)
          : localStorage.removeItem('nd-avatar');
    } catch (e) {}
    paintAvatar(uri);
    const clk = $('#clock');
    if (clk) {
      const tick = () => clk.textContent = new Date().toLocaleString();
      tick(); setInterval(tick, 30000);
    }
    go(location.hash.slice(1) || 'ops');
  }).catch(() => { API.clearSession(); location.replace('/login'); });
})();
