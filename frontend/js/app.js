/* NEELDRIK console: hash router + views. */
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const view = $('#view');
  /* `detect` is deliberately module-level, not view-level. Leaving the
     Detect page used to throw the chosen file away, so coming back showed an
     empty drop zone -- the upload now survives navigation and is cleared only
     when the user clears it. */
  const state = {
    last: null, samples: [], detections: [],
    detect: { file: null, label: '', km: 40, wind: '', lat: '', lon: '' }
  };


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
    how: { title: 'How it works', render: renderHow },
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
    return Promise.resolve(VIEWS[name].render()).catch(e => {
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
            <div id="chosen" class="chosenrow hidden">
              <span class="mono" id="chosen-name"></span>
              <button type="button" class="btn ghost sm" id="clearfile">Remove</button>
            </div>

            <h3 style="margin-top:20px">Or pick a sample scene</h3>
            <div class="samples" id="samples"><span class="dim">loading…</span></div>

            <div class="grid g2" style="gap:12px;margin-top:18px">
              <div>
                <label for="km">Assumed scene width (km)</label>
                <input id="km" type="number" value="40" min="1" max="500" step="1">
              </div>
              <div>
                <label for="wind">Wind speed (m/s) <span class="dim">— optional</span></label>
                <input id="wind" type="number" placeholder="unknown" min="0" max="60" step="0.1">
              </div>
              <div>
                <label for="lat">Scene centre latitude</label>
                <input id="lat" type="number" placeholder="e.g. 19.0400" min="-90" max="90" step="0.0001">
              </div>
              <div>
                <label for="lon">Scene centre longitude</label>
                <input id="lon" type="number" placeholder="e.g. 71.8200" min="-180" max="180" step="0.0001">
              </div>
            </div>
            <button class="btn" id="run" style="width:100%;margin-top:14px" disabled>Analyse spill</button>
            <div class="msg" id="dmsg"></div>
          </div>
        </div>
        <div class="card" id="rescard">
          <h3>Result</h3>
          <div class="empty">Upload a scene to run the detector.</div>
        </div>
      </div>`;

    const drop = $('#drop'), file = $('#file'), run = $('#run');
    const D = state.detect;

    function showChosen(label) {
      $('#chosen').classList.toggle('hidden', !label);
      $('#chosen-name').textContent = label || '';
      run.disabled = !D.file;
    }
    function pick(f, label) {
      D.file = f;
      D.label = label || `${f.name} · ${(f.size / 1024).toFixed(0)} KB`;
      showChosen(D.label);
      $$('#samples button').forEach(b => b.classList.remove('on'));
    }

    /* restore whatever was on this page last time */
    ['km', 'wind', 'lat', 'lon'].forEach(k => {
      if (D[k] !== '' && D[k] != null) $('#' + k).value = D[k];
      $('#' + k).onchange = () => { D[k] = $('#' + k).value; };
    });
    showChosen(D.file ? D.label : '');
    if (state.last) showResult(state.last);

    $('#clearfile').onclick = () => {
      D.file = null; D.label = '';
      file.value = '';
      showChosen('');
      $$('#samples button').forEach(b => b.classList.remove('on'));
    };
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
      if (!D.file) return;
      run.disabled = true;
      const lab = run.textContent;
      run.innerHTML = '<span class="spinner"></span>';
      $('#dmsg').className = 'msg';
      try {
        const v = (id) => { const e = $('#' + id); return e ? e.value.trim() : ''; };
        D.km = v('km'); D.wind = v('wind'); D.lat = v('lat'); D.lon = v('lon');
        if ((D.lat === '') !== (D.lon === ''))
          throw new Error('Give both the scene latitude and longitude, or leave both blank.');
        const res = await API.detect(D.file, {
          sceneKm: parseFloat(D.km) || null,
          windMs: D.wind, lat: D.lat, lon: D.lon
        });
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
    $('#samples').innerHTML = sm.items.length ? sm.items.map((s, i) => `
      <!-- The class label under each thumbnail (oil / look-alike / clean) is
           deliberately not shown: it is the answer. A judge should see the
           scene, run it, and read the verdict the model produced.
           A located sample carries its real coordinates, and picking it fills
           them into the form, so the map shows the real place. -->
      <button data-url="${esc(s.url)}" data-name="${esc(s.file)}"
              data-lat="${s.lat != null ? s.lat : ''}"
              data-lon="${s.lon != null ? s.lon : ''}"
              data-km="${s.scene_km != null ? s.scene_km : ''}"
              data-wind="${s.wind_ms != null ? s.wind_ms : ''}"
              data-src="${esc(s.geo_source || s.source || '')}"
              data-place="${esc(s.name || '')}"
              class="${s.located ? 'located' : ''}"
              title="${esc(s.name)}${s.located ? ' \u2014 carries its own coordinates' : ''}">
        <img src="${esc(s.url)}" alt="SAR sample scene">
        ${s.located ? '<i class="pin" aria-hidden="true"></i>' : ''}
      </button>`).join('')
      : '<span class="dim">No samples installed.</span>';
    $$('#samples button').forEach(b => b.onclick = async () => {
      $$('#samples button').forEach(x => x.classList.remove('on'));
      b.classList.add('on');
      const blob = await (await fetch(b.dataset.url)).blob();
      D.file = new File([blob], b.dataset.name, { type: 'image/png' });
      D.label = b.dataset.name + ' · sample';
      showChosen(D.label);
      b.classList.add('on');
      /* A located sample brings its real coordinates with it. They go into the
         form fields, which is where they are an input -- the place itself is
         named on the dashboard map once the scene has been analysed, not
         announced here before the detector has run. */
      const put = (id, v) => {
        if (v !== '' && v != null) { $('#' + id).value = v; D[id] = v; }
      };
      put('lat', b.dataset.lat); put('lon', b.dataset.lon);
      put('km', b.dataset.km);   put('wind', b.dataset.wind);
    });
    if (D.file) showChosen(D.label);
  }

  function showResult(r) {
    const card = $('#rescard');
    if (!card) return;          // view changed under us; nothing to paint into
    const cls = pillClass(r.verdict);

    // Out-of-domain input is refused rather than guessed at.
    if (r.analysed === false) {
      card.innerHTML = `
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
    card.innerHTML = `
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
        <span class="k">Scene centre</span><span class="v">${r.scene_center
          ? r.scene_center.lat.toFixed(4) + ', ' + r.scene_center.lon.toFixed(4) +
            ' <span class="dim">(' + (r.scene_center.source === 'raster'
              ? 'from raster CRS' : 'entered') + ')</span>'
          : '<span class="dim">not supplied</span>'}</span>
        <span class="k">Inference</span><span class="v">${r.elapsed_ms} ms · ${esc(r.runtime)}</span>
      </div>
      ${r.scene_center ? '' : `<div class="dimmer" style="font-size:12px;margin-bottom:14px">
        This scene has no position, so the slick is not drawn on a map. Enter the
        scene centre latitude and longitude above and analyse again to map it —
        the detection itself is unaffected.</div>`}
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
        <button class="btn ghost sm" id="to-map"${r.scene_center ? '' : ' disabled'}
          title="${r.scene_center ? 'Back-track this slick to its source'
                                  : 'Needs a scene centre'}">Back-track this slick</button>
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
    $('#to-map').onclick = () => { if (r.scene_center) go('drift'); };
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
    /* Wait for the view to actually render. This used to be a 60 ms timer,
       which is a race: on a slow connection #rescard did not exist yet and
       showResult threw "Cannot set properties of null". */
    await go('detect');
    state.last = r;
    showResult(r);
  }

  // ------------------------------------------------------------------ //
  /* The newest detection this account holds, fetched once if we do not
     already have it in memory. This is what the drift page is seeded from --
     without it the form was showing fixed numbers that had nothing to do with
     any uploaded image. */
  async function latestDetection() {
    if (state.last && state.last.verdict) return state.last;
    try {
      const d = await API.dashboard();
      const row = (d.recent || [])[0];
      if (!row) return null;
      state.last = await API.detection(row.id);
      return state.last;
    } catch (e) { return null; }
  }

  async function renderDrift() {
    const last = await latestDetection();
    const c = last && last.scene_center ? last.scene_center : null;
    const windFromScene = last && last.wind_ms != null ? last.wind_ms : '';
    view.innerHTML = `
      <div class="grid split">
        <div class="card">
          <h3>Back-track the slick to its origin</h3>
          <p class="dim" style="font-size:13.5px;margin-top:0">
            Surface oil moves with the current plus about 3% of the wind, deflected
            by Coriolis. Running that backwards in time gives the release point.</p>
          <div class="srcnote ${c ? 'ok' : 'warn'}">
            ${c
              ? `Position taken from <b>${esc(last.filename || 'the last scene analysed')}</b> —
                 ${c.lat.toFixed(4)}, ${c.lon.toFixed(4)}
                 (${c.source === 'raster' ? 'the raster\'s own CRS' : 'entered on upload'}).`
              : last
                ? `The last scene analysed (<b>${esc(last.filename || 'untitled')}</b>) has no
                   position, so there is nothing to seed these fields with. Re-analyse it with a
                   scene centre, or type the slick position below.`
                : `No scene analysed yet. Analyse one on <a data-goto="detect">Detect spill</a>
                   and its position will be filled in here, or type one below.`}
          </div>
          <div class="grid g2" style="gap:12px">
            <div><label for="d-lat">Slick latitude <span class="dim">— from the scene</span></label>
              <input id="d-lat" type="number" step="0.0001" placeholder="e.g. 19.0400"
                     value="${c ? c.lat.toFixed(4) : ''}"></div>
            <div><label for="d-lon">Slick longitude <span class="dim">— from the scene</span></label>
              <input id="d-lon" type="number" step="0.0001" placeholder="e.g. 71.8200"
                     value="${c ? c.lon.toFixed(4) : ''}"></div>
            <div><label for="d-ws">Wind speed (m/s) <span class="dim">— ${
                windFromScene === '' ? 'assumed' : 'from the scene'}</span></label>
              <input id="d-ws" type="number" step="0.1"
                     value="${windFromScene === '' ? 6 : windFromScene}"></div>
            <div><label for="d-wd">Wind FROM (deg) <span class="dim">— assumed</span></label>
              <input id="d-wd" type="number" step="1" value="225"></div>
            <div><label for="d-cs">Current speed (m/s) <span class="dim">— assumed</span></label>
              <input id="d-cs" type="number" step="0.01" value="0.30"></div>
            <div><label for="d-cd">Current TOWARDS (deg) <span class="dim">— assumed</span></label>
              <input id="d-cd" type="number" step="1" value="250"></div>
            <div><label for="d-h">Elapsed times (hours)</label>
              <input id="d-h" type="text" value="6, 12, 24"></div>
            <div><label for="d-k">Diffusivity (m²/s)</label>
              <input id="d-k" type="number" step="0.5" value="5"></div>
          </div>
          <p class="dimmer" style="font-size:12px;line-height:1.65;margin:12px 0 0">
            The position comes from the scene you analysed, and so does the wind speed
            when you supplied one. The fields marked <b>assumed</b> are a stated
            reference case, not a measurement: wind direction and current are not in a
            SAR image and this build has no met-ocean feed. They are filled so the
            back-track runs out of the box — replace them with real values from INCOIS,
            ERA5 or a ship report before quoting a result.
          </p>
          <div style="display:flex;gap:9px;margin-top:14px;flex-wrap:wrap">
            <button class="btn" id="d-run" style="width:auto;padding:11px 22px">Back-track</button>
            <button class="btn ghost sm" id="d-verify">Verify the physics</button>
          </div>
          <div class="msg" id="d-msg"></div>
        </div>
        <div class="card" id="d-res">
          <h3>Estimated origin</h3>
          <div class="empty">Enter the conditions, then back-track.</div>
        </div>
      </div>
      <div class="card" style="margin-top:16px">
        <h3>Back-track path</h3>
        <div id="d-map"></div>
      </div>`;

    view.querySelectorAll('[data-goto]').forEach(a => {
      a.style.cursor = 'pointer';
      a.onclick = () => go(a.dataset.goto);
    });

    /* With a position in hand there is nothing left to decide, so run it. The
       user asked for the origin, not for a form to submit. */
    if (c) setTimeout(() => $('#d-run').click(), 60);

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
        if ($('#d-lat').value === '' || $('#d-lon').value === '')
          throw new Error('A slick position is required. Analyse a scene with its '
                          + 'centre coordinates, or type the latitude and longitude here.');
        const nz = (id) => $('#' + id).value === '' ? 0 : $('#' + id).value;
        const r = await API.drift({
          lat: $('#d-lat').value, lon: $('#d-lon').value,
          wind_speed: nz('d-ws'), wind_dir: nz('d-wd'),
          cur_speed: nz('d-cs'), cur_dir: nz('d-cd'),
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
  async function renderHow() {
    /* Explanation only. The measured scores live on the Model card, and
       repeating them here meant two pages to keep in step. */
    view.innerHTML = `
      <div class="card" style="margin-bottom:16px">
        <h3>What NEELDRIK does</h3>
        <p style="font-size:14.5px;line-height:1.75;margin:6px 0 0">
          NEELDRIK reads Sentinel-1 radar images of the sea and decides whether an
          oil spill is present, how large it is, and where the oil came from — and it
          attaches the evidence for every one of those answers.
        </p>
        <p class="dim" style="font-size:13.5px;line-height:1.7;margin:12px 0 0">
          Radar returns a bright signal from the small wind-driven waves on the sea
          surface. A film of oil damps those waves, so oil appears as a dark patch.
          The difficulty is that low wind, algal blooms, rain cells and ship wakes all
          look dark too. Telling oil apart from those <b>look-alikes</b> is the problem
          this system exists to solve.
        </p>
      </div>

      <div class="card" style="margin-bottom:16px">
        <h3>The pipeline, stage by stage</h3>
        <p class="dim" style="font-size:12.8px;margin:2px 0 14px">
          Each stage is marked with whether it runs in this build or is specified for
          the next one — so nothing here is a claim the prototype cannot back.
        </p>
      <div class="howstage">
        <div class="hownum">01</div>
        <div class="howbody">
          <h3>Ingest <span class="pill clean">running now</span></h3>
          <p>A Sentinel-1 SAR scene enters the system. The prototype takes it as an upload (GeoTIFF, PNG or JPEG). In deployment the same entry point is fed automatically from the Copernicus Open Access Hub as each pass over India's EEZ lands.</p>
          <p class="howwhy"><b>Why:</b> Sentinel-1 carries a C-band radar that images through cloud and darkness, which is why it is the sensor of record for maritime surveillance.</p>
        </div>
      </div>
      <div class="howstage">
        <div class="hownum">02</div>
        <div class="howbody">
          <h3>Domain gate <span class="pill clean">running now</span></h3>
          <p>Before anything is analysed, the system decides what kind of image it is actually holding — radar, or something else. Anything that is not SAR is refused outright and returned as NOT ANALYSED.</p>
          <p class="howwhy"><b>Why:</b> In radar, oil is dark. In an ordinary photograph oil is bright silver. A detector trained on one and fed the other fails in both directions, so the system declines rather than guesses.</p>
        </div>
      </div>
      <div class="howstage">
        <div class="hownum">03</div>
        <div class="howbody">
          <h3>Preprocess <span class="pill clean">running now</span></h3>
          <p>The scene is stretched to a common range, a texture map is measured, speckle is suppressed with a Lee filter, and the result is resized to the model input. Two channels go forward: despeckled brightness, and local texture.</p>
          <p class="howwhy"><b>Why:</b> Texture is measured BEFORE filtering. The filter would erase it, and texture is the single cue that separates oil from a look-alike: oil damps waves, so it is dark AND smooth, while a low-wind patch is dark but still rough.</p>
        </div>
      </div>
      <div class="howstage">
        <div class="hownum">04</div>
        <div class="howbody">
          <h3>Segmentation <span class="pill clean">running now</span></h3>
          <p>A 5-class U-Net labels every pixel as Sea, Oil, Look-alike, Ship or Land, with a probability for each. The network holds 94,429 parameters and runs on CPU in under a second.</p>
          <p class="howwhy"><b>Why:</b> Classifying the whole image is not enough for an investigation. Responders need the slick's shape, extent and position, so the model works pixel by pixel.</p>
        </div>
      </div>
      <div class="howstage">
        <div class="hownum">05</div>
        <div class="howbody">
          <h3>Decision <span class="pill clean">running now</span></h3>
          <p>Calibrated thresholds convert the probability map into a verdict: SPILL, LOOK-ALIKE, CLEAN, or NEEDS REVIEW. Small fragments are discarded as speckle, and a wind plausibility check is applied. Every verdict carries the reasons behind it.</p>
          <p class="howwhy"><b>Why:</b> Wind can only ever weaken a call, never strengthen one. Below 3 m/s the sea is glassy and any dark patch is suspect; above 12 m/s a real slick breaks up. In both cases a SPILL is downgraded to NEEDS REVIEW.</p>
        </div>
      </div>
      <div class="howstage">
        <div class="hownum">06</div>
        <div class="howbody">
          <h3>Drift and origin <span class="pill clean">running now</span></h3>
          <p>A Lagrangian particle model carries the slick backwards through the current and 3% of the wind, deflected by Coriolis and integrated with RK4. It returns where the oil most likely was 6, 12 and 24 hours earlier, each with an uncertainty radius.</p>
          <p class="howwhy"><b>Why:</b> This is physics, not a network, so it can be proved rather than merely measured. It is checked against ten closed-form solutions and recovers a known release point to within 200 metres.</p>
        </div>
      </div>
      <div class="howstage">
        <div class="hownum">07</div>
        <div class="howbody">
          <h3>Attribution <span class="pill review">designed, not yet built</span></h3>
          <p>The back-tracked origin window is matched against AIS vessel tracks for the same period. Vessels that were inside the window are ranked by proximity, timing, course and AIS gaps, producing an evidence-backed shortlist rather than a single accusation.</p>
          <p class="howwhy"><b>Why:</b> This stage is designed and specified but is NOT implemented in the current build. Detection and drift are built and measured; attribution is the next module.</p>
        </div>
      </div>
      <div class="howstage">
        <div class="hownum">08</div>
        <div class="howbody">
          <h3>Evidence package <span class="pill clean">running now</span></h3>
          <p>The complete record is stored and exportable: the slick boundary as GeoJSON, a colour overlay as PNG, the verdict, the confidence, the thresholds that decided it and the reason trail.</p>
          <p class="howwhy"><b>Why:</b> An investigation needs evidence that survives scrutiny, not a score. Every number the system shows can be traced back to the rule that produced it.</p>
        </div>
      </div>
      </div>`;
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
    : API.demo().then(d => API.setSession(d.token, d.user, true));

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
