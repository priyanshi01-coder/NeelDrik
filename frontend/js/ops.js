/* ==========================================================================
   NEELDRIK — Operations console
   --------------------------------------------------------------------------
   The layout from the design: map in the centre, the detected-spill summary
   top right, drift forecast, priority alerts and nearby vessels down the
   right, and an imagery strip along the bottom.

   HONESTY RULE FOR THIS FILE
   Panels fall into two kinds and they are never mixed:

     LIVE   — driven by /api/dashboard and /api/detections. The spill summary,
              the slick polygon, the SAR thumbnail and the confidence all come
              from a real analysis this system performed.

     DEMO   — AIS vessel tracks, vessel identities, optical imagery and the
              alert feed. The backend has none of these. They render with a
              visible DEMO badge and placeholder identifiers, never as
              plausible-looking real vessel records.

   If a panel ever gets a real data source, move it out of DEMO_PANELS and
   drop the badge. Do not remove the badge before the data is real.
   ========================================================================== */
(function () {
  'use strict';

  const esc = (s) => String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');

  const DEMO = '<span class="demotag" title="Illustrative only — this panel ' +
               'has no live data source yet">DEMO</span>';



  /* small leading glyphs, as in the reference design */
  const IC = {
    alert:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M12 9v4M12 17h.01"/><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/></svg>',
    wave :'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M2 16c3-4 6 3 10 0s6-4 10 0"/><path d="M2 9c3-4 6 3 10 0s6-4 10 0"/></svg>',
    bell :'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M18 8a6 6 0 1 0-12 0c0 7-3 8-3 8h18s-3-1-3-8"/><path d="M13.7 21a2 2 0 0 1-3.4 0"/></svg>',
    pin  :'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 10c0 6-9 12-9 12s-9-6-9-12a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>',
    area :'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 3v18"/></svg>',
    drop :'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2.7 6.7 9a7.5 7.5 0 1 0 10.6 0z"/></svg>',
    gauge:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 14l4-4"/><path d="M4 18a9 9 0 1 1 16 0"/></svg>',
    clock:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>',
    wind :'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M3 8h11a3 3 0 1 0-3-3"/><path d="M3 16h14a3 3 0 1 1-3 3"/></svg>',
    grid :'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>'
  };
  const K = (ico, label) => '<span class="kk">' + IC[ico] + esc(label) + '</span>';

  function fmtTime(t) {
    return t ? new Date(t * 1000).toLocaleString() : '—';
  }

  /* ---------------------------------------------------------------- */
  window.renderOps = async function renderOps(ctx) {
    const { view, API, pillClass, pillText, go } = ctx;

    const d = await API.dashboard();
    const latestRow = (d.recent || [])[0] || null;
    let det = null;
    if (latestRow) {
      try { det = await API.detection(latestRow.id); } catch (e) { det = null; }
    }
    const p = det && det.payload ? det.payload : (det || {});
    // the detail endpoint returns the payload flat and omits `created`;
    // the dashboard row is what carries the timestamp
    const created = latestRow ? latestRow.created : null;

    const hasSpill = !!(p && p.verdict);
    const conf = p.confidence != null ? (p.confidence * 100).toFixed(0) + '%' : '—';
    const area = p.estimated_area_km2 != null
      ? p.estimated_area_km2.toFixed(1) + ' km²' : '—';

    /* Estimated volume is DERIVED, not measured. SAR cannot see film
       thickness, so this assumes a 1 um mean film at 0.85 t/m3 and is only an
       order-of-magnitude figure. The tooltip on the row says so. */
    const volume = p.estimated_area_km2 != null
      ? '~' + Math.round(p.estimated_area_km2 * 1e6 * 1e-6 * 0.85) + ' t'
      : '—';

    view.innerHTML = `
      <div class="ops">

       <div class="ops-left">
        <section class="card ops-map">
          <div class="ops-maphead">
            <div class="ops-maptitle">Detected slick and back-tracked origin</div>
          </div>
          <div class="mapwrap">
            <div id="opsmap"></div>
            ${hasSpill ? `
            <div class="callout c-spill">
              <b>Detected spill</b>
              <span>Area: ${area}</span>
              <span>Confidence: ${conf}</span>
              <span>${fmtTime(created)}</span>
            </div>` : ''}
            <div class="maplegend">
              <span><i class="sq" style="background:var(--c-spill)"></i>Oil spill (detected)</span>
              <span><i class="dash"></i>Back-tracked drift path</span>
            </div>
          </div>
        </section>


        <section class="ops-thumbs">
          <figure class="thumb">
            <div class="thumbimg">${(p.scene_thumb || p.scene_png)
              ? `<img src="${p.scene_thumb || p.scene_png}" alt="SAR scene">`
              : '<span>Raw scene not retained</span>'}
              <span class="stamp">${fmtTime(created)}</span><span class="scalebar">0 — 10 km</span></div>
            <figcaption>SAR Image (Enhanced) <b class="live">live</b></figcaption>
          </figure>
          <figure class="thumb">
            <div class="thumbimg">${(p.overlay_thumb || p.overlay_png)
              ? `<img src="${p.overlay_thumb || p.overlay_png}" alt="Detection overlay">`
              : '<span>No overlay</span>'}
              <span class="stamp">${fmtTime(created)}</span><span class="scalebar">0 — 10 km</span></div>
            <figcaption>Spill Trajectory Model <b class="live">live</b></figcaption>
          </figure>
        </section>
       </div>

                <aside class="ops-side">

          <div class="card spillcard ${hasSpill ? '' : 'idle'}">
            <div class="cardtop">
              <h3 class="ttl"><i class="ic red">${IC.alert}</i>${
                hasSpill ? 'Spill Detected' : 'No Active Spill'}</h3>
              ${hasSpill ? `<span class="pill ${pillClass(p.verdict)}">${pillText(p.verdict)}</span>` : ''}
            </div>
            ${hasSpill ? `
              <div class="kv">
                <span class="k">${K('grid','Scene')}</span><span class="v">${esc(p.filename || '—')}</span>
                <span class="k">${K('pin','Location')}</span><span class="v">${
                  p.scene_center
                    ? p.scene_center.lat.toFixed(4) + ', ' + p.scene_center.lon.toFixed(4)
                    : 'not supplied'}</span>
                <span class="k">${K('area','Area')}</span><span class="v">${area}</span>
                <span class="k">${K('drop','Est. volume')}<i class="qmark" title="Derived, not measured: area x 1 um mean film thickness x 0.85 t/m3. Film thickness is not observable from SAR, so treat this as an order-of-magnitude figure.">?</i></span><span class="v">${volume}</span>
                <span class="k">${K('gauge','Oil coverage')}</span><span class="v">${p.oil_area_frac != null ? (p.oil_area_frac * 100).toFixed(2) + '%' : '—'}</span>
                <span class="k">${K('grid','Regions')}</span><span class="v">${p.candidate_count != null ? p.candidate_count : '—'}</span>
                <span class="k">${K('wind','Wind')}</span><span class="v">${p.wind_ms != null ? p.wind_ms + ' m/s' : 'not supplied'}</span>
                <span class="k">${K('clock','Detected')}</span><span class="v">${fmtTime(created)}</span>
              </div>
              <div class="confbar">
                <div class="confbar-top"><span>Confidence</span><b>${conf}</b></div>
                <div class="bar"><i style="width:${p.confidence != null ? p.confidence * 100 : 0}%;
                   background:var(--c-accent)"></i></div>
              </div>
              ${(p.reasons || []).length ? `<ul class="reasonlist">
                 ${p.reasons.map(r => `<li>${esc(r)}</li>`).join('')}</ul>` : ''}
            ` : `<div class="empty" style="padding:26px 12px">
                   Nothing analysed yet.<br>
                   <button class="btn sm" id="ops-first" style="margin-top:12px">Analyse a scene</button>
                 </div>`}
          </div>

          <div class="card">
            <div class="cardtop"><h3 class="ttl"><i class="ic">${IC.wave}</i>Back-tracked origin</h3></div>
            <div id="ops-origin" class="dim" style="font-size:13px;line-height:1.7">
              ${hasSpill && p.scene_center
                ? 'Running the back-track…'
                : 'The back-track needs a slick position. ' +
                  (hasSpill ? 'This scene was analysed without one.'
                            : 'Analyse a scene to begin.')}
            </div>
            <p class="dim" style="font-size:12.5px;line-height:1.6;margin:12px 0 0">
              Lagrangian transport — current plus 3% of wind, Coriolis-deflected,
              RK4 integrated. Open <a data-goto="drift">Drift &amp; origin</a> to
              set the met-ocean conditions and run a full back-track.
            </p>
          </div>

        </aside>

      </div>`;

    const first = document.getElementById('ops-first');
    if (first) first.onclick = () => go('detect');
    view.querySelectorAll('[data-goto]').forEach(a => {
      a.style.cursor = 'pointer';
      a.onclick = () => go(a.dataset.goto);
    });
    drawMap(p);
  };

  /* ---------------------------------------------------------------- */
  /* One place where the map says why it cannot show something.
     The note goes in a strip of its own and the legend and the spill callout
     are hidden with it -- they describe symbols that are not on screen, and
     they used to sit on top of the note. */
  function mapNotice(el, html, overlay) {
    const wrap = el.closest('.mapwrap');
    if (overlay) {
      el.classList.add('nomap');
      el.innerHTML = '<img src="' + overlay + '" alt="Detection overlay">';
    }
    if (!wrap) return;
    wrap.classList.add('unlocated');
    wrap.querySelectorAll('.mapnotice').forEach(n => n.remove());
    wrap.insertAdjacentHTML('beforeend', '<div class="mapnotice">' + html + '</div>');
  }

  function drawMap(p) {
    const el = document.getElementById('opsmap');
    if (!el) return;
    const hasSpill = !!(p && p.verdict);
    if (typeof L === 'undefined') {
      const img = p && (p.overlay_png || p.overlay_thumb);
      mapNotice(el, 'Offline \u2014 the tile map needs a network connection.' +
                    (img ? ' Showing the detection overlay instead.' : ''), img);
      return;
    }
    /* WHERE THIS MAP IS CENTRED
       It used to be hard-coded to 20.35 N, 70.15 E off Gujarat, so every scene
       -- and even an empty account -- drew a red circle in the same place. The
       centre now comes from the scene itself: the GeoTIFF's CRS, or the centre
       coordinates the analyst entered on upload.

       When a scene has NO position the map is still drawn -- an operations
       console without a map is useless -- but it opens on India's EEZ at a
       wide zoom with NOTHING plotted on it, and a banner says why. The rule is
       "never draw a slick where we do not know one is": showing the sea is
       fine, inventing a marker on it is not. */
    const sc = p && p.scene_center;
    const geo = p && p.geojson && p.geojson.features && p.geojson.features.length
      && p.geojson.crs == null                 // local 0..1 grid is not degrees
      ? p.geojson : null;

    const centre = sc ? [sc.lat, sc.lon] : [15.5, 73.0];   // India's EEZ
    const map = L.map(el, { zoomControl: true }).setView(centre, sc ? 9 : 5);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 18, attribution: '&copy; OpenStreetMap'
    }).addTo(map);
    setTimeout(() => map.invalidateSize(), 120);

    if (!sc) {
      /* Nothing is plotted, so the legend would describe symbols that are not
         on the map. Hide it and put the reason in its place -- that is what was
         overlapping the legend before. */
      mapNotice(el, hasSpill
        ? 'This scene has no position, so nothing is plotted here. Pick a located '
          + 'sample, or re-analyse with the scene centre latitude and longitude.'
        : 'No located scene yet. Analyse a scene that carries its coordinates and '
          + 'the slick and its back-tracked origin appear here.');
      return;                                  // a map, but nothing invented on it
    }

    if (geo) {
      const layer = L.geoJSON(geo, {
        style: { color: '#E11D48', weight: 2, fillColor: '#E11D48', fillOpacity: .42 }
      }).addTo(map);
      try { map.fitBounds(layer.getBounds().pad(0.6)); } catch (e) { /* unbounded */ }
    } else if (p.scene_bounds) {
      // no slick polygon (clean / look-alike scene) -- show the footprint only
      const bb = p.scene_bounds;
      L.rectangle([[bb[1], bb[0]], [bb[3], bb[2]]], {
        color: '#64748B', weight: 1, dashArray: '5 5', fillOpacity: .05
      }).addTo(map).bindPopup('Scene footprint');
    }

    const c = L.latLng(centre[0], centre[1]);

    /* The back-track runs through the SAME /api/drift endpoint the Drift &
       Origin page uses, from the real slick position. Wind and current are
       not in a SAR image, so the values below are a stated reference case --
       the panel says so. If the call fails, nothing fake is drawn. */
    const REF = { wind_speed: 6, wind_dir: 225, cur_speed: 0.3, cur_dir: 250 };
    API.drift(Object.assign({ lat: c.lat, lon: c.lng, hours: [6, 12, 24] }, REF))
      .then(res => {
        const gj = res && res.geojson;
        if (!gj || !gj.features) return;

        gj.features.forEach(f => {
          const k = f.properties && f.properties.kind;
          if (k === 'backtrack') {
            const pts = f.geometry.coordinates.map(xy => [xy[1], xy[0]]);
            L.polyline(pts, { color: '#0891B2', weight: 3, dashArray: '7 6' })
              .addTo(map).bindPopup('Back-tracked path');
          } else if (k === 'origin') {
            const xy = f.geometry.coordinates;
            const pr = f.properties || {};
            L.circleMarker([xy[1], xy[0]], {
              radius: 6, color: '#F59E0B', weight: 2,
              fillColor: '#F59E0B', fillOpacity: .85
            }).addTo(map).bindPopup(
              'Likely origin ' + (pr.hours_before || '?') + ' h before<br>' +
              (pr.distance_km != null ? pr.distance_km.toFixed(1) + ' km away<br>' : '') +
              (pr.uncertainty_m != null
                ? '&plusmn;' + Math.round(pr.uncertainty_m) + ' m uncertainty' : ''));
            L.circle([xy[1], xy[0]], {
              radius: pr.uncertainty_m || 500, color: '#F59E0B',
              weight: 1, fillColor: '#F59E0B', fillOpacity: .10
            }).addTo(map);
          }
        });

        const note = document.querySelector('.ops-map .maplegend');
        if (note) note.insertAdjacentHTML('beforeend',
          '<span><i class="sq" style="background:#F59E0B"></i>Back-tracked origin</span>');

        const box = document.getElementById('ops-origin');
        const best = (res.candidates || [])[res.candidates.length - 1];
        if (box && best) box.innerHTML =
          '<b>' + best.hours_before.toFixed(0) + ' h before:</b> ' +
          best.lat.toFixed(4) + ', ' + best.lon.toFixed(4) + '<br>' +
          best.distance_from_slick_km.toFixed(1) + ' km from the slick, search ' +
          'radius ' + (best.uncertainty_m / 1000).toFixed(2) + ' km.' +
          '<br><span class="dimmer" style="font-size:12px">Reference conditions: ' +
          REF.wind_speed + ' m/s wind from ' + REF.wind_dir + '\u00B0, ' +
          REF.cur_speed + ' m/s current towards ' + REF.cur_dir + '\u00B0. ' +
          'Enter the real met-ocean values on Drift &amp; origin.</span>';
      })
      .catch(() => {
        const box = document.getElementById('ops-origin');
        if (box) box.textContent = 'Back-track unavailable for this scene.';
      });

    setTimeout(() => map.invalidateSize(), 120);
  }
})();
