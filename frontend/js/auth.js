/* Sign-in / registration screen. */
(() => {
  const $ = (id) => document.getElementById(id);
  let mode = 'in';

  if (API.token()) location.replace('/app');

  function setMode(m) {
    mode = m;
    $('tab-in').classList.toggle('on', m === 'in');
    $('tab-up').classList.toggle('on', m === 'up');
    document.querySelectorAll('.only-up')
      .forEach(el => el.classList.toggle('hidden', m !== 'up'));
    $('ttl').textContent = m === 'in' ? 'Sign in' : 'Create account';
    $('sub').textContent = m === 'in'
      ? 'Access the detection console.'
      : 'Your detections stay private to this account.';
    $('go').textContent = m === 'in' ? 'Sign in' : 'Create account';
    $('password').setAttribute('autocomplete',
      m === 'in' ? 'current-password' : 'new-password');
    msg('');
  }

  function msg(text, kind) {
    const el = $('msg');
    el.className = 'msg' + (text ? ' ' + (kind || 'err') : '');
    el.textContent = text || '';
  }

  $('tab-in').onclick = () => setMode('in');
  $('tab-up').onclick = () => setMode('up');

  $('form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = $('go');
    btn.disabled = true;
    const label = btn.textContent;
    btn.innerHTML = '<span class="spinner"></span>';
    msg('');
    try {
      const payload = {
        email: $('email').value.trim(),
        password: $('password').value
      };
      let out;
      if (mode === 'up') {
        payload.name = $('name').value.trim();
        payload.org = $('org').value.trim();
        out = await API.register(payload);
      } else {
        out = await API.login(payload);
      }
      API.setSession(out.token, out.user);
      msg('Signed in. Opening console…', 'ok');
      setTimeout(() => location.replace('/app'), 250);
    } catch (err) {
      msg(err.message);
      btn.disabled = false;
      btn.textContent = label;
    }
  });

  /* model facts on the left panel */
  API.model().then(m => {
    /* The label says "spill precision", so show precision -- and show the
       REAL-SAR one when it exists. The synthetic figure is several points
       higher, and putting it under this label would overstate the system. */
    const rh = m.real_holdout;
    const p = rh ? rh.precision : m.precision;
    if (p != null) {
      $('f-acc').textContent = (p * 100).toFixed(1) + '%';
      const lbl = $('f-acc').nextElementSibling;
      if (lbl) lbl.textContent = rh ? 'spill precision (real SAR)'
                                    : 'spill precision (synthetic)';
    }
    if (m.parameters) $('f-par').textContent = (m.parameters / 1000).toFixed(0) + 'K';
    $('f-rt').textContent = m.runtime === 'pytorch' ? 'PyTorch' : 'NumPy';
  }).catch(() => {});

  /* animated swell — decorative, pure canvas so it needs no network */
  const cv = document.getElementById('waves');
  if (cv) {
    const ctx = cv.getContext('2d');
    let t = 0, raf;
    function resize() {
      cv.width = cv.offsetWidth * devicePixelRatio;
      cv.height = cv.offsetHeight * devicePixelRatio;
    }
    function draw() {
      const w = cv.width, h = cv.height;
      ctx.clearRect(0, 0, w, h);
      for (let k = 0; k < 5; k++) {
        ctx.beginPath();
        const base = h * (0.42 + k * 0.11);
        const amp = h * 0.035 * (1 + k * 0.22);
        ctx.moveTo(0, base);
        for (let x = 0; x <= w; x += 8) {
          const y = base + Math.sin((x / w) * 6.5 + t * (0.5 + k * 0.12) + k) * amp
                         + Math.sin((x / w) * 13 + t * 0.7) * amp * 0.28;
          ctx.lineTo(x, y);
        }
        ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
        ctx.fillStyle = `rgba(46,127,158,${0.05 + k * 0.022})`;
        ctx.fill();
      }
      t += 0.012;
      raf = requestAnimationFrame(draw);
    }
    resize(); draw();
    addEventListener('resize', resize);
    addEventListener('pagehide', () => cancelAnimationFrame(raf));
  }

  setMode('in');
})();
