/* Bank term-deposit project: renders every number and chart from data/results.js
   (written by analysis/run_analysis.py). No framework, no network calls. */
(function () {
  'use strict';
  var R = window.RESULTS;
  if (!R) {
    document.body.insertAdjacentHTML('afterbegin',
      '<p style="padding:16px;font:16px sans-serif">The results file is missing. Run <code>python analysis/run_analysis.py</code> (see the Reproduce section).</p>');
    return;
  }

  /* ---------------------------------------------------------------- helpers */
  var NS = 'http://www.w3.org/2000/svg';
  function $(s, r) { return (r || document).querySelector(s); }
  function $$(s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); }
  function svg(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function txt(parent, x, y, s, cls, anchor) {
    var t = svg('text', { x: x, y: y, class: cls || 'lbl', 'text-anchor': anchor || 'start' }, parent);
    t.textContent = s;
    return t;
  }
  var F = {
    p0: function (v) { return (v * 100).toFixed(0) + '%'; },
    p1: function (v) { return (v * 100).toFixed(1) + '%'; },
    f1: function (v) { return v.toFixed(1); },
    f2: function (v) { return v.toFixed(2); },
    f3: function (v) { return v.toFixed(3); },
    int: function (v) { return Math.round(v).toLocaleString('en-US'); },
    sg3: function (v) { return (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(3); },
    x1: function (v) { return v.toFixed(1) + '×'; }
  };
  function cap(s) { return s ? s.charAt(0).toUpperCase() + s.slice(1) : s; }
  function spanLabel(s) { return s.split(' - ').map(function (p) { return cap(p); }).join(' – '); }
  function get(o, p) {
    return p.split('.').reduce(function (a, k) { return a == null ? a : a[k]; }, o);
  }
  function scale(d0, d1, r0, r1) { return function (v) { return r0 + (r1 - r0) * (v - d0) / (d1 - d0); }; }
  function el(html) { var d = document.createElement('div'); d.innerHTML = html.trim(); return d.firstChild; }

  function table(headers, rows, opts) {
    opts = opts || {};
    var h = '<table class="tbl"' + (opts.cls ? ' ' + opts.cls : '') + '>';
    if (opts.caption) h += '<caption>' + opts.caption + '</caption>';
    h += '<thead><tr>' + headers.map(function (x, i) {
      return '<th' + (opts.right && opts.right.indexOf(i) >= 0 ? ' class="r"' : '') + '>' + x + '</th>';
    }).join('') + '</tr></thead><tbody>';
    rows.forEach(function (r, ri) {
      h += '<tr' + (opts.hl === ri ? ' class="hl"' : '') + '>' + r.map(function (c, i) {
        return '<td' + (opts.right && opts.right.indexOf(i) >= 0 ? ' class="r"' : '') + '>' + c + '</td>';
      }).join('') + '</tr>';
    });
    return h + '</tbody></table>';
  }
  function put(id, html) { var n = document.getElementById(id); if (n) n.innerHTML = html; return n; }
  function addTV(figId, headers, rows, label) {
    var f = document.getElementById(figId);
    if (!f) return;
    var d = el('<details class="tv"><summary>' + (label || 'View this chart as a table') + '</summary><div class="tablewrap">' +
      table(headers, rows, { right: headers.map(function (_, i) { return i; }).slice(1) }) + '</div></details>');
    f.appendChild(d);
  }
  function tipFor(host) {
    var t = document.createElement('div');
    t.className = 'tip';
    t.setAttribute('role', 'status');
    host.appendChild(t);
    return {
      show: function (html, x, y) {
        t.innerHTML = html; t.classList.add('on');
        var w = host.clientWidth, tw = t.offsetWidth, th = t.offsetHeight;
        t.style.left = Math.max(0, Math.min(x + 14, w - tw)) + 'px';
        t.style.top = Math.max(0, y - th - 10) + 'px';
      },
      hide: function () { t.classList.remove('on'); }
    };
  }
  function mount(host, draw) {
    var last = 0;
    function run(force) {
      var w = Math.floor(host.clientWidth);
      if (!w || (w === last && !force)) return;
      last = w;
      $$('svg', host).forEach(function (s) { s.remove(); });
      draw(host, w);
    }
    if (window.ResizeObserver) new ResizeObserver(function () { run(false); }).observe(host);
    else window.addEventListener('resize', function () { run(false); });
    run(true);
    return function () { run(true); };
  }

  /* ---------------------------------------------------------------- derived values */
  var S = R.split, M = R.test.metrics, YR = R.eda.year_rows;
  var totalRows = 0, posRows = 0;
  Object.keys(YR).forEach(function (y) { totalRows += YR[y].rows; posRows += YR[y].rows * YR[y].rate; });
  var D = {
    overall_rate: posRows / totalRows,
    test_majority_acc: 1 - S.test.prevalence,
    train_majority_acc: 1 - S.train.prevalence,
    val_majority_acc: 1 - S.val.prevalence,
    prevalence_ratio: S.test.prevalence / S.train.prevalence
  };
  var P = R.primary_model || 'logreg';
  var rs = R.random_split;
  function mean(a) { return a.reduce(function (x, y) { return x + y; }, 0) / a.length; }
  D.rand = {};
  ['logreg', 'hgb', 'rule'].forEach(function (m) {
    D.rand[m] = {
      auc: mean(rs.map(function (r) { return r[m].auc; })),
      ap: mean(rs.map(function (r) { return r[m].ap; }))
    };
  });
  D.rand.prev = mean(rs.map(function (r) { return r.test_prev; }));
  D.lr_minus_rule_auc = M.logreg.auc - M.rule.auc;
  D.rolling_min_auc = Math.min.apply(null, R.rolling_origin.map(function (r) { return Math.min(r.logreg.auc, r.hgb.auc); }));
  D.macro_lr_auc = R.test.ablation[0].logreg.auc;
  D.nomacro_lr_auc = R.test.ablation[1].logreg.auc;
  D.dur_lr_auc = R.test.ablation[3].logreg.auc;
  D.brier_const = S.test.prevalence * (1 - S.test.prevalence);
  D.dur_gain_auc =R.test.ablation[3].logreg.auc - R.test.ablation[1].logreg.auc;
  var ROOT = { R: R, D: D };

  /* ---------------------------------------------------------------- scored rows engine (test block) */
  var RW = R.rows, N = RW.n, Y = RW.y;
  var SC = { logreg: RW.logreg, hgb: RW.hgb, rule: RW.rule };
  var NAMES = { logreg: 'Logistic regression', hgb: 'Gradient boosting', rule: 'Previous-success rule', prior: 'No-skill baseline' };
  var SHORT = { logreg: 'Logistic', hgb: 'Boosting', rule: 'Rule', prior: 'No skill' };
  var COL = { logreg: 'var(--c-lr)', hgb: 'var(--c-gbm)', rule: 'var(--c-rule)', prior: 'var(--c-prior)' };
  var nSub = 0; for (var i0 = 0; i0 < N; i0++) nSub += Y[i0];
  var prev = nSub / N;
  var ORD = {}, CUM = {};
  Object.keys(SC).forEach(function (m) {
    var idx = new Array(N); for (var i = 0; i < N; i++) idx[i] = i;
    idx.sort(function (a, b) { return SC[m][b] - SC[m][a] || a - b; });
    var groups = [], s = 0;
    for (var j = 1; j <= N; j++) {
      if (j === N || SC[m][idx[j]] !== SC[m][idx[s]]) { groups.push([s, j]); s = j; }
    }
    var cum = new Float64Array(N + 1);
    for (var q = 0; q < N; q++) cum[q + 1] = cum[q] + Y[idx[q]];
    ORD[m] = { idx: idx, groups: groups }; CUM[m] = cum;
  });
  /* Expected subscribers reached when the top-k clients are called. Tied scores are
     split proportionally, i.e. ties are broken at random in expectation. */
  function hitsAt(m, k) {
    var g = ORD[m].groups, c = CUM[m];
    for (var i = 0; i < g.length; i++) {
      var a = g[i][0], b = g[i][1];
      if (k <= b) return c[a] + (k - a) / (b - a) * (c[b] - c[a]);
    }
    return c[N];
  }
  function weights(m, k) {
    var o = ORD[m], w = new Float32Array(N), left = k;
    for (var gi = 0; gi < o.groups.length && left > 0; gi++) {
      var a = o.groups[gi][0], b = o.groups[gi][1], f = Math.min(1, left / (b - a));
      for (var i = a; i < b; i++) w[o.idx[i]] = f;
      left -= (b - a);
    }
    return w;
  }
  function topK(m, share) {
    var k = Math.round(share * N), h = hitsAt(m, k);
    return { calls: k, hits: h, precision: h / k, recall: h / nSub, lift: (h / k) / prev };
  }
  D.top20 = {};
  ['logreg', 'hgb', 'rule'].forEach(function (m) { D.top20[m] = topK(m, 0.20); });
  D.prior_top20 = { precision: prev, recall: 0.2, lift: 1 };

  /* ---------------------------------------------------------------- bind data-v spans */
  $$('[data-v]').forEach(function (n) {
    var path = n.getAttribute('data-v'), f = n.getAttribute('data-f') || 'f3';
    var v = path.indexOf('D.') === 0 ? get(ROOT, path) : get(R, path);
    if (v == null || !F[f]) { n.textContent = '?'; return; }
    n.textContent = F[f](v);
  });
  $$('[data-name]').forEach(function (n) { n.textContent = NAMES[n.getAttribute('data-name')]; });

  /* ---------------------------------------------------------------- theme + nav */
  var root = document.documentElement, tb = $('#theme');
  function setTheme(t, save) {
    if (t) root.setAttribute('data-theme', t); else root.removeAttribute('data-theme');
    var dark = t === 'dark' || (!t && window.matchMedia && matchMedia('(prefers-color-scheme: dark)').matches);
    if (tb) { tb.textContent = dark ? 'Light theme' : 'Dark theme'; tb.setAttribute('aria-pressed', dark ? 'true' : 'false'); }
    if (save) { try { localStorage.setItem('theme', t); } catch (e) { /* storage unavailable */ } }
  }
  var saved = null; try { saved = localStorage.getItem('theme'); } catch (e) { /* ignore */ }
  setTheme(saved === 'dark' || saved === 'light' ? saved : null, false);
  if (tb) tb.addEventListener('click', function () {
    var dark = root.getAttribute('data-theme') === 'dark' ||
      (!root.getAttribute('data-theme') && matchMedia('(prefers-color-scheme: dark)').matches);
    setTheme(dark ? 'light' : 'dark', true);
  });
  var links = $$('.rail nav a'), secs = links.map(function (a) { return $(a.getAttribute('href')); });
  if ('IntersectionObserver' in window) {
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        if (e.isIntersecting) {
          links.forEach(function (a) { a.removeAttribute('aria-current'); });
          var a = links[secs.indexOf(e.target)]; if (a) a.setAttribute('aria-current', 'true');
        }
      });
    }, { rootMargin: '-20% 0px -70% 0px' });
    secs.forEach(function (s) { if (s) io.observe(s); });
  }

  /* ---------------------------------------------------------------- generic horizontal bars */
  function hbars(host, W, items, o) {
    o = o || {};
    var rowH = o.rowH || 26, labW = Math.min(o.labelW || 150, W * 0.4), r = o.right || 10, t = o.top || ((o.ref && o.ref.label) ? 24 : 8);
    var H = t + items.length * rowH + (o.axis === false ? 6 : 26);
    var s = svg('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': o.aria || 'Bar chart' }, host);
    var d0 = o.domain[0], d1 = o.domain[1], x = scale(d0, d1, labW, W - r - (o.valueW || 0));
    var ax = svg('g', { class: 'ax' }, s);
    (o.ticks || []).forEach(function (tk) {
      svg('line', { class: 'grid', x1: x(tk), x2: x(tk), y1: t, y2: t + items.length * rowH }, ax);
      txt(ax, x(tk), t + items.length * rowH + 16, o.tickFmt ? o.tickFmt(tk) : String(tk), 'lbl', 'middle');
    });
    svg('line', { class: 'base', x1: x(d0), x2: x(d0), y1: t, y2: t + items.length * rowH }, ax);
    items.forEach(function (it, i) {
      var y = t + i * rowH, bh = Math.min(14, rowH - 10), by = y + (rowH - bh) / 2;
      var lab = txt(s, labW - 10, y + rowH / 2 + 4.5, it.label, 'lbl', 'end');
      var w = Math.max(1, x(Math.max(d0, it.value)) - x(d0));
      var rect = svg('rect', { x: x(d0), y: by, width: w, height: bh, rx: 2, fill: it.color || 'var(--accent)' }, s);
      if (it.title) { var tt = svg('title', {}, rect); tt.textContent = it.title; }
      if (it.text) txt(s, x(it.value) + 7, y + rowH / 2 + 4.5, it.text, 'lbl', 'start');
    });
    if (o.ref) {
      var rx = x(o.ref.value);
      svg('line', { class: 'ref', x1: rx, x2: rx, y1: t - 2, y2: t + items.length * rowH, stroke: 'var(--ink-3)' }, ax);
      if (o.ref.label) txt(ax, rx + 4, t - 9, o.ref.label, 'lbl', 'start').setAttribute('style', 'font-style:italic');
    }
    return s;
  }

  /* ---------------------------------------------------------------- hero timeline */
  (function () {
    var host = $('#chart-timeline'); if (!host) return;
    var T = R.eda.timeline, n = T.length, tip = tipFor(host);
    mount(host, function (h, W) {
      var m = { l: 44, r: 44, t: 36 }, h1 = 150, gap = 40, h2 = 92, H = m.t + h1 + gap + h2 + 30;
      var s = svg('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img',
        'aria-label': 'Subscription rate and 3-month Euribor over the campaign period, with training, validation and test blocks shaded.' }, h);
      var x = function (i) { return m.l + (W - m.l - m.r) * (i / (n - 1)); };
      var maxR = Math.max.apply(null, T.map(function (p) { return p.rate; })) * 1.12;
      var yr = scale(0, maxR, m.t + h1, m.t), ye = scale(0, 5.6, m.t + h1 + gap + h2, m.t + h1 + gap);
      // bands
      var blocks = [['train', 'Train'], ['val', 'Validation'], ['test', 'Test']];
      blocks.forEach(function (b) {
        var idx = []; T.forEach(function (p, i) { if (p.split === b[0]) idx.push(i); });
        if (!idx.length) return;
        var a = idx[0] === 0 ? m.l : x(idx[0] - 0.5), z = idx[idx.length - 1] === n - 1 ? W - m.r : x(idx[idx.length - 1] + 0.5);
        svg('rect', { class: 'band-' + b[0], x: a, y: m.t - 4, width: z - a, height: h1 + gap + h2 + 8 }, s);
        var sp = S[b[0]], lab = b[1];
        var long = lab + ' · ' + F.p1(sp.prevalence) + ' yes';
        txt(s, a + 6, m.t - 14, (z - a) > 150 ? long : (z - a) > 70 ? lab : '', 'lbl-strong');
      });
      var ax = svg('g', { class: 'ax' }, s);
      [0, 0.2, 0.4, 0.6].forEach(function (v) {
        if (v > maxR) return;
        svg('line', { class: 'grid', x1: m.l, x2: W - m.r, y1: yr(v), y2: yr(v) }, ax);
        txt(ax, m.l - 8, yr(v) + 4, (v * 100).toFixed(0) + '%', 'lbl', 'end');
      });
      [0, 2, 4].forEach(function (v) {
        svg('line', { class: 'grid', x1: m.l, x2: W - m.r, y1: ye(v), y2: ye(v) }, ax);
        txt(ax, m.l - 8, ye(v) + 4, v + '%', 'lbl', 'end');
      });
      txt(s, m.l, m.t + 14, 'Clients who subscribed', 'lbl-strong').setAttribute('transform', 'translate(8,0)');
      txt(s, m.l, m.t + h1 + gap - 10, 'Euribor 3-month rate (the economy)', 'lbl-strong').setAttribute('transform', 'translate(8,0)');
      var path = function (key, sc) { return T.map(function (p, i) { return (i ? 'L' : 'M') + x(i).toFixed(1) + ',' + sc(p[key]).toFixed(1); }).join(''); };
      svg('path', { d: path('rate', yr), fill: 'none', stroke: 'var(--accent)', 'stroke-width': 2.5, 'stroke-linejoin': 'round' }, s);
      svg('path', { d: path('euribor', ye), fill: 'none', stroke: 'var(--ink-2)', 'stroke-width': 2.5, 'stroke-linejoin': 'round' }, s);
      txt(s, x(n - 1) + 8, yr(T[n - 1].rate) + 4, F.p0(T[n - 1].rate), 'lbl-strong');
      txt(s, x(n - 1) + 8, ye(T[n - 1].euribor) + 4, T[n - 1].euribor.toFixed(1) + '%', 'lbl-strong');
      txt(s, x(0) + 8, yr(T[0].rate) - 8, F.p0(T[0].rate), 'lbl-strong');
      var shown = '';
      [0, 6, 12, 18, 24, 30, 40].forEach(function (i) {
        if (T[i].label === shown) return; shown = T[i].label;
        txt(ax, x(i), H - 6, cap(T[i].label), 'lbl', i === 0 ? 'start' : i === n - 1 ? 'end' : 'middle');
      });
      var cross = svg('g', { style: 'display:none' }, s);
      svg('line', { x1: 0, x2: 0, y1: m.t - 4, y2: m.t + h1 + gap + h2, stroke: 'var(--ink)', 'stroke-width': 1 }, cross);
      var d1 = svg('circle', { r: 4.5, fill: 'var(--accent)', stroke: 'var(--surface)', 'stroke-width': 2 }, cross);
      var d2 = svg('circle', { r: 4.5, fill: 'var(--ink-2)', stroke: 'var(--surface)', 'stroke-width': 2 }, cross);
      var ov = svg('rect', { x: m.l, y: m.t - 4, width: W - m.l - m.r, height: h1 + gap + h2 + 8, fill: 'transparent', style: 'cursor:crosshair' }, s);
      ov.addEventListener('pointermove', function (e) {
        var b = ov.getBoundingClientRect(), i = Math.round((e.clientX - b.left) / b.width * (n - 1));
        i = Math.max(0, Math.min(n - 1, i)); var p = T[i];
        cross.style.display = ''; cross.firstChild.setAttribute('x1', x(i)); cross.firstChild.setAttribute('x2', x(i));
        d1.setAttribute('cx', x(i)); d1.setAttribute('cy', yr(p.rate)); d2.setAttribute('cx', x(i)); d2.setAttribute('cy', ye(p.euribor));
        var hb = host.getBoundingClientRect();
        tip.show('<b>' + cap(p.label) + '</b> · ' + { train: 'training', val: 'validation', test: 'test' }[p.split] + ' block<br>rows ' + F.int(p.start + 1) + '–' + F.int(p.end + 1) +
          '<br>subscribed: <b>' + F.p1(p.rate) + '</b><br>Euribor 3m: <b>' + p.euribor.toFixed(2) + '%</b>', e.clientX - hb.left, e.clientY - hb.top);
      });
      ov.addEventListener('pointerleave', function () { cross.style.display = 'none'; tip.hide(); });
    });
    addTV('fig-timeline', ['Period (most common month)', 'Rows', 'Subscribed', 'Euribor 3m (%)', 'Block'],
      T.map(function (p) { return [cap(p.label), F.int(p.start + 1) + '–' + F.int(p.end + 1), F.p1(p.rate), p.euribor.toFixed(2), p.split]; }));
  })();

  /* ---------------------------------------------------------------- facts + static tables */
  put('tbl-quality', table(['Check', 'Finding', 'What I did'], [
    ['Missing values', '<b>' + R.quality.missing_cells + '</b> empty cells', 'Nothing to impute. Unknowns are coded as the text "unknown" (next row).'],
    ['"unknown" categories', Object.keys(R.quality.unknown_counts).map(function (k) { return k + ' ' + F.int(R.quality.unknown_counts[k]); }).join(', '), 'Kept as their own category. Whether a client’s default status is unknown is itself information, and filling it in would invent data.'],
    ['<code>pdays</code> sentinel', F.p1(R.quality.pdays_999_share) + ' of rows hold 999 (never contacted before)', 'Split into a "never contacted" flag plus days-since (0 when never). 999 is a label, not a duration.'],
    ['Near-constant column', '<code>default</code> is "yes" in only ' + R.quality.default_yes + ' rows', 'Kept; one-hot encoding handles it and regularisation shrinks it.'],
    ['Exact duplicates', R.quality.exact_duplicate_rows + ' rows (' + F.p1(R.quality.exact_duplicate_rows / R.quality.rows).replace('0.0%', '0.03%') + ')', 'Kept: identical records can be different clients. Their effect is too small to matter.'],
    ['Extremes', 'age ' + R.quality.age_min + '–' + R.quality.age_max + '; <code>campaign</code> up to ' + R.quality.campaign_max + ' (99th pct ' + R.quality.campaign_p99 + ')', 'No clipping. <code>campaign</code> is not a model input (see Question).'],
    ['Class balance', F.p1(D.overall_rate) + ' yes overall, but ' + F.p1(S.train.prevalence) + ' in the training block and ' + F.p1(S.test.prevalence) + ' in the test block', 'This drift is the central finding; see Evaluation and Results.']
  ], { right: [] }));

  var corr = R.eda.macro_corr;
  put('tbl-corr', '<table class="tbl corr"><thead><tr><th></th>' + corr.labels.map(function (l) { return '<th>' + l + '</th>'; }).join('') + '</tr></thead><tbody>' +
    corr.matrix.map(function (row, i) {
      return '<tr><th>' + corr.labels[i] + '</th>' + row.map(function (v) {
        var a = Math.abs(v), bg = 'color-mix(in srgb, var(--accent) ' + Math.round(a * 42) + '%, var(--surface))';
        return '<td style="background:' + bg + '">' + v.toFixed(2) + '</td>';
      }).join('') + '</tr>';
    }).join('') + '</tbody></table>');

  var dict = R.eda.dictionary;
  put('tbl-dict', table(['Column', 'Meaning', 'Used by the main models?'], Object.keys(dict).map(function (k) {
    var use = k === 'y' ? 'target' : k === 'duration' ? 'no: known only after the call' : k === 'campaign' ? 'no: counts the last call too'
      : k === 'pdays' ? 'yes (as two features)' : 'yes';
    return ['<code>' + k + '</code>', dict[k], use];
  })));

  put('tbl-split', table(['Block', 'Rows', 'Period', 'Share subscribing', 'Mean Euribor 3m'], [
    ['Train', F.int(S.train.rows), spanLabel(S.train.span), F.p1(S.train.prevalence), S.train.euribor_mean.toFixed(2) + '%'],
    ['Validation', F.int(S.val.rows), spanLabel(S.val.span), F.p1(S.val.prevalence), S.val.euribor_mean.toFixed(2) + '%'],
    ['Test', F.int(S.test.rows), spanLabel(S.test.span), F.p1(S.test.prevalence), S.test.euribor_mean.toFixed(2) + '%']
  ], { right: [1, 3, 4] }));
  var sb = $('#splitbar');
  if (sb) sb.innerHTML = '<div class="a" style="flex:' + S.train.rows + '">Train · ' + F.int(S.train.rows) + '</div><div class="b" style="flex:' + S.val.rows + '">Validation · ' + F.int(S.val.rows) +
    '</div><div class="c" style="flex:' + S.test.rows + '">Test · ' + F.int(S.test.rows) + '</div>';

  var B = R.tuning.best, VM = R.tuning.val_metrics;
  put('tbl-tuning', table(['Method', 'Features chosen', 'Hyperparameters chosen', 'Validation PR-AUC', 'Validation ROC-AUC'], [
    ['<span class="sw" style="background:var(--c-prior)"></span>No-skill baseline', '—', '—', F.f3(VM.prior_ap) + ' <small>(= prevalence)</small>', '0.500'],
    ['<span class="sw" style="background:var(--c-rule)"></span>Previous-success rule', '<code>poutcome</code> only', '—', F.f3(VM.rule.ap), F.f3(VM.rule.auc)],
    ['<span class="sw" style="background:var(--c-lr)"></span>Logistic regression', B.logreg.feature_set === 'pre_contact' ? 'with economy indicators' : 'without economy indicators', 'C = ' + B.logreg.params.C, F.f3(B.logreg.val_ap), F.f3(B.logreg.val_auc)],
    ['<span class="sw" style="background:var(--c-gbm)"></span>Gradient boosting', B.hgb.feature_set === 'pre_contact' ? 'with economy indicators' : 'without economy indicators',
      'learning rate ' + B.hgb.params.lr + ', ' + B.hgb.params.leaves + ' leaves, L2 ' + B.hgb.params.l2 + ', ' + B.hgb.params.iters + ' trees', F.f3(B.hgb.val_ap), F.f3(B.hgb.val_auc)]
  ], { right: [3, 4] }));

  /* headline results table */
  var rowsRes = ['prior', 'rule', 'logreg', 'hgb'].map(function (m) {
    var mt = M[m], t = m === 'prior' ? D.prior_top20 : D.top20[m];
    var ap = m === 'prior' ? F.f3(mt.ap) + ' <small>(= prevalence)</small>' : F.f3(mt.ap) + ' <small>[' + F.f3(mt.ap_ci[0]) + ', ' + F.f3(mt.ap_ci[1]) + ']</small>';
    var au = m === 'prior' ? '0.500' : F.f3(mt.auc) + ' <small>[' + F.f3(mt.auc_ci[0]) + ', ' + F.f3(mt.auc_ci[1]) + ']</small>';
    return ['<span class="sw" style="background:' + COL[m] + '"></span>' + NAMES[m], ap, au, mt.brier == null ? '—' : F.f3(mt.brier),
      F.p1(t.precision), F.p1(t.recall)];
  });
  put('tbl-results', table(['Method', 'PR-AUC [95% CI]', 'ROC-AUC [95% CI]', 'Brier', 'Precision, top 20%', 'Subscribers reached, top 20%'], rowsRes, { right: [1, 2, 3, 4, 5], hl: P === 'logreg' ? 2 : 3 }));

  put('tbl-diffs', table(['Paired comparison (PR-AUC)', 'Mean difference', '95% interval', 'Resamples where it was positive'], [
    ['Logistic regression − previous-success rule', F.sg3(R.test.paired_diffs.logreg_minus_rule.mean), '[' + F.sg3(R.test.paired_diffs.logreg_minus_rule.ci[0]) + ', ' + F.sg3(R.test.paired_diffs.logreg_minus_rule.ci[1]) + ']', F.p0(R.test.paired_diffs.logreg_minus_rule.share_positive)],
    ['Gradient boosting − logistic regression', F.sg3(R.test.paired_diffs.hgb_minus_logreg.mean), '[' + F.sg3(R.test.paired_diffs.hgb_minus_logreg.ci[0]) + ', ' + F.sg3(R.test.paired_diffs.hgb_minus_logreg.ci[1]) + ']', F.p0(R.test.paired_diffs.hgb_minus_logreg.share_positive)],
    ['Gradient boosting − previous-success rule', F.sg3(R.test.paired_diffs.hgb_minus_rule.mean), '[' + F.sg3(R.test.paired_diffs.hgb_minus_rule.ci[0]) + ', ' + F.sg3(R.test.paired_diffs.hgb_minus_rule.ci[1]) + ']', F.p0(R.test.paired_diffs.hgb_minus_rule.share_positive)]
  ], { right: [1, 2, 3] }));

  var FZ = R.test.frozen_threshold;
  put('tbl-frozen', table(['Model (trained on train block only)', 'Cut-off picked on validation', 'Called on validation', 'Called on test', 'Test precision', 'Test reach', 'Test F1', 'Best possible test F1'],
    ['logreg', 'hgb'].map(function (m) {
      var f = FZ[m];
      return ['<span class="sw" style="background:' + COL[m] + '"></span>' + NAMES[m], F.f3(f.threshold), F.p0(f.val_pred_pos_share), F.p0(f.test_pred_pos_share), F.p0(f.test_precision), F.p0(f.test_recall), F.f3(f.test_f1), F.f3(f.oracle_test_f1)];
    }), { right: [1, 2, 3, 4, 5, 6, 7] }));

  var AB = { pre_contact: 'Main set with economy indicators', pre_contact_no_macro: 'Without economy indicators (chosen on validation)', plus_campaign: '+ <code>campaign</code>', plus_campaign_duration: '+ <code>campaign</code> + <code>duration</code> (not usable before a call)' };
  put('tbl-ablation', table(['Feature set (hyperparameters fixed)', 'Logistic ROC-AUC', 'Logistic PR-AUC', 'Boosting ROC-AUC', 'Boosting PR-AUC'],
    R.test.ablation.map(function (a) { return [AB[a.feature_set], F.f3(a.logreg.auc), F.f3(a.logreg.ap), F.f3(a.hgb.auc), F.f3(a.hgb.ap)]; }), { right: [1, 2, 3, 4] }));

  put('tbl-chunks', table(['Test period', 'Rows', 'Share subscribing', 'Logistic ROC-AUC', 'Boosting ROC-AUC', 'Rule ROC-AUC'],
    R.test.chunks.map(function (c) { return [spanLabel(c.span), F.int(c.n), F.p1(c.prevalence), F.f3(c.logreg.auc), F.f3(c.hgb.auc), F.f3(c.rule.auc)]; }), { right: [1, 2, 3, 4, 5] }));

  put('tbl-random', table(['', 'Time-ordered test (primary)', 'Random-split test (mean of 3 seeds)'], [
    ['Share subscribing in the test rows', F.p1(S.test.prevalence), F.p1(D.rand.prev)],
    ['<span class="sw" style="background:var(--c-rule)"></span>Rule: ROC-AUC / PR-AUC', F.f3(M.rule.auc) + ' / ' + F.f3(M.rule.ap), F.f3(D.rand.rule.auc) + ' / ' + F.f3(D.rand.rule.ap)],
    ['<span class="sw" style="background:var(--c-lr)"></span>Logistic: ROC-AUC / PR-AUC', F.f3(M.logreg.auc) + ' / ' + F.f3(M.logreg.ap), F.f3(D.rand.logreg.auc) + ' / ' + F.f3(D.rand.logreg.ap)],
    ['<span class="sw" style="background:var(--c-gbm)"></span>Boosting: ROC-AUC / PR-AUC', F.f3(M.hgb.auc) + ' / ' + F.f3(M.hgb.ap), F.f3(D.rand.hgb.auc) + ' / ' + F.f3(D.rand.hgb.ap)]
  ], { right: [1, 2] }));

  /* ---------------------------------------------------------------- EDA rate bars with tabs */
  (function () {
    var host = $('#chart-rates'), sel = $('#rates-dim'); if (!host || !sel) return;
    var DIMS = { poutcome: 'Outcome of previous campaign', job: 'Job', month: 'Month of contact', contact: 'Contact channel', age: 'Age', education: 'Education', marital: 'Marital status' };
    var cur = 'poutcome';
    var redraw = mount(host, function (h, W) {
      var items = R.eda.train_rates[cur].map(function (r) {
        return { label: cap(r.label.replace(/\./g, ' ')), value: r.rate, text: F.p1(r.rate) + '  (n=' + F.int(r.n) + ')', title: cap(r.label) + ': ' + F.p1(r.rate) + ' of ' + F.int(r.n) };
      });
      var mx = Math.max.apply(null, items.map(function (i) { return i.value; }));
      hbars(h, W, items, { domain: [0, Math.max(0.1, mx * 1.35)], ticks: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6].filter(function (t) { return t <= mx * 1.35; }), tickFmt: F.p0, rowH: 25,
        ref: { value: S.train.prevalence, label: 'training-block average ' + F.p1(S.train.prevalence) }, aria: 'Subscription rate by ' + DIMS[cur] });
    });
    sel.addEventListener('change', function (e) {
      if (e.target.name === 'rates-dim') { cur = e.target.value; redraw(); renderTV(); }
    });
    function renderTV() {
      var f = $('#fig-rates'); var old = $('details.tv', f); if (old) old.remove();
      addTV('fig-rates', ['Group', 'Clients (train block)', 'Share subscribing'], R.eda.train_rates[cur].map(function (r) { return [cap(r.label), F.int(r.n), F.p1(r.rate)]; }));
    }
    renderTV();
  })();

  /* leakage bars: duration and campaign */
  (function () {
    var h1 = $('#chart-duration'), h2 = $('#chart-campaign');
    if (h1) mount(h1, function (h, W) {
      hbars(h, W, R.eda.train_rates.duration.map(function (r) { return { label: r.label, value: r.rate, text: F.p0(r.rate), title: r.label + ': ' + F.p1(r.rate) }; }),
        { domain: [0, 0.75], ticks: [0, 0.25, 0.5, 0.75], tickFmt: F.p0, labelW: 56, aria: 'Subscription rate by call length' });
    });
    if (h2) mount(h2, function (h, W) {
      hbars(h, W, R.eda.train_rates.campaign.map(function (r) { return { label: r.label, value: r.rate, text: F.p1(r.rate), title: r.label + ' contacts: ' + F.p1(r.rate) }; }),
        { domain: [0, 0.75], ticks: [0, 0.25, 0.5, 0.75], tickFmt: F.p0, labelW: 56, aria: 'Subscription rate by contacts in the campaign' });
    });
    addTV('fig-duration', ['Call length', 'Share subscribing (train block)'], R.eda.train_rates.duration.map(function (r) { return [r.label, F.p1(r.rate)]; }));
    addTV('fig-campaign', ['Contacts this campaign', 'Share subscribing (train block)'], R.eda.train_rates.campaign.map(function (r) { return [r.label, F.p1(r.rate)]; }));
  })();

  /* unknown counts */
  (function () {
    var host = $('#chart-unknown'); if (!host) return;
    var u = R.quality.unknown_counts, keys = Object.keys(u).sort(function (a, b) { return u[b] - u[a]; });
    mount(host, function (h, W) {
      hbars(h, W, keys.map(function (k) { return { label: k, value: u[k] / R.quality.rows, text: F.p1(u[k] / R.quality.rows) + ' (' + F.int(u[k]) + ')' }; }),
        { domain: [0, 0.3], ticks: [0, 0.1, 0.2, 0.3], tickFmt: F.p0, labelW: 90, rowH: 24, aria: 'Share of rows coded unknown, by column' });
    });
    addTV('fig-unknown', ['Column', 'Rows "unknown"', 'Share'], keys.map(function (k) { return [k, F.int(u[k]), F.p1(u[k] / R.quality.rows)]; }));
  })();

  /* ---------------------------------------------------------------- tuning strip */
  (function () {
    var host = $('#chart-tuning'); if (!host) return;
    var rowsDef = [['logreg', 'pre_contact', 'Logistic · with economy'], ['logreg', 'pre_contact_no_macro', 'Logistic · without'],
      ['hgb', 'pre_contact', 'Boosting · with economy'], ['hgb', 'pre_contact_no_macro', 'Boosting · without']];
    var tb2 = R.tuning.table;
    var all = tb2.map(function (r) { return r.val_ap; }).concat([VM.prior_ap, VM.rule.ap]);
    var lo = Math.floor((Math.min.apply(null, all) - 0.01) * 50) / 50, hi = Math.ceil((Math.max.apply(null, all) + 0.01) * 50) / 50;
    mount(host, function (h, W) {
      var m = { l: Math.min(170, W * .34), r: 16, t: 34 }, rh = 34, H = m.t + rowsDef.length * rh + 30;
      var s = svg('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'Validation PR-AUC of every configuration tried' }, h);
      var x = scale(lo, hi, m.l, W - m.r), ax = svg('g', { class: 'ax' }, s);
      for (var v = lo; v <= hi + 1e-9; v += 0.02) {
        svg('line', { class: 'grid', x1: x(v), x2: x(v), y1: m.t, y2: m.t + rowsDef.length * rh }, ax);
        txt(ax, x(v), m.t + rowsDef.length * rh + 16, v.toFixed(2), 'lbl', 'middle');
      }
      txt(ax, (m.l + W - m.r) / 2, H - 2, 'Validation PR-AUC (higher is better)', 'lbl', 'middle');
      [[VM.prior_ap, 'no skill', 'var(--c-prior)'], [VM.rule.ap, 'rule', 'var(--c-rule)']].forEach(function (r, i) {
        svg('line', { x1: x(r[0]), x2: x(r[0]), y1: m.t - 4, y2: m.t + rowsDef.length * rh, stroke: r[2], 'stroke-width': 2, 'stroke-dasharray': '5 3' }, s);
        txt(s, x(r[0]) + (i ? 4 : -4), m.t - 10, r[1] + ' ' + r[0].toFixed(3), 'lbl-strong', i ? 'start' : 'end');
      });
      rowsDef.forEach(function (d, i) {
        var y = m.t + i * rh + rh / 2;
        txt(s, m.l - 10, y + 4, d[2], 'lbl', 'end');
        tb2.filter(function (r) { return r.model === d[0] && r.feature_set === d[1]; }).forEach(function (r, k) {
          var best = B[d[0]] && B[d[0]].feature_set === d[1] && B[d[0]].val_ap === r.val_ap;
          var c = svg('circle', { cx: x(r.val_ap), cy: y + ((k % 3) - 1) * 4, r: best ? 6.5 : 4.5, fill: COL[d[0]], 'fill-opacity': best ? 1 : .55, stroke: best ? 'var(--ink)' : 'var(--surface)', 'stroke-width': best ? 2 : 1 }, s);
          var tt = svg('title', {}, c); tt.textContent = JSON.stringify(r.params) + ' → validation PR-AUC ' + r.val_ap.toFixed(3) + ', ROC-AUC ' + r.val_auc.toFixed(3) + (best ? ' (chosen)' : '');
        });
      });
    });
    var top = tb2.slice().sort(function (a, b) { return b.val_ap - a.val_ap; }).slice(0, 12);
    addTV('fig-tuning', ['Model / features / settings', 'Validation PR-AUC', 'Validation ROC-AUC'], top.map(function (r) {
      return [NAMES[r.model] + ' · ' + (r.feature_set === 'pre_contact' ? 'with economy' : 'without') + ' · ' + JSON.stringify(r.params).replace(/[{}"]/g, ''), F.f3(r.val_ap), F.f3(r.val_auc)];
    }), 'View the 12 best configurations as a table');
  })();

  /* ---------------------------------------------------------------- results: forest panels */
  function forest(hostId, key, dom, refVal, refLabel, ticks) {
    var host = document.getElementById(hostId); if (!host) return;
    var order = ['prior', 'rule', 'logreg', 'hgb'];
    mount(host, function (h, W) {
      var m = { l: Math.min(132, W * .32), r: 14, t: 22 }, rh = 38, H = m.t + order.length * rh + 38;
      var s = svg('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': key.toUpperCase() + ' on the test block with 95% intervals' }, h);
      var x = scale(dom[0], dom[1], m.l, W - m.r), ax = svg('g', { class: 'ax' }, s);
      ticks.forEach(function (t) {
        svg('line', { class: 'grid', x1: x(t), x2: x(t), y1: m.t - 6, y2: m.t + order.length * rh }, ax);
        txt(ax, x(t), m.t + order.length * rh + 16, t.toFixed(2), 'lbl', 'middle');
      });
      svg('line', { class: 'ref', x1: x(refVal), x2: x(refVal), y1: m.t - 6, y2: m.t + order.length * rh, stroke: 'var(--ink-3)' }, ax);
      txt(ax, x(refVal), m.t + order.length * rh + 32, refLabel, 'lbl', 'middle');
      order.forEach(function (mm, i) {
        var y = m.t + i * rh + rh / 2, v = key === 'ap' ? M[mm].ap : M[mm].auc, ci = key === 'ap' ? M[mm].ap_ci : M[mm].auc_ci;
        txt(s, m.l - 10, y + 4, NAMES[mm], 'lbl', 'end');
        if (ci && mm !== 'prior') svg('line', { x1: x(ci[0]), x2: x(ci[1]), y1: y, y2: y, stroke: COL[mm], 'stroke-width': 3, 'stroke-linecap': 'round' }, s);
        var c = svg('circle', { cx: x(v), cy: y, r: 6, fill: COL[mm], stroke: 'var(--surface)', 'stroke-width': 2 }, s);
        var tt = svg('title', {}, c); tt.textContent = NAMES[mm] + ': ' + v.toFixed(3) + (ci && mm !== 'prior' ? ' [' + ci[0].toFixed(3) + ', ' + ci[1].toFixed(3) + ']' : '');
        txt(s, x(v), y - 12, v.toFixed(3), 'lbl-strong', 'middle');
      });
    });
  }
  (function () {
    var apAll = ['rule', 'logreg', 'hgb'].reduce(function (a, m) { return a.concat(M[m].ap_ci); }, [M.prior.ap]);
    var auAll = ['rule', 'logreg', 'hgb'].reduce(function (a, m) { return a.concat(M[m].auc_ci); }, [0.5]);
    forest('chart-forest-ap', 'ap', [Math.floor((Math.min.apply(null, apAll) - .02) * 20) / 20, Math.ceil((Math.max.apply(null, apAll) + .02) * 20) / 20], S.test.prevalence, 'no skill = prevalence', [0.3, 0.35, 0.4, 0.45, 0.5, 0.55].filter(function (t) { return t >= Math.floor((Math.min.apply(null, apAll) - .02) * 20) / 20 && t <= Math.ceil((Math.max.apply(null, apAll) + .02) * 20) / 20; }));
    forest('chart-forest-auc', 'auc', [0.45, 0.75], 0.5, 'chance', [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]);
    addTV('fig-forest', ['Method', 'PR-AUC', 'PR-AUC 95% interval', 'ROC-AUC', 'ROC-AUC 95% interval'], ['prior', 'rule', 'logreg', 'hgb'].map(function (m) {
      var mt = M[m];
      return [NAMES[m], F.f3(mt.ap), mt.ap_ci ? '[' + F.f3(mt.ap_ci[0]) + ', ' + F.f3(mt.ap_ci[1]) + ']' : '—', F.f3(mt.auc), mt.auc_ci ? '[' + F.f3(mt.auc_ci[0]) + ', ' + F.f3(mt.auc_ci[1]) + ']' : '—'];
    }));
  })();

  /* ---------------------------------------------------------------- calibration */
  (function () {
    var host = $('#chart-calib'); if (!host) return;
    var C = R.test.calibration;
    mount(host, function (h, W) {
      var m = { l: 46, r: 100, t: 14, b: 40 }, Hh = Math.min(340, Math.max(240, W * .62)), H = Hh + m.t + m.b;
      var s = svg('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'Mean predicted score against observed subscription rate in ten equal bins of the test block' }, h);
      var d1 = 0.8, x = scale(0, d1, m.l, W - m.r), y = scale(0, d1, m.t + Hh, m.t), ax = svg('g', { class: 'ax' }, s);
      [0, 0.2, 0.4, 0.6, 0.8].forEach(function (t) {
        svg('line', { class: 'grid', x1: m.l, x2: W - m.r, y1: y(t), y2: y(t) }, ax);
        svg('line', { class: 'grid', x1: x(t), x2: x(t), y1: m.t, y2: m.t + Hh }, ax);
        txt(ax, m.l - 8, y(t) + 4, (t * 100).toFixed(0) + '%', 'lbl', 'end');
        txt(ax, x(t), m.t + Hh + 16, (t * 100).toFixed(0) + '%', 'lbl', 'middle');
      });
      txt(ax, (m.l + W - m.r) / 2, H - 4, 'Average predicted probability in the bin', 'lbl', 'middle');
      var yl = txt(ax, 12, m.t + Hh / 2, 'Share who actually subscribed', 'lbl', 'middle'); yl.setAttribute('transform', 'rotate(-90 12 ' + (m.t + Hh / 2) + ')');
      svg('line', { class: 'ref', x1: x(0), y1: y(0), x2: x(d1), y2: y(d1), stroke: 'var(--ink-3)' }, ax);
      txt(ax, x(0.5) + 6, y(0.5) + 18, 'perfectly calibrated', 'lbl').setAttribute('style', 'font-style:italic');
      ['logreg', 'hgb'].forEach(function (mm) {
        var pts = C[mm];
        svg('path', { d: pts.map(function (p, i) { return (i ? 'L' : 'M') + x(p.pred).toFixed(1) + ',' + y(p.obs).toFixed(1); }).join(''), fill: 'none', stroke: COL[mm], 'stroke-width': 2.5 }, s);
        pts.forEach(function (p) {
          var c = svg('circle', { cx: x(p.pred), cy: y(p.obs), r: 4.5, fill: COL[mm], stroke: 'var(--surface)', 'stroke-width': 1.5 }, s);
          var tt = svg('title', {}, c); tt.textContent = NAMES[mm] + ': predicted ' + F.p1(p.pred) + ', observed ' + F.p1(p.obs) + ' (' + F.int(p.n) + ' clients)';
        });
        var last = pts[pts.length - 1];
        txt(s, x(last.pred) + 9, y(last.obs) + (mm === 'logreg' ? -2 : 14), SHORT[mm], 'lbl-strong');
      });
    });
    var rows = [];
    for (var i = 0; i < C.logreg.length; i++) rows.push([String(i + 1), F.p1(C.logreg[i].pred), F.p1(C.logreg[i].obs), F.p1(C.hgb[i].pred), F.p1(C.hgb[i].obs)]);
    addTV('fig-calib', ['Bin (low to high score)', 'Logistic: predicted', 'Logistic: observed', 'Boosting: predicted', 'Boosting: observed'], rows);
  })();

  /* ---------------------------------------------------------------- call-list explorer */
  (function () {
    var host = $('#chart-gains'), segHost = $('#chart-seg'); if (!host) return;
    var st = { m: P, pct: 20, dim: 'poutcome' }, tip = tipFor(host);
    var curves = {};
    ['logreg', 'hgb', 'rule'].forEach(function (m) { curves[m] = []; for (var p = 0; p <= 100; p++) curves[m].push(hitsAt(m, p / 100 * N) / nSub); });
    var kslider = $('#ex-k'), kout = $('#ex-kout'), radios = $$('input[name="ex-model"]'), dimSel = $('#ex-dim');
    radios.forEach(function (r) { r.checked = r.value === st.m; });

    function drawGains(h, W) {
      var m = { l: 46, r: 14, t: 14, b: 42 }, Hh = Math.min(330, Math.max(230, W * .6)), H = Hh + m.t + m.b;
      var s = svg('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'Share of all subscribers reached against share of clients called' }, h);
      var x = scale(0, 100, m.l, W - m.r), y = scale(0, 1, m.t + Hh, m.t), ax = svg('g', { class: 'ax' }, s);
      [0, 25, 50, 75, 100].forEach(function (t) {
        svg('line', { class: 'grid', x1: x(t), x2: x(t), y1: m.t, y2: m.t + Hh }, ax);
        txt(ax, x(t), m.t + Hh + 16, t + '%', 'lbl', 'middle');
        svg('line', { class: 'grid', x1: m.l, x2: W - m.r, y1: y(t / 100), y2: y(t / 100) }, ax);
        txt(ax, m.l - 8, y(t / 100) + 4, t + '%', 'lbl', 'end');
      });
      txt(ax, (m.l + W - m.r) / 2, H - 4, 'Clients called, highest score first', 'lbl', 'middle');
      var yl = txt(ax, 12, m.t + Hh / 2, 'Subscribers reached', 'lbl', 'middle'); yl.setAttribute('transform', 'rotate(-90 12 ' + (m.t + Hh / 2) + ')');
      svg('line', { class: 'ref', x1: x(0), y1: y(0), x2: x(100), y2: y(1), stroke: 'var(--ink-3)' }, ax);
      txt(ax, x(62), y(0.62) + 18, 'calling at random', 'lbl').setAttribute('style', 'font-style:italic');
      ['rule', 'hgb', 'logreg'].forEach(function (mm) {
        var on = mm === st.m;
        svg('path', { d: curves[mm].map(function (v, i) { return (i ? 'L' : 'M') + x(i).toFixed(1) + ',' + y(v).toFixed(1); }).join(''), fill: 'none', stroke: COL[mm], 'stroke-width': on ? 3 : 1.8, 'stroke-opacity': on ? 1 : .7, 'stroke-dasharray': mm === 'rule' ? '6 3' : '' }, s);
      });
      var kx = x(st.pct);
      svg('line', { x1: kx, x2: kx, y1: m.t, y2: m.t + Hh, stroke: 'var(--ink)', 'stroke-width': 1.5 }, s);
      var labs = ['logreg', 'hgb', 'rule'].map(function (mm) { return { m: mm, v: curves[mm][st.pct], y: y(curves[mm][st.pct]) }; }).sort(function (a, b) { return a.y - b.y; });
      for (var i = 1; i < labs.length; i++) if (labs[i].y - labs[i - 1].y < 15) labs[i].ly = labs[i - 1].y + 15; else labs[i].ly = labs[i].y;
      labs[0].ly = labs[0].y;
      var right = st.pct < 55;
      labs.forEach(function (l) {
        svg('circle', { cx: kx, cy: l.y, r: 5, fill: COL[l.m], stroke: 'var(--surface)', 'stroke-width': 2 }, s);
        txt(s, kx + (right ? 10 : -10), l.ly + 4, SHORT[l.m] + ' ' + F.p0(l.v), l.m === st.m ? 'lbl-strong' : 'lbl', right ? 'start' : 'end');
      });
      var ov = svg('rect', { x: m.l, y: m.t, width: W - m.l - m.r, height: Hh, fill: 'transparent', style: 'cursor:pointer' }, s);
      function pctAt(e) { var b = ov.getBoundingClientRect(); return Math.max(1, Math.min(100, Math.round((e.clientX - b.left) / b.width * 100))); }
      ov.addEventListener('pointermove', function (e) {
        var p = pctAt(e), hb = host.getBoundingClientRect();
        tip.show('<b>Call the top ' + p + '%</b> (' + F.int(p / 100 * N) + ' clients)<br>' + ['logreg', 'hgb', 'rule'].map(function (mm) { return SHORT[mm] + ': reaches <b>' + F.p0(curves[mm][p]) + '</b> of subscribers'; }).join('<br>') + '<br><i>click to use this</i>', e.clientX - hb.left, e.clientY - hb.top);
      });
      ov.addEventListener('pointerleave', function () { tip.hide(); });
      ov.addEventListener('click', function (e) { st.pct = pctAt(e); kslider.value = st.pct; update(); });
    }
    var redrawG = mount(host, drawGains);
    var redrawS = segHost ? mount(segHost, drawSeg) : function () {};

    function drawSeg(h, W) {
      var calls = Math.round(st.pct / 100 * N), w = weights(st.m, calls), D2 = RW.dims[st.dim], agg = {};
      D2.codes.forEach(function (c, i) {
        var a = agg[c] || (agg[c] = { n: 0, sub: 0, called: 0, reached: 0 });
        a.n++; a.sub += Y[i]; a.called += w[i]; a.reached += w[i] * Y[i];
      });
      var list = Object.keys(agg).map(function (c) { var a = agg[c]; a.label = String(D2.values[c]); return a; })
        .filter(function (a) { return a.sub > 0; }).sort(function (a, b) { return b.n - a.n; }).slice(0, 10);
      var overall = hitsAt(st.m, calls) / nSub;
      hbars(h, W, list.map(function (a) {
        return { label: cap(a.label), value: a.reached / a.sub, text: F.p0(a.reached / a.sub) + ' of ' + F.int(a.sub),
          color: COL[st.m], title: cap(a.label) + ': ' + F.int(a.n) + ' clients, ' + F.int(a.sub) + ' subscribers, ' + F.int(a.called) + ' called, ' + F.int(a.reached) + ' subscribers reached' };
      }), { domain: [0, 1.0], ticks: [0, 0.25, 0.5, 0.75, 1], tickFmt: F.p0, valueW: 70, labelW: 130, rowH: 25, ref: { value: overall, label: 'overall ' + F.p0(overall) }, aria: 'Share of each group’s subscribers who are reached' });
      var note = $('#ex-seg-note');
      if (note) {
        var worst = list.filter(function (a) { return a.sub >= 40; }).sort(function (a, b) { return a.reached / a.sub - b.reached / b.sub; })[0];
        note.textContent = worst ? 'At this budget the list reaches the fewest subscribers in "' + cap(worst.label) + '" (' + F.p0(worst.reached / worst.sub) + ' of ' + F.int(worst.sub) + '). Bars show only groups with at least one subscriber, up to 10 groups by size.' : '';
      }
    }

    function update() {
      var calls = Math.round(st.pct / 100 * N), h = hitsAt(st.m, calls), prec = h / calls, rec = h / nSub;
      kout.textContent = st.pct + '% (' + F.int(calls) + ' clients)';
      kslider.setAttribute('aria-valuetext', st.pct + ' percent of clients called');
      put('ex-readout', '<div class="big">' + F.int(h) + ' subscribers reached</div><div class="sub">from ' + F.int(calls) + ' calls out of ' + F.int(N) + ' clients, ' + F.int(nSub) + ' of whom subscribed</div>' +
        '<div class="rd-grid"><div><b>' + F.p1(prec) + '</b><span>of calls end in a subscription (precision)</span></div><div><b>' + F.p1(rec) + '</b><span>of all subscribers reached (recall)</span></div>' +
        '<div><b>' + F.x1(prec / prev) + '</b><span>better than calling at random (' + F.p1(prev) + ')</span></div><div><b>' + (calls / Math.max(h, 1)).toFixed(1) + '</b><span>calls per subscriber</span></div></div>');
      var fp = calls - h, fn = nSub - h, tn = N - nSub - fp;
      put('ex-cm', '<thead><tr><th></th><th>Called</th><th>Not called</th></tr></thead><tbody><tr><th>Subscribed</th><td class="good"><b>' + F.int(h) + '</b><small>reached</small></td><td><b>' + F.int(fn) + '</b><small>missed</small></td></tr>' +
        '<tr><th>Did not</th><td><b>' + F.int(fp) + '</b><small>wasted calls</small></td><td class="good"><b>' + F.int(tn) + '</b><small>correctly left</small></td></tr></tbody>');
      redrawG(); redrawS();
    }
    kslider.addEventListener('input', function () { st.pct = +kslider.value; update(); });
    radios.forEach(function (r) { r.addEventListener('change', function () { if (r.checked) { st.m = r.value; update(); } }); });
    dimSel.addEventListener('change', function () { st.dim = dimSel.value; update(); });
    kslider.value = st.pct; update();

    var rows = []; [5, 10, 20, 30, 50, 75].forEach(function (p) { rows.push([p + '%', F.p1(curves.logreg[p]), F.p1(curves.hgb[p]), F.p1(curves.rule[p])]); });
    addTV('fig-gains', ['Clients called', 'Logistic: subscribers reached', 'Boosting: subscribers reached', 'Rule: subscribers reached'], rows, 'View selected points as a table');
  })();

  /* ---------------------------------------------------------------- move the cut-off date */
  (function () {
    var host1 = $('#chart-roll-auc'), host2 = $('#chart-roll-ap'), box = $('#roll-ctl'); if (!host1 || !box) return;
    var RO = R.rolling_origin, cur = 1;
    box.innerHTML = RO.map(function (r, i) {
      return '<label><input type="radio" name="roll" value="' + i + '"' + (i === cur ? ' checked' : '') + '><span>First ' + Math.round(r.origin * 100) + '%</span></label>';
    }).join('');
    function info() {
      var r = RO[cur];
      put('roll-info', 'Train on the first <b>' + F.int(r.train_n) + '</b> rows (' + F.p1(r.train_prev) + ' subscribed). Test on the next <b>' + F.int(r.test_n) + '</b> rows, <b>' + spanLabel(r.span) + '</b> (' + F.p1(r.test_prev) + ' subscribed).');
    }
    var d1 = mount(host1, function (h, W) {
      var r = RO[cur];
      hbars(h, W, ['logreg', 'hgb', 'rule'].map(function (m) { return { label: NAMES[m], value: r[m].auc, color: COL[m], text: F.f3(r[m].auc), title: NAMES[m] + ' ROC-AUC ' + F.f3(r[m].auc) }; }),
        { domain: [0.3, 0.8], ticks: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8], tickFmt: function (t) { return t.toFixed(1); }, labelW: 140, valueW: 40, ref: { value: 0.5, label: 'chance' }, aria: 'ROC-AUC for the chosen cut-off' });
    });
    var d2 = mount(host2, function (h, W) {
      var r = RO[cur];
      hbars(h, W, ['logreg', 'hgb', 'rule'].map(function (m) { return { label: NAMES[m], value: r[m].ap, color: COL[m], text: F.f3(r[m].ap), title: NAMES[m] + ' PR-AUC ' + F.f3(r[m].ap) }; }),
        { domain: [0, 0.8], ticks: [0, 0.2, 0.4, 0.6, 0.8], tickFmt: function (t) { return t.toFixed(1); }, labelW: 140, valueW: 40, ref: { value: r.test_prev, label: 'no skill ' + F.f3(r.test_prev) }, aria: 'PR-AUC for the chosen cut-off' });
    });
    box.addEventListener('change', function (e) { cur = +e.target.value; info(); d1(); d2(); });
    info();
    addTV('fig-roll', ['Train on first', 'Test period', 'Share subscribing (test)', 'Logistic ROC-AUC', 'Boosting ROC-AUC', 'Rule ROC-AUC', 'Logistic PR-AUC', 'Boosting PR-AUC', 'Rule PR-AUC'],
      RO.map(function (r) { return [Math.round(r.origin * 100) + '%', spanLabel(r.span), F.p1(r.test_prev), F.f3(r.logreg.auc), F.f3(r.hgb.auc), F.f3(r.rule.auc), F.f3(r.logreg.ap), F.f3(r.hgb.ap), F.f3(r.rule.ap)]; }));
  })();

  /* ---------------------------------------------------------------- time-ordered vs random split dumbbell */
  (function () {
    var host = $('#chart-dumbbell'); if (!host) return;
    var order = ['rule', 'logreg', 'hgb'];
    mount(host, function (h, W) {
      var m = { l: Math.min(140, W * .3), r: 40, t: 18 }, rh = 40, H = m.t + order.length * rh + 40;
      var s = svg('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'ROC-AUC under a time-ordered test and under a random split' }, h);
      var x = scale(0.5, 0.85, m.l, W - m.r), ax = svg('g', { class: 'ax' }, s);
      [0.5, 0.6, 0.7, 0.8].forEach(function (t) {
        svg('line', { class: 'grid', x1: x(t), x2: x(t), y1: m.t - 6, y2: m.t + order.length * rh }, ax);
        txt(ax, x(t), m.t + order.length * rh + 16, t.toFixed(1), 'lbl', 'middle');
      });
      txt(ax, (m.l + W - m.r) / 2, H - 4, 'ROC-AUC on the test rows', 'lbl', 'middle');
      order.forEach(function (mm, i) {
        var y = m.t + i * rh + rh / 2, a = M[mm].auc, b = D.rand[mm].auc;
        txt(s, m.l - 10, y + 4, NAMES[mm], 'lbl', 'end');
        svg('line', { x1: x(a), x2: x(b), y1: y, y2: y, stroke: COL[mm], 'stroke-width': 2, 'stroke-opacity': .6 }, s);
        var c1 = svg('circle', { cx: x(a), cy: y, r: 6.5, fill: COL[mm], stroke: 'var(--surface)', 'stroke-width': 2 }, s);
        var c2 = svg('circle', { cx: x(b), cy: y, r: 6, fill: 'var(--surface)', stroke: COL[mm], 'stroke-width': 3 }, s);
        svg('title', {}, c1).textContent = NAMES[mm] + ', time-ordered test: ' + a.toFixed(3);
        svg('title', {}, c2).textContent = NAMES[mm] + ', random split: ' + b.toFixed(3);
        txt(s, x(a), y - 12, a.toFixed(2), 'lbl-strong', 'middle');
        txt(s, x(b), y - 12, b.toFixed(2), 'lbl', 'middle');
      });
    });
    addTV('fig-dumbbell', ['Method', 'ROC-AUC, time-ordered test', 'ROC-AUC, random split (mean of 3)'], order.map(function (m) { return [NAMES[m], F.f3(M[m].auc), F.f3(D.rand[m].auc)]; }));
  })();

  /* ---------------------------------------------------------------- errors section */
  (function () {
    var E = R.errors, op = E.operating_point;
    put('tbl-errcm', '<thead><tr><th></th><th>Called (top ' + F.p0(op.budget) + ')</th><th>Not called</th></tr></thead><tbody><tr><th>Subscribed</th><td class="good"><b>' + F.int(op.tp) + '</b><small>reached</small></td><td><b>' + F.int(op.fn) + '</b><small>missed</small></td></tr>' +
      '<tr><th>Did not</th><td><b>' + F.int(op.fp) + '</b><small>wasted calls</small></td><td class="good"><b>' + F.int(op.tn) + '</b><small>correctly left</small></td></tr></tbody>');
    function exRows(a) {
      return a.map(function (e) {
        return [cap(e.when), cap(e.job.replace('.', ' ')), String(e.age), e.poutcome, e.contact, F.f3(e.score) + ' <small>(top ' + (100 - e.score_percentile).toFixed(0) + '%)</small>'];
      });
    }
    var H6 = ['When', 'Job', 'Age', 'Previous outcome', 'Contact', 'Score'];
    put('tbl-fp-worst', table(H6, exRows(E.worst_false_positives)));
    put('tbl-fn-worst', table(H6, exRows(E.worst_false_negatives)));
    put('tbl-fp-typ', table(H6, exRows(E.typical_false_positives)));
    put('tbl-fn-typ', table(H6, exRows(E.typical_false_negatives)));
    function segTbl(k, title) {
      return table([title, 'Clients', 'Subscribers', 'Reached', 'Reach rate', 'Precision'], E.segments[k].filter(function (s) { return s.subscribers > 0; }).map(function (s) {
        return [cap(s.label), F.int(s.n), F.int(s.subscribers), F.int(s.captured), F.p0(s.recall), s.called ? F.p0(s.precision) : '—'];
      }), { right: [1, 2, 3, 4, 5] });
    }
    put('tbl-seg-poutcome', segTbl('poutcome', 'Previous campaign'));
    put('tbl-seg-contact', segTbl('contact', 'Contact channel'));
    put('tbl-seg-year', segTbl('year', 'Year'));
    put('tbl-seg-month', segTbl('month', 'Month'));
    var ec = E.segments.poutcome.reduce(function (a, s) { a[s.label] = s; return a; }, {});
    var mo = E.segments.month.reduce(function (a, s) { a[s.label] = s; return a; }, {});
    var octTrain = R.eda.train_rates.month.filter(function (r) { return r.label === 'oct'; })[0];
    ROOT.D.err = { fn_nonexist: E.fn_share_poutcome_nonexistent, sub_nonexist: E.subscribers_nonexistent_share,
      nonexist_reach: ec.nonexistent ? ec.nonexistent.recall : 0, success_reach: ec.success ? ec.success.recall : 0, success_prec: ec.success ? ec.success.precision : 0,
      may_reach: mo.may ? mo.may.recall : 0, may_rate: mo.may ? mo.may.subscribers / mo.may.n : 0,
      oct_rate: mo.oct ? mo.oct.subscribers / mo.oct.n : 0, oct_train_n: octTrain ? octTrain.n : 0 };
    $$('[data-v^="D.err"]').forEach(function (n) { n.textContent = F[n.getAttribute('data-f') || 'p0'](get(ROOT, n.getAttribute('data-v'))); });
  })();

  /* ---------------------------------------------------------------- feature list + environment */
  put('code-versions', 'scikit-learn ' + R.meta.sklearn + ', pandas ' + R.meta.pandas + ', numpy ' + R.meta.numpy + ', seed ' + R.meta.seed);
})();
