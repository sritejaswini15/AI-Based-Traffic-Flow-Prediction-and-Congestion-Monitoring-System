// Dashboard: fetches one snapshot per replay moment and renders it.
// All text goes in through textContent; no HTML is built from data.
(function () {
  "use strict";

  const root = document.getElementById("dashboard");
  if (!root) return;

  const SNAPSHOT_URL = root.dataset.snapshotUrl;
  const CSV_URL = root.dataset.csvUrl; // ".../0.csv": the day is substituted
  const FIRST_DAY = Number(root.dataset.firstDay);
  const LAST_DAY = Number(root.dataset.lastDay);
  const STEP_MIN = 5;
  const DAY_MIN = 24 * 60;
  const LIVE_POLL_MS = 30000;
  const PLAY_MS = 1200;
  const LEVEL_TEXT = { low: "Low", moderate: "Moderate", heavy: "Heavy" };

  const $ = (id) => document.getElementById(id);
  const ctl = {
    form: $("controls"), day: $("ctl-day"), time: $("ctl-time"), back: $("ctl-back"), fwd: $("ctl-fwd"),
    play: $("ctl-play"), live: $("ctl-live"), junction: $("ctl-junction"), outcomes: $("ctl-outcomes"), csv: $("ctl-csv"),
  };
  const body = $("dash-body");
  const nf = new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 });

  const state = { mode: "live", day: FIRST_DAY, minutes: 0, focus: 0, outcomes: false, playing: false };
  let chart = null;
  let pollTimer = null;
  let playTimer = null;
  let inflight = null;
  let last = null;

  // ------------------------------------------------------------ helpers
  const fmt = (v) => (v === null || v === undefined ? "–" : nf.format(v));
  const pad = (n) => String(n).padStart(2, "0");
  const clockText = (minutes) => pad(Math.floor(minutes / 60)) + ":" + pad(minutes % 60);
  const toMinutes = (text) => {
    const m = /^(\d{2}):(\d{2})$/.exec(text || "");
    if (!m) return null;
    const total = Number(m[1]) * 60 + Number(m[2]);
    return total < DAY_MIN ? total - (total % STEP_MIN) : null;
  };
  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const withAlpha = (hex, alpha) => {
    const h = hex.replace("#", "");
    if (h.length !== 6) return hex;
    const n = parseInt(h, 16);
    return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
  };
  const make = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  };
  const levelBadge = (level) => {
    const badge = make("span", "level level--" + (level || "none"));
    badge.append(make("span", "level__icon"), document.createTextNode(LEVEL_TEXT[level] || "No reading"));
    return badge;
  };
  const changeText = (to, from) => {
    if (to === null || from === null || from === 0) return "";
    const pct = Math.round(((to - from) / from) * 100);
    if (Math.abs(pct) < 3) return "about the same as now";
    return (pct > 0 ? "▲ " : "▼ ") + Math.abs(pct) + "% " + (pct > 0 ? "more" : "fewer") + " than now";
  };

  // ---------------------------------------------------------- URL state
  function readUrl() {
    const p = new URLSearchParams(location.search);
    const day = Number(p.get("day"));
    const minutes = toMinutes(p.get("time"));
    const focus = Number(p.get("junction"));
    if (Number.isInteger(focus) && focus >= 0 && focus < ctl.junction.options.length) state.focus = focus;
    if (Number.isInteger(day) && day >= FIRST_DAY && day <= LAST_DAY && minutes !== null) {
      state.mode = "fixed";
      state.day = day;
      state.minutes = minutes;
    }
    state.outcomes = p.get("outcomes") === "1";
  }

  function writeUrl() {
    const p = new URLSearchParams();
    if (state.mode === "fixed") {
      p.set("day", state.day);
      p.set("time", clockText(state.minutes));
    }
    if (state.focus) p.set("junction", state.focus);
    if (state.outcomes) p.set("outcomes", "1");
    const qs = p.toString();
    history.replaceState(null, "", qs ? "?" + qs : location.pathname);
  }

  // ------------------------------------------------------------ loading
  async function load() {
    const p = new URLSearchParams({ junction: state.focus });
    if (state.mode === "fixed") {
      p.set("day", state.day);
      p.set("time", clockText(state.minutes));
    }
    if (state.outcomes) p.set("outcomes", "1");
    writeUrl();

    if (inflight) inflight.abort();
    const request = new AbortController();
    inflight = request;
    body.classList.add("is-loading"); // keep the previous frame, dimmed
    try {
      const res = await fetch(SNAPSHOT_URL + "?" + p.toString(), {
        headers: { Accept: "application/json" },
        credentials: "same-origin",
        signal: request.signal,
      });
      if (res.status === 401) {
        location.reload(); // session ended: the server sends the sign-in page
        return;
      }
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || "the server answered " + res.status);
      last = data;
      render(data);
      showError(null);
    } catch (err) {
      if (err.name === "AbortError") return;
      showError("Could not load forecasts (" + err.message + "). " +
        (state.mode === "live" ? "Retrying automatically." : "Try again in a moment."));
    } finally {
      if (inflight === request) {
        inflight = null;
        body.classList.remove("is-loading");
      }
    }
  }

  function showError(message) {
    const box = $("load-error");
    box.hidden = !message;
    box.textContent = message || "";
  }

  // ----------------------------------------------------------- controls
  function setPlaying(on) {
    state.playing = on;
    ctl.play.setAttribute("aria-pressed", String(on));
    ctl.play.textContent = on ? "Pause" : "Play";
    clearInterval(playTimer);
    playTimer = on ? setInterval(() => shift(STEP_MIN, true), PLAY_MS) : null;
  }

  function setMode(mode) {
    state.mode = mode;
    clearInterval(pollTimer);
    pollTimer = null;
    if (mode === "live") {
      setPlaying(false);
      pollTimer = setInterval(() => { if (!document.hidden) load(); }, LIVE_POLL_MS);
    }
  }

  function freezeAtCurrent() {
    if (state.mode === "live" && last) {
      state.day = last.day;
      state.minutes = toMinutes(last.clock);
    }
    setMode("fixed");
  }

  function shift(deltaMinutes, fromPlay) {
    if (!fromPlay) setPlaying(false);
    freezeAtCurrent();
    let day = state.day;
    let minutes = state.minutes + deltaMinutes;
    if (minutes < 0) { day -= 1; minutes += DAY_MIN; }
    if (minutes >= DAY_MIN) { day += 1; minutes -= DAY_MIN; }
    if (day < FIRST_DAY) { day = FIRST_DAY; minutes = 0; }
    if (day > LAST_DAY) {
      day = LAST_DAY;
      minutes = DAY_MIN - STEP_MIN;
      setPlaying(false); // the replay has ended
    }
    state.day = day;
    state.minutes = minutes;
    load();
  }

  ctl.form.addEventListener("submit", (e) => e.preventDefault());
  ctl.day.addEventListener("change", () => {
    freezeAtCurrent();
    state.day = Number(ctl.day.value);
    load();
  });
  ctl.time.addEventListener("change", () => {
    const minutes = toMinutes(ctl.time.value);
    if (minutes === null) return;
    setPlaying(false);
    freezeAtCurrent();
    state.minutes = minutes;
    load();
  });
  ctl.back.addEventListener("click", () => shift(-STEP_MIN));
  ctl.fwd.addEventListener("click", () => shift(STEP_MIN));
  ctl.play.addEventListener("click", () => {
    if (state.playing) { setPlaying(false); renderStatus(); return; }
    freezeAtCurrent();
    setPlaying(true);
    renderStatus();
  });
  ctl.live.addEventListener("click", () => { setMode("live"); load(); });
  ctl.junction.addEventListener("change", () => { state.focus = Number(ctl.junction.value); load(); });
  ctl.outcomes.addEventListener("change", () => { state.outcomes = ctl.outcomes.checked; load(); });
  $("junction-grid").addEventListener("click", (e) => {
    const card = e.target.closest(".junction");
    if (!card) return;
    state.focus = Number(card.dataset.index);
    load();
  });
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && state.mode === "live") load();
  });

  // ---------------------------------------------------------- rendering
  function render(d) {
    renderStatus(d);
    renderKpis(d);
    renderJunctions(d);
    renderFocus(d);
  }

  function renderStatus(d) {
    d = d || last;
    if (!d) return;
    const pill = $("mode-pill");
    const live = state.mode === "live";
    pill.textContent = live ? "Live replay" : state.playing ? "Playing" : "Paused";
    pill.classList.toggle("pill--paused", !live && !state.playing);
    $("replay-label").textContent = "Day " + d.day + " · " + d.clock;
    $("replay-note").textContent = live
      ? "Recorded counts from held-out day " + d.day + ", in step with your clock. Not a live sensor feed."
      : "Recorded counts from held-out day " + d.day + ". Choose “Follow clock” to return to the live replay.";
    ctl.live.disabled = live;
    if (document.activeElement !== ctl.time) ctl.time.value = d.clock;
    ctl.day.value = String(d.day);
    ctl.junction.value = String(state.focus);
    ctl.outcomes.checked = state.outcomes;
    ctl.csv.href = CSV_URL.replace(/\/0\.csv$/, "/" + d.day + ".csv");
    ctl.csv.textContent = "Download day " + d.day + " as CSV";
  }

  function renderKpis(d) {
    const net = d.network;
    const longest = String(d.horizons_minutes[d.horizons_minutes.length - 1]);
    const n = d.junctions.length;

    $("kpi-now").textContent = net.now === null ? "No reading" : fmt(net.now);
    $("kpi-now-sub").textContent = net.now === null
      ? "A sensor outage at this moment"
      : "per 5-minute reading · typical at " + d.clock + ": " + fmt(net.typical_now);

    const ahead = net.forecast[longest];
    $("kpi-60").textContent = fmt(ahead);
    $("kpi-60-sub").textContent = changeText(ahead, net.now) || "across all junctions";

    $("kpi-heavy").textContent = net.heavy_now + " of " + n;
    $("kpi-heavy-sub").textContent = "In " + longest + " min: " + net.heavy_in[longest] + " of " + n + " forecast heavy";

    const busiest = net.busiest ? d.junctions[net.busiest.index] : null;
    $("kpi-busiest").textContent = busiest ? busiest.name : "–";
    $("kpi-busiest-sub").textContent = busiest
      ? fmt(busiest.now) + " vehicles · busier than " + fmt(busiest.percentile) + "% of its readings"
      : "No readings at this moment";
  }

  function renderJunctions(d) {
    const longestIndex = d.horizons_minutes.length - 1;
    const minutes = d.horizons_minutes[longestIndex];
    document.querySelectorAll("#junction-grid .junction").forEach((card) => {
      const j = d.junctions[Number(card.dataset.index)];
      if (!j) return;
      const next = j.forecasts[longestIndex];
      const selected = j.index === state.focus;
      card.setAttribute("aria-pressed", String(selected));
      card.replaceChildren();

      const now = make("span", "junction__now", j.now === null ? "–" : fmt(j.now));
      if (j.now !== null) now.append(make("small", null, "vehicles"));
      const meter = make("span", "meter meter--" + (j.level || "none"));
      const fill = make("span", "meter__fill");
      fill.style.width = (j.percentile || 0) + "%";
      meter.append(fill);
      const nextLine = make("span", "junction__next", "In " + minutes + " min: ");
      nextLine.append(make("strong", null, fmt(next.value)), document.createTextNode(" · " + LEVEL_TEXT[next.level]));

      card.append(
        make("span", "junction__name", j.name),
        now,
        levelBadge(j.level),
        meter,
        make("span", "meter__caption", j.percentile === null ? "No reading" : "Busier than " + fmt(j.percentile) + "% of its readings"),
        nextLine,
      );
      card.setAttribute("aria-label",
        j.name + ": " + (j.now === null ? "no reading" : fmt(j.now) + " vehicles, " + LEVEL_TEXT[j.level]) +
        ". In " + minutes + " minutes: " + fmt(next.value) + ", " + LEVEL_TEXT[next.level] + "." +
        (selected ? " Shown in the chart." : " Select to chart it."));
    });
  }

  function renderFocus(d) {
    const f = d.focus;
    const j = d.junctions[f.index];
    $("focus-title").textContent = j.name;
    $("focus-accuracy").replaceChildren();
    if (f.today_mae === null) {
      $("focus-accuracy").textContent = "Today's accuracy appears once the first forecast can be checked.";
    } else {
      $("focus-accuracy").append(
        document.createTextNode(f.earlier_minutes + "-minute forecasts so far on day " + d.day + ": off by "),
        make("strong", null, fmt(f.today_mae) + " vehicles"),
        document.createTextNode(" on average, over " + f.today_points + " readings."),
      );
    }
    renderForecastTable(j);
    renderChartTable(f);
    renderChart(f);
  }

  function renderForecastTable(j) {
    const table = $("forecast-table");
    const hasOutcomes = state.outcomes && j.forecasts.some((x) => x.actual !== undefined);
    table.classList.toggle("show-outcomes", hasOutcomes);
    const rows = j.forecasts.map((x) => {
      const tr = make("tr");
      const th = make("th", null, "In " + x.minutes + " min ");
      th.scope = "row";
      th.append(make("span", "muted", "(" + x.target_clock + ")"));
      const level = make("td");
      level.append(levelBadge(x.level));
      const actual = x.actual === undefined || x.actual === null ? null : x.actual;
      const error = actual === null ? "–" : (x.value - actual >= 0 ? "+" : "−") + fmt(Math.abs(x.value - actual));
      tr.append(th, make("td", null, fmt(x.value)), make("td", null, fmt(x.lower) + "–" + fmt(x.upper)), level,
        make("td", "outcome-col", fmt(actual)), make("td", "outcome-col", error));
      return tr;
    });
    table.tBodies[0].replaceChildren(...rows);
  }

  function renderChartTable(f) {
    const rows = f.labels.map((label, i) => {
      const tr = make("tr");
      if (i === f.now_index) tr.className = "is-now";
      const th = make("th", null, label + (i === f.now_index ? " (now)" : ""));
      th.scope = "row";
      const ahead = i > f.now_index ? f.ahead[i] : null;
      const range = ahead === null ? "–" : fmt(f.ahead_lower[i]) + "–" + fmt(f.ahead_upper[i]);
      tr.append(th, make("td", null, fmt(f.recorded[i])), make("td", null, fmt(f.earlier_forecast[i])),
        make("td", null, fmt(ahead)), make("td", null, range), make("td", null, fmt(f.typical[i])));
      return tr;
    }).filter((tr, i) => f.recorded[i] !== null || f.ahead[i] !== null);
    $("chart-table").tBodies[0].replaceChildren(...rows);
  }

  // -------------------------------------------------------------- chart
  function colors() {
    return {
      recorded: cssVar("--series-recorded"), ahead: cssVar("--series-ahead"), earlier: cssVar("--series-earlier"),
      typical: cssVar("--series-typical"), text: cssVar("--text"), text2: cssVar("--text-2"), grid: cssVar("--grid"),
      axis: cssVar("--axis"), surface: cssVar("--surface"), border: cssVar("--border-strong"),
    };
  }

  const markers = {
    id: "markers",
    afterDatasetsDraw(c) {
      const { ctx, chartArea, scales } = c;
      const nowIndex = c.$nowIndex;
      const col = c.$colors;
      if (nowIndex === undefined || !col) return;
      ctx.save();
      const x = Math.round(scales.x.getPixelForValue(nowIndex)) + 0.5;
      ctx.strokeStyle = col.text2;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, chartArea.top);
      ctx.lineTo(x, chartArea.bottom);
      ctx.stroke();
      ctx.fillStyle = col.text2;
      ctx.font = "600 12px system-ui, sans-serif";
      ctx.textAlign = "right";
      ctx.fillText("Now", x - 6, chartArea.top + 12);
      const active = c.tooltip && c.tooltip.getActiveElements();
      if (active && active.length) {
        const cx = Math.round(active[0].element.x) + 0.5;
        ctx.strokeStyle = col.axis;
        ctx.beginPath();
        ctx.moveTo(cx, chartArea.top);
        ctx.lineTo(cx, chartArea.bottom);
        ctx.stroke();
      }
      ctx.restore();
    },
  };

  function buildChart() {
    if (!window.Chart) return null;
    const col = colors();
    const line = { borderWidth: 2, pointRadius: 0, pointHoverRadius: 4, borderCapStyle: "round", borderJoinStyle: "round", tension: 0.25 };
    const instance = new window.Chart($("focus-chart"), {
      type: "line",
      data: {
        labels: [],
        datasets: [
          { ...line, key: "typical", label: "Typical for this time of day", data: [], borderColor: col.typical, borderDash: [2, 4], order: 4 },
          { ...line, key: "earlier", label: "Forecast made 60 min before", data: [], borderColor: col.earlier, borderDash: [6, 4], order: 3 },
          { ...line, key: "recorded", label: "Recorded", data: [], borderColor: col.recorded, order: 1 },
          { key: "upper", label: "80% range, upper", data: [], borderWidth: 0, pointRadius: 0, pointHoverRadius: 0, fill: "+1", spanGaps: true, backgroundColor: withAlpha(col.ahead, 0.16), order: 5, tension: 0 },
          { key: "lower", label: "80% range, lower", data: [], borderWidth: 0, pointRadius: 0, pointHoverRadius: 0, fill: false, spanGaps: true, order: 5, tension: 0 },
          {
            ...line, key: "ahead", label: "Forecast now", data: [], borderColor: col.ahead, spanGaps: true, tension: 0, order: 0,
            pointRadius: (ctx) => (ctx.dataIndex === ctx.chart.$nowIndex || ctx.raw === null ? 0 : 4),
            pointHoverRadius: 5, pointBackgroundColor: col.ahead, pointBorderColor: col.surface, pointBorderWidth: 2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: "index", intersect: false },
        layout: { padding: { top: 4 } },
        scales: {
          x: {
            grid: { display: false },
            border: { color: col.axis },
            ticks: { color: col.text2, autoSkip: true, maxTicksLimit: 9, maxRotation: 0 },
          },
          y: {
            beginAtZero: true,
            grid: { color: col.grid },
            border: { display: false },
            ticks: { color: col.text2, precision: 0, maxTicksLimit: 6 },
            title: { display: true, text: "Vehicles per reading", color: col.text2 },
          },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: col.surface,
            borderColor: col.border,
            borderWidth: 1,
            titleColor: col.text2,
            bodyColor: col.text,
            padding: 10,
            boxWidth: 14,
            boxHeight: 2,
            itemSort: (a, b) => a.dataset.order - b.dataset.order,
            filter: (item) =>
              item.raw !== null && item.raw !== undefined &&
              item.dataset.key !== "upper" && item.dataset.key !== "lower" &&
              !(item.dataset.key === "ahead" && item.dataIndex === item.chart.$nowIndex),
            callbacks: {
              label: (item) => {
                const value = fmt(item.raw);
                if (item.dataset.key !== "ahead") return value + "  " + item.dataset.label;
                const data = item.chart.data.datasets;
                const lo = data[4].data[item.dataIndex];
                const hi = data[3].data[item.dataIndex];
                return value + "  Forecast now (80% range " + fmt(lo) + "–" + fmt(hi) + ")";
              },
              labelColor: (item) => ({ borderColor: item.dataset.borderColor, backgroundColor: item.dataset.borderColor }),
            },
          },
        },
      },
      plugins: [markers],
    });
    instance.$colors = col;
    return instance;
  }

  function renderChart(f) {
    if (!chart) chart = buildChart();
    if (!chart) return;
    const upper = f.ahead_upper.slice();
    const lower = f.ahead_lower.slice();
    upper[f.now_index] = lower[f.now_index] = f.recorded[f.now_index]; // the range opens from now
    const series = { typical: f.typical, earlier: f.earlier_forecast, recorded: f.recorded, upper, lower, ahead: f.ahead };
    chart.$nowIndex = f.now_index;
    chart.data.labels = f.labels;
    chart.data.datasets.forEach((ds) => { ds.data = series[ds.key]; });
    chart.data.datasets[1].label = "Forecast made " + f.earlier_minutes + " min before";
    chart.update();
  }

  function rebuildChart() {
    if (chart) { chart.destroy(); chart = null; }
    if (last) renderChart(last.focus);
  }
  document.addEventListener("themechange", rebuildChart);
  if (window.matchMedia) {
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", rebuildChart);
  }

  // --------------------------------------------------------------- start
  readUrl();
  setMode(state.mode);
  ctl.junction.value = String(state.focus);
  ctl.outcomes.checked = state.outcomes;
  load();
})();
