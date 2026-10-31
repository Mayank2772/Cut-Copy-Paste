/**
 * Factory Memory UI
 * - Loads ../outputs/<clip>/events.json
 * - Plays video from a path relative to the frontend (usually ../egohack-data/...)
 * - Search + jump-to-timestamp
 */
let allEvents = [];
let filtered = [];

const $ = (id) => document.getElementById(id);

function fmtTime(s) {
  s = Math.max(0, Number(s) || 0);
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  const ds = Math.floor((s % 1) * 10);
  return `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}.${ds}`;
}

function confBadge(label) {
  const map = {
    High: "bg-emerald-100 text-emerald-800",
    Medium: "bg-amber-100 text-amber-800",
    Low: "bg-rose-100 text-rose-800",
  };
  const cls = map[label] || "bg-slate-100 text-slate-700";
  return `<span class="text-[10px] font-medium px-1.5 py-0.5 rounded ${cls}">${label || "—"}</span>`;
}

function eventBadge(type) {
  const map = {
    possible_pickup: "bg-emerald-50 text-emerald-700 border-emerald-200",
    hold: "bg-sky-50 text-sky-700 border-sky-200",
    possible_release: "bg-amber-50 text-amber-700 border-amber-200",
    approach: "bg-violet-50 text-violet-700 border-violet-200",
    contact: "bg-slate-50 text-slate-700 border-slate-200",
    tracking_uncertain: "bg-rose-50 text-rose-700 border-rose-200",
  };
  const cls = map[type] || "bg-slate-50 text-slate-600 border-slate-200";
  return `<span class="text-[11px] px-1.5 py-0.5 rounded border ${cls}">${type}</span>`;
}

function renderResults() {
  const box = $("results");
  $("resultCount").textContent = `${filtered.length} event${filtered.length === 1 ? "" : "s"}`;

  if (!filtered.length) {
    box.innerHTML = `<p class="p-4 text-sm text-slate-500">No matching events.</p>`;
    return;
  }

  box.innerHTML = filtered
    .map(
      (e, idx) => `
    <button type="button"
            class="event-row w-full text-left px-3 py-2.5 flex gap-3 items-start transition"
            data-idx="${idx}"
            data-t="${e.timestamp_start}">
      <div class="shrink-0 w-14 text-sm font-mono text-blue-700 pt-0.5">
        ${fmtTime(e.timestamp_start)}
      </div>
      <div class="min-w-0 flex-1">
        <div class="flex flex-wrap items-center gap-1.5 mb-0.5">
          ${eventBadge(e.event)}
          ${confBadge(e.confidence_label)}
        </div>
        <div class="text-sm font-medium truncate">${e.object || "unknown"}</div>
        <div class="text-xs text-slate-500">
          ${e.hand || "?"} hand · ${fmtTime(e.timestamp_start)} → ${fmtTime(e.timestamp_end)}
        </div>
      </div>
    </button>`
    )
    .join("");

  box.querySelectorAll(".event-row").forEach((row) => {
    row.addEventListener("click", () => {
      box.querySelectorAll(".event-row").forEach((r) => r.classList.remove("active"));
      row.classList.add("active");
      jumpTo(Number(row.dataset.t));
    });
  });
}

function searchEvents(query) {
  const q = (query || "").toLowerCase().trim();
  if (!q) {
    filtered = allEvents.slice();
    renderResults();
    return;
  }
  const tokens = q.split(/\s+/).filter(Boolean);
  filtered = allEvents.filter((e) => {
    const hay = [
      e.object,
      e.hand,
      e.event,
      e.confidence_label,
      e.source,
      e.event_id,
    ]
      .join(" ")
      .toLowerCase();
    return tokens.every((t) => hay.includes(t));
  });
  renderResults();
}

function jumpTo(t) {
  const v = $("player");
  if (!v || !Number.isFinite(t)) return;
  const go = () => {
    v.currentTime = Math.max(0, t);
    v.play().catch(() => {});
    $("videoStatus").textContent = `Jumped to ${fmtTime(t)}`;
  };
  if (v.readyState >= 1) go();
  else v.addEventListener("loadedmetadata", go, { once: true });
}

function showPathHint(msg) {
  const el = $("pathHint");
  if (!msg) {
    el.classList.add("hidden");
    el.textContent = "";
    return;
  }
  el.textContent = msg;
  el.classList.remove("hidden");
}

function setVideo(path) {
  const v = $("player");
  let src = (path || $("videoPath").value || "").trim();
  if (!src) return;

  // Normalise: if user pastes absolute path or missing ../, fix common cases
  if (src.startsWith("egohack-data/")) src = "../" + src;
  if (src.startsWith("/Users/") || src.startsWith("/home/")) {
    // Absolute OS path won't work in the browser — convert to relative if possible
    const marker = "egohack-data/";
    const i = src.indexOf(marker);
    if (i >= 0) src = "../" + src.slice(i);
    else {
      showPathHint("Use a path relative to the frontend folder, e.g. ../egohack-data/clips_480p/your.mp4");
    }
  }

  $("videoPath").value = src;
  v.src = src;
  v.load();

  v.onerror = () => {
    $("videoStatus").textContent = "Video failed to load";
    showPathHint(
      "Video not found at: " + src +
      " — put the file under egohack-data/clips_480p/ and use ../egohack-data/clips_480p/<name>.mp4"
    );
  };
  v.onloadeddata = () => {
    $("videoStatus").textContent = "Video ready · " + src;
    showPathHint("");
  };
}

async function loadEvents() {
  const clip = $("clipName").value.trim().replace(/\.mp4$/i, "");
  if (!clip) {
    alert("Enter a clip folder name (outputs/<clip>/ without .mp4)");
    return;
  }
  $("clipName").value = clip;
  const url = `../outputs/${clip}/events.json`;
  $("resultCount").textContent = "Loading…";
  try {
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(
        `HTTP ${res.status} — missing ${url}\n\n` +
        `1) Run: python run.py events egohack-data/clips_480p/${clip}.mp4\n` +
        `2) Clip folder name must be exactly: ${clip}`
      );
    }
    allEvents = await res.json();
    if (!Array.isArray(allEvents)) allEvents = [];
    filtered = allEvents.slice();
    renderResults();
    $("videoStatus").textContent = `Loaded ${allEvents.length} events from ${clip}`;

    // Auto video path
    const defaultVid = `../egohack-data/clips_480p/${clip}.mp4`;
    $("videoPath").value = defaultVid;
    setVideo(defaultVid);
  } catch (err) {
    allEvents = [];
    filtered = [];
    renderResults();
    $("resultCount").textContent = "Failed to load";
    alert(String(err.message || err));
  }
}

$("loadBtn").addEventListener("click", loadEvents);
$("setVideoBtn").addEventListener("click", () => setVideo());
$("searchBox").addEventListener("input", (e) => searchEvents(e.target.value));
document.querySelectorAll(".chip").forEach((btn) => {
  btn.addEventListener("click", () => {
    $("searchBox").value = btn.dataset.q;
    searchEvents(btn.dataset.q);
  });
});

// Auto-load default clip on open
window.addEventListener("DOMContentLoaded", () => {
  loadEvents();
});