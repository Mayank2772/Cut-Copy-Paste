// this is the app.js

let allEvents = [];
let filtered = [];
const $ = (id) => document.getElementById(id);
let activeJob = null;
let pollTimer = null;

function fmtTime(value) {
  const s = Math.max(0, Number(value) || 0);
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}.${Math.floor((s % 1) * 10)}`;
}
function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;", "'":"&#39;"}[c]));
}
function setStatus(message, state = "info", progress = null) {
  $("statusText").textContent = message;
  const pill = $("statusPill");
  pill.textContent = state === "processing" ? "Processing" : state === "completed" ? "Complete" : state === "failed" ? "Failed" : state === "uploading" ? "Uploading" : "Ready";
  pill.className = "text-xs rounded-full px-3 py-1 " + (state === "completed" ? "bg-emerald-100 text-emerald-800" : state === "failed" ? "bg-rose-100 text-rose-800" : state === "processing" || state === "uploading" ? "bg-blue-100 text-blue-800" : "bg-slate-100 text-slate-600");
  if (progress !== null) $("progressBar").style.width = `${Math.max(0, Math.min(100, progress))}%`;
}
function eventBadge(type) {
  const styles = { possible_pickup:"bg-emerald-50 text-emerald-700 border-emerald-200", hold:"bg-sky-50 text-sky-700 border-sky-200", possible_release:"bg-amber-50 text-amber-700 border-amber-200", approach:"bg-violet-50 text-violet-700 border-violet-200", contact:"bg-slate-50 text-slate-700 border-slate-200", tracking_uncertain:"bg-rose-50 text-rose-700 border-rose-200" };
  const cls = styles[type] || "bg-slate-50 text-slate-600 border-slate-200";
  return `<span class="text-[11px] px-1.5 py-0.5 rounded border ${cls}">${escapeHtml(type || "event")}</span>`;
}
function renderResults() {
  const box = $("results");
  $("resultCount").textContent = `${filtered.length} event${filtered.length === 1 ? "" : "s"}`;
  if (!filtered.length) { box.innerHTML = `<p class="p-4 text-sm text-slate-500">No matching events were found for this video.</p>`; return; }
  box.innerHTML = filtered.map((e, idx) => `<button type="button" class="event-row w-full text-left px-3 py-3 flex gap-3 items-start transition" data-idx="${idx}" data-t="${Number(e.timestamp_start) || 0}">
    <div class="shrink-0 w-16 text-sm font-mono text-blue-700 pt-0.5">${fmtTime(e.timestamp_start)}</div>
    <div class="min-w-0 flex-1"><div class="flex flex-wrap items-center gap-1.5 mb-1">${eventBadge(e.event)}<span class="text-[10px] text-slate-500">${escapeHtml(e.confidence_label || "")}</span></div>
    <div class="text-sm font-medium truncate">${escapeHtml(e.object || "unknown object")}</div>
    <div class="text-xs text-slate-500">${escapeHtml(e.hand || "unknown")} hand · ${fmtTime(e.timestamp_start)} – ${fmtTime(e.timestamp_end)}</div></div></button>`).join("");
  box.querySelectorAll(".event-row").forEach((row) => row.addEventListener("click", () => {
    box.querySelectorAll(".event-row").forEach((r) => r.classList.remove("active"));
    row.classList.add("active"); jumpTo(Number(row.dataset.t));
  }));
}
function searchEvents(query) {
  const tokens = (query || "").toLowerCase().trim().split(/\s+/).filter(Boolean);
  filtered = !tokens.length ? allEvents.slice() : allEvents.filter((e) => tokens.every((t) => [e.object,e.hand,e.event,e.confidence_label,e.source,e.event_id].join(" ").toLowerCase().includes(t)));
  renderResults();
}
function jumpTo(t) {
  const v = $("player");
  if (!v || !Number.isFinite(t)) return;
  const seek = () => { v.currentTime = Math.max(0, t); v.play().catch(() => {}); $("videoStatus").textContent = `Jumped to ${fmtTime(t)}`; };
  if (v.readyState >= 1) seek(); else v.addEventListener("loadedmetadata", seek, { once: true });
}
function showVideo(url, label) {
  const v = $("player");
  v.src = url;
  v.load();
  v.onerror = () => { $("videoStatus").textContent = "Could not play this video format in the browser"; };
  v.onloadeddata = () => { $("videoStatus").textContent = label || "Video ready"; };
}
function uploadVideo(file) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `/api/upload?filename=${encodeURIComponent(file.name)}`);
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) setStatus(`Uploading ${file.name}: ${Math.round(e.loaded / e.total * 100)}%`, "uploading", e.loaded / e.total * 100); };
    xhr.onload = () => { let data = {}; try { data = JSON.parse(xhr.responseText); } catch {} if (xhr.status >= 200 && xhr.status < 300) resolve(data); else reject(new Error(data.error || xhr.responseText || `Upload failed (${xhr.status})`)); };
    xhr.onerror = () => reject(new Error("Network error while uploading video"));
    xhr.send(file);
  });
}
async function startAnalysis() {
  const file = $("videoFile").files[0];
  if (!file) { alert("Choose a video file first."); return; }
  const allowed = /\.(mp4|mov|avi|webm|m4v)$/i;
  if (!allowed.test(file.name)) { alert("Choose an MP4, MOV, AVI, WebM or M4V file."); return; }
  $("uploadBtn").disabled = true;
  $("errorDetails").classList.add("hidden");
  allEvents = []; filtered = []; renderResults();
  setStatus("Uploading video…", "uploading", 0);
  try {
    const result = await uploadVideo(file);
    activeJob = result.job_id;
    const originalUrl = `/media/uploads/${encodeURIComponent(result.stored_name || "")}`;
    $("originalVideo").href = originalUrl;
    $("originalVideoWrap").classList.remove("hidden");
    setStatus("Upload complete. Starting ML analysis…", "processing", 3);
    pollJob();
  } catch (err) {
    setStatus(err.message, "failed", 0);
    $("uploadBtn").disabled = false;
  }
}
async function pollJob() {
  if (!activeJob) return;
  try {
    const res = await fetch(`/api/status/${activeJob}`, { cache: "no-store" });
    const job = await res.json();
    if (!res.ok) throw new Error(job.error || "Could not read processing status");
    if (job.status === "queued" || job.status === "processing") {
      setStatus(`${job.message || "Processing video"}. This can take a while for long clips.`, "processing", 10);
      pollTimer = setTimeout(pollJob, 2500);
      return;
    }
    if (job.status === "failed") {
      setStatus(job.message || "Processing failed", "failed", 0);
      if (job.log_available) {
        const logRes = await fetch(`/api/log/${activeJob}`, { cache: "no-store" });
        const logData = await logRes.json();
        $("errorDetails").classList.remove("hidden");
        $("errorLog").textContent = logData.log || "No pipeline log was recorded.";
      }
      $("uploadBtn").disabled = false;
      return;
    }
    if (job.status === "completed") {
      setStatus(job.message || "Analysis complete", "completed", 100);
      if (job.output_video) showVideo(job.output_video, "Processed output ready");
      const eventRes = await fetch(job.events || `/api/events/${activeJob}`, { cache: "no-store" });
      allEvents = eventRes.ok ? await eventRes.json() : [];
      if (!Array.isArray(allEvents)) allEvents = [];
      searchEvents($("searchBox").value);
      $("uploadBtn").disabled = false;
      if (!job.output_video) $("videoStatus").textContent = "No rendered output video was generated";
    }
  } catch (err) {
    setStatus(err.message, "failed");
    $("uploadBtn").disabled = false;
  }
}

$("uploadBtn").addEventListener("click", startAnalysis);
$("searchBox").addEventListener("input", (e) => searchEvents(e.target.value));
document.querySelectorAll(".chip").forEach((btn) => btn.addEventListener("click", () => { $("searchBox").value = btn.dataset.q; searchEvents(btn.dataset.q); }));
