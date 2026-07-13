// ==========================================================================
// Sentinel — frontend logic. Every button below calls a real Flask endpoint.
// ==========================================================================
(function () {
  "use strict";

  const API = {
    status: "/api/model-status",
    upload: "/api/upload-dataset",
    train: "/api/train",
    predict: "/api/predict",
    batch: "/api/predict-batch",
    history: "/api/history",
    clearHistory: "/api/history/clear",
  };

  const state = {
    uploadedFilename: null,
    batchFile: null,
    lastBatchResults: null,
  };

  // ---------------- utilities ----------------
  const $ = (id) => document.getElementById(id);
  const scanBar = $("scanProgress");

  function startProgress() {
    scanBar.classList.add("active");
    scanBar.style.width = "20%";
    setTimeout(() => { if (scanBar.classList.contains("active")) scanBar.style.width = "65%"; }, 250);
  }
  function endProgress() {
    scanBar.style.width = "100%";
    setTimeout(() => {
      scanBar.classList.remove("active");
      scanBar.style.width = "0%";
    }, 300);
  }

  async function api(url, opts) {
    startProgress();
    try {
      const res = await fetch(url, opts);
      let data;
      try { data = await res.json(); } catch (e) { data = {}; }
      if (!res.ok) {
        const err = new Error(data.error || `Request failed (${res.status})`);
        err.status = res.status;
        throw err;
      }
      return data;
    } finally {
      endProgress();
    }
  }

  let toastTimer = null;
  function toast(message, isError) {
    const el = $("toast");
    el.textContent = message;
    el.classList.toggle("error", !!isError);
    el.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.remove("show"), 3600);
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function featureRows(features) {
    const labelMap = {
      url_count: "URLs found",
      exclamation_count: "Exclamations",
      caps_word_count: "ALL-CAPS words",
      length: "Character length",
      word_count: "Word count",
      avg_word_length: "Avg. word length",
      keyword_hit_count: "Spam keywords",
      digit_count: "Digit tokens",
    };
    return Object.keys(labelMap)
      .map((k) => `<div class="feature-item"><span class="fk">${labelMap[k]}</span><span class="fv">${features[k]}</span></div>`)
      .join("");
  }

  function renderVerdict(badgeEl, confEl, result) {
    badgeEl.textContent = result.is_spam ? "Spam" : "Legitimate";
    badgeEl.className = "verdict-badge " + (result.is_spam ? "spam" : "legit");
    confEl.textContent = `${result.confidence}% confidence`;
  }

  // ---------------- reveal-on-scroll ----------------
  function initReveal() {
    const targets = document.querySelectorAll(".section-inner > *, .step, .card");
    targets.forEach((t) => t.classList.add("reveal"));
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("in-view");
            io.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.1, rootMargin: "0px 0px -60px 0px" }
    );
    targets.forEach((t) => io.observe(t));
  }

  // ---------------- model status / dashboard / hero stats ----------------
  async function refreshStatus() {
    try {
      const data = await api(API.status);
      const ready = data.ready;
      const m = data.metrics;

      $("dashStatus").textContent = ready ? "Trained & ready" : "Untrained";
      $("dashAlgo").textContent = m ? (m.algorithm === "nb" ? "Naive Bayes" : "Logistic Regression") : "—";
      $("dashVocab").textContent = m ? m.vocabulary_size.toLocaleString() : "—";
      $("dashTotal").textContent = m ? m.total_size.toLocaleString() : "—";
      $("dashSplit").textContent = m ? `${m.spam_count} / ${m.ham_count}` : "—";
      $("dashF1").textContent = m ? m.f1_score : "—";

      document.querySelector('[data-stat="accuracy"]').textContent = m ? `${(m.accuracy * 100).toFixed(1)}%` : "—";
      document.querySelector('[data-stat="total"]').textContent = m ? m.total_size.toLocaleString() : "—";
      document.querySelector('[data-stat="status"]').textContent = ready ? "Ready" : "Untrained";

      if (m) renderMetricsCard(m);
      updateActionAvailability(ready);
      return ready;
    } catch (e) {
      toast("Could not reach the server. Is app.py running?", true);
      return false;
    }
  }

  function updateActionAvailability(ready) {
    $("detectBtn").disabled = false; // still allow click -> shows clear error if untrained
    $("heroScanBtn").disabled = false;
  }

  function renderMetricsCard(m) {
    $("metricsEmpty").hidden = true;
    $("metricsBlock").hidden = false;
    $("mAccuracy").textContent = `${(m.accuracy * 100).toFixed(1)}%`;
    $("mPrecision").textContent = `${(m.precision * 100).toFixed(1)}%`;
    $("mRecall").textContent = `${(m.recall * 100).toFixed(1)}%`;
    $("mF1").textContent = `${(m.f1_score * 100).toFixed(1)}%`;

    const cm = m.confusion_matrix;
    $("confusionTable").innerHTML = `
      <tr><th></th><th>Predicted legit</th><th>Predicted spam</th></tr>
      <tr><th>Actual legit</th><td>${cm[0][0]}</td><td>${cm[0][1]}</td></tr>
      <tr><th>Actual spam</th><td>${cm[1][0]}</td><td>${cm[1][1]}</td></tr>
    `;
  }

  // ---------------- dataset upload ----------------
  function initDatasetUpload() {
    const dropzone = $("dropzone");
    const fileInput = $("fileInput");
    const statusEl = $("dropzoneStatus");

    const openPicker = () => fileInput.click();
    dropzone.addEventListener("click", openPicker);
    dropzone.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openPicker(); }
    });

    ["dragover", "dragenter"].forEach((evt) =>
      dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add("dragover"); })
    );
    ["dragleave", "drop"].forEach((evt) =>
      dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); })
    );
    dropzone.addEventListener("drop", (e) => {
      const file = e.dataTransfer.files[0];
      if (file) handleDatasetFile(file);
    });
    fileInput.addEventListener("change", () => {
      if (fileInput.files[0]) handleDatasetFile(fileInput.files[0]);
    });

    async function handleDatasetFile(file) {
      if (!file.name.toLowerCase().endsWith(".csv")) {
        statusEl.textContent = "Only .csv files are supported.";
        statusEl.style.color = "var(--brick)";
        return;
      }
      statusEl.textContent = "Uploading…";
      statusEl.style.color = "var(--ink-soft)";

      const formData = new FormData();
      formData.append("file", file);
      try {
        const data = await api(API.upload, { method: "POST", body: formData });
        state.uploadedFilename = data.filename;
        statusEl.textContent = `Uploaded: ${file.name} (${data.row_count.toLocaleString()} rows)`;
        statusEl.style.color = "var(--teal)";
        renderDatasetPreview(data);
        $("sourcePill").textContent = `${file.name} (${data.row_count.toLocaleString()} rows)`;
        toast("Dataset uploaded. Ready to train.");
      } catch (e) {
        statusEl.textContent = e.message;
        statusEl.style.color = "var(--brick)";
        toast(e.message, true);
      }
    }
  }

  function renderDatasetPreview(data) {
    $("datasetPreviewEmpty").hidden = true;
    const wrap = $("datasetPreviewTable");
    wrap.hidden = false;
    const cols = data.columns;
    let html = "<table><tr>" + cols.map((c) => `<th>${escapeHtml(c)}</th>`).join("") + "</tr>";
    data.preview.forEach((row) => {
      html += "<tr>" + cols.map((c) => `<td>${escapeHtml(String(row[c] ?? "")).slice(0, 80)}</td>`).join("") + "</tr>";
    });
    html += "</table>";
    wrap.innerHTML = html;
    $("datasetRowCount").textContent = `${data.row_count.toLocaleString()} rows detected`;
  }

  // ---------------- train ----------------
  function initTrain() {
    $("trainBtn").addEventListener("click", async () => {
      const algo = $("algoSelect").value;
      const btn = $("trainBtn");
      const progress = $("trainProgress");
      const errEl = $("trainError");
      errEl.hidden = true;

      btn.disabled = true;
      progress.hidden = false;
      $("trainBarFill").style.width = "15%";
      $("trainStatusText").textContent = "Vectorizing text and fitting the model…";

      try {
        const body = { algorithm: algo };
        if (state.uploadedFilename) body.filename = state.uploadedFilename;

        const fillTimer = setTimeout(() => { $("trainBarFill").style.width = "70%"; }, 300);
        const data = await api(API.train, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        clearTimeout(fillTimer);
        $("trainBarFill").style.width = "100%";
        $("trainStatusText").textContent = `Done — accuracy ${(data.metrics.accuracy * 100).toFixed(1)}%`;
        renderMetricsCard(data.metrics);
        toast("Model trained successfully.");
        await refreshStatus();
      } catch (e) {
        progress.hidden = true;
        errEl.textContent = e.message;
        errEl.hidden = false;
        toast(e.message, true);
      } finally {
        btn.disabled = false;
        setTimeout(() => { progress.hidden = true; }, 1200);
      }
    });
  }

  // ---------------- live detection (main section) ----------------
  function initDetect() {
    $("detectBtn").addEventListener("click", async () => {
      const text = $("detectInput").value.trim();
      const errEl = $("detectError");
      errEl.hidden = true;
      if (!text) {
        errEl.textContent = "Paste an email's text before analyzing.";
        errEl.hidden = false;
        return;
      }
      $("detectBtn").disabled = true;
      try {
        const result = await api(API.predict, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });
        $("detectEmpty").hidden = true;
        $("detectResult").hidden = false;
        renderVerdict($("detectBadge"), $("detectConfidence"), result);

        const fill = $("confidenceBarFill");
        fill.style.width = `${result.confidence}%`;
        fill.style.background = result.is_spam ? "var(--brick)" : "var(--teal)";

        $("detectFeatures").innerHTML = featureRows(result.features);

        const kwSection = $("keywordSection");
        if (result.features.keyword_hits.length) {
          kwSection.hidden = false;
          $("keywordChips").innerHTML = result.features.keyword_hits
            .map((k) => `<span class="keyword-chip">${escapeHtml(k)}</span>`)
            .join("");
        } else {
          kwSection.hidden = true;
        }
        refreshHistory();
      } catch (e) {
        errEl.textContent = e.message;
        errEl.hidden = false;
        toast(e.message, true);
      } finally {
        $("detectBtn").disabled = false;
      }
    });
  }

  // ---------------- hero mini console ----------------
  function initHeroConsole() {
    $("heroScanBtn").addEventListener("click", async () => {
      const text = $("heroInput").value.trim();
      if (!text) { toast("Paste some text into the console first.", true); return; }
      const consoleEl = $("heroConsole");
      consoleEl.classList.add("scanning");
      $("heroScanBtn").disabled = true;
      try {
        const result = await api(API.predict, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });
        $("heroResult").hidden = false;
        renderVerdict($("heroBadge"), $("heroConfidence"), result);
        $("heroFeatures").innerHTML = featureRows(result.features);
        refreshHistory();
      } catch (e) {
        toast(e.message, true);
      } finally {
        consoleEl.classList.remove("scanning");
        $("heroScanBtn").disabled = false;
      }
    });
  }

  // ---------------- batch analysis ----------------
  function initBatch() {
    const input = $("batchFileInput");
    $("batchUploadBtn").addEventListener("click", () => input.click());
    input.addEventListener("change", () => {
      if (input.files[0]) {
        state.batchFile = input.files[0];
        $("batchFileName").textContent = input.files[0].name;
      }
    });

    $("batchRunBtn").addEventListener("click", async () => {
      const errEl = $("batchError");
      errEl.hidden = true;
      if (!state.batchFile) {
        errEl.textContent = "Choose a CSV file first.";
        errEl.hidden = false;
        return;
      }
      const btn = $("batchRunBtn");
      btn.disabled = true;
      const originalLabel = btn.textContent;
      btn.textContent = "Scanning…";

      const formData = new FormData();
      formData.append("file", state.batchFile);
      try {
        const data = await api(API.batch, { method: "POST", body: formData });
        state.lastBatchResults = data.results;
        renderBatchResults(data);
        toast(`Scanned ${data.total} messages.`);
      } catch (e) {
        errEl.textContent = e.message;
        errEl.hidden = false;
        toast(e.message, true);
      } finally {
        btn.disabled = false;
        btn.textContent = originalLabel;
      }
    });

    $("batchDownloadBtn").addEventListener("click", () => {
      if (!state.lastBatchResults) return;
      const rows = [["text", "prediction", "confidence"]];
      state.lastBatchResults.forEach((r) => rows.push([r.text.replace(/"/g, '""'), r.prediction, r.confidence]));
      const csv = rows.map((r) => r.map((c) => `"${c}"`).join(",")).join("\n");
      const blob = new Blob([csv], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "sentinel_batch_results.csv";
      a.click();
      URL.revokeObjectURL(url);
    });
  }

  function renderBatchResults(data) {
    const summary = $("batchSummary");
    summary.hidden = false;
    summary.innerHTML = `
      <div class="batch-summary-item"><span class="batch-summary-value">${data.total}</span><span class="batch-summary-label">Messages scanned</span></div>
      <div class="batch-summary-item"><span class="batch-summary-value" style="color:var(--brick)">${data.spam_count}</span><span class="batch-summary-label">Flagged spam</span></div>
      <div class="batch-summary-item"><span class="batch-summary-value" style="color:var(--teal)">${data.legitimate_count}</span><span class="batch-summary-label">Legitimate</span></div>
    `;
    const wrap = $("batchTableWrap");
    wrap.hidden = false;
    let html = "<table><tr><th>Message</th><th>Verdict</th><th>Confidence</th></tr>";
    data.results.slice(0, 200).forEach((r) => {
      html += `<tr><td>${escapeHtml(r.text)}</td><td class="cell-verdict ${r.prediction === "spam" ? "spam" : "legit"}">${r.prediction}</td><td>${r.confidence}%</td></tr>`;
    });
    html += "</table>";
    wrap.innerHTML = html;
    $("batchDownloadBtn").hidden = false;
  }

  // ---------------- history ----------------
  async function refreshHistory() {
    try {
      const items = await api(API.history);
      const wrap = $("historyTableWrap");
      if (!items.length) {
        $("historyEmpty").hidden = false;
        wrap.hidden = true;
        return;
      }
      $("historyEmpty").hidden = true;
      wrap.hidden = false;
      let html = "<table><tr><th>Time</th><th>Message</th><th>Verdict</th><th>Confidence</th></tr>";
      items.forEach((h) => {
        const time = new Date(h.timestamp).toLocaleString();
        html += `<tr><td>${time}</td><td>${escapeHtml(h.text)}</td><td class="cell-verdict ${h.prediction === "spam" ? "spam" : "legit"}">${h.prediction}</td><td>${h.confidence}%</td></tr>`;
      });
      html += "</table>";
      wrap.innerHTML = html;
    } catch (e) {
      /* silent — history is non-critical */
    }
  }

  function initHistory() {
    $("clearHistoryBtn").addEventListener("click", async () => {
      try {
        await api(API.clearHistory, { method: "POST" });
        toast("History cleared.");
        refreshHistory();
      } catch (e) {
        toast(e.message, true);
      }
    });
  }

  // ---------------- init ----------------
  document.addEventListener("DOMContentLoaded", () => {
    $("year").textContent = new Date().getFullYear();
    initReveal();
    initDatasetUpload();
    initTrain();
    initDetect();
    initHeroConsole();
    initBatch();
    initHistory();
    refreshStatus();
    refreshHistory();
  });
})();
