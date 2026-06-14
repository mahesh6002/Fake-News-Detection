const API_BASE = "";   // Empty string = same origin

async function analyze() {
  const text = document.getElementById("articleText").value.trim();
  const resultDiv = document.getElementById("result");
  const errorBox = document.getElementById("errorBox");
  const spinner = document.getElementById("spinner");

  resultDiv.classList.add("d-none");
  errorBox.classList.add("d-none");
  const sourcePanel = document.getElementById("sourceCheckPanel");
  if (sourcePanel) {
    sourcePanel.classList.add("d-none");
  }

  if (text.length < 10) {
    showError("Please enter a longer article (at least 10 characters).");
    return;
  }

  spinner.classList.remove("d-none");
  document.getElementById("analyzeBtn").disabled = true;

  try {
    const resp = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!resp.ok) {
      const err = await resp.json();
      // Handle array detail structure if Pydantic validation error
      let errMsg = "Prediction failed";
      if (err && err.detail) {
        if (Array.isArray(err.detail)) {
          errMsg = err.detail.map(d => d.msg).join(", ");
        } else {
          errMsg = err.detail;
        }
      }
      throw new Error(errMsg);
    }

    const data = await resp.json();
    displayResult(data);
  } catch (e) {
    showError(e.message);
  } finally {
    spinner.classList.add("d-none");
    document.getElementById("analyzeBtn").disabled = false;
  }
}

function displayResult(data) {
  const isFake = data.label === "FAKE";
  document.getElementById("labelBadge").textContent = data.label;
  
  // Custom styled classes instead of standard bootstrap colors
  document.getElementById("labelBadge").className =
    `badge fs-4 px-3 py-2 ${isFake ? "bg-danger" : "bg-success"}`;
  
  document.getElementById("confidenceText").textContent =
    `${data.confidence.toFixed(1)}%`;
  
  if (document.getElementById("confidencePercent")) {
    document.getElementById("confidencePercent").textContent = `${data.confidence.toFixed(1)}%`;
  }

  const bar = document.getElementById("confidenceBar");
  bar.style.width = `${data.confidence}%`;
  bar.className = `progress-bar ${isFake ? "bg-danger" : "bg-success"}`;

  document.getElementById("fakeProb").textContent = `${data.fake_probability.toFixed(1)}%`;
  document.getElementById("realProb").textContent = `${data.real_probability.toFixed(1)}%`;
  
  // Display processing time in ms
  document.getElementById("processingTime").textContent = `Latency: ${data.processing_time_ms} ms`;

  resultDiv = document.getElementById("result");
  resultDiv.classList.remove("d-none");
  
  const text = document.getElementById("articleText").value.trim();
  triggerSourceCheck(text);
}

function showError(msg) {
  const box = document.getElementById("errorBox");
  box.textContent = `Error: ${msg}`;
  box.classList.remove("d-none");
  box.scrollIntoView({ behavior: 'smooth' });
}

async function triggerSourceCheck(text) {
  const sourcePanel = document.getElementById("sourceCheckPanel");
  const sourceList = document.getElementById("sourceList");
  if (!sourcePanel || !sourceList) return;

  sourcePanel.classList.add("d-none");
  sourceList.innerHTML = "";

  const words = text.split(/\s+/).slice(0, 10).join(" ");
  if (!words) return;

  try {
    const resp = await fetch(`/verify?query=${encodeURIComponent(words)}`);
    if (!resp.ok) return;

    const data = await resp.json();
    const articles = data.articles || [];

    if (articles.length === 0) {
      sourceList.innerHTML = '<li class="source-empty">No matching sources found in news index</li>';
    } else {
      articles.forEach(art => {
        const dateStr = art.published_at ? new Date(art.published_at).toLocaleDateString(undefined, {
          month: 'short',
          day: 'numeric',
          year: 'numeric'
        }) : "Recent Date";
        
        const li = document.createElement("li");
        li.className = "source-item";
        li.innerHTML = `
          <div class="source-meta">
            <span>${art.source}</span>
            <span class="source-meta-dot">·</span>
            <span>${dateStr}</span>
            <span class="source-meta-dot">·</span>
            <span class="source-relevance">Match: ${art.relevance_score.toFixed(0)}%</span>
          </div>
          <a href="${art.url}" target="_blank" rel="noopener noreferrer" class="source-link">${art.title}</a>
        `;
        sourceList.appendChild(li);
      });
    }
    sourcePanel.classList.remove("d-none");
  } catch (e) {
    console.error("News verification request failed:", e);
  }
}
