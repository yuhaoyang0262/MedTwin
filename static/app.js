const state = { meta: null };

const featureIds = [
  "education", "insurance", "breastCancerFamilyHistory", "benignBreastDisease",
  "cancerFamilyHistory", "metabolicSyndrome", "diabetes", "hormoneTherapy",
  "bmiCategory", "waistHipRatio", "sedentaryTime", "exercise",
  "sleepHours", "depressionScore",
];

function $(id) { return document.getElementById(id); }

async function api(path, options = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  return res.json();
}

function showView(view) {
  document.querySelectorAll(".view").forEach(node => node.classList.remove("active"));
  document.querySelectorAll(".nav__btn").forEach(btn => btn.classList.toggle("active", btn.dataset.view === view));
  $(`${view}View`).classList.add("active");
  if (view === "doctor") loadCases();
}

function collectPayload() {
  const features = {};
  for (const id of featureIds) features[id] = Number($(id).value);
  return {
    name: $("name").value.trim() || "未命名病例",
    age: Number($("age").value || 45),
    role: "patient",
    features,
  };
}

function riskClass(level) {
  if (level.includes("极高")) return "critical";
  if (level.includes("高")) return "high";
  if (level.includes("中")) return "middle";
  return "low";
}

function renderPrediction(prediction) {
  const percent = Math.round(prediction.probability * 1000) / 10;
  const factors = prediction.factors.slice(0, 3).map(f => `<span class="factor">${f.name}</span>`).join("");
  $("predictionBox").className = "result-card";
  $("predictionBox").innerHTML = `
    <div class="score">${percent.toFixed(1)}%</div>
    <span class="pill ${riskClass(prediction.risk_level)}">${prediction.risk_level}</span>
    <div class="progress"><span style="width:${Math.max(4, Math.min(100, percent))}%"></span></div>
    <p>${prediction.explanation}</p>
    <div class="factor-list">${factors || '<span class="factor">暂无明显因素</span>'}</div>
  `;
}

async function loadMeta() {
  state.meta = await api("/api/meta");
  const cases = await api("/api/cases?sort=time");
  $("modelStatus").textContent = state.meta.modelStatus.replace("选用模型：", "");
  $("thresholdValue").textContent = Number(state.meta.threshold).toFixed(3);
  $("caseCount").textContent = cases.cases.length;
  $("queueCount").textContent = state.meta.queue.length;
  $("homeStatus").textContent = `${state.meta.modelStatus} · 当前病例 ${cases.cases.length} · 待复核 ${state.meta.queue.length}`;
}

async function previewRisk() {
  const result = await api("/api/predict", { method: "POST", body: JSON.stringify(collectPayload()) });
  renderPrediction(result);
}

async function submitCase() {
  await api("/api/cases", { method: "POST", body: JSON.stringify(collectPayload()) });
  await loadMeta();
  showView("doctor");
}

async function loadCases() {
  const q = encodeURIComponent($("searchInput").value || "");
  const sort = $("sortSelect").value;
  const role = $("roleSelect").value;
  const data = await api(`/api/cases?q=${q}&sort=${sort}&role=${role}`);
  $("queueLine").textContent = data.queue.length ? `待复核队列：${data.queue.join(" → ")}` : "当前无待复核病例";
  $("caseRows").innerHTML = data.cases.map(item => {
    const p = item.prediction;
    const percent = Math.round(p.probability * 1000) / 10;
    const factors = p.factors.slice(0, 3).map(f => `<span class="factor">${f.name}</span>`).join("");
    return `
      <tr>
        <td>#${item.id}</td>
        <td><div class="patient-name">${item.name}</div><div class="sub">${item.age} 岁 · ${item.role === "doctor" ? "医生端" : "患者端"}<br>${item.created_at_text}</div></td>
        <td><strong>${percent.toFixed(1)}%</strong><br><span class="pill ${riskClass(p.risk_level)}">${p.risk_level}</span><div class="progress"><span style="width:${Math.max(4, Math.min(100, percent))}%"></span></div></td>
        <td>${item.status}</td>
        <td><div class="factor-list">${factors || '<span class="factor">暂无</span>'}</div></td>
      </tr>`;
  }).join("") || `<tr><td colspan="5">没有匹配病例</td></tr>`;
}

async function popQueue() {
  await api("/api/queue/pop", { method: "POST", body: "{}" });
  await loadMeta();
  await loadCases();
}

async function undo() {
  await api("/api/undo", { method: "POST", body: "{}" });
  await loadMeta();
  await loadCases();
}

document.addEventListener("click", event => {
  const btn = event.target.closest("[data-view]");
  if (btn) showView(btn.dataset.view);
});

$("previewBtn").addEventListener("click", previewRisk);
$("submitBtn").addEventListener("click", submitCase);
$("popQueueBtn").addEventListener("click", popQueue);
$("undoBtn").addEventListener("click", undo);
$("searchInput").addEventListener("input", loadCases);
$("sortSelect").addEventListener("change", loadCases);
$("roleSelect").addEventListener("change", loadCases);

loadMeta().then(loadCases);
