const state = {
  meta: null,
  staticReady: null,
  artifacts: null,
  xgb: null,
};

const featureIds = [
  "education", "insurance", "breastCancerFamilyHistory", "benignBreastDisease",
  "cancerFamilyHistory", "metabolicSyndrome", "diabetes", "hormoneTherapy",
  "bmiCategory", "waistHipRatio", "sedentaryTime", "exercise",
  "sleepHours", "depressionScore",
];

const storageKey = "medtwin-static-store-v1";
const binaryOneTwoFields = new Set(["cancerFamilyHistory", "metabolicSyndrome", "diabetes", "hormoneTherapy"]);
const clinicalWeights = {
  breastCancerFamilyHistory: 0.70,
  benignBreastDisease: 0.65,
  cancerFamilyHistory: 0.45,
  metabolicSyndrome: 0.35,
  diabetes: 0.30,
  hormoneTherapy: 0.25,
  bmiCategory: 0.15,
  waistHipRatio: 0.12,
  sedentaryTime: 0.12,
  exercise: 0.10,
  depressionScore: 0.08,
};

function $(id) { return document.getElementById(id); }

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

function riskLevelFor(probability) {
  if (probability >= 0.70) return "极高风险";
  if (probability >= 0.40) return "高风险";
  if (probability >= 0.20) return "中风险";
  return "低风险";
}

function riskClass(level) {
  if (level.includes("极高")) return "critical";
  if (level.includes("高")) return "high";
  if (level.includes("中")) return "middle";
  return "low";
}

async function ensureStaticModel() {
  if (!state.staticReady) {
    state.staticReady = Promise.all([
      fetch("static/model_artifacts.json").then(res => res.json()),
      fetch("static/xgboost_model_SJ.json").then(res => res.json()),
    ]).then(([artifacts, xgb]) => {
      state.artifacts = artifacts;
      state.xgb = xgb;
    });
  }
  return state.staticReady;
}

function normalizeFeatures(raw = {}) {
  const defaults = state.artifacts.defaultFeatures;
  const values = { ...defaults, ...raw };
  return values;
}

function modelEncodedValue(key, value) {
  if (binaryOneTwoFields.has(key)) return value > 0 ? 2 : 1;
  if (key === "exercise") return value > 0 ? 1 : 2;
  return value;
}

function toModelVector(raw = {}) {
  const values = normalizeFeatures(raw);
  const modelValues = Object.fromEntries(
    Object.entries(values).map(([key, value]) => [key, modelEncodedValue(key, Number(value))]),
  );
  modelValues.interactionFeature = Number(values.breastCancerFamilyHistory > 0)
    + Number(values.benignBreastDisease > 0)
    + Number(values.cancerFamilyHistory > 0)
    + Number(values.metabolicSyndrome > 0)
    + Number(values.diabetes > 0);
  const labelToId = Object.fromEntries(
    Object.entries(state.artifacts.fieldMap).map(([id, label]) => [label, id]),
  );
  return state.artifacts.featureNames.map((label, index) => {
    const id = labelToId[label];
    let value = id ? Number(modelValues[id]) : Number.NaN;
    if (Number.isNaN(value)) value = state.artifacts.imputerStatistics[index];
    return (value - state.artifacts.scalerMean[index]) / state.artifacts.scalerScale[index];
  });
}

function predictWithXgBoost(rawFeatures = {}) {
  const vector = toModelVector(rawFeatures);
  const model = state.xgb.learner.gradient_booster.model;
  let margin = 0;

  const treeLimit = state.artifacts.treeLimit || 738;
  for (const tree of model.trees.slice(0, treeLimit)) {
    let node = 0;
    while (tree.left_children[node] !== -1) {
      const featureValue = vector[tree.split_indices[node]];
      const goLeft = Number.isNaN(featureValue)
        ? tree.default_left[node] === 1
        : featureValue < tree.split_conditions[node];
      node = goLeft ? tree.left_children[node] : tree.right_children[node];
    }
    margin += tree.base_weights[node];
  }

  return sigmoid(margin);
}

function clinicallyCalibratedProbability(modelProbability, rawFeatures = {}) {
  const values = normalizeFeatures(rawFeatures);
  let probability = Math.max(0.001, Math.min(0.999, modelProbability));
  let score = Math.log(probability / (1 - probability));
  if (values.breastCancerFamilyHistory > 0) score += clinicalWeights.breastCancerFamilyHistory;
  if (values.benignBreastDisease > 0) score += clinicalWeights.benignBreastDisease;
  if (values.cancerFamilyHistory > 0) score += clinicalWeights.cancerFamilyHistory;
  if (values.metabolicSyndrome > 0) score += clinicalWeights.metabolicSyndrome;
  if (values.diabetes > 0) score += clinicalWeights.diabetes;
  if (values.hormoneTherapy > 0) score += clinicalWeights.hormoneTherapy;
  if (values.bmiCategory >= 3) score += clinicalWeights.bmiCategory;
  if (values.waistHipRatio >= 0.85) score += clinicalWeights.waistHipRatio;
  if (values.sedentaryTime >= 8) score += clinicalWeights.sedentaryTime;
  if (values.exercise <= 0) score += clinicalWeights.exercise;
  if (values.depressionScore >= 10) score += clinicalWeights.depressionScore;
  return sigmoid(score);
}

function topFactors(rawFeatures = {}) {
  const values = normalizeFeatures(rawFeatures);
  return Object.entries(state.artifacts.factorLabels)
    .filter(([id]) => Number(values[id] || 0) > 0)
    .map(([id, meta]) => ({ key: id, name: meta.name, weight: meta.weight }))
    .sort((a, b) => b.weight - a.weight)
    .slice(0, 3);
}

function buildPrediction(rawFeatures = {}) {
  const probability = clinicallyCalibratedProbability(predictWithXgBoost(rawFeatures), rawFeatures);
  const risk_level = riskLevelFor(probability);
  const factors = topFactors(rawFeatures);
  const factorText = factors.length ? factors.map(item => item.name).join("、") : "未发现明显高权重因素";
  return {
    probability: Math.round(probability * 1000000) / 1000000,
    risk_level,
    factors,
    explanation: `风险概率为 ${(probability * 100).toFixed(1)}%，等级为${risk_level}。主要影响因素：${factorText}。该结果仅用于课程演示和辅助筛查，不替代临床诊断。`,
    model_status: "选用模型：XGBoost Risk",
    threshold: state.artifacts.threshold,
  };
}

function readStore() {
  const fallback = { cases: [], queue: [], undo: [] };
  try {
    return { ...fallback, ...JSON.parse(localStorage.getItem(storageKey) || "{}") };
  } catch {
    return fallback;
  }
}

function writeStore(store) {
  localStorage.setItem(storageKey, JSON.stringify(store));
}

function makeCase(payload, id) {
  const prediction = buildPrediction(payload.features);
  const created = new Date();
  const riskQueue = prediction.probability >= state.artifacts.threshold;
  return {
    id,
    name: payload.name || "未命名病例",
    age: payload.age || 45,
    role: payload.role || "patient",
    status: riskQueue ? "待医生复核" : "已归档",
    created_at: created.toISOString(),
    created_at_text: created.toLocaleString("zh-CN", { hour12: false }),
    features: payload.features,
    prediction,
  };
}

async function staticApi(path, options = {}) {
  await ensureStaticModel();
  const url = new URL(path, window.location.origin);
  const store = readStore();

  if (url.pathname.endsWith("/api/meta")) {
    return {
      modelStatus: "选用模型：XGBoost Risk",
      threshold: state.artifacts.threshold,
      queue: store.queue,
      undoCount: store.undo.length,
    };
  }

  if (url.pathname.endsWith("/api/predict")) {
    const payload = JSON.parse(options.body || "{}");
    return buildPrediction(payload.features || payload);
  }

  if (url.pathname.endsWith("/api/cases") && options.method === "POST") {
    const payload = JSON.parse(options.body || "{}");
    store.undo.push(JSON.stringify(store));
    const nextId = store.cases.reduce((max, item) => Math.max(max, item.id), 1000) + 1;
    const item = makeCase(payload, nextId);
    store.cases.unshift(item);
    if (item.status === "待医生复核") store.queue.push(item.id);
    writeStore(store);
    return { ok: true, case: item };
  }

  if (url.pathname.endsWith("/api/cases")) {
    let cases = [...store.cases];
    const q = (url.searchParams.get("q") || "").trim();
    const role = url.searchParams.get("role") || "all";
    const sort = url.searchParams.get("sort") || "risk";
    if (q) cases = cases.filter(item => item.name.includes(q) || String(item.id).includes(q));
    if (role !== "all") cases = cases.filter(item => item.role === role);
    if (sort === "risk") cases.sort((a, b) => b.prediction.probability - a.prediction.probability);
    else cases.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
    return { cases, queue: store.queue };
  }

  if (url.pathname.endsWith("/api/queue/pop")) {
    store.undo.push(JSON.stringify(store));
    const caseId = store.queue.shift();
    const item = store.cases.find(entry => entry.id === caseId);
    if (item) item.status = "医生已复核";
    writeStore(store);
    return { caseId };
  }

  if (url.pathname.endsWith("/api/undo")) {
    const previous = store.undo.pop();
    if (previous) {
      writeStore(JSON.parse(previous));
      return { ok: true };
    }
    return { ok: false, message: "暂无可撤销操作" };
  }

  return { error: "not found" };
}

async function api(path, options = {}) {
  try {
    const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch {
    return staticApi(path, options);
  }
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
  state.meta = await api("api/meta");
  const cases = await api("api/cases?sort=time");
  $("modelStatus").textContent = state.meta.modelStatus.replace("选用模型：", "");
  $("thresholdValue").textContent = Number(state.meta.threshold).toFixed(3);
  $("caseCount").textContent = cases.cases.length;
  $("queueCount").textContent = state.meta.queue.length;
  $("homeStatus").textContent = `${state.meta.modelStatus} · 当前病例 ${cases.cases.length} · 待复核 ${state.meta.queue.length}`;
}

async function previewRisk() {
  const result = await api("api/predict", { method: "POST", body: JSON.stringify(collectPayload()) });
  renderPrediction(result);
}

async function submitCase() {
  await api("api/cases", { method: "POST", body: JSON.stringify(collectPayload()) });
  await loadMeta();
  showView("doctor");
}

async function loadCases() {
  const q = encodeURIComponent($("searchInput").value || "");
  const sort = $("sortSelect").value;
  const role = $("roleSelect").value;
  const data = await api(`api/cases?q=${q}&sort=${sort}&role=${role}`);
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
  await api("api/queue/pop", { method: "POST", body: "{}" });
  await loadMeta();
  await loadCases();
}

async function undo() {
  await api("api/undo", { method: "POST", body: "{}" });
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
