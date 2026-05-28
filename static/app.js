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
const reviewRiskCutoff = 0.40;
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

  const treeLimit = state.artifacts.treeLimit || model.trees.length;
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
    model_status: state.artifacts.modelStatus || "选用模型：Dual Distilled",
    threshold: state.artifacts.threshold,
    review_cutoff: reviewRiskCutoff,
  };
}

function pendingCaseIds(cases = []) {
  return cases
    .filter(item => item && item.status === "待医生复核")
    .map(item => item.id);
}

function normalizeStore(store = {}) {
  const cases = Array.isArray(store.cases) ? store.cases.filter(Boolean) : [];
  const existingQueue = Array.isArray(store.queue) ? store.queue : [];
  for (const item of cases) {
    const probability = Number(item.prediction && item.prediction.probability);
    if (item.status === "已归档" && probability >= reviewRiskCutoff) item.status = "待医生复核";
  }
  const pendingIds = pendingCaseIds(cases);
  const queue = existingQueue.filter(id => pendingIds.includes(id));
  for (const id of pendingIds) {
    if (!queue.includes(id)) queue.push(id);
  }
  return {
    cases,
    queue,
    undo: Array.isArray(store.undo) ? store.undo : [],
  };
}

function readStore() {
  const fallback = { cases: [], queue: [], undo: [] };
  try {
    return normalizeStore({ ...fallback, ...JSON.parse(localStorage.getItem(storageKey) || "{}") });
  } catch {
    return fallback;
  }
}

function writeStore(store) {
  localStorage.setItem(storageKey, JSON.stringify(normalizeStore(store)));
}

function buildCaseHashIndex(cases = []) {
  const byId = new Map();
  const byName = new Map();
  for (const item of cases) {
    byId.set(String(item.id), item);
    const key = String(item.name || "").trim().toLowerCase();
    if (!key) continue;
    if (!byName.has(key)) byName.set(key, []);
    byName.get(key).push(item);
  }
  return { byId, byName };
}

function searchWithHashIndex(cases = [], keyword = "") {
  const q = keyword.trim().toLowerCase();
  if (!q) return cases;
  const index = buildCaseHashIndex(cases);
  const exact = [];
  const byId = index.byId.get(q);
  if (byId) exact.push(byId);
  const byName = index.byName.get(q) || [];
  exact.push(...byName);
  if (exact.length) return [...new Map(exact.map(item => [item.id, item])).values()];
  return cases.filter(item => String(item.name || "").toLowerCase().includes(q) || String(item.id).includes(q));
}

function buildRiskTree(cases = []) {
  const tree = {
    name: "全部病例",
    children: [
      { name: "极高风险", min: 0.70, max: 1.01, cases: [] },
      { name: "高风险", min: 0.40, max: 0.70, cases: [] },
      { name: "中风险", min: 0.20, max: 0.40, cases: [] },
      { name: "低风险", min: 0, max: 0.20, cases: [] },
    ],
  };
  for (const item of cases) {
    const probability = Number(item.prediction && item.prediction.probability) || 0;
    const node = tree.children.find(entry => probability >= entry.min && probability < entry.max)
      || tree.children[tree.children.length - 1];
    node.cases.push(item);
  }
  return tree;
}

function renderDataStructurePanel(allCases = [], visibleCases = []) {
  const hashStat = $("hashIndexStat");
  const hashHint = $("hashIndexHint");
  const riskTree = $("riskTree");
  if (!hashStat || !riskTree) return;
  const index = buildCaseHashIndex(allCases);
  hashStat.textContent = `${index.byId.size} 个编号键 · ${index.byName.size} 个姓名键`;
  hashHint.textContent = `当前命中 ${visibleCases.length} 条病例，查询优先使用 Map 哈希索引。`;
  const tree = buildRiskTree(allCases);
  riskTree.innerHTML = `<div class="tree-root">${tree.name}<b>${allCases.length}</b></div>` + tree.children.map(node => `
    <div class="tree-node ${node.cases.length ? "active" : ""}">
      <span>${node.name}</span><b>${node.cases.length}</b>
    </div>`).join("");
}

function isStaticHosting() {
  return location.protocol === "file:" || location.hostname.endsWith("github.io");
}

function makeCase(payload, id) {
  const prediction = buildPrediction(payload.features);
  const created = new Date();
  const riskQueue = prediction.probability >= reviewRiskCutoff;
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
      modelStatus: state.artifacts.modelStatus || "选用模型：Dual Distilled",
      threshold: state.artifacts.threshold,
      queue: pendingCaseIds(store.cases),
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
    if (item.status === "待医生复核" && !store.queue.includes(item.id)) store.queue.push(item.id);
    writeStore(store);
    return { ok: true, case: item };
  }

  if (url.pathname.endsWith("/api/cases")) {
    let cases = [...store.cases];
    const q = (url.searchParams.get("q") || "").trim();
    const role = url.searchParams.get("role") || "all";
    const sort = url.searchParams.get("sort") || "risk";
    if (q) cases = searchWithHashIndex(cases, q);
    if (role !== "all") cases = cases.filter(item => item.role === role);
    if (sort === "risk") cases.sort((a, b) => b.prediction.probability - a.prediction.probability);
    else cases.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
    return { cases, queue: pendingCaseIds(store.cases) };
  }

  if (url.pathname.endsWith("/api/queue/pop")) {
    const queue = pendingCaseIds(store.cases);
    if (!queue.length) {
      return { caseId: null, message: "当前无待复核病例。请先在患者端录入高风险病例并点击“预测并入库”。" };
    }
    store.undo.push(JSON.stringify(store));
    const caseId = queue[0];
    const item = store.cases.find(entry => entry.id === caseId);
    if (item) item.status = "医生已复核";
    store.queue = pendingCaseIds(store.cases);
    writeStore(store);
    return { caseId, message: `已处理队首病例 #${caseId}` };
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
  if (isStaticHosting()) return staticApi(path, options);
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
  if (view === "doctor") {
    loadCases();
  }
}

function collectPayload() {
  const features = {};
  for (const id of featureIds) features[id] = Number($(id).value);
  features.age = Number($("age").value || 45);
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
  const result = await api("api/cases", { method: "POST", body: JSON.stringify(collectPayload()) });
  await loadMeta();
  showView("doctor");
  await loadCases();
  if (result.case) {
    $("queueLine").textContent = result.case.status === "待医生复核"
      ? `已新增病例 #${result.case.id}，已进入待复核队列。`
      : `已新增病例 #${result.case.id}，当前风险未达到复核标准。`;
  }
}

async function loadCases() {
  const q = encodeURIComponent($("searchInput").value || "");
  const sort = $("sortSelect").value;
  const data = await api(`api/cases?q=${q}&sort=${sort}&role=all`);
  const allData = q ? await api(`api/cases?sort=${sort}&role=all`) : data;
  renderDataStructurePanel(allData.cases, data.cases);
  const hasQueue = data.queue.length > 0;
  $("queueLine").textContent = hasQueue
    ? `当前显示全部病例 ${data.cases.length} 条；待复核队列：${data.queue.join(" → ")}`
    : `当前显示全部病例 ${data.cases.length} 条；暂无待复核病例。`;
  $("popQueueBtn").disabled = false;
  $("popQueueBtn").title = hasQueue ? "处理复核队列中的第一位病例" : "当前没有待复核病例，点击可查看提示";
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
  const result = await api("api/queue/pop", { method: "POST", body: "{}" });
  await loadMeta();
  await loadCases();
  $("queueLine").textContent = result.message || (result.caseId
    ? `已处理队首病例 #${result.caseId}`
    : "当前无待复核病例。请先在患者端录入高风险病例并点击“预测并入库”。");
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

loadMeta().then(loadCases);
