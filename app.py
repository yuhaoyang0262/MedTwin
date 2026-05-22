#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MedTwin v1

Clean stdlib backend:
- Serves index.html + static JS/CSS/assets
- Provides case management APIs
- Calls the existing trained XGBoost breast cancer risk model
"""

from __future__ import annotations

import json
import math
import mimetypes
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
MODEL_SUBDIR = Path("models") / "xgboost" / "models_balanced_recall_precision_SJ_5%(best)"
MODEL_DIR = APP_DIR / MODEL_SUBDIR
if not MODEL_DIR.exists():
    MODEL_DIR = ROOT / MODEL_SUBDIR
MODEL_FILES = {
    "model": "xgboost_model_SJ.json",
    "imputer": "imputer_1_1223.pkl",
    "scaler": "scaler_1_1223.pkl",
    "features": "train_features_1_1223.pkl",
    "results": "final_results_1_1223.json",
}


FIELD_MAP = {
    "education": "学历",
    "marriage": "婚姻",
    "income": "人均月收入",
    "insurance": "医保类型",
    "economicStatus": "主观经济地位",
    "socialStatus": "主观社会地位",
    "bmiCategory": "BMI分类(推荐)",
    "breastfeeding": "哺乳时长分类",
    "menopause": "是否绝经",
    "waistHipRatio": "腰臀比",
    "smoking": "吸烟分类",
    "drinking": "饮酒分类",
    "menarcheAge": "初次月经年龄分类",
    "regularMenstruation": "月经是否规律",
    "dysmenorrhea": "痛经",
    "secondhandSmoke": "二手烟",
    "redMeat": "高红肉饮食分类",
    "vitamin": "维生素",
    "sedentaryTime": "总静态行为时间",
    "exercise": "是否每周进行中高强度体育锻炼",
    "sleepHours": "睡眠时长",
    "hormoneTherapy": "激素替代疗法",
    "benignBreastDisease": "良性乳腺病二分类",
    "abortionHistory": "流产分类",
    "birthCount": "生育次数分类",
    "cancerFamilyHistory": "恶性肿瘤家族史",
    "metabolicSyndrome": "代谢综合征（高血糖+高血压+高血脂）",
    "firstBirthAge": "头胎年龄分类",
    "diabetes": "二型糖尿病",
    "breastCancerFamilyHistory": "乳腺癌家族史",
    "depressionScore": "抑郁得分",
    "interactionFeature": "交互特征",
}


DEFAULT_FEATURES = {
    "education": 2,
    "marriage": 1,
    "income": 2,
    "insurance": 2,
    "economicStatus": 3,
    "socialStatus": 3,
    "bmiCategory": 2,
    "breastfeeding": 1,
    "menopause": 0,
    "waistHipRatio": 0.84,
    "smoking": 0,
    "drinking": 0,
    "menarcheAge": 2,
    "regularMenstruation": 1,
    "dysmenorrhea": 0,
    "secondhandSmoke": 0,
    "redMeat": 0,
    "vitamin": 0,
    "sedentaryTime": 6,
    "exercise": 1,
    "sleepHours": 7,
    "hormoneTherapy": 0,
    "benignBreastDisease": 0,
    "abortionHistory": 0,
    "birthCount": 1,
    "cancerFamilyHistory": 0,
    "metabolicSyndrome": 0,
    "firstBirthAge": 2,
    "diabetes": 0,
    "breastCancerFamilyHistory": 0,
    "depressionScore": 0,
}


FACTOR_LABELS = {
    "breastCancerFamilyHistory": ("乳腺癌家族史", 0.18),
    "benignBreastDisease": ("良性乳腺病史", 0.16),
    "cancerFamilyHistory": ("恶性肿瘤家族史", 0.12),
    "metabolicSyndrome": ("代谢综合征", 0.10),
    "diabetes": ("二型糖尿病", 0.08),
    "hormoneTherapy": ("激素替代疗法", 0.08),
    "menopause": ("绝经状态", 0.06),
    "bmiCategory": ("BMI分类", 0.04),
    "waistHipRatio": ("腰臀比", 0.04),
    "sedentaryTime": ("静态行为时间", 0.03),
}

BINARY_ONE_TWO_FIELDS = {
    "cancerFamilyHistory",
    "metabolicSyndrome",
    "diabetes",
    "hormoneTherapy",
}

CLINICAL_WEIGHTS = {
    "breastCancerFamilyHistory": 0.70,
    "benignBreastDisease": 0.65,
    "cancerFamilyHistory": 0.45,
    "metabolicSyndrome": 0.35,
    "diabetes": 0.30,
    "hormoneTherapy": 0.25,
    "bmiCategory": 0.15,
    "waistHipRatio": 0.12,
    "sedentaryTime": 0.12,
    "exercise": 0.10,
    "depressionScore": 0.08,
}


@dataclass
class Prediction:
    probability: float
    risk_level: str
    factors: list[dict[str, Any]]
    explanation: str
    model_status: str
    threshold: float


@dataclass
class Case:
    id: int
    name: str
    role: str
    age: int
    status: str
    created_at: float
    features: dict[str, float]
    prediction: Prediction


@dataclass
class Store:
    cases: list[Case] = field(default_factory=list)
    queue: deque[int] = field(default_factory=deque)
    undo_stack: list[dict[str, Any]] = field(default_factory=list)
    next_id: int = 1001

    def snapshot(self) -> dict[str, Any]:
        return {
            "cases": [case_to_dict(case) for case in self.cases],
            "queue": list(self.queue),
            "next_id": self.next_id,
        }

    def restore(self, state: dict[str, Any]) -> None:
        self.cases = [dict_to_case(item) for item in state["cases"]]
        self.queue = deque(state["queue"])
        self.next_id = state["next_id"]

    def push_undo(self, action: str) -> None:
        self.undo_stack.append({"action": action, "state": self.snapshot()})
        if len(self.undo_stack) > 30:
            self.undo_stack.pop(0)


class XGBoostRiskModel:
    def __init__(self) -> None:
        self.threshold = 0.5
        self.model = None
        self.imputer = None
        self.scaler = None
        self.features: list[str] = []
        self.status = "模型未加载"
        self._load()

    def _load(self) -> None:
        try:
            import joblib  # type: ignore
            import xgboost as xgb  # type: ignore
        except Exception as exc:
            self.status = f"缺少模型依赖：{exc.__class__.__name__}"
            return

        try:
            with (MODEL_DIR / MODEL_FILES["results"]).open("r", encoding="utf-8") as f:
                self.threshold = float(json.load(f).get("threshold", self.threshold))

            self.features = list(joblib.load(MODEL_DIR / MODEL_FILES["features"]))
            self.imputer = joblib.load(MODEL_DIR / MODEL_FILES["imputer"])
            self.scaler = joblib.load(MODEL_DIR / MODEL_FILES["scaler"])
            self.model = xgb.Booster()
            self.model.load_model(str(MODEL_DIR / MODEL_FILES["model"]))
            self.status = "选用模型：XGBoost Risk"
        except Exception as exc:
            self.status = f"模型加载失败：{exc}"
            self.model = None

    def predict(self, raw_features: dict[str, Any]) -> Prediction:
        ui_features = normalize_ui_features(raw_features)
        features = to_model_features(ui_features)
        if self.model is None:
            probability = fallback_score(features)
        else:
            probability = self._predict_model(features)
        probability = clinically_calibrated_probability(probability, ui_features)

        risk_level = risk_level_for(probability)
        factors = top_factors(ui_features)
        explanation = (
            f"风险概率为 {probability:.1%}，等级为{risk_level}。"
            f"主要影响因素：{format_factor_names(factors)}。"
            "该结果仅用于课程演示和辅助筛查，不替代临床诊断。"
        )
        return Prediction(
            probability=round(float(probability), 6),
            risk_level=risk_level,
            factors=factors,
            explanation=explanation,
            model_status=self.status,
            threshold=self.threshold,
        )

    def _predict_model(self, features: dict[str, float]) -> float:
        import pandas as pd  # type: ignore
        import xgboost as xgb  # type: ignore

        row = {name: features.get(name, 0.0) for name in self.features}
        frame = pd.DataFrame([row], columns=self.features)
        x_imputed = self.imputer.transform(frame)
        x_scaled = self.scaler.transform(x_imputed)
        dmatrix = xgb.DMatrix(x_scaled, feature_names=self.features)
        return float(self.model.predict(dmatrix)[0])


def normalize_ui_features(raw: dict[str, Any]) -> dict[str, float]:
    values = dict(DEFAULT_FEATURES)
    for key, value in raw.items():
        if key in values:
            try:
                values[key] = float(value)
            except (TypeError, ValueError):
                pass

    return values


def model_encoded_value(key: str, value: float) -> float:
    if key in BINARY_ONE_TWO_FIELDS:
        return 2.0 if value > 0 else 1.0
    if key == "exercise":
        return 1.0 if value > 0 else 2.0
    return value


def to_model_features(ui_values: dict[str, float]) -> dict[str, float]:
    values = {key: model_encoded_value(key, value) for key, value in ui_values.items()}

    values["interactionFeature"] = (
        values["breastCancerFamilyHistory"]
        + values["benignBreastDisease"]
        + (1.0 if ui_values["cancerFamilyHistory"] > 0 else 0.0)
        + (1.0 if ui_values["metabolicSyndrome"] > 0 else 0.0)
        + (1.0 if ui_values["diabetes"] > 0 else 0.0)
    )

    return {
        model_name: float(values.get(front_key, 0.0))
        for front_key, model_name in FIELD_MAP.items()
    }


def clinically_calibrated_probability(model_probability: float, ui_features: dict[str, float]) -> float:
    probability = min(max(float(model_probability), 0.001), 0.999)
    score = math.log(probability / (1.0 - probability))
    if ui_features.get("breastCancerFamilyHistory", 0.0) > 0:
        score += CLINICAL_WEIGHTS["breastCancerFamilyHistory"]
    if ui_features.get("benignBreastDisease", 0.0) > 0:
        score += CLINICAL_WEIGHTS["benignBreastDisease"]
    if ui_features.get("cancerFamilyHistory", 0.0) > 0:
        score += CLINICAL_WEIGHTS["cancerFamilyHistory"]
    if ui_features.get("metabolicSyndrome", 0.0) > 0:
        score += CLINICAL_WEIGHTS["metabolicSyndrome"]
    if ui_features.get("diabetes", 0.0) > 0:
        score += CLINICAL_WEIGHTS["diabetes"]
    if ui_features.get("hormoneTherapy", 0.0) > 0:
        score += CLINICAL_WEIGHTS["hormoneTherapy"]
    if ui_features.get("bmiCategory", 0.0) >= 3:
        score += CLINICAL_WEIGHTS["bmiCategory"]
    if ui_features.get("waistHipRatio", 0.0) >= 0.85:
        score += CLINICAL_WEIGHTS["waistHipRatio"]
    if ui_features.get("sedentaryTime", 0.0) >= 8:
        score += CLINICAL_WEIGHTS["sedentaryTime"]
    if ui_features.get("exercise", 1.0) <= 0:
        score += CLINICAL_WEIGHTS["exercise"]
    if ui_features.get("depressionScore", 0.0) >= 10:
        score += CLINICAL_WEIGHTS["depressionScore"]
    return 1.0 / (1.0 + math.exp(-score))


def fallback_score(features: dict[str, float]) -> float:
    score = -2.8
    score += 0.35 * features.get("乳腺癌家族史", 0.0)
    score += 0.45 * features.get("良性乳腺病二分类", 0.0)
    score += 0.28 * features.get("恶性肿瘤家族史", 0.0)
    score += 0.22 * features.get("代谢综合征（高血糖+高血压+高血脂）", 0.0)
    score += 0.08 * max(features.get("BMI分类(推荐)", 2.0) - 2.0, 0.0)
    score += 0.04 * max(features.get("总静态行为时间", 6.0) - 6.0, 0.0)
    score -= 0.18 * features.get("是否每周进行中高强度体育锻炼", 0.0)
    return 1.0 / (1.0 + pow(2.718281828, -score))


def top_factors(ui_features: dict[str, float]) -> list[dict[str, Any]]:
    scored = []
    for front_key, (label, weight) in FACTOR_LABELS.items():
        value = ui_features.get(front_key, 0.0)
        baseline = DEFAULT_FEATURES.get(front_key, 0.0)
        impact = abs(value - float(baseline)) * weight
        if front_key in {"breastCancerFamilyHistory", "benignBreastDisease", "cancerFamilyHistory", "metabolicSyndrome", "diabetes", "hormoneTherapy"}:
            impact = weight if value > 0 else 0.0
        if front_key == "exercise":
            impact = weight if value <= 0 else 0.0
        if impact > 0:
            scored.append({"name": label, "value": value, "impact": round(impact, 3)})
    scored.sort(key=lambda item: item["impact"], reverse=True)
    return scored[:5]


def format_factor_names(factors: list[dict[str, Any]]) -> str:
    return "、".join(item["name"] for item in factors[:3]) if factors else "暂无明显高权重因素"


def risk_level_for(probability: float) -> str:
    if probability >= 0.70:
        return "极高风险"
    if probability >= 0.40:
        return "高风险"
    if probability >= 0.20:
        return "中风险"
    return "低风险"


def case_to_dict(case: Case) -> dict[str, Any]:
    data = asdict(case)
    data["created_at_text"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(case.created_at))
    return data


def dict_to_case(data: dict[str, Any]) -> Case:
    prediction = Prediction(**data["prediction"])
    return Case(
        id=int(data["id"]),
        name=data["name"],
        role=data.get("role", "patient"),
        age=int(data["age"]),
        status=data["status"],
        created_at=float(data["created_at"]),
        features={k: float(v) for k, v in data["features"].items()},
        prediction=prediction,
    )


model = XGBoostRiskModel()
store = Store()


def seed_cases() -> None:
    examples = [
        ("张女士", "patient", 46, {"breastCancerFamilyHistory": 1, "benignBreastDisease": 1, "bmiCategory": 3, "sedentaryTime": 8, "exercise": 0}),
        ("李女士", "patient", 38, {"breastCancerFamilyHistory": 0, "benignBreastDisease": 0, "bmiCategory": 2, "sedentaryTime": 4, "exercise": 1}),
        ("王女士", "doctor", 55, {"breastCancerFamilyHistory": 1, "menopause": 1, "hormoneTherapy": 1, "metabolicSyndrome": 1, "diabetes": 1}),
    ]
    for name, role, age, features in examples:
        create_case({"name": name, "role": role, "age": age, "features": features}, push_undo=False)


def create_case(payload: dict[str, Any], push_undo: bool = True) -> Case:
    if push_undo:
        store.push_undo("新增病例")

    raw_features = dict(DEFAULT_FEATURES)
    raw_features.update(payload.get("features", {}))
    prediction = model.predict(raw_features)
    status = "待医生复核" if prediction.probability >= 0.40 else "已归档"
    case = Case(
        id=store.next_id,
        name=str(payload.get("name", f"病例{store.next_id}")),
        role=str(payload.get("role", "patient")),
        age=int(payload.get("age", 45)),
        status=status,
        created_at=time.time(),
        features={k: float(v) for k, v in raw_features.items()},
        prediction=prediction,
    )
    store.next_id += 1
    store.cases.append(case)
    if status == "待医生复核":
        store.queue.append(case.id)
    return case


def list_cases(query: dict[str, list[str]]) -> dict[str, Any]:
    keyword = query.get("q", [""])[0].strip().lower()
    role = query.get("role", ["all"])[0]
    sort_key = query.get("sort", ["risk"])[0]
    cases = store.cases
    if keyword:
        cases = [case for case in cases if keyword in case.name.lower() or keyword in str(case.id)]
    if role != "all":
        cases = [case for case in cases if case.role == role]

    def sort_value(case: Case) -> Any:
        if sort_key == "time":
            return case.created_at
        if sort_key == "age":
            return case.age
        if sort_key == "name":
            return case.name
        return case.prediction.probability

    return {
        "cases": [case_to_dict(case) for case in sorted(cases, key=sort_value, reverse=sort_key != "name")],
        "queue": list(store.queue),
        "undoCount": len(store.undo_stack),
    }


def send_json(handler: BaseHTTPRequestHandler, data: Any, status: int = 200) -> None:
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_file(APP_DIR / "index.html", "text/html; charset=utf-8")
            return
        if parsed.path.startswith("/static/"):
            self._send_file(APP_DIR / parsed.path.removeprefix("/"), None)
            return
        if parsed.path.startswith("/assets/"):
            self._send_file(APP_DIR / parsed.path.removeprefix("/"), None)
            return
        if parsed.path == "/api/meta":
            send_json(self, {
                "modelStatus": model.status,
                "threshold": model.threshold,
                "queue": list(store.queue),
                "undoCount": len(store.undo_stack),
            })
            return
        if parsed.path == "/api/cases":
            send_json(self, list_cases(parse_qs(parsed.query)))
            return
        send_json(self, {"error": "not found"}, 404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        payload = self._read_json()
        if parsed.path == "/api/predict":
            send_json(self, asdict(model.predict(payload.get("features", payload))))
            return
        if parsed.path == "/api/cases":
            case = create_case(payload)
            send_json(self, {"ok": True, "case": case_to_dict(case)})
            return
        if parsed.path == "/api/queue/pop":
            store.push_undo("处理队列")
            case_id = store.queue.popleft() if store.queue else None
            for case in store.cases:
                if case.id == case_id:
                    case.status = "医生已复核"
                    break
            send_json(self, {"caseId": case_id})
            return
        if parsed.path == "/api/undo":
            if not store.undo_stack:
                send_json(self, {"ok": False, "message": "暂无可撤销操作"})
                return
            item = store.undo_stack.pop()
            store.restore(item["state"])
            send_json(self, {"ok": True, "action": item["action"]})
            return
        send_json(self, {"error": "not found"}, 404)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_file(self, path: Path, content_type: str | None) -> None:
        try:
            resolved = path.resolve()
            if APP_DIR.resolve() not in [resolved, *resolved.parents]:
                raise FileNotFoundError
            body = resolved.read_bytes()
        except FileNotFoundError:
            send_json(self, {"error": "not found"}, 404)
            return

        self.send_response(200)
        self.send_header("Content-Type", content_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    seed_cases()
    port = int(os.getenv("PORT", os.getenv("MEDTWIN_PORT", "8000")))
    host = os.getenv("HOST", "0.0.0.0" if os.getenv("PORT") else "127.0.0.1")
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"MedTwin running at http://{host}:{port}")
    print(model.status)
    server.serve_forever()


if __name__ == "__main__":
    main()
