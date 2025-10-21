# app/services/gemini_service.py
from __future__ import annotations

import json
from typing import Any, Dict, Optional, List
import google.generativeai as genai

from app.config import get_settings


def _safe_extract_json(text: str) -> Dict[str, Any]:
    """Gemini가 코드펜스/서문을 붙여도 JSON만 뽑아낸다."""
    if not text:
        return {}
    s = text.strip()

    # 코드펜스 제거
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 2:
            s = parts[1]
            if s.lstrip().startswith("json"):
                s = s.split("\n", 1)[1] if "\n" in s else s.replace("json", "", 1).strip()

    # 첫 '{' ~ 마지막 '}' 범위
    try:
        start = s.index("{")
        end = s.rindex("}")
        candidate = s[start:end + 1]
        return json.loads(candidate)
    except Exception:
        pass

    # 통째로 파싱 시도
    try:
        return json.loads(s)
    except Exception:
        return {}


class GeminiExplanationService:
    """
    - 실제 SDK 호출.
    - 모델 ID 자동 해석(예: 'gemini-1.5-flash' → 'gemini-1.5-flash-002' / '-latest' 폴백).
    - 반환은 항상 dict(summary/actions/xai).
    """
    def __init__(self, api_key: Optional[str] = None):
        settings = get_settings()
        self.api_key = api_key or getattr(settings, 'GEMINI_API_KEY', None)
        self.enabled = bool(getattr(settings, 'GEMINI_ENABLED', False))
        preferred = getattr(settings, 'GEMINI_MODEL', 'gemini-2.5-flash')
        self.timeout = int(getattr(settings, 'GEMINI_TIMEOUT', 30))

        self.initialized = False
        self.model = None
        self.model_name = None  # 실제 선택된 모델 ID

        if not (self.enabled and self.api_key):
            print("Gemini API is not configured or disabled")
            return

        try:
            genai.configure(api_key=self.api_key)
            self.model_name = self._resolve_model_id(preferred)
            self.model = genai.GenerativeModel(
                self.model_name,
                generation_config={
                    # ✅ JSON만 반환
                    "response_mime_type": "application/json",
                    # 필요 시: "temperature": 0.2, "top_p": 0.9, "max_output_tokens": 1024,
                },
            )
            self.initialized = True
            print(f"✓ Gemini API initialized successfully (model: {self.model_name})")
        except Exception as e:
            print(f"Failed to initialize Gemini API: {e}")
            self.initialized = False

    def _resolve_model_id(self, preferred: str) -> str:
        """
        사용 가능한 모델 중에서 generateContent 지원 모델을 선택한다.
        우선순위:
          1) preferred 그대로
          2) preferred-002, preferred-001, preferred-latest
          3) flash-002, pro-002 (기본 폴백)
        """
        # 후보군 구성
        candidates: List[str] = [preferred]
        for suffix in ("-002", "-001", "-latest"):
            if not preferred.endswith(suffix):
                candidates.append(preferred + suffix)
        # 널리 쓰이는 최신들 (사용 가능한 모델들)
        candidates += ["gemini-2.5-flash", "gemini-2.0-flash-exp", "gemini-1.5-flash-002", "gemini-1.5-pro-002", "gemini-pro"]

        # 사용 가능한 모델 나열
        try:
            available = list(genai.list_models())
            supported = {m.name for m in available if "generateContent" in getattr(m, "supported_generation_methods", [])}
        except Exception:
            # 모델 리스트 요청이 실패해도 그냥 후보군 순서대로 시도
            supported = set()

        # 지원 목록이 비어 있으면 후보군 첫 번째 반환
        if not supported:
            return candidates[0]

        for c in candidates:
            # list_models()는 'models/<id>'로 준다.
            if c in supported or ("models/" + c) in supported:
                # SDK에선 그냥 'gemini-1.5-flash-002' 형태로 넘겨도 동작
                return c

        # 아무 것도 못 찾으면 첫 후보
        return candidates[0]

    def _prepare_payload(self, ai_prediction: Dict[str, Any], virustotal: Optional[Dict[str, Any]], feature_importance: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ai = (ai_prediction or {}).get("ai_analysis", {})
        hard = ai.get("predicted_types", [])
        probs = ai.get("confidence_scores") or ai.get("class_probabilities") or {}

        classes = ['Backdoor', 'Botnet', 'Download', 'Infiltration', 'Normal']
        # 0~1 스케일 그대로 전달 (프롬프트에서 %로 설명하도록 지시)
        class_probs = {c: float(probs.get(c, 0.0)) for c in classes}

        vt_scan = (virustotal or {}).get("scan_summary", {}) if virustotal else {}
        malicious = int(vt_scan.get("malicious", 0) or 0)
        total = int(vt_scan.get("total", 0) or 0)

        # 클래스별 top3 특징 중요도 구조화
        class_features = {}
        if feature_importance:
            print(f"🔍 Processing feature_importance with {len(feature_importance)} features")
            # 클래스별로 그룹화
            for feature_name, importance in feature_importance.items():
                # 클래스 이름 추출
                if '_' in feature_name:
                    parts = feature_name.split('_', 1)
                    class_name = parts[0]
                    if class_name.lower() == 'soft':
                        # soft_Backdoor_feature 형태
                        sub_parts = parts[1].split('_', 1)
                        if len(sub_parts) > 1:
                            class_name = f"Soft_{sub_parts[0]}"
                            feature = sub_parts[1]
                        else:
                            class_name = "Soft"
                            feature = parts[1]
                    else:
                        feature = parts[1]
                    
                    if class_name not in class_features:
                        class_features[class_name] = []
                    class_features[class_name].append({
                        "feature": feature.replace('_', ' '),
                        "importance": round(importance * 100, 1)
                    })
                    print(f"  Added: {class_name} -> {feature}")

            print(f"📊 class_features keys: {list(class_features.keys())}")

        return {
            "hard_labels": hard,
            "class_probs": class_probs,
            "virustotal": {"malicious": malicious, "total": total},
            "class_features": class_features  # 클래스별 top3 특징들
        }

    def _build_prompt(self, payload: Dict[str, Any]) -> str:
        """
        더 단단한 프롬프트:
        - JSON 이외 출력 금지
        - 키/타입/길이 제약
        - 숫자 표현(확률→퍼센트, 소수점 1자리) 지시
        - 과도한 단정/판단 금지(보안 도메인 톤)
        """
        # class_features 개수에 맞게 bullets 개수 결정
        num_classes = len(payload.get('class_features', {}))
        bullets_count = max(num_classes, 3)  # 최소 3개
        
        schema_hint = {
            "summary": "string (한국어 2~3문장, 확률은 %로 소수점 1자리, 과장 금지, VirusTotal 언급 금지, 구체적인 위협 설명 포함)",
            "actions": {
                "immediate": "string (1문장, 사용자 행동 지침, '주의가 필요합니다' 같은 모호한 표현 금지)",
                "investigate": "string (1문장, 분석자용 조사 지침, 낮은 확률 패턴 언급 금지)",
                "deep": "string (1문장, 장기 대응/차단 지침)"
            },
            "xai": {
                "bullets": ["string"] * bullets_count  # 클래스 개수만큼, one_sentence 제거
            }
        }
        return (
            "아래 입력(모델 예측/VT 요약/SHAP 분석)을 바탕으로 JSON만 출력하세요. **코드펜스/서문/후기/설명 금지.**\n"
            "출력은 아래 키를 반드시 포함해야 합니다.\n"
            f"{json.dumps(schema_hint, ensure_ascii=False)}\n\n"
            "입력 데이터:\n"
            "- hard_labels: 모델이 예측한 악성 클래스 목록\n"
            "- class_probs: 각 클래스의 확률(0~1). 설명할 때는 0~100%로 환산하여 소수점 1자리로 표현\n"
            "- class_features: 각 예측된 클래스별로 SHAP이 계산한 top3 특징과 중요도(%)\n"
            "  예: {\"Backdoor\": [{\"feature\": \"total line count\", \"importance\": 23.4}, ...]}\n\n"
            "Summary 작성 가이드:\n"
            "1) 모델 예측 결과와 확률만 언급 (VirusTotal 결과는 언급하지 말 것)\n"
            "2) 악성일 경우:\n"
            "   - 감지된 악성 패턴과 확률을 언급\n"
            "   - 해당 악성코드 유형의 특성과 위험성을 2-3문장으로 구체적으로 설명\n"
            "   - 예: \"이 문서는 Backdoor(98.7%), Download(98.0%) 등의 악성 패턴이 감지되었습니다. Backdoor는 공격자가 시스템에 원격으로 접근할 수 있는 백도어를 설치하려는 시도이며, Download는 추가 악성 파일을 다운로드하여 감염을 확산시킬 수 있습니다. 이러한 패턴들이 복합적으로 나타나는 것은 다단계 공격을 시사합니다.\"\n"
            "3) **중요**: 매우 낮은 확률(10% 미만)의 악성 패턴은 언급하지 말 것\n"
            "4) **중요**: 모호한 표현('잠재적인 위협을 나타냅니다', '주의가 필요합니다') 사용 금지. 구체적인 위협 설명 필수\n\n"
            "XAI 설명 작성 가이드:\n"
            "1) xai.bullets: **class_features에 있는 모든 클래스(각각 하나씩)**의 top 특징들을 언급하여 '왜 이 클래스로 판단했는지' 설명\n"
            f"   - **필수**: class_features의 {num_classes}개 클래스를 각각 1개 bullet으로 설명 (총 {bullets_count}개 bullets)\n"
            "   - **중요**: class_features에 포함된 모든 클래스를 반드시 언급해야 합니다. 하나라도 빠뜨리지 마세요!\n"
            "   - 예: \"Backdoor로 판단된 이유는 total line count(70.0%), obfuscated strings count(19.4%) 등의 특징이 높게 나타났기 때문입니다.\"\n"
            "   - 예: \"Botnet으로 판단된 이유는 total line count(20.8%), entropy max(17.7%) 등의 특징이 관찰되었기 때문입니다.\"\n"
            "2) 실제 class_features 데이터를 활용하여 구체적인 특징 이름과 중요도(%)를 정확히 언급할 것\n\n"
            "Actions 작성 가이드:\n"
            "1) immediate: 파일 실행/열기 금지, 격리 등 즉각적인 조치만 언급\n"
            "2) investigate: 높은 확률(30% 이상)의 악성 패턴에 대한 조사만 언급. 낮은 확률(<10%) 패턴 언급 금지\n"
            "3) deep: 시스템 전체 스캔, 네트워크 차단 등 장기 대응책 언급\n"
            "4) **금지**: '주의가 필요합니다', '검증해야 합니다' 같은 모호한 표현\n\n"
            "제약:\n"
            f"1) JSON 외 텍스트 금지, 2) summary는 2~3문장 (VirusTotal 언급 금지), 3) xai.bullets는 정확히 {bullets_count}개 (클래스 개수만큼), 4) 과도한 확신/단정 금지.\n"
            "5) 위험 표현은 근거가 있을 때만, 모호하면 '가능성' 표현 사용.\n"
            "6) 10% 미만의 낮은 확률 패턴은 summary와 actions에서 언급 금지.\n\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )

    def explain(self, ai_prediction: Dict[str, Any], virustotal: Optional[Dict[str, Any]] = None, feature_importance: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not (self.enabled and self.initialized):
            raise RuntimeError("Gemini API is not configured or disabled")

        payload = self._prepare_payload(ai_prediction, virustotal, feature_importance)
        print(f"📤 Gemini payload: hard_labels={payload['hard_labels']}, class_features keys={list(payload.get('class_features', {}).keys())}")
        prompt = self._build_prompt(payload)

        # === 실제 SDK 호출 ===
        try:
            resp = self.model.generate_content(prompt)
            text = getattr(resp, "text", "") or ""
        except Exception as e:
            print(f"Gemini call failed: {e}")
            return {"summary": "", "actions": {}, "xai": {"bullets": [], "one_sentence": ""}}

        data = _safe_extract_json(text)
        if not isinstance(data, dict):
            data = {}
        # 최소 스키마 보정
        data.setdefault("summary", "")
        data.setdefault("actions", {})
        data.setdefault("xai", {"bullets": [], "one_sentence": ""})
        return data


# === 싱글톤 + 외부 진입점 ===
_gemini_service_singleton: Optional[GeminiExplanationService] = None

def get_gemini_service() -> GeminiExplanationService:
    global _gemini_service_singleton
    if _gemini_service_singleton is None:
        _gemini_service_singleton = GeminiExplanationService()
    return _gemini_service_singleton

def generate_explanation(report_or_ai_prediction: dict, virustotal_result: dict | None = None) -> dict:
    svc = get_gemini_service()
    if not (svc and svc.enabled and svc.initialized):
        raise RuntimeError("Gemini API is not configured or disabled")
    ai_pred = report_or_ai_prediction.get("ai_prediction") if isinstance(report_or_ai_prediction, dict) and "ai_prediction" in report_or_ai_prediction else report_or_ai_prediction
    return svc.explain(ai_pred, virustotal_result)
