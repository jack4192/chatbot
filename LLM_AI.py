import re
import json
from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any, List, Tuple

import streamlit as st
import openai
from pydantic import BaseModel, Field


# =========================
# 1) 재설계 지표 계산 (Ai, Ii, Ri, S, Band)
# =========================

def clip01(x: float) -> float:
    return max(0.0, min(1.0, x))

@dataclass
class Inputs:
    income_avg_3m: Optional[float] = None     # 최근 3개월 월평균 소득(원)
    income_avg_12m: Optional[float] = None    # 최근 12개월 월평균 소득(원)

    activity_avg_3m: Optional[float] = None   # 최근 3개월 월평균 활동량(건수/횟수 등)
    activity_avg_12m: Optional[float] = None  # 최근 12개월 월평균 활동량

    no_income_months_3m: Optional[int] = None # 최근 3개월 무소득 월 수(0~3)
    payment_stop: Optional[bool] = None       # 납부중단 여부
    wage_arrears: Optional[bool] = None       # 체불 경험 여부

class IndicatorResult(BaseModel):
    A: float
    I: float
    R: float
    S: float
    band: Literal["A", "B", "C"]
    likelihood_label: str
    notes: List[str] = Field(default_factory=list)

def compute_indicators(inp: Inputs, eps: float = 1e-9) -> IndicatorResult:
    notes: List[str] = []

    # 소득 지표(Ii) 필수
    if inp.income_avg_3m is None or inp.income_avg_12m is None:
        raise ValueError("income_avg_3m, income_avg_12m는 필수 입력입니다.")

    income_ratio = inp.income_avg_3m / (inp.income_avg_12m + eps)
    I = clip01(1.0 - income_ratio)

    # 활동 지표(Ai): 미입력 허용(보수적으로 0 처리 + 불확실성 노트)
    if inp.activity_avg_3m is None or inp.activity_avg_12m is None:
        A = 0.0
        notes.append("활동량(최근3/12개월 평균) 미입력 → 활동중단 지수(Ai)=0으로 처리(불확실성↑).")
    else:
        activity_ratio = inp.activity_avg_3m / (inp.activity_avg_12m + eps)
        A = clip01(1.0 - activity_ratio)

    # 위험 지표(Ri): 무소득월수/3, 납부중단, 체불
    if inp.no_income_months_3m is None:
        g_i = 0.0
        notes.append("최근 3개월 무소득 월 수 미입력 → g_i=0 처리(불확실성↑).")
    else:
        g_i = clip01(inp.no_income_months_3m / 3.0)

    if inp.payment_stop is None:
        p_i = 0.0
        notes.append("납부중단 여부 미입력 → p_i=0 처리(불확실성↑).")
    else:
        p_i = 1.0 if inp.payment_stop else 0.0

    if inp.wage_arrears is None:
        w_i = 0.0
        notes.append("체불 경험 여부 미입력 → w_i=0 처리(불확실성↑).")
    else:
        w_i = 1.0 if inp.wage_arrears else 0.0

    R = clip01(0.5 * g_i + 0.3 * p_i + 0.2 * w_i)

    # 종합 점수
    S = 100.0 * (0.5 * A + 0.3 * I + 0.2 * R)

    if S >= 70.0:
        band = "A"
        likelihood_label = "지원 가능성(자가진단) 높음: 우선 검토 구간"
    elif S >= 50.0:
        band = "B"
        likelihood_label = "지원 가능성(자가진단) 중간: 추가 확인 필요"
    else:
        band = "C"
        likelihood_label = "지원 가능성(자가진단) 낮음: 기준 미달 가능"

    return IndicatorResult(A=A, I=I, R=R, S=S, band=band, likelihood_label=likelihood_label, notes=notes)


# =========================
# 2) 질문 시나리오
# =========================

QuestionKey = Literal[
    "income_avg_3m",
    "income_avg_12m",
    "activity_avg_3m",
    "activity_avg_12m",
    "no_income_months_3m",
    "payment_stop",
    "wage_arrears",
]

QUESTIONS: Dict[QuestionKey, str] = {
    "income_avg_3m": "최근 3개월의 월평균 소득이 대략 얼마였나요? (예: 120만원 / 1500000원 / 소득 없음)",
    "income_avg_12m": "최근 12개월의 월평균 소득이 대략 얼마였나요? (예: 250만원)",
    "activity_avg_3m": "최근 3개월의 월평균 활동량(월 작업/배차/콜 등)은 어느 정도였나요? (모르면 '모름')",
    "activity_avg_12m": "최근 12개월의 월평균 활동량은 어느 정도였나요? (모르면 '모름')",
    "no_income_months_3m": "최근 3개월 중 ‘무소득’인 달이 몇 달이었나요? (0~3)",
    "payment_stop": "최근에 사회보험/공공요금/대출 등의 ‘납부를 중단한 적’이 있나요? (예/아니오)",
    "wage_arrears": "최근에 ‘체불(미지급)’을 경험한 적이 있나요? (예/아니오)",
}

ORDER: List[QuestionKey] = [
    "income_avg_3m",
    "income_avg_12m",
    "activity_avg_3m",
    "activity_avg_12m",
    "no_income_months_3m",
    "payment_stop",
    "wage_arrears",
]

OPTIONAL_KEYS = {"activity_avg_3m", "activity_avg_12m"}  # 미입력 허용


# =========================
# 3) 로컬 파서(우선)
# =========================

def parse_korean_money_to_won(text: str) -> Optional[float]:
    t = text.replace(",", "").strip().lower()

    if any(k in t for k in ["소득 없음", "무소득", "없음", "0원", "0 원"]):
        return 0.0

    # 숫자 + 단위(원/만원/천원/백만원 등)
    m = re.search(r"(-?\d+(\.\d+)?)\s*(원|만원|천원|백만|백만원)?", t)
    if not m:
        return None

    val = float(m.group(1))
    unit = m.group(3)

    if unit in [None, "원"]:
        return val
    if unit == "천원":
        return val * 1_000
    if unit == "만원":
        return val * 10_000
    if unit in ["백만", "백만원"]:
        return val * 1_000_000

    return None

def parse_yes_no(text: str) -> Optional[bool]:
    t = text.strip().lower()
    if any(k in t for k in ["예", "네", "맞", "있", "y", "yes", "응"]):
        return True
    if any(k in t for k in ["아니", "없", "n", "no"]):
        return False
    return None

def parse_int_0_3(text: str) -> Optional[int]:
    m = re.search(r"(\d+)", text)
    if not m:
        return None
    v = int(m.group(1))
    return v if 0 <= v <= 3 else None

def try_parse_locally(key: QuestionKey, answer: str) -> Optional[Any]:
    a = answer.strip()

    if key in ["income_avg_3m", "income_avg_12m"]:
        return parse_korean_money_to_won(a)

    if key in ["activity_avg_3m", "activity_avg_12m"]:
        if "모름" in a or "몰라" in a:
            return None
        m = re.search(r"(-?\d+(\.\d+)?)", a.replace(",", ""))
        return float(m.group(1)) if m else None

    if key == "no_income_months_3m":
        return parse_int_0_3(a)

    if key in ["payment_stop", "wage_arrears"]:
        return parse_yes_no(a)

    return None


# =========================
# 4) LLM 추출(로컬 파싱 실패 시)
#    - Structured Outputs(Pydantic) 우선
#    - 실패 시 JSON-only fallback
# =========================

class Extraction(BaseModel):
    income_avg_3m: Optional[float] = None
    income_avg_12m: Optional[float] = None
    activity_avg_3m: Optional[float] = None
    activity_avg_12m: Optional[float] = None
    no_income_months_3m: Optional[int] = Field(None, description="0~3")
    payment_stop: Optional[bool] = None
    wage_arrears: Optional[bool] = None
    clarification_question: Optional[str] = None

def json_dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)

def openai_client() -> Any:
    # 사용자 요청대로 import openai를 사용
    return openai.OpenAI()

def llm_extract_field(client: Any, key: QuestionKey, user_answer: str) -> Tuple[Optional[Any], Optional[str]]:
    system = (
        "You extract structured fields from a Korean user's answer for a pre-check chatbot.\n"
        "Do NOT make any official eligibility decision.\n"
        "If ambiguous, set value to null and ask ONE clarification question in Korean."
    )
    payload = {
        "target_field": key,
        "user_answer": user_answer,
        "rules": [
            "Money -> convert to KRW (원). If 만원/백만원/천원, convert.",
            "no_income_months_3m must be an integer 0~3.",
            "Yes/No -> boolean.",
            "Return only structured fields."
        ],
        "examples": [
            {"user_answer": "최근 3개월은 월 120만원 정도요", "income_avg_3m": 1200000},
            {"user_answer": "무소득 월은 2달", "no_income_months_3m": 2},
            {"user_answer": "납부중단은 없어요", "payment_stop": False},
        ],
    }

    # 1) Structured outputs
    try:
        resp = client.responses.parse(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": json_dump(payload)},
            ],
            text_format=Extraction,
        )
        parsed: Extraction = resp.output_parsed
        value = getattr(parsed, key)
        return value, parsed.clarification_question
    except Exception:
        # 2) JSON-only fallback
        fallback_system = system + "\nReturn ONLY valid JSON. No markdown. No extra text."
        resp2 = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": fallback_system},
                {"role": "user", "content": json_dump(payload)},
            ],
        )
        text = getattr(resp2, "output_text", "") or ""
        if not text:
            return None, "답변이 애매해요. 숫자(예: 120만원) 또는 예/아니오 형태로 다시 말해줄래요?"
        try:
            parsed = Extraction.model_validate_json(text)
            value = getattr(parsed, key)
            return value, parsed.clarification_question
        except Exception:
            return None, "답변이 애매해요. 숫자(예: 120만원) 또는 예/아니오 형태로 다시 말해줄래요?"


# =========================
# 5) 결과 설명 생성(선택)
# =========================

class Explanation(BaseModel):
    headline: str
    reasons: List[str]
    caution: str

def llm_explain_result(client: Any, ind: IndicatorResult, inp: Inputs) -> Explanation:
    system = (
        "You are a Korean pre-check assistant.\n"
        "Do NOT claim official eligibility. Say it's a self-check signal.\n"
        "Be concise and practical."
    )
    payload = {
        "computed": ind.model_dump(),
        "inputs": inp.__dict__,
        "requirements": {
            "headline": "한 줄 요약",
            "reasons": "3~6개 근거 bullet",
            "caution": "공식 확인 권고 문장",
        }
    }
    try:
        resp = client.responses.parse(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": json_dump(payload)},
            ],
            text_format=Explanation,
        )
        return resp.output_parsed
    except Exception:
        # 실패 시 최소 설명(LLM 없이)
        return Explanation(
            headline=ind.likelihood_label,
            reasons=[
                f"S 점수 {ind.S:.1f} (A:{ind.A:.2f}, I:{ind.I:.2f}, R:{ind.R:.2f})",
                "최근/과거 대비 급감 정도를 기반으로 ‘우선 검토 신호’를 만든 결과입니다.",
            ],
            caution="이 결과는 비공식 자가진단이며, 실제 자격은 제도 요건에 따라 달라집니다. 공식 안내로 최종 확인하세요."
        )


# =========================
# 6) Streamlit UI (챗봇)
# =========================

def get_next_question_key(inp: Inputs) -> Optional[QuestionKey]:
    for k in ORDER:
        if getattr(inp, k) is None:
            # 활동량은 미입력 허용: 사용자가 '모름'이면 그냥 넘어갈 수 있게 UI에서 처리
            return k
    return None

def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "inputs" not in st.session_state:
        st.session_state.inputs = Inputs()
    if "client_ready" not in st.session_state:
        st.session_state.client_ready = True
    if "client" not in st.session_state:
        try:
            st.session_state.client = openai_client()
        except Exception:
            st.session_state.client = None
            st.session_state.client_ready = False

def add_message(role: str, content: str):
    st.session_state.messages.append({"role": role, "content": content})

def render_chat():
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

def main():
    st.set_page_config(page_title="비정형 노동자 실업급여 자가진단 챗봇", layout="centered")
    st.title("비정형 노동자 실업급여(재설계 지표) 자가진단 챗봇")
    st.caption("※ 비공식 자가진단입니다. 실제 수급 자격은 제도 요건에 따라 달라질 수 있습니다.")

    init_state()

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("대화 초기화"):
            st.session_state.messages = []
            st.session_state.inputs = Inputs()
    with col2:
        st.write("")

    render_chat()

    inp: Inputs = st.session_state.inputs
    next_key = get_next_question_key(inp)

    # 시작 안내/첫 질문 자동 출력
    if len(st.session_state.messages) == 0:
        add_message("assistant", "안녕하세요. 몇 가지 질문에 답하면 ‘지원 가능성(자가진단)’을 계산해드릴게요.")
        next_key = get_next_question_key(inp)
        add_message("assistant", QUESTIONS[next_key])

    # 사용자 입력
    user_text = st.chat_input("여기에 답변을 입력하세요")
    if user_text is not None and user_text.strip():
        add_message("user", user_text)

        # 현재 질문 키 재확인(사용자 입력 시점)
        next_key = get_next_question_key(inp)
        if next_key is None:
            add_message("assistant", "이미 필요한 정보가 모두 입력됐어요. 아래 결과를 확인하세요.")
        else:
            # 활동량 질문에서 '모름' 허용
            if next_key in OPTIONAL_KEYS and ("모름" in user_text or "몰라" in user_text):
                # 그대로 None 유지하고 다음 질문으로
                add_message("assistant", "활동량은 미입력으로 진행할게요.")
                # 다음 질문
                nk = get_next_question_key(inp)
                if nk is not None:
                    add_message("assistant", QUESTIONS[nk])

            else:
                # 1) 로컬 파싱
                v = try_parse_locally(next_key, user_text)

                # 2) 로컬 파싱 실패 → LLM 추출
                if v is None and st.session_state.client_ready and st.session_state.client is not None:
                    value, clar_q = llm_extract_field(st.session_state.client, next_key, user_text)
                    if value is None:
                        add_message("assistant", clar_q or "답변이 애매해요. 숫자/예-아니오 형태로 다시 말해줄래요?")
                    else:
                        setattr(inp, next_key, value)
                        add_message("assistant", f"확인했어요. ({next_key} 입력 완료)")
                        nk = get_next_question_key(inp)
                        if nk is not None:
                            add_message("assistant", QUESTIONS[nk])
                elif v is None:
                    add_message("assistant", "답변에서 값을 해석하기 어려워요. 숫자/예-아니오 형태로 다시 말해줄래요?")
                else:
                    setattr(inp, next_key, v)
                    add_message("assistant", f"확인했어요. ({next_key} 입력 완료)")
                    nk = get_next_question_key(inp)
                    if nk is not None:
                        add_message("assistant", QUESTIONS[nk])

        st.rerun()

    # 모든 필수값(소득 2개) 입력되면 계산 영역 표시
    #ready = (inp.income_avg_3m is not None) and (inp.income_avg_12m is not None)
    ready = (next_key is None)  # 모든 질문이 끝났을 때만
    if ready:
        st.divider()
        st.subheader("자가진단 결과")

        try:
            ind = compute_indicators(inp)
        except Exception as e:
            st.error(f"계산 오류: {e}")
            return

        st.metric("종합 점수 S", f"{ind.S:.1f} / 100", help="S=100(0.5A+0.3I+0.2R)")
        st.write(f"- 구간: **{ind.band}**")
        st.write(f"- 해석: **{ind.likelihood_label}**")
        st.write(f"- Ai(활동): {ind.A:.3f} / Ii(소득): {ind.I:.3f} / Ri(위험): {ind.R:.3f}")

        if ind.notes:
            with st.expander("불확실성/주의 사항(입력 누락 등)"):
                for n in ind.notes:
                    st.write(f"- {n}")

        # LLM 설명(선택)
        if st.session_state.client_ready and st.session_state.client is not None:
            expl = llm_explain_result(st.session_state.client, ind, inp)
            st.write("**요약**")
            st.write(expl.headline)
            st.write("**근거**")
            for r in expl.reasons:
                st.write(f"- {r}")
            st.write("**유의**")
            st.write(expl.caution)
        else:
            st.info("OPENAI_API_KEY가 설정되지 않아 LLM 설명 기능은 비활성화되었습니다. 점수 계산은 로컬에서 수행됩니다.")


if __name__ == "__main__":
    main()
