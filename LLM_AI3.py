import streamlit as st
from typing import Optional, Dict, Any, List

# =========================
# Config
# =========================
st.set_page_config(page_title="나도 실엽급여줘 챗봇", layout="wide")
EPS = 1e-9

# =========================
# Utils: 지표 계산
# =========================
def clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def compute_indices(
    avg_income_12m: float,
    avg_income_3m: float,
    avg_activity_12m: float,
    avg_activity_3m: float,
    no_income_months_last3: int,
    pay_stop: bool,
    wage_arrears: bool,
) -> Dict[str, Any]:
    rA = avg_activity_3m / (avg_activity_12m + EPS)
    Ai = clip01(1.0 - rA)

    rI = avg_income_3m / (avg_income_12m + EPS)
    Ii = clip01(1.0 - rI)

    g = clip01(no_income_months_last3 / 3.0)
    p = 1.0 if pay_stop else 0.0
    w = 1.0 if wage_arrears else 0.0
    Ri = clip01(0.5 * g + 0.3 * p + 0.2 * w)

    S = 100.0 * (0.5 * Ai + 0.3 * Ii + 0.2 * Ri)

    if S >= 70:
        tier = "A"
        verdict = "실질 실업 가능성 높음 → 우선 검토 대상"
    elif S >= 50:
        tier = "B"
        verdict = "실업 위험 상태 → 추가 확인(서류 검토 병행)"
    else:
        tier = "C"
        verdict = "현 입력 기준, 실업 상태로 판단되기 어려움"

    return {
        "Ai": Ai, "Ii": Ii, "Ri": Ri, "S": S, "tier": tier, "verdict": verdict,
        "rA": rA, "rI": rI, "g": g, "p": p, "w": w
    }

# =========================
# Utils: 규칙 기반 서류 추천
# =========================
def recommend_docs_rule_based(
    worker_type: str,
    industries: List[str],
    score_detail: Dict[str, Any],
    pay_stop: bool,
    wage_arrears: bool,
    no_income_months_last3: int
) -> str:
    tier = score_detail.get("tier", "?")
    ind = ", ".join(industries) if industries else "미선택"

    lines = []
    lines.append(f"### 준비 서류 추천(규칙 기반) — {worker_type} / {ind} / 구간 {tier}\n")
    lines.append("아래 체크리스트는 **‘소득 감소’(Ii), ‘활동 감소’(Ai), ‘위험 신호’(Ri)를 입증**하는 자료로 구성했습니다.\n")

    docs = [
        ("플랫폼/거래 소득 증빙",
         "정산서(월별), 통장 입금내역, 세금신고(종소세/부가세/원천징수) 중 가능한 것",
         "Ii(소득 급감)을 객관적으로 입증",
         "정산서가 없으면 통장 입금내역 + 매출 캡처로 대체"),
        ("활동량(업무 수행) 증빙",
         "완료건수/콜 수/운행내역/작업건수 리포트(다운로드 파일 + 캡처)",
         "Ai(활동 중단)을 직접 입증",
         "리포트가 없으면 앱 내 통계 화면 캡처 + 날짜별 로그"),
        ("계약/관계 증빙(플랫폼/발주처)",
         "가입/계약/위수탁/용역 계약서, 약관 동의, 계정 정보",
         "‘노무 제공 관계’의 실체 설명에 도움",
         "계약서 없으면 가입 화면/약관 캡처, 발주처 안내문"),
        ("업무 중단 정황(객관 이벤트)",
         "계정 정지/배정 중단 공지, 고객센터 문의·답변, 계약 종료 통지(문자/메일)",
         "‘본인 의사’가 아닌 중단 정황 보강",
         "증빙이 없으면 상담 로그/통화 기록 요약이라도 확보"),
    ]

    if no_income_months_last3 >= 1:
        docs.append(("무소득 공백 증빙",
                    "최근 3개월 정산 0원(또는 급락) 내역, 통장 거래 공백",
                    "Ri의 g(무소득 공백) 근거 강화",
                    "정산 0원 캡처 + 통장 거래내역(입금 없음) 조합 권장"))
    if pay_stop:
        docs.append(("납부 중단(해당 시)",
                    "사회보험/공공요금 고지서, 미납 안내, 납부 이력 화면",
                    "Ri의 p(납부 중단) 근거 확보",
                    "고지서 없으면 앱/홈페이지 납부내역 캡처"))
    if wage_arrears:
        docs.append(("체불/미지급(해당 시)",
                    "미지급 정산내역, 지급예정일 안내, 이의제기/신고 자료, 대화 기록",
                    "Ri의 w(체불) 근거 + ‘불가피한 중단’ 정황",
                    "정산 화면 + CS 답변 캡처만 있어도 1차 근거"))

    if "배달" in industries:
        docs.append(("배달업 특화 증빙",
                    "플랫폼별 월정산 리포트, 운행기록, 콜 수/완료 수",
                    "활동량·소득 변화 원인 설명에 유리",
                    "앱 통계 캡처 + 월정산 PDF 조합"))
    if "프리랜서(디자인/개발/콘텐츠)" in industries:
        docs.append(("프리랜서 특화 증빙",
                    "프로젝트 계약/발주서, 납품·검수 기록, 인보이스, 클라이언트 메시지",
                    "작업 건수/매출 감소를 구조적으로 설명",
                    "계약서 없으면 메일·메신저 발주/검수 기록"))
    if "대리/운전" in industries:
        docs.append(("대리/운전 특화 증빙",
                    "운행 내역, 콜 수, 정산 내역, 수수료/정책 변경 공지",
                    "수요 감소(콜 감소) 정황을 수치로 입증",
                    "운행 캡처 + 정산내역 조합"))

    docs = docs[:7]

    lines.append("#### 우선순위 TOP 7\n")
    for i, (title, what, why, alt) in enumerate(docs, 1):
        lines.append(f"{i}. **{title}**")
        lines.append(f"   - 무엇: {what}")
        lines.append(f"   - 왜: {why}")
        lines.append(f"   - 대체: {alt}")

    lines.append("\n#### 판정을 더 정확히 하려면 추가로 확인할 질문 3개")
    lines.append("1) 최근 18개월 동안 고용보험/노무 제공 이력(가입·납부 또는 유사 증빙)이 있나요?")
    lines.append("2) 활동 감소가 ‘본인 선택’인지 ‘배정/수요 감소·계정 제한’ 같은 외부 요인인지요?")
    lines.append("3) 소득/활동 급락의 ‘시점(월)’이 명확한가요? 그 달의 공지/대화/정산을 묶을 수 있나요?")

    return "\n".join(lines)

# =========================
# Utils: 규칙 기반 챗
# =========================
FAQ_KB = [
    {
        "keywords": ["보강", "어떤 부분", "뭘 준비", "강화", "입증"],
        "answer": (
            "점수 구조상 **Ai(활동)**, **Ii(소득)**, **Ri(위험신호)** 중에서 "
            "본인 점수가 크게 나온 축을 ‘증빙’하는 게 효율적입니다.\n\n"
            "- **Ai가 높다(활동 급감)** → 완료건수/콜수/운행내역 같은 *활동 로그*를 월별로 확보\n"
            "- **Ii가 높다(소득 급감)** → 정산서 + 통장입금(월별)로 *소득 급감*을 월별로 정리\n"
            "- **Ri가 높다(무소득/납부중단/체불)** → 0원 정산, 미납 안내, 미지급 정산/CS 기록 같은 *사건성 증빙* 첨부\n\n"
            "핵심은 ‘최근 3개월’과 ‘과거 12개월’의 대비를 **같은 포맷(월별 표/캡처)**으로 만드는 겁니다."
        ),
    },
    {
        "keywords": ["정산", "정산내역", "배달", "콜", "완료", "라이더"],
        "answer": (
            "배달 업종이면 설득력 높은 조합은 보통:\n\n"
            "1) **월별 정산서(PDF/다운로드)**\n"
            "2) **월별 완료건수/콜수 통계(앱 통계 캡처)**\n"
            "3) **통장 입금내역(정산 입금 매칭)**\n\n"
            "이 3개가 함께 있으면 소득(Ii)+활동(Ai)+실지급(보강)이 한 번에 묶입니다."
        ),
    },
    {
        "keywords": ["A", "B", "C", "구간", "점수", "왜"],
        "answer": (
            "점수 S는 **S=100*(0.5Ai + 0.3Ii + 0.2Ri)** 구조라서 "
            "가중치가 큰 **Ai(활동 중단)**이 크게 작동합니다.\n\n"
            "최근 3개월이 과거 12개월 대비 얼마나 급락했는지가 점수 대부분을 결정합니다."
        ),
    },
]

def rule_based_chat(user_text: str, score_detail: Optional[Dict[str, Any]] = None) -> str:
    t = (user_text or "").strip().lower()

    for item in FAQ_KB:
        if any(k in t for k in item["keywords"]):
            return item["answer"]

    if score_detail:
        Ai, Ii, Ri = score_detail["Ai"], score_detail["Ii"], score_detail["Ri"]
        top = max([("Ai(활동)", Ai), ("Ii(소득)", Ii), ("Ri(위험신호)", Ri)], key=lambda x: x[1])[0]
        return (
            f"현재 입력 기준으로는 **{top}** 쪽이 가장 크게 작동했습니다.\n\n"
            "더 정확히 안내하려면:\n"
            "- 활동/소득 급락이 ‘수요 감소/배정 중단/계정 제한’ 같은 외부 요인인지\n"
            "- 아니면 ‘개인 사정으로 일을 줄임’인지\n"
            "중 어느 쪽인지 알려주세요.\n\n"
            "그리고 현재 보유한 자료(정산서/활동로그/통장입금) 중 무엇이 있는지도 말해주면 "
            "그 기준으로 ‘최소 서류 세트’를 정리해줄게요."
        )

    return "업종과(배달/프리랜서 등) 궁금한 포인트(서류/점수해석/보강)를 함께 적어주면 더 정확히 답할 수 있어요."

# =========================
# Session State Init
# =========================
if "step" not in st.session_state:
    st.session_state.step = 1  # 1~4 입력 단계, 5 결과
if "score_detail" not in st.session_state:
    st.session_state.score_detail = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "결과 페이지에서 질문하면, 점수(Ai/Ii/Ri)를 기준으로 규칙 기반 안내를 제공합니다."}
    ]

# 기본값 저장(입력 단계 넘어가도 값 유지)
defaults = {
    "worker_type": "비정형(플랫폼/프리랜서/특고/단기·단시간 등)",
    "industries": ["배달"],
    "avg_income_12m": 240.0,
    "avg_income_3m": 50.0,
    "avg_activity_12m": 180.0,
    "avg_activity_3m": 20.0,
    "no_income_months_last3": 2,
    "pay_stop": False,
    "wage_arrears": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =========================
# Header
# =========================
st.title("나도 실업급여줘 챗봇")
st.caption("입력은 단계별로 진행되고, 계산 후 실엽급여 지원 가능성을 결과 페이지에서 2)판정·3)서류·4)챗으로 함께 표시됩니다.")

# =========================
# Navigation Helpers
# =========================
def go_next():
    st.session_state.step = min(st.session_state.step + 1, 5)

def go_prev():
    st.session_state.step = max(st.session_state.step - 1, 1)

def go_result():
    st.session_state.step = 5

# =========================
# Step UI
# =========================
step = st.session_state.step

# ---- STEP 1: 고용 형태 + 업종
if step == 1:
    st.subheader("1단계: 고용 형태 / 종사업종")
    st.session_state.worker_type = st.radio(
        "고용 형태(택1)",
        ["정형(정규직/표준고용)", "비정형(플랫폼/프리랜서/특고/단기·단시간 등)"],
        index=1 if st.session_state.worker_type.startswith("비정형") else 0,
        key="worker_type_radio",
    )

    st.session_state.industries = st.multiselect(
        "종사업종(복수 선택 가능)",
        ["배달", "프리랜서(디자인/개발/콘텐츠)", "대리/운전", "퀵/물류", "학습지/방문판매", "보험설계", "기타 플랫폼 노동", "기타"],
        default=st.session_state.industries,
        key="industries_select",
    )

    c1, c2 = st.columns([1, 1])
    with c2:
        st.button("다음 →", on_click=go_next, type="primary")

# ---- STEP 2: 소득 입력
elif step == 2:
    st.subheader("2단계: 소득 입력(단위: 만원, 월평균)")
    st.session_state.avg_income_12m = st.number_input(
        "과거 12개월 평균 소득",
        min_value=0.0,
        value=float(st.session_state.avg_income_12m),
        step=10.0,
        key="avg_income_12m_input",
    )
    st.session_state.avg_income_3m = st.number_input(
        "최근 3개월 평균 소득",
        min_value=0.0,
        value=float(st.session_state.avg_income_3m),
        step=10.0,
        key="avg_income_3m_input",
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        st.button("← 이전", on_click=go_prev)
    with c2:
        st.button("다음 →", on_click=go_next, type="primary")

# ---- STEP 3: 활동량 입력
elif step == 3:
    st.subheader("3단계: 활동량 입력(단위: 건/월)")
    st.session_state.avg_activity_12m = st.number_input(
        "과거 12개월 평균 활동량",
        min_value=0.0,
        value=float(st.session_state.avg_activity_12m),
        step=10.0,
        key="avg_activity_12m_input",
    )
    st.session_state.avg_activity_3m = st.number_input(
        "최근 3개월 평균 활동량",
        min_value=0.0,
        value=float(st.session_state.avg_activity_3m),
        step=5.0,
        key="avg_activity_3m_input",
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        st.button("← 이전", on_click=go_prev)
    with c2:
        st.button("다음 →", on_click=go_next, type="primary")

# ---- STEP 4: 위험 신호
elif step == 4:
    st.subheader("4단계: 위험 신호(최근 3개월)")
    st.session_state.no_income_months_last3 = st.slider(
        "무소득 월 수",
        min_value=0,
        max_value=3,
        value=int(st.session_state.no_income_months_last3),
        key="no_income_slider",
    )
    st.session_state.pay_stop = st.checkbox(
        "사회보험/공공요금 납부 중단(있음)",
        value=bool(st.session_state.pay_stop),
        key="pay_stop_check",
    )
    st.session_state.wage_arrears = st.checkbox(
        "체불 경험(있음)",
        value=bool(st.session_state.wage_arrears),
        key="wage_arrears_check",
    )

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.button("← 이전", on_click=go_prev)
    with c2:
        st.button("계산하기", type="primary", on_click=go_result)


# ---- STEP 5: 결과 페이지 (2+3+4 통합)
else:
    st.subheader("결과(2)판정 + (3)서류 추천 + (4)추가 질문")

    # 점수 계산 (결과 페이지 들어올 때마다 최신 상태로 계산)
    st.session_state.score_detail = compute_indices(
        avg_income_12m=float(st.session_state.avg_income_12m),
        avg_income_3m=float(st.session_state.avg_income_3m),
        avg_activity_12m=float(st.session_state.avg_activity_12m),
        avg_activity_3m=float(st.session_state.avg_activity_3m),
        no_income_months_last3=int(st.session_state.no_income_months_last3),
        pay_stop=bool(st.session_state.pay_stop),
        wage_arrears=bool(st.session_state.wage_arrears),
    )
    sd = st.session_state.score_detail

    # 상단 요약
    st.metric("종합 점수 S", f'{sd["S"]:.1f}', help="S = 100*(0.5Ai + 0.3Ii + 0.2Ri)")
    st.write(f"**구간:** {sd['tier']}  \n**판정:** {sd['verdict']}")

    # 2) 지표 상세
    with st.expander("지표 상세(왜 이렇게 나왔나?)", expanded=True):
        st.write(
            f"- Ai(활동 중단) = clip(1 - (최근3개월 활동/과거12개월 활동), 0, 1) = **{sd['Ai']:.3f}**\n"
            f"- Ii(소득 급감) = clip(1 - (최근3개월 소득/과거12개월 소득), 0, 1) = **{sd['Ii']:.3f}**\n"
            f"- Ri(위험 신호) = 0.5*g + 0.3*p + 0.2*w = **{sd['Ri']:.3f}**\n"
            f"  - g(무소득 공백)={sd['g']:.3f}, p(납부중단)={int(bool(st.session_state.pay_stop))}, w(체불)={int(bool(st.session_state.wage_arrears))}"
        )

    st.markdown("---")

    # 결과 2열 레이아웃: 왼쪽=서류, 오른쪽=챗
    left, right = st.columns([1.2, 1], gap="large")

    with left:
        st.subheader("3) 준비 서류 추천(체크리스트)")
        st.markdown(
            recommend_docs_rule_based(
                worker_type=str(st.session_state.worker_type),
                industries=list(st.session_state.industries),
                score_detail=sd,
                pay_stop=bool(st.session_state.pay_stop),
                wage_arrears=bool(st.session_state.wage_arrears),
                no_income_months_last3=int(st.session_state.no_income_months_last3),
            )
        )

    with right:
        st.subheader("4) 추가 질문(챗)")
        st.caption("점수 결과를 바탕으로 ‘어떤 부분을 보강할지’, ‘어떤 서류가 센지’를 질문해보세요.")

        for m in st.session_state.chat_messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        user_msg = st.chat_input("예: 배달 플랫폼 정산내역은 어떤 형식이 제일 설득력 있어?")
        if user_msg:
            st.session_state.chat_messages.append({"role": "user", "content": user_msg})
            with st.chat_message("user"):
                st.markdown(user_msg)

            reply = rule_based_chat(user_msg, score_detail=sd)
            st.session_state.chat_messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)

    st.markdown("---")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.button("← 입력 수정(이전 단계로)", on_click=go_prev)
    with c2:
        st.button("처음부터 다시 입력", on_click=lambda: setattr(st.session_state, "step", 1))

    st.caption("주의: 본 도구는 행정 판단을 대체하지 않으며, 실제 수급 요건/증빙은 관할 기관 안내를 따르세요.")
