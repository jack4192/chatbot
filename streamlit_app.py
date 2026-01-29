import time
import random
import streamlit as st
from openai import OpenAI
from openai import RateLimitError, APIError, APITimeoutError

# -----------------------------
# 설정
# -----------------------------
DEFAULT_MODEL = "gpt-4o-mini"   # 가능하면 이걸 권장 (gpt-3.5-turbo는 구환경에서 에러/제한이 더 잦을 수 있음)
COOLDOWN_SEC = 1.0             # 연속 전송 방지(세션당)

# -----------------------------
# 백오프 래퍼
# -----------------------------
def with_backoff(call_fn, max_retries: int = 5):
    """
    429(RateLimit)나 일시적인 네트워크 오류에 대해 지수 백오프 재시도.
    """
    for i in range(max_retries):
        try:
            return call_fn()
        except (RateLimitError, APITimeoutError, APIError) as e:
            # 마지막이면 그대로 raise
            if i == max_retries - 1:
                raise
            sleep = (2 ** i) + random.random()
            time.sleep(sleep)

# -----------------------------
# UI
# -----------------------------
st.title("💬 Chatbot")
st.write(
    "This is a simple chatbot that uses OpenAI to generate responses. "
    "To use this app, provide an OpenAI API key."
)

# Streamlit Cloud에서는 Secrets 권장:
# st.secrets["OPENAI_API_KEY"] 를 먼저 시도하고, 없으면 입력받기
secret_key = None
try:
    secret_key = st.secrets.get("OPENAI_API_KEY", None)
except Exception:
    secret_key = None

openai_api_key = secret_key or st.text_input("OpenAI API Key", type="password")

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
    st.stop()

# Create an OpenAI client.
client = OpenAI(api_key=openai_api_key)

# Session state 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_send_ts" not in st.session_state:
    st.session_state.last_send_ts = 0.0

# 기존 메시지 렌더
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 입력
if prompt := st.chat_input("What is up?"):
    # 쿨다운 (너무 빠른 연속 전송 방지)
    now = time.time()
    if now - st.session_state.last_send_ts < COOLDOWN_SEC:
        st.warning("잠깐만요. 너무 빠르게 연속 전송 중이에요.")
        st.stop()
    st.session_state.last_send_ts = now

    # 사용자 메시지 저장/표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # assistant 응답 생성
    with st.chat_message("assistant"):
        try:
            # 스트리밍 호출 (백오프 포함)
            def _stream_call():
                return client.chat.completions.create(
                    model=DEFAULT_MODEL,
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    stream=True,
                    # 필요하면 토큰 제한 추가:
                    # max_tokens=400,
                )

            stream = with_backoff(_stream_call)

            # 스트리밍 출력
            response = st.write_stream(stream)

        except RateLimitError:
            # 429면 안내 메시지
            st.error(
                "요청이 너무 많아(OpenAI Rate Limit) 잠시 차단됐어요. "
                "몇 초 후 다시 시도해 주세요."
            )
            st.stop()

        except Exception as e:
            # 스트리밍이 실패할 수 있으니 non-stream fallback
            st.warning("스트리밍 응답에 실패해서 일반 응답으로 재시도할게요.")
            try:
                def _non_stream_call():
                    return client.chat.completions.create(
                        model=DEFAULT_MODEL,
                        messages=[
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages
                        ],
                        stream=False,
                    )
                resp = with_backoff(_non_stream_call)
                response = resp.choices[0].message.content
                st.markdown(response)
            except Exception as e2:
                st.error(f"오류가 발생했어요: {type(e2).__name__}")
                st.stop()

    # 응답 저장
    st.session_state.messages.append({"role": "assistant", "content": response})
