import re
import torch
import speech_recognition as sr
import requests
import json
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


# Question logic
QUESTION_WORDS = {
    "what", "why", "how", "when", "where", "who", "whom", "which", "whose",
    "is", "are", "do", "does", "did", "can", "could", "will", "would",
    "should", "shall", "may", "might", "have", "has", "had", "am", "was", "were"
}

QUESTION_PHRASES = [
    "can you", "do you", "did you", "have you", "would you", "is it", "are you",
    "could you", "will you", "should i", "why is", "how does", "what if", "am i"
]

IMPERATIVE_VERBS = [
    "explain", "describe", "tell", "list", "define", "compare", "differentiate",
    "outline", "summarize", "elaborate", "discuss"
]

DATA_KEYWORDS = {
    "sql", "select", "insert", "update", "delete", "drop", "alter", "create", "table",
    "join", "joins", "left join", "right join", "inner join", "outer join", "merge",
    "group by", "having", "where", "order by", "window function", "cte", "union", "rank",
    "dense_rank", "lead", "lag", "partition", "null", "vlookup", "xlookup", "excel",
    "power bi", "powerbi", "pivot", "chart", "dax", "dashboard", "measure", "kpi",
    "etl", "pipeline", "airflow", "dag", "job", "task", "bigquery", "snowflake", "spark",
    "pandas", "numpy", "dataframe", "python", "read_csv", "fillna", "dropna", "query", "program"
}

# Question detection
def is_question(text: str) -> bool:
    text = text.lower().strip()
    words = re.findall(r'\b\w+\b', text)

    if text.endswith("?"):
        return True
    if words and words[0] in QUESTION_WORDS:
        return True
    if words and words[0] in IMPERATIVE_VERBS:
        return True
    if any(phrase in text for phrase in QUESTION_PHRASES):
        return True
    if any(keyword in text for keyword in DATA_KEYWORDS):
        return True
    return False

def ask_deepseek(question: str) -> str:
    st.info("Asking DeepSeek LLM...")
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": "Bearer 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",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": "deepseek/deepseek-r1:free",  # Better for coding/sql
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an interview assistant bot. "
                            "Answer each question in a very concise way, ideal for quick memory refresh. "
                            "Limit answers to a 15 to 20 words"
                            "Support coding and SQL questions too. Do not explain unless asked."
                        )
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ]
            })
        )
        return response.json()['choices'][0]["message"]["content"]
    except Exception as e:
        return f"❌ LLM error: {e}"

# Streamlit UI
st.set_page_config(page_title="Real-time Audio Transcriber & Interview Assistant", layout="centered")

st.title("🗣️ Real-time Audio Transcriber")
st.markdown("---")

st.write(
    "This application transcribes your speech in real-time. "
    "If it detects a question, it will attempt to answer using the DeepSeek LLM. "
    "Click 'Start Listening' to begin."
)

if 'listening' not in st.session_state:
    st.session_state.listening = False
if 'transcribed_text' not in st.session_state:
    st.session_state.transcribed_text = ""
if 'llm_answer' not in st.session_state:
    st.session_state.llm_answer = ""
if 'status_message' not in st.session_state:
    st.session_state.status_message = "Ready to start listening."

def start_listening_callback():
    st.session_state.listening = True
    st.session_state.transcribed_text = ""
    st.session_state.llm_answer = ""
    st.session_state.status_message = "🎤 Listening..."

def stop_listening_callback():
    st.session_state.listening = False
    st.session_state.status_message = "Stopped listening."

col1, col2 = st.columns(2)

with col1:
    start_button = st.button("Start Listening", on_click=start_listening_callback, use_container_width=True)
with col2:
    stop_button = st.button("Stop Listening", on_click=stop_listening_callback, use_container_width=True)

st.markdown("---")

st.subheader("Status")
status_placeholder = st.empty()
status_placeholder.write(st.session_state.status_message)

st.subheader("Transcribed Text")
transcribed_text_placeholder = st.empty()
transcribed_text_placeholder.write(st.session_state.transcribed_text)

st.subheader("LLM Answer (if question detected)")
llm_answer_placeholder = st.empty()
llm_answer_placeholder.write(st.session_state.llm_answer)

# Listen and respond loop within Streamlit
if st.session_state.listening:
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
            st.session_state.status_message = "⏳ Transcribing..."
            status_placeholder.write(st.session_state.status_message)
            
            text = recognizer.recognize_google(audio)
            st.session_state.transcribed_text = f"📝 You said: {text}"
            transcribed_text_placeholder.write(st.session_state.transcribed_text)

            if is_question(text):
                st.session_state.status_message = "✅ Detected as a question! Getting answer..."
                status_placeholder.write(st.session_state.status_message)
                answer = ask_deepseek(text)
                st.session_state.llm_answer = f"📘 Answer:\n{answer}"
                llm_answer_placeholder.write(st.session_state.llm_answer)
                st.session_state.status_message = "🎤 Listening..." # Resume listening message
                status_placeholder.write(st.session_state.status_message)
            else:
                st.session_state.status_message = "No question detected. 🎤 Listening..."
                status_placeholder.write(st.session_state.status_message)

            # Re-run the Streamlit app to continue listening
            st.rerun()

        except sr.UnknownValueError:
            st.session_state.status_message = "⚠️ Could not understand audio. 🎤 Listening..."
            status_placeholder.write(st.session_state.status_message)
            st.rerun()
        except sr.WaitTimeoutError:
            st.session_state.status_message = "⏰ Timeout — no speech. 🎤 Listening..."
            status_placeholder.write(st.session_state.status_message)
            st.rerun()
        except Exception as e:
            st.session_state.status_message = f"❌ Error: {e}. Stopping listening."
            status_placeholder.write(st.session_state.status_message)
            st.session_state.listening = False
            st.rerun()
