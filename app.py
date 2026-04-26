import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide"
)

st.write("App starting...")
st.write(f"faiss_index exists: {os.path.exists('faiss_index')}")
st.write(f"Files: {os.listdir('.')}")

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { color: #e94560; font-size: 2.5rem; margin: 0; font-weight: 700; }
    .main-header p { color: #a8b2d8; margin: 0.5rem 0 0; font-size: 1rem; }
    .stat-card {
        background: #16213e;
        border: 1px solid #0f3460;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        color: white;
    }
    .stat-card h3 { color: #e94560; font-size: 1.8rem; margin: 0; }
    .stat-card p { color: #a8b2d8; margin: 0; font-size: 0.85rem; }
    .footer {
        text-align: center;
        color: #a8b2d8;
        font-size: 0.8rem;
        margin-top: 2rem;
        padding: 1rem;
        border-top: 1px solid #0f3460;
    }
    .verify-box {
        background: #0f3460;
        border-left: 3px solid #e94560;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin-top: 0.5rem;
        color: #a8b2d8;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    vectorstore = FAISS.load_local(
        "faiss_index", embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY")
    )

def get_answer(question, vectorstore, llm):
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""You are a helpful multilingual document assistant.
The context below is in English. Read it and answer the question in the SAME language as the question.
If question is in Bengali, answer in Bengali.
If question is in Hindi, answer in Hindi.
If question is in Tamil, answer in Tamil.
If question is in Telugu, answer in Telugu.
If question is in Marathi, answer in Marathi.
If question is in Kannada, answer in Kannada.
If question is in Punjabi, answer in Punjabi.
If question is in Gujarati, answer in Gujarati.
If question is in Malayalam, answer in Malayalam.
If question is in Urdu, answer in Urdu.
If question is in any other language, answer in that same language.
Translate the relevant information from the context and answer naturally.
If the topic is genuinely not in the context, say so politely in the same language.

Context:
{context}

Question: {question}

Answer:"""
    response = llm.invoke(prompt)
    return response.content, docs

def translate_to_english(text, llm):
    prompt = f"""Translate the following text to English.
Only return the translation, nothing else.

Text: {text}

English Translation:"""
    response = llm.invoke(prompt)
    return response.content

# --- UI ---
st.markdown("""
<div class="main-header">
    <h1>DocuMind AI</h1>
    <p>Intelligent Document Assistant — Ask in any language, get answers in the same language</p>
</div>
""", unsafe_allow_html=True)

vectorstore = load_vectorstore()
llm = load_llm()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="stat-card"><h3>22+</h3><p>Indian Languages</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="stat-card"><h3>50+</h3><p>Total Languages</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="stat-card"><h3>RAG</h3><p>Powered by AI</p></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="stat-card"><h3>Fast</h3><p>Groq LPU Engine</p></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### DocuMind AI")
    st.markdown("---")
    st.markdown("**Indian Languages**")
    for lang in ["🇮🇳 Hindi", "🇮🇳 Bengali", "🇮🇳 Tamil", "🇮🇳 Telugu",
                 "🇮🇳 Marathi", "🇮🇳 Gujarati", "🇮🇳 Kannada",
                 "🇮🇳 Malayalam", "🇮🇳 Punjabi", "🇮🇳 Urdu",
                 "🇮🇳 Odia", "🇮🇳 Assamese"]:
        st.markdown(f"• {lang}")
    st.markdown("**Other Languages**")
    for lang in ["🌍 English", "🌍 Spanish", "🌍 French",
                 "🌍 German", "🌍 Arabic", "🌍 Chinese", "🌍 Japanese"]:
        st.markdown(f"• {lang}")
    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown("1. Upload PDF to docs/")
    st.markdown("2. Run python ingest.py")
    st.markdown("3. Ask in any language!")
    st.markdown("---")
    st.markdown("**Test All Languages**")
    if st.button("Run Language Test ↗"):
        st.session_state.messages = []
        st.session_state.translations = {}
        test_questions = [
            ("Hindi", "डेटा साइंस क्या है?"),
            ("Bengali", "ডেটা সায়েন্স কী?"),
            ("Tamil", "தரவு அறிவியல் என்றால் என்ன?"),
            ("Telugu", "డేటా సైన్స్ అంటే ఏమిటి?"),
            ("Marathi", "डेटा सायन्स म्हणजे काय?"),
            ("Kannada", "ಡೇಟಾ ಸೈನ್ಸ್ ಎಂದರೇನು?"),
            ("Gujarati", "ડેટા સાયન્સ શું છે?"),
            ("Punjabi", "ਡੇਟਾ ਸਾਇੰਸ ਕੀ ਹੈ?"),
            ("Malayalam", "ഡാറ്റ സയൻസ് എന്താണ്?"),
            ("Urdu", "ڈیٹا سائنس کیا ہے؟"),
        ]
        progress = st.progress(0)
        for i, (lang, q) in enumerate(test_questions):
            with st.spinner(f"Testing {lang}..."):
                answer, _ = get_answer(q, vectorstore, llm)
                st.session_state.messages.append({"role": "user", "content": f"[{lang}] {q}"})
                st.session_state.messages.append({"role": "assistant", "content": answer})
            progress.progress((i + 1) / len(test_questions))
        st.rerun()
    st.markdown("---")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.translations = {}
        st.rerun()

st.markdown("### Ask your document anything")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "translations" not in st.session_state:
    st.session_state.translations = {}

if not st.session_state.messages:
    st.info("Try: 'What is RAG?' | 'RAG kya hai?' | 'ডেটা সায়েন্স কি?' | Or click 'Run Language Test' in sidebar!")

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant":
            col_btn, col_empty = st.columns([1, 3])
            with col_btn:
                btn_key = f"verify_{i}"
                if st.button("🔍 Verify in English", key=btn_key):
                    with st.spinner("Translating..."):
                        translation = translate_to_english(msg["content"], llm)
                        st.session_state.translations[i] = translation
                        st.rerun()
            if i in st.session_state.translations:
                st.markdown(f'<div class="verify-box">📋 <strong>English Translation:</strong><br>{st.session_state.translations[i]}</div>', unsafe_allow_html=True)

if question := st.chat_input("Koi bhi bhasha mein poochho..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = get_answer(question, vectorstore, llm)
        st.write(answer)
        with st.expander("View Sources"):
            for i, doc in enumerate(sources):
                st.caption(f"Source {i+1} — Page {doc.metadata.get('page', '?')}: {doc.page_content[:250]}...")
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()

st.markdown('<div class="footer">DocuMind AI — Built with LangChain, FAISS, Groq & Streamlit | Supports 50+ Languages</div>', unsafe_allow_html=True)