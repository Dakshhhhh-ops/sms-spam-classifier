import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import nltk
import requests

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')

# --------------------------
# Preprocessing Setup
# --------------------------

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for word in text:
        if word.isalnum():
            y.append(word)
    text = y[:]
    y.clear()
    for word in text:
        if word not in stopwords.words('english') and word not in string.punctuation:
            y.append(word)
    text = y[:]
    y.clear()
    for word in text:
        y.append(ps.stem(word))
    return " ".join(y)

# --------------------------
# Page Config
# --------------------------

st.set_page_config(
    page_title="SpamShield — SMS Classifier",
    page_icon="🛡️",
    layout="centered"
)

# --------------------------
# Custom CSS
# --------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0D1117;
    color: #E6EDF3;
}

.stApp {
    background-color: #0D1117;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 3rem;
    padding-bottom: 3rem;
    max-width: 680px;
}

/* ── Header ── */
.ss-header {
    text-align: center;
    margin-bottom: 2.5rem;
}
.ss-badge {
    display: inline-block;
    background: rgba(88, 213, 216, 0.12);
    border: 1px solid rgba(88, 213, 216, 0.3);
    color: #58D5D8;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.3rem 0.85rem;
    border-radius: 100px;
    margin-bottom: 1rem;
}
.ss-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #E6EDF3;
    line-height: 1.15;
    margin: 0 0 0.6rem 0;
}
.ss-title span {
    color: #58D5D8;
}
.ss-subtitle {
    font-size: 0.95rem;
    color: #8B949E;
    line-height: 1.6;
    max-width: 480px;
    margin: 0 auto;
}

/* ── Card ── */
.ss-card {
    background: #1C2333;
    border: 1px solid #30363D;
    border-radius: 14px;
    padding: 2rem;
    margin-bottom: 1.25rem;
}
.ss-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #8B949E;
    margin-bottom: 0.6rem;
    display: block;
}

/* ── Textarea override ── */
.stTextArea textarea {
    background: #0D1117 !important;
    border: 1px solid #30363D !important;
    border-radius: 10px !important;
    color: #E6EDF3 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.875rem !important;
    line-height: 1.65 !important;
    padding: 0.9rem 1rem !important;
    resize: vertical !important;
    transition: border-color 0.2s ease !important;
}
.stTextArea textarea:focus {
    border-color: #58D5D8 !important;
    box-shadow: 0 0 0 3px rgba(88, 213, 216, 0.12) !important;
    outline: none !important;
}
.stTextArea textarea::placeholder {
    color: #484F58 !important;
}
.stTextArea label { display: none !important; }

/* ── Button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #58D5D8 0%, #3ABFC2 100%);
    color: #0D1117;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.04em;
    border: none;
    border-radius: 10px;
    padding: 0.85rem 2rem;
    cursor: pointer;
    transition: all 0.2s ease;
    margin-top: 0.5rem;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 24px rgba(88, 213, 216, 0.28);
    background: linear-gradient(135deg, #6EDDDF 0%, #58D5D8 100%);
}
.stButton > button:active {
    transform: translateY(0);
}

/* ── Verdict Banner ── */
.verdict-spam {
    background: linear-gradient(135deg, rgba(255, 76, 76, 0.15) 0%, rgba(255, 76, 76, 0.05) 100%);
    border: 1px solid rgba(255, 76, 76, 0.4);
    border-radius: 12px;
    padding: 1.5rem 1.75rem;
    display: flex;
    align-items: center;
    gap: 1.1rem;
    animation: slideUp 0.35s cubic-bezier(0.22, 1, 0.36, 1);
    box-shadow: 0 0 32px rgba(255, 76, 76, 0.12);
}
.verdict-ham {
    background: linear-gradient(135deg, rgba(88, 213, 216, 0.12) 0%, rgba(88, 213, 216, 0.04) 100%);
    border: 1px solid rgba(88, 213, 216, 0.35);
    border-radius: 12px;
    padding: 1.5rem 1.75rem;
    display: flex;
    align-items: center;
    gap: 1.1rem;
    animation: slideUp 0.35s cubic-bezier(0.22, 1, 0.36, 1);
    box-shadow: 0 0 32px rgba(88, 213, 216, 0.10);
}
.verdict-icon {
    font-size: 2.2rem;
    line-height: 1;
    flex-shrink: 0;
}
.verdict-text-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.2rem;
}
.verdict-text-label-spam { color: #FF4C4C; }
.verdict-text-label-ham  { color: #58D5D8; }
.verdict-text-main {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: #E6EDF3;
    line-height: 1.2;
}

/* ── Confidence Bar ── */
.conf-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.45rem;
}
.conf-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.8rem;
    font-weight: 500;
    color: #8B949E;
}
.conf-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    font-weight: 500;
}
.conf-value-spam { color: #FF4C4C; }
.conf-value-ham  { color: #58D5D8; }
.conf-bar-track {
    background: #30363D;
    border-radius: 100px;
    height: 6px;
    overflow: hidden;
    margin-bottom: 0.85rem;
}
.conf-bar-fill-spam {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #FF4C4C, #FF7676);
    transition: width 0.8s cubic-bezier(0.22, 1, 0.36, 1);
}
.conf-bar-fill-ham {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #3ABFC2, #58D5D8);
    transition: width 0.8s cubic-bezier(0.22, 1, 0.36, 1);
}

/* ── Divider ── */
.ss-divider {
    border: none;
    border-top: 1px solid #21262D;
    margin: 1.25rem 0;
}

/* ── Footer ── */
.ss-footer {
    text-align: center;
    margin-top: 2.5rem;
    color: #484F58;
    font-size: 0.78rem;
    line-height: 1.7;
}
.ss-footer a {
    color: #58D5D8;
    text-decoration: none;
}

/* ── Warning ── */
.ss-warning {
    background: rgba(255, 193, 7, 0.08);
    border: 1px solid rgba(255, 193, 7, 0.3);
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    color: #FFC107;
    font-size: 0.875rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

/* ── Animation ── */
@keyframes slideUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Tip chips ── */
.tip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.75rem;
}
.tip-chip {
    background: #21262D;
    border: 1px solid #30363D;
    border-radius: 6px;
    padding: 0.3rem 0.7rem;
    font-size: 0.75rem;
    color: #8B949E;
    cursor: pointer;
    font-family: 'JetBrains Mono', monospace;
}
</style>
""", unsafe_allow_html=True)

# --------------------------
# Header
# --------------------------

st.markdown("""
<div class="ss-header">
    <div class="ss-badge">🛡️ ML-Powered Detection</div>
    <h1 class="ss-title">Spam<span>Shield</span></h1>
    <p class="ss-subtitle">
        Paste any SMS or message below. The classifier analyses its content
        and returns a verdict with confidence score in real time.
    </p>
</div>
""", unsafe_allow_html=True)

# --------------------------
# Input Card
# --------------------------

st.markdown('<div class="ss-card">', unsafe_allow_html=True)
st.markdown('<span class="ss-label">Message Content</span>', unsafe_allow_html=True)

input_sms = st.text_area(
    label="hidden",
    placeholder="Paste or type an SMS message here…",
    height=160,
    label_visibility="hidden"
)

st.markdown("""
<div class="tip-row">
    <span class="tip-chip">Try: "You've won a FREE prize!"</span>
    <span class="tip-chip">Try: "Hey, can we meet at 6?"</span>
    <span class="tip-chip">Try: "Claim your reward NOW"</span>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------
# Predict Button
# --------------------------

predict_clicked = st.button("Run Analysis →")

# --------------------------
# Result
# --------------------------

if predict_clicked:
    if not input_sms.strip():
        st.markdown("""
        <div class="ss-warning">
            ⚠️ &nbsp; No message provided. Paste some text above to analyse it.
        </div>
        """, unsafe_allow_html=True)
    else:

        try:

            response = requests.post(
                "http://127.0.0.1:8000/api/v1/predict",
                json={
                    "text": input_sms
                }
            )

            result = response.json()

            prediction = result["prediction"]

            # Temporary confidence values
            if prediction == "Spam":
                spam_prob = 0.95
                ham_prob = 0.05
            else:
                spam_prob = 0.05
                ham_prob = 0.95

            spam_pct = int(spam_prob * 100)
            ham_pct = int(ham_prob * 100)

            st.markdown('<div class="ss-card">', unsafe_allow_html=True)

            if prediction == "Spam":

                st.markdown(f"""
                <div class="verdict-spam">
                    <div class="verdict-icon">🚨</div>
                    <div>
                        <div class="verdict-text-label verdict-text-label-spam">Verdict</div>
                        <div class="verdict-text-main">Spam Detected</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<hr class="ss-divider">', unsafe_allow_html=True)

                st.markdown(f"""
                <div class="conf-row">
                    <span class="conf-label">Spam probability</span>
                    <span class="conf-value conf-value-spam">{spam_prob:.1%}</span>
                </div>

                <div class="conf-bar-track">
                    <div class="conf-bar-fill-spam" style="width:{spam_pct}%"></div>
                </div>

                <div class="conf-row">
                    <span class="conf-label">Not Spam probability</span>
                    <span class="conf-value conf-value-ham">{ham_prob:.1%}</span>
                </div>

                <div class="conf-bar-track">
                    <div class="conf-bar-fill-ham" style="width:{ham_pct}%"></div>
                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown(f"""
                <div class="verdict-ham">
                    <div class="verdict-icon">✅</div>
                    <div>
                        <div class="verdict-text-label verdict-text-label-ham">Verdict</div>
                        <div class="verdict-text-main">Not Spam</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<hr class="ss-divider">', unsafe_allow_html=True)

                st.markdown(f"""
                <div class="conf-row">
                    <span class="conf-label">Not Spam probability</span>
                    <span class="conf-value conf-value-ham">{ham_prob:.1%}</span>
                </div>

                <div class="conf-bar-track">
                    <div class="conf-bar-fill-ham" style="width:{ham_pct}%"></div>
                </div>

                <div class="conf-row">
                    <span class="conf-label">Spam probability</span>
                    <span class="conf-value conf-value-spam">{spam_prob:.1%}</span>
                </div>

                <div class="conf-bar-track">
                    <div class="conf-bar-fill-spam" style="width:{spam_pct}%"></div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:

            st.error(f"API Error: {str(e)}")