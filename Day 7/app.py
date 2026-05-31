import streamlit as st
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import re

st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
<style>
.block-container { padding-top: 1.5rem !important; max-width: 1100px !important; }

.frd-title { font-size: 30px; font-weight: 800; color: inherit; margin: 0; }
.frd-sub   { font-size: 14px; opacity: 0.55; margin-top: 4px; }

.big-pct   { font-size: 40px; font-weight: 800; text-align: center; margin: 0; line-height: 1; }
.big-label { font-size: 12px; text-align: center; opacity: 0.55; margin-top: 5px; }
.pct-fake    { color: #ff5252; }
.pct-genuine { color: #00e676; }
.pct-conf    { color: #448aff; }

.verdict-fake {
    background: rgba(229,57,53,0.12);
    border-left: 5px solid #e53935;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.25rem; margin-top: 0.75rem;
}
.verdict-genuine {
    background: rgba(0,200,83,0.10);
    border-left: 5px solid #00c853;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.25rem; margin-top: 0.75rem;
}
.verdict-fake strong    { color: #ff5252; font-size: 17px; }
.verdict-genuine strong { color: #00e676; font-size: 17px; }
.verdict-fake p, .verdict-genuine p { margin: 6px 0 0; font-size: 13px; opacity: 0.8; }

.spam-box {
    background: rgba(255,152,0,0.10);
    border-left: 4px solid #ff9800;
    border-radius: 0 10px 10px 0;
    padding: .85rem 1.1rem; margin-top: .5rem;
}
.spam-word {
    display: inline-block;
    background: rgba(255,82,82,0.2);
    color: #ff5252;
    border-radius: 6px;
    padding: 2px 8px;
    margin: 2px 3px;
    font-size: 13px;
    font-weight: 600;
    border: 1px solid rgba(255,82,82,0.3);
}
.reason-item {
    background: rgba(255,255,255,0.05);
    border-radius: 8px;
    padding: 8px 12px;
    margin: 5px 0;
    font-size: 13px;
    border-left: 3px solid;
}
.reason-bad  { border-color: #ff5252; }
.reason-good { border-color: #00c853; }
.reason-warn { border-color: #ff9800; }

.sample-card {
    background: rgba(255,255,255,0.04);
    border: 0.5px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: .85rem 1rem;
    margin-bottom: 8px;
    cursor: pointer;
}
.sample-card:hover { background: rgba(255,255,255,0.08); }

.review-echo {
    background: rgba(100,149,237,0.08);
    border-radius: 10px;
    padding: .75rem 1rem;
    font-size: 13px;
    border-left: 3px solid cornflowerblue;
    margin-top: .5rem;
    opacity: 0.85;
}
.meter-wrap {
    background: rgba(255,255,255,0.08);
    border-radius: 8px;
    height: 14px;
    overflow: hidden;
    margin-top: 4px;
}
.meter-fill {
    height: 100%;
    border-radius: 8px;
    transition: width .6s ease;
}
</style>
""", unsafe_allow_html=True)

# ── Spam signal words ───────────────────────────────────────────
SPAM_WORDS = [
    "must buy","must-buy","best ever","changed my life","life changing",
    "absolutely amazing","totally perfect","100% recommend","highly recommend",
    "best product","incredible","fantastic","amazing","awesome","love it",
    "5 stars","five stars","exceeded expectations","world class","unbelievable",
    "don't hesitate","buy now","order now","hurry","limited time","best deal",
    "no complaints","perfect product","exactly as described","fast delivery",
    "great quality","top quality","superb","outstanding","excellent","wonderful",
    "mind blowing","mind-blowing","blown away","beyond expectations"
]

GENUINE_SIGNALS = [
    "however","but","although","issue","problem","returned","disappointed",
    "average","okay","decent","not bad","could be better","mixed feelings",
    "expected more","slight issue","minor problem","would have liked",
    "not perfect","takes time","learning curve","improvement needed"
]

SAMPLE_REVIEWS = [
    {"text": "This product is absolutely amazing! Best purchase I have ever made in my life. Must buy for everyone! 100% recommend, totally perfect and incredible quality!", "label": "Likely Fake"},
    {"text": "The laptop works well for basic tasks. Battery life is decent around 6 hours. However the keyboard feels a bit stiff and took some getting used to. Overall okay for the price.", "label": "Likely Genuine"},
    {"text": "Wow! Life changing product! Exceeded all my expectations! World class quality! Don't hesitate, buy now! Five stars! Unbelievable value!", "label": "Likely Fake"},
    {"text": "I bought this for my home office. Setup was straightforward. There is a slight issue with the cable being short but manageable. Sound quality is good, not perfect but decent for the price.", "label": "Likely Genuine"},
    {"text": "Outstanding product! Mind blowing results! Blown away by the quality! Best deal ever! Order now before it runs out!", "label": "Likely Fake"},
]

# ── Helpers ─────────────────────────────────────────────────────
def detect_spam_words(text):
    found = []
    lower = text.lower()
    for w in SPAM_WORDS:
        if w in lower:
            found.append(w)
    return list(set(found))

def detect_genuine_signals(text):
    found = []
    lower = text.lower()
    for w in GENUINE_SIGNALS:
        if w in lower:
            found.append(w)
    return list(set(found))

def build_reasons(text, fake_pct, spam_words, genuine_signals, rating, word_count):
    reasons = []
    if spam_words:
        reasons.append(("bad", f"Contains {len(spam_words)} spam/hype word(s): {', '.join(spam_words[:4])}"))
    if fake_pct > 70:
        reasons.append(("bad", "Very high fake probability from NLP model"))
    elif fake_pct > 50:
        reasons.append(("warn", "Moderate fake signals detected by model"))
    if word_count < 20:
        reasons.append(("bad", f"Very short review ({word_count} words) — often fake"))
    elif word_count > 80:
        reasons.append(("good", f"Detailed review ({word_count} words) — genuine signal"))
    exclaim = text.count("!")
    if exclaim >= 3:
        reasons.append(("bad", f"Excessive exclamation marks ({exclaim}!) — spam pattern"))
    if genuine_signals:
        reasons.append(("good", f"Contains balanced language: {', '.join(genuine_signals[:3])}"))
    if rating == 5 and fake_pct > 60:
        reasons.append(("warn", "5-star rating combined with hype language — suspicious"))
    elif rating in [3, 4] and fake_pct < 50:
        reasons.append(("good", "Mid-range rating suggests honest opinion"))
    upper_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    if upper_ratio > 0.15:
        reasons.append(("bad", "High ratio of CAPITAL LETTERS — aggressive/fake tone"))
    return reasons

def star_display(rating):
    return "⭐" * rating + "☆" * (5 - rating)

@st.cache_resource
def load_resources():
    model    = joblib.load("fake_review_model.pkl")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return model, embedder

model, embedder = load_resources()

# ── Layout ──────────────────────────────────────────────────────
st.markdown("""
<div style="padding:.5rem 0 1rem">
    <div class="frd-title">🔍 Fake Review Detector</div>
    <div class="frd-sub">AI-powered multimodal fake review analysis · NLP + Spam signals + Rating analysis</div>
</div>
""", unsafe_allow_html=True)

left_col, right_col = st.columns([3, 2], gap="large")

# ══════════════════════════════════════════════════════════════
# LEFT COLUMN — Input & Results
# ══════════════════════════════════════════════════════════════
with left_col:

    st.markdown("#### ✍️ Enter review")

    rating = st.select_slider(
        "⭐ Product rating given by reviewer",
        options=[1, 2, 3, 4, 5],
        value=5,
        format_func=lambda x: f"{star_display(x)}  ({x}/5)"
    )

    review = st.text_area(
        "Review text",
        height=150,
        placeholder="Paste or type a product review here…",
        label_visibility="collapsed"
    )

    c1, c2 = st.columns(2)
    with c1:
        analyze = st.button("🚀 Analyze Review", type="primary", use_container_width=True)
    with c2:
        clear = st.button("🗑️ Clear", use_container_width=True)

    if clear:
        st.rerun()

    # ── Analysis ───────────────────────────────────────────────
    if analyze:
        if not review.strip():
            st.warning("⚠️ Please enter a review first.")
        else:
            with st.spinner("Running AI analysis…"):
                embedding    = embedder.encode([review])
                prediction   = model.predict(embedding)[0]
                probs        = model.predict_proba(embedding)[0]
                genuine_pct  = round(float(probs[0]) * 100, 1)
                fake_pct     = round(float(probs[1]) * 100, 1)
                confidence   = max(genuine_pct, fake_pct)
                is_fake      = int(prediction) == 1
                spam_words   = detect_spam_words(review)
                gen_signals  = detect_genuine_signals(review)
                word_count   = len(review.split())
                reasons      = build_reasons(review, fake_pct, spam_words, gen_signals, rating, word_count)

                # boost fake score if many spam words
                spam_boost = min(len(spam_words) * 3.5, 20)
                display_fake = min(round(fake_pct + spam_boost, 1), 99.0) if is_fake else fake_pct
                display_gen  = round(100 - display_fake, 1)

            st.divider()

            # ── Big 3 metrics ───────────────────────────────
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f'<p class="big-pct pct-genuine">{display_gen}%</p><p class="big-label">✅ Genuine</p>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<p class="big-pct pct-fake">{display_fake}%</p><p class="big-label">🚨 Fake</p>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<p class="big-pct pct-conf">{confidence}%</p><p class="big-label">🎯 Confidence</p>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Visual meter bars ───────────────────────────
            st.markdown(f"""
            <div style="font-size:13px;opacity:.7;margin-bottom:4px">Genuine probability</div>
            <div class="meter-wrap">
                <div class="meter-fill" style="width:{display_gen}%;background:#00c853"></div>
            </div>
            <div style="font-size:13px;opacity:.7;margin:8px 0 4px">Fake probability</div>
            <div class="meter-wrap">
                <div class="meter-fill" style="width:{display_fake}%;background:#ff5252"></div>
            </div>
            <div style="font-size:13px;opacity:.7;margin:8px 0 4px">Model confidence</div>
            <div class="meter-wrap">
                <div class="meter-fill" style="width:{confidence}%;background:#448aff"></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Spam word highlight ─────────────────────────
            if spam_words:
                spam_pills = " ".join([f'<span class="spam-word">{w}</span>' for w in spam_words])
                st.markdown(f"""
                <div class="spam-box">
                    <strong style="color:#ff9800">⚠️ {len(spam_words)} spam/hype signal(s) detected</strong><br>
                    <div style="margin-top:8px">{spam_pills}</div>
                    <div style="font-size:12px;opacity:.6;margin-top:8px">
                        These words are commonly found in fake or paid reviews.
                        Each adds to the fake probability score.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

            # ── Verdict ─────────────────────────────────────
            if is_fake or display_fake > 55:
                st.markdown(f"""
                <div class="verdict-fake">
                    <strong>🚨 FAKE REVIEW — {display_fake}% probability this is FAKE</strong>
                    <p>The AI model detected strong fake/spam patterns in this review.
                    It is <b>{display_fake}%</b> likely this is NOT a genuine customer review.
                    Found <b>{len(spam_words)}</b> hype word(s) and
                    <b>{review.count('!')}</b> exclamation mark(s).
                    Rating given: {star_display(rating)}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-genuine">
                    <strong>✅ GENUINE REVIEW — {display_gen}% probability this is REAL</strong>
                    <p>This review appears to be written by a real customer.
                    The model found balanced language and genuine patterns.
                    Confidence: <b>{confidence}%</b>.
                    Rating given: {star_display(rating)}</p>
                </div>
                """, unsafe_allow_html=True)

            # ── Review echo ─────────────────────────────────
            st.markdown(f"""
            <div class="review-echo">
                <small style="opacity:.5">Review analyzed ({word_count} words, rating {rating}/5):</small><br>
                "{review[:200]}{"…" if len(review)>200 else ""}"
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# RIGHT COLUMN — Reasons + Dataset Samples
# ══════════════════════════════════════════════════════════════
with right_col:

    # ── Detection reasons (only after analysis) ────────────────
    if analyze and review.strip():
        st.markdown("#### 🧠 Why this prediction?")
        icon_map = {"bad": "🔴", "good": "🟢", "warn": "🟡"}
        css_map  = {"bad": "reason-bad", "good": "reason-good", "warn": "reason-warn"}
        if reasons:
            for kind, text_r in reasons:
                st.markdown(f"""
                <div class="reason-item {css_map[kind]}">
                    {icon_map[kind]} {text_r}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="reason-item reason-warn">🟡 No strong signals found either way</div>', unsafe_allow_html=True)

        st.divider()

    # ── Dataset sample predictions ─────────────────────────────
    st.markdown("#### 📋 Try sample reviews")
    st.caption("Click a sample to load it into the analyzer")

    for i, sample in enumerate(SAMPLE_REVIEWS):
        label_color = "#ff5252" if "Fake" in sample["label"] else "#00c853"
        if st.button(
            f"{'🚨' if 'Fake' in sample['label'] else '✅'} {sample['label']} — \"{sample['text'][:55]}…\"",
            key=f"sample_{i}",
            use_container_width=True
        ):
            st.session_state["loaded_sample"] = sample["text"]
            st.rerun()

    if "loaded_sample" in st.session_state:
        st.info(f"✅ Sample loaded into the text box above. Click **Analyze Review** to run it.")

    st.divider()

    # ── Quick stats legend ─────────────────────────────────────
    st.markdown("#### 📖 How scoring works")
    st.markdown("""
    | Signal | Effect on score |
    |--------|----------------|
    | Spam/hype words | +Fake % |
    | Short review (<20 words) | +Fake % |
    | Excessive `!!!` | +Fake % |
    | ALL CAPS usage | +Fake % |
    | Balanced language | +Genuine % |
    | Detailed review (>80 words) | +Genuine % |
    | Mid-range rating (3-4★) | +Genuine % |
    | NLP model embedding | Base score |
    """)

st.markdown("---")
st.caption("🤖 AI Fake Review Detector · NLP + Spam Analysis + Rating Signals · Multimodal System")