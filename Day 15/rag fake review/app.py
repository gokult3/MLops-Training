import streamlit as st

st.set_page_config(page_title="Fake Review Checker", page_icon="🕵️‍♀️", layout="centered")
st.title("Fake Review Checker")
st.write("Paste a product review below and the app will give a quick authenticity check.")

SAMPLE_REVIEWS = [
    {
        "review": "This product is amazing! Best purchase ever — buy now and thank me later.",
        "label": "fake",
    },
    {
        "review": "The sound quality is great and the battery lasts all day. I use it every morning.",
        "label": "genuine",
    },
    {
        "review": "Fast shipping, perfect condition, and customer service was very helpful.",
        "label": "genuine",
    },
    {
        "review": "Guaranteed results in 7 days! You must buy this before it sells out.",
        "label": "fake",
    },
]

FAKE_KEYWORDS = [
    "best product ever",
    "buy now",
    "guaranteed",
    "must buy",
    "100%",
    "no risk",
    "miracle",
    "instant",
    "perfect",
    "amazing",
    "thank me later",
]


def predict_review(review: str) -> tuple[str, int]:
    text = review.lower()
    score = sum(1 for keyword in FAKE_KEYWORDS if keyword in text)
    if score >= 2 or any(word in text for word in ["cheap", "free", "lowest price"]):
        return "FAKE", min(95, 50 + score * 15)

    if any(word in text for word in ["terrible", "worst", "broken", "damaged"]):
        return "GENUINE", 80

    return "GENUINE", 65


def retrieve_similar_reviews(review: str, top_k: int = 3):
    text = set(review.lower().split())
    scored = []
    for item in SAMPLE_REVIEWS:
        words = set(item["review"].lower().split())
        overlap = len(text & words)
        scored.append((overlap, item))
    scored.sort(reverse=True, key=lambda pair: pair[0])
    return [item for _, item in scored[:top_k]]


def generate_explanation(review: str, similar: list[dict], label: str) -> str:
    review_lower = review.lower()
    explanation = []

    if label == "FAKE":
        explanation.append("This review looks suspicious because it uses strong sales language and vague claims.")
        explanation.append("Phrases like ‘buy now’, ‘best product ever’, or ‘guaranteed’ are common in fake reviews.")
    else:
        explanation.append("This review appears more balanced and includes concrete details about the product.")
        explanation.append("Genuine reviews often describe real experience, not just promotional language.")

    explanation.append("\nSimilar reviews found:")
    for item in similar:
        explanation.append(f"- {item['review']} ({item['label']})")

    return "\n".join(explanation)


user_review = st.text_area("Paste a product review here:", height=180)

if st.button("Analyze Review"):
    if not user_review.strip():
        st.warning("Please enter a review first.")
    else:
        with st.spinner("Analyzing review..."):
            label, confidence = predict_review(user_review)
            similar = retrieve_similar_reviews(user_review, top_k=3)
            explanation = generate_explanation(user_review, similar, label)

        col1, col2 = st.columns(2)
        with col1:
            color = "red" if label == "FAKE" else "green"
            st.markdown(f"### Prediction: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)

        with col2:
            st.metric("Confidence", f"{confidence}%")

        st.markdown("### 🤖 AI Explanation")
        st.info(explanation)

        st.markdown("### 📚 Similar Reviews")
        for i, item in enumerate(similar, start=1):
            badge = "🔴 FAKE" if item["label"] == "fake" else "🟢 GENUINE"
            with st.expander(f"Similar Review {i} — {badge}"):
                st.write(item["review"])
