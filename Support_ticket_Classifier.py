import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# -----------------------------
# Sample Training Data
# -----------------------------
data = {
    "text": [
        "I want refund",
        "payment failed",
        "app is not working",
        "need help with account",
        "charged twice",
        "bug in application"
    ],
    "category": [
        "Refund",
        "Payment",
        "Technical",
        "Support",
        "Payment",
        "Technical"
    ]
}

df = pd.DataFrame(data)

# Train Model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])

model = MultinomialNB()
model.fit(X, df["category"])

# -----------------------------
# UI DESIGN
# -----------------------------
st.set_page_config(page_title="Support Ticket Classifier", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center;'>🧾 Support Ticket Classifier</h1>
    <p style='text-align: center; color: gray;'>
    Enter a customer support ticket to automatically classify and assign priority.
    </p>
    """,
    unsafe_allow_html=True
)

# Input Box
ticket = st.text_area("📌 Ticket Description", height=150)

product = st.text_input("📦 Product Name (Optional)")

# Button
if st.button(" Classify Ticket"):
    if ticket:
        transformed = vectorizer.transform([ticket])
        prediction = model.predict(transformed)[0]

        # Priority logic
        if prediction == "Refund":
            priority = "🔴 CRITICAL"
        elif prediction == "Payment":
            priority = "🟠 HIGH"
        else:
            priority = "🟢 MEDIUM"

        st.success("✅ Ticket processed successfully!")

        # -----------------------------
        # RESULT DISPLAY
        # -----------------------------
        st.markdown("---")
        st.markdown("##  Classification Results")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                f"""
                <div style="padding:15px; border-radius:10px; background-color:#1f2937;">
                    <h4 style="color:white;">📂 Ticket Category</h4>
                    <p style="font-size:18px; color:#60a5fa;"><b>{prediction}</b></p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f"""
                <div style="padding:15px; border-radius:10px; background-color:#1f2937;">
                    <h4 style="color:white;"> Assigned Priority</h4>
                    <p style="font-size:18px; color:#f87171;"><b>{priority}</b></p>
                </div>
                """,
                unsafe_allow_html=True
            )

    else:
        st.error("⚠️ Please enter ticket description")