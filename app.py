import streamlit as st
import cv2
import pytesseract
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter
from fpdf import FPDF

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Smart Food Label Analyzer", layout="wide")

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = True  # auto login for demo

if "chat" not in st.session_state:
    st.session_state.chat = []

if "done" not in st.session_state:
    st.session_state.done = False

# ---------------- LOGIN (SIMPLE DEMO) ----------------
if not st.session_state.logged_in:
    st.title("Login")
    u = st.text_input("User")
    p = st.text_input("Pass", type="password")

    if st.button("Login"):
        if u == "admin" and p == "admin":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Wrong credentials")

else:

    # =====================================================
    # HEADER
    # =====================================================
    st.title(" Smart Food Label Analyzer ")

    # =====================================================
    # SIDEBAR
    # =====================================================
    st.sidebar.title("Controls")

    uploaded_file = st.sidebar.file_uploader("Upload Food Label")

    show_chat = st.sidebar.checkbox("🤖 AI Chatbot")
    show_pdf = st.sidebar.checkbox("📄 PDF Report")

    # =====================================================
    # ANALYSIS ENGINE
    # =====================================================
    if uploaded_file is not None:

        if st.button("🔍 Analyze Food"):

            with open("temp.jpg", "wb") as f:
                f.write(uploaded_file.getbuffer())

            img = cv2.imread("temp.jpg")
            text = pytesseract.image_to_string(img)

            # CLEAN OCR
            text = text.lower()
            words = re.findall(r'\b[a-z]{2,}\b', text)

            noise = {"and","the","this","with","contains","nutrition","value","per","serving"}
            words = [w for w in words if w not in noise]

            valid = {
                "salt","sugar","oil","palm","wheat","flour",
                "msg","starch","spices","cream","butter",
                "milk","soy","preservatives","flavour"
            }

            ingredients = list(set([w for w in words if w in valid]))

            # =====================================================
            # MULTI-LABEL PROBABILITY SYSTEM (REAL AI STYLE)
            # =====================================================
            diseases = {
                "Hypertension": 0,
                "Diabetes": 0,
                "Heart Disease": 0,
                "Obesity": 0,
                "Cancer Risk": 0
            }

            weights = {
                "salt": ("Hypertension", 0.6),
                "sugar": ("Diabetes", 0.7),
                "palm": ("Heart Disease", 0.6),
                "oil": ("Heart Disease", 0.4),
                "flour": ("Obesity", 0.5),
                "wheat": ("Obesity", 0.4),
                "msg": ("Hypertension", 0.3),
                "preservatives": ("Cancer Risk", 0.6),
                "glucose": ("Diabetes", 0.5)
            }

            for ing in ingredients:

                if ing in weights:
                    d, w = weights[ing]
                    diseases[d] += w

                try:
                    X = vectorizer.transform([ing])
                    pred = model.predict(X)[0]

                    if pred != "safe":
                        if pred in diseases:
                            diseases[pred] += 0.3

                except:
                    pass

            # normalize probabilities
            for k in diseases:
                diseases[k] = min(round(diseases[k], 2), 1.0)

            st.session_state.ingredients = ingredients
            st.session_state.diseases = diseases
            st.session_state.done = True

    # =====================================================
    # DISPLAY RESULTS
    # =====================================================
    if st.session_state.done:

        ingredients = st.session_state.ingredients
        diseases = st.session_state.diseases

        st.subheader("📊 Dashboard")

        score = 10 - sum(diseases.values()) * 2
        score = max(0, round(score, 1))

        col1, col2, col3 = st.columns(3)
        col1.metric("Health Score", f"{score}/10")
        col2.metric("Ingredients", len(ingredients))
        col3.metric("Risk Factors", sum(1 for v in diseases.values() if v > 0.3))

        st.markdown("---")

        # =====================================================
        # INGREDIENTS
        # =====================================================
        st.subheader("🧾 Ingredients")
        st.write(ingredients)

        # =====================================================
        # AI DISEASE PROBABILITY
        # =====================================================
        st.subheader("🧠 AI Disease Probability")

        for d, p in diseases.items():
            st.write(f"{d} → {int(p*100)}%")

        # =====================================================
        # CHARTS
        # =====================================================
        st.subheader("📊 Risk Visualization")

        col1, col2 = st.columns(2)

        with col1:
            df = pd.DataFrame({
                "Disease": list(diseases.keys()),
                "Risk": list(diseases.values())
            })
            st.bar_chart(df.set_index("Disease"))

        with col2:
            fig, ax = plt.subplots()
            ax.pie(list(diseases.values()),
                   labels=list(diseases.keys()),
                   autopct='%1.1f%%')
            ax.axis("equal")
            st.pyplot(fig)

        # =====================================================
        # HEALTH SCORE
        # =====================================================
        st.subheader("🧠 Final Score")
        st.progress(score/10)
        st.metric("Score", score)

        # =====================================================
        # CHATBOT (SIMPLE AI)
        # =====================================================
        if show_chat:

            st.subheader("🤖 AI Health Chatbot")

            user_input = st.text_input("Ask something about your food")

            if user_input:

                response = "Based on your food analysis, avoid excess salt and oil."

                if "sugar" in user_input:
                    response = "High sugar detected → risk of diabetes."

                if "safe" in user_input:
                    response = "Your food is moderately safe depending on portion size."

                st.session_state.chat.append(("You", user_input))
                st.session_state.chat.append(("AI", response))

            for role, msg in st.session_state.chat:
                st.write(f"**{role}:** {msg}")

        # =====================================================
        # PDF REPORT
        # =====================================================
        if show_pdf:

            st.subheader("📄 Download Report")

            if st.button("Generate PDF"):

                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)

                pdf.cell(200,10,"AI FOOD HEALTH REPORT", ln=True)
                pdf.cell(200,10,f"Health Score: {score}", ln=True)

                pdf.cell(200,10,"Ingredients:", ln=True)
                pdf.cell(200,10,str(ingredients), ln=True)

                pdf.cell(200,10,"Disease Probabilities:", ln=True)
                pdf.cell(200,10,str(diseases), ln=True)

                pdf.output("report.pdf")

                with open("report.pdf","rb") as f:
                    st.download_button("Download PDF", f, file_name="AI_health_report.pdf")