import streamlit as st
import pandas as pd
import os

from rag import (
    process_pdf,
    generate_notes,
    generate_quiz,
    generate_flashcards,
    generate_study_plan
)

# -------------------------
# PAGE CONFIG
# -------------------------

st.set_page_config(
    page_title="AI Educational Generator",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Educational Generator using RAG")

# -------------------------
# ANALYTICS SETUP
# -------------------------

if not os.path.exists("analytics.csv"):
    pd.DataFrame(columns=["Topic"]).to_csv(
        "analytics.csv",
        index=False
    )


def save_topic(topic):

    if topic.strip() == "":
        return

    df = pd.read_csv("analytics.csv")

    new_row = pd.DataFrame(
        {
            "Topic": [topic]
        }
    )

    df = pd.concat(
        [df, new_row],
        ignore_index=True
    )

    df.to_csv(
        "analytics.csv",
        index=False
    )


# -------------------------
# PDF UPLOAD
# -------------------------

st.header("📄 Upload Study Material")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    os.makedirs("uploads", exist_ok=True)

    pdf_path = os.path.join(
        "uploads",
        uploaded_file.name
    )

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("Process PDF"):

        with st.spinner(
            "Creating Embeddings and Vector Database..."
        ):

            process_pdf(pdf_path)

        st.success(
            "✅ PDF Processed Successfully!"
        )

# -------------------------
# TOPIC INPUT
# -------------------------

st.header("🔍 Enter Topic")

topic = st.text_input(
    "Topic",
    placeholder="Example: Machine Learning"
)

difficulty = st.selectbox(
    "Select Difficulty",
    ["Easy", "Medium", "Hard"]
)

# -------------------------
# BUTTONS
# -------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:

    if st.button("📝 Generate Notes"):

        if topic:

            save_topic(topic)

            with st.spinner(
                "Generating Notes..."
            ):

                result = generate_notes(topic)

            st.subheader("Generated Notes")

            st.write(result)

            st.download_button(
                label="Download Notes",
                data=result,
                file_name="notes.txt",
                mime="text/plain"
            )

with col2:

    if st.button("❓ Generate Quiz"):

        if topic:

            save_topic(topic)

            with st.spinner(
                "Generating Quiz..."
            ):

                quiz = generate_quiz(
                    topic,
                    difficulty
                )

            st.subheader("Quiz")

            st.write(quiz)

            st.download_button(
                label="Download Quiz",
                data=quiz,
                file_name="quiz.txt",
                mime="text/plain"
            )

with col3:

    if st.button("🧠 Generate Flashcards"):

        if topic:

            save_topic(topic)

            with st.spinner(
                "Generating Flashcards..."
            ):

                flashcards = generate_flashcards(
                    topic
                )

            st.subheader("Flashcards")

            st.write(flashcards)

            st.download_button(
                label="Download Flashcards",
                data=flashcards,
                file_name="flashcards.txt",
                mime="text/plain"
            )

with col4:

    if st.button("📅 Study Plan"):

        if topic:

            save_topic(topic)

            with st.spinner(
                "Generating Study Plan..."
            ):

                plan = generate_study_plan(
                    topic
                )

            st.subheader("7-Day Study Plan")

            st.write(plan)

            st.download_button(
                label="Download Plan",
                data=plan,
                file_name="study_plan.txt",
                mime="text/plain"
            )

# -------------------------
# ANALYTICS
# -------------------------

st.markdown("---")

st.header("📊 Learning Analytics")

df = pd.read_csv("analytics.csv")

if len(df) > 0:

    st.write(
        f"Total Searches: {len(df)}"
    )

    st.bar_chart(
        df["Topic"].value_counts()
    )

    st.dataframe(
        df.tail(10),
        use_container_width=True
    )

else:

    st.info(
        "No analytics data available."
    )

# -------------------------
# FOOTER
# -------------------------

st.markdown("---")

st.caption(
    "AI Educational Generator | RAG + FAISS + LangChain + Gemini"
)