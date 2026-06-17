import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import google.generativeai as genai

# -----------------------------
# LOAD ENV
# -----------------------------

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found in .env file"
    )

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-1.5-flash")


# -----------------------------
# PDF PROCESSING
# -----------------------------

def process_pdf(pdf_path):

    loader = PyPDFLoader(pdf_path)

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(
        chunks,
        embeddings
    )

    os.makedirs(
        "vectorstore",
        exist_ok=True
    )

    db.save_local("vectorstore")


# -----------------------------
# LOAD CONTEXT FROM VECTOR DB
# -----------------------------

def get_context(topic):

    if not os.path.exists("vectorstore"):
        return (
            "No PDF has been processed yet. "
            "Please upload and process a PDF first."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = db.similarity_search(
        topic,
        k=4
    )

    context = "\n".join(
        [doc.page_content for doc in docs]
    )

    return context


# -----------------------------
# NOTES
# -----------------------------

def generate_notes(topic):

    context = get_context(topic)

    prompt = f"""
You are an educational assistant.

Using the context below:

{context}

Generate:

1. Detailed Notes
2. Summary
3. Key Points
4. Important Definitions
5. Real-world Applications

Topic:
{topic}
"""

    response = model.generate_content(prompt)

    return response.text


# -----------------------------
# QUIZ
# -----------------------------

def generate_quiz(topic, difficulty):

    context = get_context(topic)

    prompt = f"""
Using the context below:

{context}

Generate 10 {difficulty} level MCQs.

Format:

Question:
A)
B)
C)
D)

Correct Answer:
Explanation:

Topic:
{topic}
"""

    response = model.generate_content(prompt)

    return response.text


# -----------------------------
# FLASHCARDS
# -----------------------------

def generate_flashcards(topic):

    context = get_context(topic)

    prompt = f"""
Using the context below:

{context}

Generate 10 flashcards.

Format:

Q:
A:

Topic:
{topic}
"""

    response = model.generate_content(prompt)

    return response.text


# -----------------------------
# STUDY PLAN
# -----------------------------

def generate_study_plan(topic):

    context = get_context(topic)

    prompt = f"""
Using the context below:

{context}

Create a 7-Day Study Plan.

Include:

1. Day-wise Schedule
2. Topics to Learn
3. Practice Tasks
4. Revision Strategy
5. Final Assessment

Topic:
{topic}
"""

    response = model.generate_content(prompt)

    return response.text