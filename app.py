#              STUDYGEN AI - INTELLIGENT LEARNING ASSISTANT
# ============================================================

# ==============STEP 1: IMPORT LIBRARIES ================

import os
import re
import io
import time
import numpy

from google import genai
from google.genai import types
import streamlit as st

from PIL import Image
from dotenv import load_dotenv
from docx import Document

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document as LangchainDocument

# YouTube
from urllib.parse import urlparse, parse_qs

#====================STEP 2 API KEYS======================
st.set_page_config(page_title = "GEN-AI INTELLIGENT LEARNING🎓",
              layout = "wide")

st.sidebar.title("SET API CONFIG")
st.title("GEN-AI INTELLIGENT LEARNING💻")

GOOGLE_API_KEY = st.sidebar.text_input("GOOGLE_API_KEY",type = "password")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

if GOOGLE_API_KEY:
  st.sidebar.success("API key Loaded!!")
else:
  st.sidebar.info("Give API key")

if GOOGLE_API_KEY:
    model = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=GOOGLE_API_KEY
    )
else:
    st.stop()

# =================== STEP 3 : PDF BACKEND FUNCTIONS ===================
# loading pdf
def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents

# YOUTUBE VIDEO FUNCTIONS 

def extract_video_id(url):

    parsed_url = urlparse(url.strip())

    if parsed_url.hostname in ["youtu.be", "www.youtu.be"]:
        return parsed_url.path.lstrip("/").split("/")[0]

    if parsed_url.hostname in [
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com"
    ]:

        if parsed_url.path == "/watch":
            return parse_qs(parsed_url.query).get("v", [None])[0]

        if parsed_url.path.startswith(("/embed/", "/shorts/", "/live/")):
            return parsed_url.path.split("/")[2]

    return None




def load_youtube_content(url):

    video_id = extract_video_id(url)

    if not video_id:
        raise ValueError("Invalid YouTube URL.")

    client = genai.Client(api_key=GOOGLE_API_KEY)

    response = client.interactions.create(
        model="gemini-3.8-flash",
        input=[
            {
                "type": "text",
                "text": "Summarize this video in 5 bullet points."
            },
            {
                "type": "video",
                "uri": url
            }
        ]
    )

    video_text = response.output_text

    if not video_text:
        raise ValueError("Gemini returned no video summary.")

    document = LangchainDocument(
        page_content=video_text,
        metadata={
            "source": url,
            "type": "YouTube Video",
            "video_id": video_id
        }
    )

    return [document]

# =================== STEP 4 : CREATE TEXT CHUNKS ===================

def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    return chunks

# ================= STEP 5 : CREATE EMBEDDINGS =================

def create_vectorstore(chunks, embeddings):
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )
    return vectorstore

def create_retriever(vectorstore, k_value):
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k_value}
    )
    return retriever
  
@st.cache_resource
def create_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings

# ================= STEP 9 : CREATE LCEL RAG CHAIN =================

def create_rag_chain(retriever):
    """
    This function creates the LCEL RAG Chain.
    """
    prompt = ChatPromptTemplate.from_template(
        """
    You are StudyGen AI, an Intelligent Learning Assistant.

    Answer the user's question ONLY using the provided context.

    If the answer is not available in the context,
    reply:
    "I couldn't find this information in the uploaded study material."
    Context:
    {context}
    Question:
    {question}
    """
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | model
        | StrOutputParser()
    )
    return rag_chain

# ================= STEP 10 : NOTES GENERATOR =================

def generate_notes(rag_chain):

    prompt = """
Generate well-structured study notes from the uploaded study material.

Include the following sections:

1. Introduction
2. Key Concepts
3. Important Definitions
4. Key Points
5. Examples (if available)
6. Summary

Use simple language.
Format the output using headings and bullet points.
"""

    try:
        notes = rag_chain.invoke(prompt)
        return notes

    except Exception as e:
        st.exception(e)
        return None
# ================= STEP 11 : QUIZ GENERATOR =================

def generate_quiz(rag_chain):
    """
    This function generates a quiz
    from the uploaded PDF.
    """

    prompt = """
Generate exactly 10 Multiple Choice Questions (MCQs) from the uploaded study material.

Follow this format STRICTLY.

Question 1:
<question>

A. Option 1
B. Option 2
C. Option 3
D. Option 4

Correct Answer:
A

Explanation:
Write the explanation in 2-3 complete sentences.

Repeat the same format for all 10 questions.

Rules:
- Put every option on a separate line.
- Leave one blank line between sections.
- Put "Correct Answer" on a separate line.
- Put "Explanation" on a separate line.
- Make the output clean, readable and well-formatted using Markdown.
"""

    quiz = rag_chain.invoke(prompt)

    return quiz
  
# ================= STEP 12 : DOUBT SOLVER =================

def solve_doubt(rag_chain, question):
    """
    This function answers the user's question
    from the uploaded study material.
    with proper spacing
    """

    answer = rag_chain.invoke(question)

    return answer

# ================= STEP 13 : STUDY PLANNER =================

def generate_study_plan(subjects, exam_date, study_hours):

    prompt = f"""
You are StudyGen AI.

Create a personalized study plan.

Subjects:
{subjects}

Exam Date:
{exam_date}

Study Hours Per Day:
{study_hours}

Generate the output in proper Markdown.

Rules:
- Use headings.
- Use bullet points.
- Use tables where possible.
- Leave blank lines between sections.
- Do NOT return JSON.
- Do NOT return Python objects.
- Return only the study plan.
"""

    response = model.invoke(prompt)

    if isinstance(response.content, list):
        plan = "".join(
            item["text"] if isinstance(item, dict) else item.text
            for item in response.content
        )
    else:
        plan = response.content

    return plan
  
def create_doc(text):

    doc = Document()
    doc.add_heading("StudyGen AI", level=1)
    doc.add_paragraph(text)

    file = io.BytesIO()
    doc.save(file)
    file.seek(0)

    return file.getvalue()
  
# ============================================================
# ============== STEP 14 : STREAMLIT USER INTERFACE ===========
# ---------------------- PAGE TITLE -----------------------
st.title("📚 StudyGen AI")
st.subheader("Intelligent Learning Assistant")

st.markdown(
  """
Welcome to **StudyGen AI**.
Upload your study material and use AI to simplify learning.
"""
)

st.image("pg.png", use_container_width=True)

st.divider()
st.markdown("## ✨ Features")


# ================= HOME PAGE FEATURES =================

col1, col2, col3 = st.columns(3)

with col1:
    st.info("📄\n\nGenerate Notes")

with col2:
    st.info("📝\n\nGenerate Quiz")

with col3:
    st.info("🧠\n\nFlashcards")

col4, col5 = st.columns(2)

with col4:
    st.info("💬\n\nSolve Doubts")

with col5:
    st.info("📅\n\nStudy Planner")

st.markdown("## 🎯 About StudyGen AI")

st.write(
  """
StudyGen AI is an AI-powered intelligent learning assistant developed using
Generative AI, Retrieval-Augmented Generation (RAG), LangChain, FAISS,
HuggingFace Embeddings, and Google's Gemini model.

The application helps students understand study materials more effectively by
providing AI-powered learning assistance from uploaded PDF documents.

### Key Features

✔ Generate well-structured study notes

✔ Create AI-powered multiple-choice quizzes

✔ Ask questions directly from the uploaded PDF

✔ Generate personalized study plans based on exam dates and study hours

StudyGen AI reduces manual study effort, improves learning efficiency,
and provides an interactive, personalized learning experience.
"""
)

st.divider()
# ==================== STEP 15 : SIDEBAR ====================

st.sidebar.title("📚 StudyGen AI")
st.sidebar.markdown("### Choose your study material")


source_type = st.sidebar.radio(
    "Select Learning Source",
    ["📄 PDF Documents", "▶️ YouTube Video"]
)

uploaded_files = None
youtube_url = None

if source_type == "📄 PDF Documents":
    uploaded_files = st.sidebar.file_uploader(
        "Choose PDF Files",
        type=["pdf"],
        accept_multiple_files=True
    )

else:
    youtube_url = st.sidebar.text_input(
        "Paste YouTube Video URL"
    )

    st.sidebar.info(
        "YouTube integration is experimental and "
        "currently under development."
    )


k_value = st.sidebar.slider(
    "Retriever Top-K",
    min_value=1,
    max_value=10,
    value=3
)

st.sidebar.divider()

st.sidebar.markdown("## ℹ️ About")

st.sidebar.info("""
📚 **StudyGen AI**

An AI-powered Intelligent Learning Assistant that transforms PDF study material into:

• 📄 Smart Notes

• 📝 AI Quizzes

• 💬 Doubt Resolution

• 📅 Personalized Study Plans

**Tech Stack**

🤖 Gemini

🔗 LangChain

🧠 FAISS

🤗 HuggingFace

🐍 Python

🎨 Streamlit
""")

# ================= STEP 16 : PROCESS PDF =================
def build_rag(pdf_path, k_value):

    documents = load_pdf(pdf_path)
    chunks = split_documents(documents)

    embeddings = create_embeddings()
    vectorstore = create_vectorstore(chunks, embeddings)
    retriever = create_retriever(vectorstore, k_value)
    return create_rag_chain(retriever)
rag_chain = None

if uploaded_files:

    save_dir = "uploaded_files"
    os.makedirs(save_dir, exist_ok=True)

    try:
        with st.spinner("Processing PDF files..."):

            all_documents = []
            for uploaded_file in uploaded_files:

                pdf_path = os.path.join(
                    save_dir,
                    uploaded_file.name
                )
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                documents = load_pdf(pdf_path)

                # Add filename to metadata
                for doc in documents:
                    doc.metadata["source"] = uploaded_file.name

                all_documents.extend(documents)

            chunks = split_documents(all_documents)
            embeddings = create_embeddings()

            vectorstore = create_vectorstore(
                chunks,
                embeddings
            )
            retriever = create_retriever(
                vectorstore,
                k_value
            )
            rag_chain = create_rag_chain(retriever)
        st.success(
            f"{len(uploaded_files)} PDF(s) processed successfully ✅"
        )
    except Exception as e:
        st.exception(e)

# ================= PROCESS YOUTUBE VIDEO =================

if youtube_url:

    try:
        with st.spinner("Processing YouTube video..."):

            documents = load_youtube_content(youtube_url)

            chunks = split_documents(documents)

            embeddings = create_embeddings()

            vectorstore = create_vectorstore(
                chunks,
                embeddings
            )

            retriever = create_retriever(
                vectorstore,
                k_value
            )

            rag_chain = create_rag_chain(retriever)

        st.success("YouTube video processed successfully ✅")

    except Exception as e:
        st.error("Could not process this YouTube video.")
        st.exception(e)
        
# ================= STEP 17 : SELECT FEATURE ==================
st.divider()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if rag_chain:
    col1, col2, col3 = st.columns(3)

    col1.metric("AI Model", "Gemini")
    col2.metric("Retriever", f"Top {k_value}")

    if source_type == "📄 PDF Documents":
        col3.metric("Source", "PDF Loaded")
    else:
        col3.metric("Source", "YouTube Loaded")

    # Display five learning features
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📄 Smart Notes",
            "📝 AI Quiz Generator",
            "🧠 Flashcards",
            "💬 AI Doubt Solver",
            "📅 Personalized Study Planner"
        ]
    )
    # ================= NOTES =================

    with tab1:

        st.subheader("📄 Generate Study Notes")

        if st.button("Generate Notes"):

            with st.spinner("Generating Notes..."):

                prompt = """
Generate well-structured study notes from the uploaded study material.

Include the following sections:

1. Introduction
2. Key Concepts
3. Important Definitions
4. Key Points
5. Examples (if available)
6. Summary

Use simple language.
Format the output using headings and bullet points.
"""

                notes = rag_chain.invoke(prompt)

                st.markdown(notes)

                st.download_button(
                    "⬇ Download Notes (.md)",
                    notes,
                    file_name="Study_Notes.md",
                    mime="text/markdown"
                )

                st.download_button(
                    "⬇ Download Notes (.docx)",
                    create_doc(notes),
                    file_name="Study_Notes.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

    # ================= QUIZ =================

    with tab2:

        st.subheader("📝 Generate Quiz")

        if st.button("Generate Quiz"):

            with st.spinner("Generating Quiz..."):

                prompt = """
Generate exactly 10 Multiple Choice Questions (MCQs) from the uploaded study material.

Follow this format STRICTLY.

Question 1:

<question>

A. Option 1
B. Option 2
C. Option 3
D. Option 4

Correct Answer:
A

Explanation:
Write the explanation in 2-3 sentences.

Repeat for all 10 questions.
"""

                quiz = rag_chain.invoke(prompt)

                st.markdown(quiz)

                st.download_button(
                    "⬇ Download Quiz (.md)",
                    quiz,
                    file_name="Quiz.md",
                    mime="text/markdown"
                )

                st.download_button(
                    "⬇ Download Quiz (.docx)",
                    create_doc(quiz),
                    file_name="Quiz.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
              
# ================= FLASHCARDS =================

    with tab3:
    
        st.subheader("🧠 AI Flashcards")
    
        number_of_cards = st.slider(
            "Number of Flashcards",
            min_value=5,
            max_value=20,
            value=10
        )
    
        if st.button("Generate Flashcards"):
            with st.spinner("Creating Flashcards..."):
    
                prompt = f"""
    Create exactly {number_of_cards} flashcards
    from the uploaded study material.
    
    Format each flashcard like this:
    
    Flashcard 1:
    Question: <question>
    Answer: <answer>
    
    Flashcard 2:
    Question: <question>
    Answer: <answer>
    
    Rules:
    - Use only information from the uploaded study material.
    - Focus on important concepts and definitions.
    - Keep questions clear.
    - Keep answers concise.
    - Do not include information outside the study material.
    """
                flashcards = rag_chain.invoke(prompt)
    
                st.markdown(flashcards)
                st.download_button(
                    "⬇ Download Flashcards (.md)",
                    flashcards,
                    file_name="Flashcards.md",
                    mime="text/markdown"
                )
    
                st.download_button(
                    "⬇ Download Flashcards (.docx)",
                    create_doc(flashcards),
                    file_name="Flashcards.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
    # ================= DOUBT SOLVER =================
    with tab4:
    
        st.subheader("💬 AI Doubt Solver")
    
        # Display previous messages
        for message in st.session_state.chat_history:
    
            if message["role"] == "user":
                st.markdown(f"**👤 You:** {message['content']}")
    
            else:
                st.markdown(f"**🤖 StudyGen AI:** {message['content']}")
    
        question = st.text_input(
            "Ask your question",
            key="doubt_question"
        )
    
        if st.button("Get Answer"):
            if question:
                with st.spinner("Thinking..."):
                  
                    answer = rag_chain.invoke(question)
    
                # Save user question
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": question
                })
    
                # Save AI answer
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer
                })
    
                st.rerun()
        if st.button("🗑 Clear Chat"):
    
            st.session_state.chat_history = []
            st.rerun()

    # ================= STUDY PLANNER =================

    with tab5:
    
        st.subheader("📅 Personalized Study Planner")
    
        subjects = st.text_input("Subjects")
    
        exam_date = st.date_input("Exam Date")
    
        study_hours = st.slider(
            "Study Hours",
            1,
            12,
            4
        )
    
        if st.button("Generate Study Plan"):
    
            plan = generate_study_plan(
                subjects,
                exam_date,
                study_hours
            )
    
            # Display the formatted study plan
            st.write(plan)
    
            # Download as Markdown
            st.download_button(
                label="⬇ Download Study Plan (.md)",
                data=plan,
                file_name="Study_Plan.md",
                mime="text/markdown"
            )
    
            # Download as Word
            st.download_button(
                label="⬇ Download Study Plan (.docx)",
                data=create_doc(plan),
                file_name="Study_Plan.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
st.divider()

st.caption(
    "🚀 Powered by Gemini 2.5 Flash Lite | LangChain | FAISS | HuggingFace Embeddings | Streamlit"
)
