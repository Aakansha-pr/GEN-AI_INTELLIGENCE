#              STUDYGEN AI - INTELLIGENT LEARNING ASSISTANT
# ============================================================

# ==============STEP 1: IMPORT LIBRARIES ================

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import streamlit as st
import numpy
import time
from PIL import Image
from dotenv import load_dotenv

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
def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents
  
# =================== STEP 4 : CREATE TEXT CHUNKS ===================
def split_pdf(documents):
    """
    This function splits PDF into smaller text chunks.
    """
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
    """
    This function generates a personalized study plan.
    """

    prompt = f"""
You are StudyGen AI, an Intelligent Learning Assistant.

Create a personalized study plan using the following details:

Subjects:
{subjects}

Exam Date:
{exam_date}

Study Hours Per Day:
{study_hours}

Instructions:
1. Create a day-wise study timetable.
2. Allocate study time for each subject.
3. Include short breaks.
4. Reserve time for revision before the exam.
5. Highlight high-priority subjects.
6. Present the output using proper headings and bullet points.

IN DIFFERENT LINES NO MESS UP. SHOULD LOOK PRESENTABLE
"""

    response = model.invoke(prompt)

    return response.content

# ============================================================
# ============== STEP 14 : STREAMLIT USER INTERFACE ===========
# ============================================================

# ---------------------- PAGE TITLE -----------------------

st.title("📚 StudyGen AI")
st.subheader("Intelligent Learning Assistant")

st.markdown(
    """
Welcome to **StudyGen AI**.

Upload your study material and use AI to:

- 📄 Generate Study Notes
- 📝 Generate Quiz
- 💬 Solve Doubts
- 📅 Create a Personalized Study Planner
"""
)

# ==================== STEP 15 : SIDEBAR ====================

st.sidebar.title("📚 StudyGen AI")
st.sidebar.markdown("### Upload your study material")

uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF File",
    type=["pdf"]
)
k_value = st.sidebar.slider(
    "Retriever Top-K",
    min_value=1,
    max_value=10,
    value=3
)

# ================= STEP 16 : PROCESS PDF =================
def build_rag(pdf_path, k_value):

    documents = load_pdf(pdf_path)
    chunks = split_pdf(documents)

    embeddings = create_embeddings()
    vectorstore = create_vectorstore(chunks, embeddings)
    retriever = create_retriever(vectorstore, k_value)
    return create_rag_chain(retriever)

rag_chain = None
if uploaded_file is not None:

    save_dir = "uploaded_files"
    os.makedirs(save_dir, exist_ok=True)

    pdf_path = os.path.join(save_dir, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        with st.spinner("Processing PDF..."):
            rag_chain = build_rag(pdf_path, k_value)

        st.success("PDF Processed Successfully ✅")
        st.write("RAG Chain Built Successfully")

    except Exception as e:
        st.exception(e)
        
# ================= STEP 17 : SELECT FEATURE ==================
st.divider()

if rag_chain:

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📄 Notes",
            "📝 Quiz",
            "💬 Doubt Solver",
            "📅 Study Planner"
        ]
    )
    # ================= NOTES =================
    with tab1:
    
        st.subheader("📄 Generate Study Notes")
    
        if st.button("Generate Notes"):
    
            with st.spinner("Generating Notes..."):
    
                    st.write_stream(
                    rag_chain.stream("""
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
    """)
                )
      # ================= QUIZ =================
    with tab2:
    
        st.subheader("📝 Generate Quiz")
    
        if st.button("Generate Quiz"):
    
            with st.spinner("Generating Quiz..."):
    
                st.write_stream(
                    rag_chain.stream("""
    Generate a quiz from the uploaded study material.
    
    Instructions:
    - Generate exactly 10 Multiple Choice Questions (MCQs).
    - Each question should have four options:
    A)
    B)
    C)
    D)
    - Mention the correct answer after each question.
    - Provide a short explanation for the correct answer.
    """)
                )
      # ================= DOUBT SOLVER =================
    with tab3:
    
        st.subheader("💬 Doubt Solver")
    
        question = st.text_input("Ask your question")
    
        if st.button("Get Answer"):
    
            with st.spinner("Generating Answer..."):
    
                st.write_stream(rag_chain.stream(question))
      # ================= STUDY PLANNER =================
  
    with tab4:

        st.subheader("📅 Study Planner")

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

            st.markdown(plan)
