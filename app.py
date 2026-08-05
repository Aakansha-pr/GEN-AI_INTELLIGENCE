
#              STUDYGEN AI - INTELLIGENT LEARNING ASSISTANT
# ============================================================

# ==============STEP 1: IMPORT LIBRARIES ================

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(layout = 'wide')
# ============== STEP 2 : LOAD API KEY ===========
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key=GOOGLE_API_KEY
)
st.success("Model Loaded Successfully!!!")

# =================== STEP 3 : PDF BACKEND FUNCTIONS ===================

def load_pdf(pdf_path):
    """
    This function loads the uploaded PDF.
    """
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

def create_embeddings():
    """
    This function loads the HuggingFace embedding model.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings

# ================= STEP 6 : CREATE VECTOR DATABASE =================

def create_vectorstore(chunks, embeddings):
    """
    This function creates the FAISS vector database.
    """

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vectorstore

# ================= STEP 7 : CREATE RETRIEVER =================

def create_retriever(vectorstore):
    """
    This function creates the retriever from the vector database.
    """

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    return retriever

# ================= STEP 8 : LCEL COMPONENTS =================

# ChatPromptTemplate
# RunnablePassthrough
# StrOutputParser

# (Already imported in Step 1)

# ================= STEP 9 : CREATE LCEL RAG CHAIN =================

def create_rag_chain(retriever):
    """
    This function creates the LCEL RAG Chain.
    """
    prompt = ChatPromptTemplate.from_template("""
    You are StudyGen AI, an Intelligent Learning Assistant.

    Answer the user's question ONLY using the provided context.

    If the answer is not available in the context,
    reply:
    "I couldn't find this information in the uploaded study material."
    Context:
    {context}
    Question:
    {question}
    """)

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
    """
    This function generates well-structured study notes
    from the uploaded PDF.
    """
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

    return notes

# ================= STEP 11 : QUIZ GENERATOR =================

def generate_quiz(rag_chain):
    """
    This function generates a quiz
    from the uploaded PDF.
    """

    prompt = """
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
- Format the output using proper headings and numbering.
"""

    quiz = rag_chain.invoke(prompt)

    return quiz
  
# ================= STEP 12 : DOUBT SOLVER =================

def solve_doubt(rag_chain, question):
    """
    This function answers the user's question
    from the uploaded study material.
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
"""

    response = model.invoke(prompt)

    return response.content