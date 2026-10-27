import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from pypdf import PdfReader
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def load_cv(file_path):
    reader = PdfReader(file_path)

    cv_text = "\n\n".join(
        page.extract_text() or ""
        for page in reader.pages
    )

    return cv_text

def load_prompt():
    return (
        Path(__file__).resolve().parent / "CV_Prompt.txt"
    ).read_text(encoding="utf-8")

def create_llm():
    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        temperature=0.7,
        max_tokens=1024
    )

    return llm

def create_chain():

    llm = create_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", load_prompt()),
        ("human", "Analyze the following CV:\n\n<cv>\n{cv}\n</cv>")
    ])

    chain = prompt | llm

    return chain









def agent_cv():
     cv_path = Path("cv.pdf").resolve().parent / "cv.pdf"
     cv_text = load_cv(cv_path)

     if not cv_text.strip():
        raise ValueError("Aucun texte trouvé dans le CV.")

     chain = create_chain()

     response = chain.invoke({
        "cv": cv_text
     })

     return response.content

