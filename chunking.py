
"""
chunking_demo.py

Trainer-friendly demonstration of:
1. Character Chunking
2. Recursive Chunking
3. Semantic Chunking
4. FAISS Vector Database
5. Semantic Search
6. Effect of Chunk Size on Retrieval Quality

Install:
pip install langchain langchain-openai langchain-community langchain-experimental
pip install faiss-cpu python-dotenv

Create .env
OPENAI_API_KEY=your_api_key
"""

import os
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

# Semantic chunker
from langchain_experimental.text_splitter import SemanticChunker

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env")

print("=" * 80)
print("CHUNKING STRATEGIES DEMO")
print("=" * 80)

# ---------------------------------------------------------------------
# SAMPLE HR DATA
# ---------------------------------------------------------------------
text = """
Resume Writing Guide

A professional resume should usually be one page if the applicant has less
than five years of experience.

Candidates with more than ten years of experience can prepare a two-page
resume.

Recruiters usually spend only six to eight seconds scanning a resume.

Always include measurable achievements.

Use bullet points instead of long paragraphs.

ATS systems prefer standard section headings like Education,
Skills, Experience and Certifications.

Avoid tables, graphics and fancy icons because ATS software may fail
to understand them.

A cover letter should explain why you are interested in the company.

Avoid mentioning salary expectations unless requested.

Interview Preparation

Research the company before the interview.

Prepare examples using the STAR method.

Dress professionally.

Ask thoughtful questions at the end of the interview.
"""

print("\nORIGINAL DOCUMENT\n")
print(text)

embeddings = OpenAIEmbeddings()

# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------
def show_chunks(title, chunks):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}")
        print("-" * 40)
        print(chunk)

    print("\nTotal Chunks :", len(chunks))


def build_vector_store(chunks):
    docs = [Document(page_content=c) for c in chunks]
    return FAISS.from_documents(docs, embeddings)


def search(db, question):
    print("\nQuestion :", question)
    docs = db.similarity_search(question, k=2)

    print("\nRetrieved Chunks\n")

    for i, d in enumerate(docs, 1):
        print(f"Result {i}")
        print("-" * 30)
        print(d.page_content)


# ---------------------------------------------------------------------
# CHARACTER CHUNKING
# ---------------------------------------------------------------------
print("\n\nSTEP 1 : CHARACTER CHUNKING")
print("Cuts text purely using character count.\n")

character_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=180,
    chunk_overlap=40
)

character_chunks = character_splitter.split_text(text)

show_chunks("CHARACTER CHUNKS", character_chunks)

character_db = build_vector_store(character_chunks)

# ---------------------------------------------------------------------
# RECURSIVE CHUNKING
# ---------------------------------------------------------------------
print("\n\nSTEP 2 : RECURSIVE CHUNKING")
print("Tries Paragraph -> Sentence -> Word -> Character")

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=180,
    chunk_overlap=40
)

recursive_chunks = recursive_splitter.split_text(text)

show_chunks("RECURSIVE CHUNKS", recursive_chunks)

recursive_db = build_vector_store(recursive_chunks)

# ---------------------------------------------------------------------
# SEMANTIC CHUNKING
# ---------------------------------------------------------------------
print("\n\nSTEP 3 : SEMANTIC CHUNKING")
print("Groups similar ideas using embeddings.")

semantic_splitter = SemanticChunker(embeddings)

semantic_docs = semantic_splitter.create_documents([text])

semantic_chunks = [doc.page_content for doc in semantic_docs]

show_chunks("SEMANTIC CHUNKS", semantic_chunks)

semantic_db = build_vector_store(semantic_chunks)

# ---------------------------------------------------------------------
# SEARCH
# ---------------------------------------------------------------------
questions = [
    "How many pages should my resume be?",
    "Should I use tables in resume?",
    "How long do recruiters read resumes?",
    "How should I prepare for interview?"
]

for q in questions:

    print("\n" + "#" * 80)
    print("CHARACTER SEARCH")
    print("#" * 80)
    search(character_db, q)

    print("\n" + "#" * 80)
    print("RECURSIVE SEARCH")
    print("#" * 80)
    search(recursive_db, q)

    print("\n" + "#" * 80)
    print("SEMANTIC SEARCH")
    print("#" * 80)
    search(semantic_db, q)

# ---------------------------------------------------------------------
# IMPACT OF CHUNK SIZE
# ---------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 4 : IMPACT OF CHUNK SIZE")
print("=" * 80)

sizes = [60, 180, 500]

question = "How many pages should my resume be?"

for size in sizes:

    print("\n" + "=" * 60)
    print("Chunk Size =", size)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=20
    )

    chunks = splitter.split_text(text)

    db = build_vector_store(chunks)

    docs = db.similarity_search(question, k=1)

    print("Total Chunks :", len(chunks))
    print("\nBest Match\n")
    print(docs[0].page_content)

print("\n" + "=" * 80)
print("TRAINER EXPLANATION")
print("=" * 80)

print("""
1. Character Chunking
   - Splits based only on number of characters.
   - Can cut sentences in half.

2. Recursive Chunking
   - Preserves paragraphs and sentences.
   - Better than Character chunking.

3. Semantic Chunking
   - Uses embeddings.
   - Groups similar ideas together.
   - Usually gives the highest retrieval quality.

4. Chunk Size Impact

Small Chunk (60)
----------------
+ Faster
+ Less tokens
- Context gets lost

Medium Chunk (180)
------------------
+ Best balance
+ Better retrieval

Large Chunk (500)
-----------------
+ More context
- Higher token cost
- More irrelevant information

Conclusion
----------
Good chunking is one of the most important parts of a RAG pipeline.
Better chunks -> Better Retrieval -> Better LLM Answers.
""")
