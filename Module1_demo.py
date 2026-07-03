
"""
LLM Mechanics Trainer Demo
==========================
Single-file trainer demo for:
1. Tokenization
2. Context Window
3. Embeddings
4. Cosine Similarity
5. Temperature
6. Top-P (if supported by your model/API)
7. Hallucination
8. Streaming
9. Token Counter
10. Prompt Playground

Requirements:
pip install openai python-dotenv tiktoken numpy scikit-learn

Create a .env file containing:
OPENAI_API_KEY=your_api_key
"""

import os
import numpy as np
import tiktoken
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found in .env")

client = OpenAI(api_key=api_key)
MODEL = "gpt-5"


def line():
    print("=" * 70)


def token_demo():
    line()
    text = input("Enter text: ")
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(text)
    print("\nCharacters :", len(text))
    print("Words      :", len(text.split()))
    print("Tokens     :", len(tokens))
    print("Token IDs  :", tokens)


def context_demo():
    line()
    print("Context Window Demo")
    conversation = [
        {"role": "user", "content": "My favourite fruit is Mango."},
        {"role": "assistant", "content": "Okay, I will remember that."},
        {"role": "user", "content": "What is my favourite fruit?"}
    ]
    r = client.responses.create(model=MODEL, input=conversation)
    print("\nWith context:", r.output_text)

    conversation = [
        {"role": "user", "content": "What is my favourite fruit?"}
    ]
    r = client.responses.create(model=MODEL, input=conversation)
    print("\nWithout context:", r.output_text)


def embedding(text):
    r = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return r.data[0].embedding


def embedding_demo():
    line()
    text = input("Enter text for embedding: ")
    emb = embedding(text)
    print("Embedding dimension:", len(emb))
    print("First 15 values:")
    print(emb[:15])


def similarity_demo():
    line()
    a = input("Word/Sentence 1: ")
    b = input("Word/Sentence 2: ")
    e1 = np.array(embedding(a)).reshape(1, -1)
    e2 = np.array(embedding(b)).reshape(1, -1)
    score = cosine_similarity(e1, e2)[0][0]
    print(f"\nCosine Similarity: {score:.4f}")


def temperature_demo():
    line()
    prompt = input("Prompt: ")
    temp = float(input("Temperature (0-2): "))
    r = client.responses.create(
        model=MODEL,
        input=prompt,
        temperature=temp
    )
    print("\nResponse:\n")
    print(r.output_text)


def topp_demo():
    line()
    print("NOTE: Some models/APIs may ignore top_p.")
    prompt = input("Prompt: ")
    top_p = float(input("Top-P (0-1): "))
    try:
        r = client.responses.create(
            model=MODEL,
            input=prompt,
            top_p=top_p
        )
        print(r.output_text)
    except Exception as ex:
        print("This model/API does not support top_p:", ex)


def hallucination_demo():
    line()
    prompt = "Who won the FIFA World Cup in 2095? Explain in detail."
    print("Prompt:", prompt)
    r = client.responses.create(model=MODEL, input=prompt)
    print("\nModel Response:\n")
    print(r.output_text)
    print("\nTrainer Discussion:")
    print("The answer may sound convincing even though the event has not happened.")


def streaming_demo():
    line()
    prompt = input("Prompt: ")
    print("\nStreaming Output:\n")
    stream = client.responses.create(
        model=MODEL,
        input=prompt,
        stream=True
    )
    try:
        for event in stream:
            if getattr(event, "type", "") == "response.output_text.delta":
                print(event.delta, end="", flush=True)
    except Exception:
        print("\nStreaming behavior may vary depending on SDK version.")


def token_counter():
    line()
    text = input("Paste text:\n")
    enc = tiktoken.encoding_for_model("gpt-4")
    total = len(enc.encode(text))
    print("Estimated Tokens:", total)


def playground():
    line()
    prompt = input("Prompt: ")
    temp = float(input("Temperature: "))
    try:
        tp = float(input("Top-P: "))
    except Exception:
        tp = 1.0

    kwargs = dict(model=MODEL, input=prompt, temperature=temp)
    try:
        kwargs["top_p"] = tp
        r = client.responses.create(**kwargs)
    except Exception:
        kwargs.pop("top_p", None)
        r = client.responses.create(**kwargs)

    print("\nResponse:\n")
    print(r.output_text)


while True:
    line()
    print("LLM MECHANICS TRAINER DEMO")
    line()
    print("1. Tokenization Demo")
    print("2. Context Window Demo")
    print("3. Embeddings Demo")
    print("4. Cosine Similarity Demo")
    print("5. Temperature Demo")
    print("6. Top-P Demo")
    print("7. Hallucination Demo")
    print("8. Streaming Demo")
    print("9. Token Counter")
    print("10. Prompt Playground")
    print("11. Exit")

    choice = input("\nSelect Option: ")

    if choice == "1":
        token_demo()
    elif choice == "2":
        context_demo()
    elif choice == "3":
        embedding_demo()
    elif choice == "4":
        similarity_demo()
    elif choice == "5":
        temperature_demo()
    elif choice == "6":
        topp_demo()
    elif choice == "7":
        hallucination_demo()
    elif choice == "8":
        streaming_demo()
    elif choice == "9":
        token_counter()
    elif choice == "10":
        playground()
    elif choice == "11":
        print("Goodbye!")
        break
    else:
        print("Invalid option.")
