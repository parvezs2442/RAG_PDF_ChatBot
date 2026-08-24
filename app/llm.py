from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set")

client = OpenAI(api_key=api_key)


def generate_answer(query, documents):

    # Convert retrieved documents into context
    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    response = client.responses.create(
        model="gpt-5-mini",

        input=[
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant. "
                    "Answer the user's question using only the provided context. "
                    "If the answer is not present in the context, "
                    "say: 'I don't know based on the provided document.'"
                )
            },
            {
                "role": "user",
                "content": f"""
Context:
{context}

Question:
{query}
"""
            }
        ]
    )

    return response.output_text