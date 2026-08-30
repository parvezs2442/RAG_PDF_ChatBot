
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set")


model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=1,
    max_tokens=None,
    max_retries=2,
)

def generate_answer(query, documents):

    # Convert retrieved documents into context
    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY the provided context.

If the answer is not present in the context, say:
"I don't know based on the provided document."

Context:
{context}

Question:
{query}
"""

    response = model.invoke(prompt)

    # Extract only the text from Gemini's structured response
    if isinstance(response.content, list):
        answer = "".join(
            item["text"]
            for item in response.content
            if isinstance(item, dict) and item.get("type") == "text"
        )
        return answer

    return response.content

