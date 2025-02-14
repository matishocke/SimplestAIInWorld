from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize OpenAI client
client = OpenAI(api_key="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")  # 🔹 Replace with your actual API key

# Dummy database with example questions and answers
dummy_database = [
    {"question": "What are the store opening hours?", "answer": "The store is open from 9 AM to 6 PM."},
    {"question": "Do you have wedding dresses in size 12?", "answer": "Yes, we have a variety of wedding dresses in size 12."},
    {"question": "Where is the store located?", "answer": "The store is located at 123 Main Street, Vejle."},
]

# Load sentence transformer model for semantic similarity search
print("🔍 Loading sentence transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Model loaded successfully!")

# Convert database questions into embeddings
print("🔍 Encoding database questions...")
database_embeddings = model.encode([item["question"] for item in dummy_database])
print("✅ Database embeddings created!")

def find_best_answer(user_question: str) -> str:
    """Finds the closest matching answer from the dummy database using cosine similarity."""
    print(f"🧠 Finding best answer for: {user_question}")
    
    user_embedding = model.encode(user_question)

    # Compute cosine similarity
    similarities = np.dot(database_embeddings, user_embedding) / (
        np.linalg.norm(database_embeddings, axis=1) * np.linalg.norm(user_embedding)
    )

    # Find the index of the most similar question
    best_index = np.argmax(similarities)
    best_match = dummy_database[best_index]

    print(f"✅ Best match found: {best_match['question']}")
    return best_match["answer"]

def ask_chatgpt(question: str, found_answer: str) -> str:
    """Sends the found answer to ChatGPT for refinement."""
    print("🔍 Sending to ChatGPT...")

    prompt = f"Q: {question}\nA (from database): {found_answer}\nCan you refine this answer to be more detailed and natural?"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    print("✅ ChatGPT response received!")
    return response.choices[0].message.content

if __name__ == "__main__":
    print("✅ Chatbot started! Type your question below.")
    
    while True:
        user_question = input("Ask a question (or type 'exit' to quit): ")
        if user_question.lower() == "exit":
            print("Goodbye!")
            break

        best_answer = find_best_answer(user_question)
        final_answer = ask_chatgpt(user_question, best_answer)

        print("\n🤖 AI Response:", final_answer, "\n")
