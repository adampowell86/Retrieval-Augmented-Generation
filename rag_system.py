from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from transformers import AutoTokenizer
import numpy as np
import os

# Step 3: Load document and generate embeddings
def load_and_chunk_document(file_path=r"C:\Users\adamp\OneDrive - Full Sail University\The Artificial Intelligence Ecosystem.CAP320-O\4.2 Retrieval-Augmented Generation\Selected_Document.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return [chunk for chunk in text.split("\n\n") if chunk.strip()]

def generate_embeddings(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    return {
        "chunks": chunks,
        "embeddings": embeddings
    }

# Step 4: Query and response system
class RAGSystem:
    def __init__(self, embedding_dict):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.generator = pipeline("text2text-generation", model="google/flan-t5-small")
        self.embedding_dict = embedding_dict

    def retrieve_chunks(self, query, top_k=3):
        query_embedding = self.embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embedding_dict["embeddings"])
        top_indices = np.argsort(similarities[0])[-top_k:][::-1]
        return [self.embedding_dict["chunks"][i] for i in top_indices]

    def generate_response(self, query, retrieved_chunks):
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        context = "\n".join(retrieved_chunks)
        prompt_prefix = (
            "Answer the following question using the provided context.\n\n"
            f"Question: {query}\n\n"
            "Context:\n"
        )
        prompt_suffix = "\n\nAnswer:"
        # Calculate the available tokens for context (adjust max_total_tokens as needed)
        max_total_tokens = 512
        prefix_tokens = tokenizer.encode(prompt_prefix, add_special_tokens=False)
        suffix_tokens = tokenizer.encode(prompt_suffix, add_special_tokens=False)
        max_context_tokens = max_total_tokens - len(prefix_tokens) - len(suffix_tokens)
        context_tokens = tokenizer.encode(context, add_special_tokens=False)
        if len(context_tokens) > max_context_tokens:
            context_tokens = context_tokens[:max_context_tokens]
            context = tokenizer.decode(context_tokens, skip_special_tokens=True)
        full_prompt = prompt_prefix + context + prompt_suffix
        return self.generator(full_prompt, max_length=200)[0]["generated_text"]

if __name__ == "__main__":
    # Load document and generate embeddings
    chunks = load_and_chunk_document()
    embedding_dict = generate_embeddings(chunks)
    
    # Initialize the RAG system
    rag = RAGSystem(embedding_dict)
    
    # Get user query
    query = input("Enter your question: ")
    
    # Retrieve relevant chunks
    retrieved = rag.retrieve_chunks(query)
    
    # Generate and display response
    response = rag.generate_response(query, retrieved)
    
    print(f"\nRetrieved Context:\n{'-'*30}")
    for i, chunk in enumerate(retrieved, 1):
        print(f"Chunk {i}: {chunk[:150]}...")  # Show first 150 characters
    print(f"\nGenerated Response:\n{'-'*30}\n{response}")