import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_pdfs(directory="documents"):
    # Store documents and keep metadata
    # (text, page number, source file)
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory, filename)
            pdf = PdfReader(file_path)
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:  # Only add if text is extracted
                    documents.append({
                        "text": text,
                        "page": page_num,
                        "source": filename
                    })
    return documents

# Create embeddings and vector store
def create_vector_store(documents):
    # Load a pre-trained embedding model
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')  # Lightweight and effective
    
    # Extract text from documents for embedding
    texts = [doc["text"] for doc in documents]
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Create a FAISS index
    dimension = embeddings.shape[1]  # Embedding size
    index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
    index.add(np.array(embeddings, dtype="float32"))
    
    return index, embeddings, model

# Query the vector store
def query_vector_store(query, index, documents, model, k=3):
    # Convert query to embedding
    query_embedding = model.encode([query])[0]
    
    # Search for top-k similar documents
    distances, indices = index.search(np.array([query_embedding], dtype="float32"), k)
    
    # Retrieve relevant documents
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        doc = documents[idx]
        results.append({
            "text": doc["text"],
            "page": doc["page"],
            "source": doc["source"],
            "similarity": float(1 - distance)  # Convert distance to similarity (optional)
        })
    return results

# Load local LLM (Mistral-7B)
def load_llm():
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"  # 7B parameters, good for French
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True)
    return model, tokenizer

# Generate response with local LLM
def generate_response(query, results, model, tokenizer):
    context = "\n\n".join([f"From {r['source']} (Page {r['page']}): {r['text']}" for r in results])
    prompt = f"Question: {query}\n\nContext from documents:\n{context}\n\nProvide a concise answer in French based on the context, and list the sources (document name and page number) used."
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    docs = load_pdfs()
    print(f"Loaded {len(docs)} pages from PDFs.")
    
    # Print a sample to verify
    # if docs:
    #     print("Sample entry:", docs[0])
    
    # Create vector store
    index, embeddings, model = create_vector_store(docs)
    print(f"Created vector store with {index.ntotal} entries.")
    
    # Optional: Save the index to disk for reuse
    faiss.write_index(index, "documents/vector_store.index")
    print("Vector store saved to 'documents/vector_store.index'.")

    # # Example query
    # query = "effectifs 2024"
    # results = query_vector_store(query, index, docs, model, k=3)
    # response = generate_response(query, results)
    # print(f"Query: {query}")
    # print(response)

    # Load local LLM
    print("Loading Mistral-7B model (this may take a minute)...")
    llm_model, tokenizer = load_llm()
    print("Model loaded successfully.")
    
    # Interactive loop
    print("\nBienvenue! Tapez votre requête (ou 'quitter' pour quitter la discussion):")
    while True:
        query = input("> ")
        if query.lower() == "quitter":
            break
        results = query_vector_store(query, index, docs, model, k=3)
        response = generate_response(query, results, llm_model, tokenizer)
        print(f"\nRéponse:\n{response}\n")
