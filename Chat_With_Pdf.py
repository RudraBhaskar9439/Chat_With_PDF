from sentence_transformers import SentenceTransformer
import fitz 
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np

import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY environment variable")

genai.configure(api_key=GOOGLE_API_KEY)

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            text += page.get_text()
    return text

def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    return splitter.split_text(text)


def sentence_encode(sentences):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(sentences)
    return embeddings

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
if __name__ == "__main__":
    pdf_path = "RudraCV.pdf"  
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text_into_chunks(text)
  

    chunk_vectors = []
    chunk_vectors = sentence_encode(chunks)
    
    while True:
        # Get user input
        query = input("\nEnter your question (or 'quit' to exit): ")
        
        if query.lower() == 'quit':
            break
            
        query_vector = sentence_encode([query])
        top_k = 3
        
        similarities = []
        for idx, chunk_vec in enumerate(chunk_vectors):
            sim = cosine_similarity(chunk_vec, query_vector[0])
            similarities.append((sim, idx))
        
        print("Similarities:", similarities)

        print("==" * 20)

        # Sort by similarity descending and get top_k indices
        top_chunks = sorted(similarities, reverse=True)[:top_k]
        top_indices = [idx for _, idx in top_chunks]

        print("Top chunk indices:", top_indices)

        new_context = ""
        for i in top_indices:
            new_context += chunks[i] + "\n"


        prompt_template = f"""You are a helpful assistant. Answer the question based on the context provided.
        Context: {new_context}
        Question: {query}"""

        try:
                # Configure the API
                genai.configure(api_key=GOOGLE_API_KEY)
                
                # Initialize the model correctly
                model = genai.GenerativeModel('gemini-2.0-flash')
                
                # Generate response with the actual prompt
                response = model.generate_content(prompt_template)
                print("\nResponse:")
                print(response.text)
        except Exception as e:
                print(f"Error generating response: {str(e)}")