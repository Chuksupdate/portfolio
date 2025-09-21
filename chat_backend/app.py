import os
import json
import numpy as np
import asyncio
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
import google.generativeai as genai
import hashlib
import time

# Load environment variables from .env
load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
RESUME_PATH = os.getenv('RESUME_PATH', 'resume.txt')

app = Flask(__name__)
CORS(app)

class OptimizedTfidfEmbeddings:
    def __init__(self):
        # Optimized TF-IDF
        self.vectorizer = TfidfVectorizer(
            max_features=800, 
            stop_words='english',
            ngram_range=(1, 2),  # Add bigrams for better context
            min_df=1,
            max_df=0.95,
            norm='l2',
            use_idf=True
        )
        self.chunks = []
        self.vectors = None
        self.vectors_normalized = None  # Pre-normalized for faster cosine similarity
        self._cache = {}  # Manual cache for async compatibility
    
    def fit_transform(self, texts):
        self.chunks = texts
        self.vectors = self.vectorizer.fit_transform(texts)
        # Pre-normalize vectors for faster cosine similarity
        self.vectors_normalized = self.vectors.copy()
        self.vectors_normalized = self.vectors_normalized.astype(np.float32)  # Use float32 for speed
        return self.vectors
    
    async def similarity_search_async(self, query, k=3):
        """Async similarity search with caching"""
        # Create hash for caching
        query_hash = hashlib.md5(query.encode()).hexdigest()
        cache_key = f"{query_hash}_{k}"
        
        # Check cache first
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Run CPU-intensive search in thread pool
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, self._perform_search, query, k)
        
        # Cache result
        self._cache[cache_key] = results
        
        # Limit cache size
        if len(self._cache) > 100:
            # Remove oldest 20 entries
            keys_to_remove = list(self._cache.keys())[:20]
            for key in keys_to_remove:
                del self._cache[key]
        
        return results
    
    def _perform_search(self, query, k=3):
        """Core search logic (runs in thread pool)"""
        if self.vectors_normalized is None:
            return []
        
        try:
            # Transform query using pre-fitted vectorizer
            query_vector = self.vectorizer.transform([query])
            query_vector = query_vector.astype(np.float32)
            
            # Fast cosine similarity with pre-normalized vectors
            similarities = cosine_similarity(query_vector, self.vectors_normalized).flatten()
            
            # Use numpy for faster top-k selection
            if len(similarities) > k:
                top_indices = np.argpartition(similarities, -k)[-k:]
                top_indices = top_indices[np.argsort(similarities[top_indices])][::-1]
            else:
                top_indices = np.argsort(similarities)[::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Lowered threshold for more results
                    results.append({
                        'page_content': self.chunks[idx],
                        'score': float(similarities[idx])
                    })
            
            return results
        except Exception as e:
            print(f"Search error: {e}")
            return []

# Load and split resume into optimized chunks
def load_resume_chunks():
    if os.path.exists(RESUME_PATH):
        try:
            with open(RESUME_PATH, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            text = f"Error reading TXT: {e}"
    else:
        text = "Resume not found."
    
    # Optimized chunking strategy
    chunks = []
    
    # Split by double newlines (paragraphs)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    for para in paragraphs:
        if len(para) > 80:  # Only substantial content
            # Smart chunking for long paragraphs
            if len(para) > 500:
                sentences = [s.strip() + '.' for s in para.split('.') if s.strip()]
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk + sentence) <= 400:
                        current_chunk += " " + sentence
                    else:
                        if len(current_chunk.strip()) > 80:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
                
                if len(current_chunk.strip()) > 80:
                    chunks.append(current_chunk.strip())
            else:
                chunks.append(para)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_chunks = []
    for chunk in chunks:
        chunk_hash = hashlib.md5(chunk.encode()).hexdigest()
        if chunk_hash not in seen:
            seen.add(chunk_hash)
            unique_chunks.append(chunk)
    
    return unique_chunks

# Initialize the embedding system
print("Loading resume chunks...")
RESUME_CHUNKS = load_resume_chunks()
print(f"Loaded {len(RESUME_CHUNKS)} chunks")

print("Building TF-IDF index...")
EMBEDDINGS = OptimizedTfidfEmbeddings()

if RESUME_CHUNKS:
    EMBEDDINGS.fit_transform(RESUME_CHUNKS)
    print("TF-IDF index built successfully")

# Async Gemini API call
async def call_gemini_async(context, question):
    """Async Gemini API call"""
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        prompt = (
            "You are an assistant with access to details about me. Don't mention the resume. "
            "IF user is trying to chat e.g user says hello or hi, no need to check context, reply with hello, what do you want to know about Abdulazeez?"
            "Answer the following question based ONLY on the provided context asides when user says hello, hi or user is greeting."            
            "If the answer is not present, say 'I could not find that information about Abdulazeez, what else would you like to know?'\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\n\n"
            "If the context is long, summarize or return only the most relevant information."
        )
        
        # Run API call in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, model.generate_content, prompt)
        
        return response.text.strip() if hasattr(response, 'text') else str(response)
    except Exception as e:
        return f"Error with Gemini API: {e}"

# Cache for responses
response_cache = {}

async def ask_gemini_async(question):
    """Main async RAG function"""
    if not GEMINI_API_KEY:
        return "Gemini API key not set."
    
    if not RESUME_CHUNKS:
        return "Resume not loaded."
    
    # Check response cache first
    question_hash = hashlib.md5(question.encode()).hexdigest()
    if question_hash in response_cache:
        return response_cache[question_hash]
    
    try:
        # Async retrieval
        start_time = time.time()
        docs = await EMBEDDINGS.similarity_search_async(question, k=3)
        retrieval_time = time.time() - start_time
        
        if not docs:
            return "I could not find that information about Abdulazeez. what else would you like to know?"
        
        # Prepare context
        rag_context = "\n---\n".join([d['page_content'] for d in docs])
        
        # Async Gemini call
        start_time = time.time()
        response = await call_gemini_async(rag_context, question)
        api_time = time.time() - start_time
        
        # Cache response
        response_cache[question_hash] = response
        
        # Limit cache size
        if len(response_cache) > 50:
            # Remove oldest entries
            keys_to_remove = list(response_cache.keys())[:10]
            for key in keys_to_remove:
                del response_cache[key]
        
        print(f"Retrieval: {retrieval_time:.3f}s, API: {api_time:.3f}s")
        return response
        
    except Exception as e:
        return f"Error: {e}"

@app.route('/', methods=['GET'])
def home():
    return f"Abdulazeez Chat API is running. Loaded {len(RESUME_CHUNKS)} chunks.", 200

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('message', '')
    
    if not question:
        return jsonify({'error': 'No message provided'}), 400
    
    # Run async function in sync context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        answer = loop.run_until_complete(ask_gemini_async(question))
    finally:
        loop.close()
    
    return jsonify({'answer': answer})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'chunks_loaded': len(RESUME_CHUNKS),
        'retrieval_cache_size': len(getattr(EMBEDDINGS, '_cache', {})),
        'response_cache_size': len(response_cache)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
