#!/usr/bin/env python3
"""
AI-Powered Research Assistant
Enhanced search interface for Arabic/English documents with streaming responses
"""

import streamlit as st
import os
from datetime import datetime
import time
from typing import List, Dict, Any, Optional, Tuple
import json
import re

# Import required libraries
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import numpy as np
from openai import OpenAI
import unicodedata

# ----------------------
# Configuration
# ----------------------
QDRANT_URL = "https://993e7bbb-cbe2-4b82-b672-90c4aed8585e.europe-west3-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.qRNtWMTIR32MSM6KUe3lE5qJ0fS5KgyAf86EKQgQ1FQ"
COLLECTION_NAME = "arabic_documents_enhanced"

# Get OpenAI API key from environment or config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")

# System prompt for the AI assistant
SYSTEM_PROMPT = """You are an intelligent research assistant with expertise in analyzing Arabic and English documents. Your role is to:

1. Provide accurate, well-structured answers based on the provided document context
2. Cite sources appropriately using [Source: filename] format
3. Maintain the original language of quotes and references
4. Be helpful, professional, and thorough in your responses
5. If you cannot find relevant information in the provided context, clearly state this
6. Organize complex answers with clear sections and bullet points when appropriate
7. Highlight key findings and important information
8. Maintain cultural sensitivity when dealing with Arabic content

Remember to always base your answers on the provided document context and cite your sources."""

# ----------------------
# Enhanced Embedding Model
# ----------------------
class EnhancedEmbedder:
    """Enhanced embedder with better Arabic support"""
    
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        self.vector_size = 768  # Model output dimension
        
    def encode(self, texts, normalize=True):
        """Encode texts with enhanced preprocessing"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess texts
        processed_texts = []
        for text in texts:
            # Clean text
            text = clean_text(text)
            
            # For Arabic text, also create normalized version
            if detect_language(text) == "arabic":
                normalized = normalize_arabic_text(text)
                # Use both original and normalized for better matching
                text = f"{text} {normalized}"
            
            processed_texts.append(text)
        
        # Encode
        embeddings = self.model.encode(
            processed_texts,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )
        
        return embeddings

# ----------------------
# Text Processing Functions
# ----------------------
def normalize_arabic_text(text):
    """Normalize Arabic text for better indexing"""
    if not text:
        return text
    
    # Remove diacritics (tashkeel)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    
    # Normalize Arabic characters
    replacements = {
        'أ': 'ا', 'إ': 'ا', 'آ': 'ا',  # Alef variations
        'ى': 'ي',  # Ya variations
        'ة': 'ه',  # Ta marbuta
        'ؤ': 'و', 'ئ': 'ي',  # Hamza
        '\u200c': '',  # Zero-width non-joiner
        '\u200d': '',  # Zero-width joiner
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove extra spaces and normalize
    text = ' '.join(text.split())
    
    return text

def clean_text(text):
    """Clean and prepare text for embedding"""
    if not text:
        return ""
    
    # Remove control characters
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
    
    # Fix common PDF extraction issues
    text = text.replace('\u200b', '')  # Zero-width space
    text = text.replace('\ufeff', '')  # BOM
    text = text.replace('\x00', '')    # Null character
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def detect_language(text):
    """Detect if text is primarily Arabic or English"""
    if not text:
        return "unknown"
    
    # Count Arabic characters
    arabic_chars = len([c for c in text if '\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F'])
    total_chars = len([c for c in text if c.isalpha()])
    
    if total_chars == 0:
        return "unknown"
    
    arabic_ratio = arabic_chars / total_chars
    
    if arabic_ratio > 0.3:
        return "arabic"
    else:
        return "english"

# ----------------------
# Search Functions
# ----------------------
def search_documents(query: str, embedder: EnhancedEmbedder, qdrant_client: QdrantClient, 
                    limit: int = 10, score_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """Search documents using enhanced embeddings"""
    
    try:
        # Generate query embedding
        query_embedding = embedder.encode(query)[0]
        
        # Search in Qdrant
        results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding.tolist(),
            limit=limit,
            score_threshold=score_threshold
        )
        
        # Process results
        processed_results = []
        for result in results:
            processed_results.append({
                'id': result.id,
                'score': result.score,
                'source': result.payload.get('source', 'Unknown'),
                'text': result.payload.get('text', ''),
                'chunk_index': result.payload.get('chunk_index', 0),
                'total_chunks': result.payload.get('total_chunks', 1),
                'language': result.payload.get('language', 'unknown'),
                'file_type': result.payload.get('file_type', 'unknown')
            })
        
        return processed_results
        
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

def rerank_results(results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """Rerank results based on relevance and diversity"""
    if not results:
        return results
    
    # Group by source
    source_groups = {}
    for result in results:
        source = result['source']
        if source not in source_groups:
            source_groups[source] = []
        source_groups[source].append(result)
    
    # Take best from each source first, then fill with remaining
    reranked = []
    
    # First pass: best from each source
    for source, group in source_groups.items():
        if group:
            reranked.append(max(group, key=lambda x: x['score']))
    
    # Second pass: fill with remaining high-score results
    remaining = []
    for source, group in source_groups.items():
        for result in group:
            if result not in reranked:
                remaining.append(result)
    
    # Sort remaining by score and add
    remaining.sort(key=lambda x: x['score'], reverse=True)
    reranked.extend(remaining)
    
    return reranked

# ----------------------
# AI Response Generation
# ----------------------
def generate_ai_response(query: str, context: List[Dict[str, Any]], client: OpenAI) -> str:
    """Generate AI response using OpenAI with streaming"""
    
    # Prepare context
    context_text = ""
    for i, doc in enumerate(context):
        context_text += f"\n[Document {i+1} - Source: {doc['source']}]\n"
        context_text += f"{doc['text']}\n"
        context_text += "-" * 50
    
    # Prepare messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"""Based on the following document context, please answer this question: {query}

Document Context:
{context_text}

Please provide a comprehensive answer based on the documents provided. Cite your sources using [Source: filename] format."""}
    ]
    
    # Create placeholder for streaming response
    response_placeholder = st.empty()
    full_response = ""
    
    try:
        # Stream the response
        stream = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=2000,
            stream=True
        )
        
        # Process stream
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                response_placeholder.markdown(full_response + "▌")
        
        # Final response without cursor
        response_placeholder.markdown(full_response)
        
        return full_response
        
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        response_placeholder.error(error_msg)
        return error_msg

# ----------------------
# UI Components
# ----------------------
def display_search_result(result: Dict[str, Any], index: int):
    """Display a single search result in an expandable format"""
    
    with st.expander(
        f"📄 {result['source']} (Score: {result['score']:.3f})",
        expanded=(index < 3)  # Expand top 3 results
    ):
        # Metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption(f"📑 Chunk: {result['chunk_index'] + 1}/{result['total_chunks']}")
        with col2:
            st.caption(f"🌐 Language: {result['language'].title()}")
        with col3:
            st.caption(f"📋 Type: {result['file_type']}")
        
        # Text content with highlighting
        text = result['text']
        
        # Simple highlight for Arabic/English
        if st.session_state.get('current_query'):
            query_terms = st.session_state.current_query.split()
            for term in query_terms:
                if len(term) > 2:  # Only highlight terms longer than 2 chars
                    text = text.replace(term, f"**{term}**")
        
        st.markdown(text)

def display_statistics(results: List[Dict[str, Any]]):
    """Display search statistics"""
    
    if not results:
        return
    
    with st.expander("📊 Search Statistics", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Results", len(results))
        
        with col2:
            unique_sources = len(set(r['source'] for r in results))
            st.metric("Unique Documents", unique_sources)
        
        with col3:
            avg_score = sum(r['score'] for r in results) / len(results)
            st.metric("Average Score", f"{avg_score:.3f}")
        
        # Language distribution
        lang_dist = {}
        for r in results:
            lang = r['language']
            lang_dist[lang] = lang_dist.get(lang, 0) + 1
        
        st.caption("Language Distribution:")
        for lang, count in lang_dist.items():
            st.write(f"- {lang.title()}: {count} chunks")

# ----------------------
# Main Application
# ----------------------
def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="AI Research Assistant",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .search-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""
    if 'show_ai_response' not in st.session_state:
        st.session_state.show_ai_response = True
    
    # Header
    st.markdown("""
    <div class="search-header">
        <h1>🔍 AI-Powered Research Assistant</h1>
        <p>Search and analyze Arabic and English documents with AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize clients
    try:
        with st.spinner("Initializing system..."):
            # Qdrant client
            qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            
            # Embedder
            if 'embedder' not in st.session_state:
                st.session_state.embedder = EnhancedEmbedder()
            
            # OpenAI client
            openai_client = OpenAI(api_key=OPENAI_API_KEY)
            print("OpenAI client initialized successfully.")  # Fixed line
            
    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Search settings
        st.subheader("Search Parameters")
        num_results = st.slider("Number of results", 5, 20, 10)
        score_threshold = st.slider("Relevance threshold", 0.0, 1.0, 0.5, 0.05)
        
        # AI settings
        st.subheader("AI Assistant")
        st.session_state.show_ai_response = st.checkbox(
            "Generate AI response", 
            value=st.session_state.show_ai_response
        )
        
        # Search history
        if st.session_state.search_history:
            st.subheader("📜 Recent Searches")
            for query in st.session_state.search_history[-5:]:
                if st.button(query, key=f"history_{query}"):
                    st.session_state.current_query = query
    
    # Main search interface
    col1, col2 = st.columns([5, 1])
    
    with col1:
        query = st.text_input(
            "Enter your search query (Arabic or English):",
            value=st.session_state.current_query,
            placeholder="e.g., ما هي أحدث التطورات في الذكاء الاصطناعي؟",
            key="search_input"
        )
    
    with col2:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # Perform search
    if search_button and query:
        st.session_state.current_query = query
        
        # Add to history
        if query not in st.session_state.search_history:
            st.session_state.search_history.append(query)
        
        # Search
        with st.spinner("Searching documents..."):
            results = search_documents(
                query=query,
                embedder=st.session_state.embedder,
                qdrant_client=qdrant_client,
                limit=num_results,
                score_threshold=score_threshold
            )
        
        if results:
            # Rerank results
            results = rerank_results(results, query)
            
            # Display statistics
            display_statistics(results)
            
            # Create tabs for different views
            if st.session_state.show_ai_response:
                tab1, tab2 = st.tabs(["🤖 AI Response", "📚 Search Results"])
                
                with tab1:
                    st.markdown("### AI-Generated Answer")
                    with st.spinner("Generating AI response..."):
                        ai_response = generate_ai_response(
                            query=query,
                            context=results[:5],  # Use top 5 results for context
                            client=openai_client
                        )
                
                with tab2:
                    st.markdown("### Document Search Results")
                    for i, result in enumerate(results):
                        display_search_result(result, i)
            else:
                st.markdown("### Document Search Results")
                for i, result in enumerate(results):
                    display_search_result(result, i)
        else:
            st.warning("No results found. Try adjusting your search query or lowering the relevance threshold.")
    
    # Footer
    st.markdown("---")
    st.caption("AI Research Assistant v2.0 | Powered by OpenAI & Qdrant")

if __name__ == "__main__":
    main()
