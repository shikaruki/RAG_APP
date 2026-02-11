# RAG Approaches & Techniques

A comprehensive guide to all working approaches for building a production-ready RAG (Retrieval-Augmented Generation) system.

---

## Table of Contents
1. [Data Ingestion](#1-data-ingestion)
2. [Data Cleaning](#2-data-cleaning)
3. [Chunking Strategies](#3-chunking-strategies)
4. [Summarization Techniques](#4-summarization-techniques)
5. [Retrieval Approaches](#5-retrieval-approaches)
6. [Re-ranking Strategies](#6-re-ranking-strategies)
7. [Current Implementation](#7-current-implementation)

---

## 1. Data Ingestion

### 1.1 PDF Extraction
```python
# Using PyPDF (current implementation)
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(file_path)
pages = loader.load()
```

### 1.2 Alternative Loaders
| Loader | Best For | Pros | Cons |
|--------|----------|------|------|
| **PyPDFLoader** | General PDFs | Fast, reliable | May miss complex layouts |
| **PDFPlumber** | Tables, structured data | Better table extraction | Slower |
| **Unstructured** | Mixed content | Handles images, tables | Heavy dependency |
| **PyMuPDF** | Fast processing | Very fast | Less accurate on complex PDFs |
| **Azure Document Intelligence** | Enterprise | Best accuracy | Paid service |

### 1.3 Best Practices
- **File size limits**: Limit uploads (e.g., 30MB) to prevent memory issues
- **Batch processing**: Process large documents in batches
- **Error handling**: Gracefully handle corrupted or encrypted PDFs

---

## 2. Data Cleaning

### 2.1 Text Preprocessing
```python
def clean_text(text: str) -> str:
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters (optional)
    text = re.sub(r'[^\w\s.,!?;:\-]', '', text)
    
    # Normalize unicode
    text = unicodedata.normalize('NFKD', text)
    
    return text.strip()
```

### 2.2 Cleaning Strategies
| Strategy | When to Use | Implementation |
|----------|-------------|----------------|
| **Whitespace normalization** | Always | `' '.join(text.split())` |
| **Empty chunk filtering** | Always | `if text.strip()` |
| **Header/Footer removal** | Formal docs | Regex patterns |
| **Page number removal** | Multi-page PDFs | `re.sub(r'Page \d+', '')` |
| **Table flattening** | Tabular data | Custom parsers |

### 2.3 Quality Filters
```python
def is_quality_chunk(text: str) -> bool:
    # Minimum length
    if len(text) < 50:
        return False
    
    # Not just numbers/symbols
    alpha_ratio = sum(c.isalpha() for c in text) / len(text)
    if alpha_ratio < 0.5:
        return False
    
    return True
```

---

## 3. Chunking Strategies

### 3.1 Fixed-Size Chunking (Current)
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,      # Characters per chunk
    chunk_overlap=300,    # Overlap for context
    separators=["\n\n", "\n", ". ", " "]
)
chunks = splitter.split_documents(pages)
```

### 3.2 Chunking Strategies Comparison

| Strategy | Chunk Size | Overlap | Best For |
|----------|------------|---------|----------|
| **Small chunks** | 500-1000 | 50-100 | Precise answers, Q&A |
| **Medium chunks** | 1000-2000 | 100-200 | Balanced retrieval |
| **Large chunks** | 2000-4000 | 200-400 | Context-heavy, summaries |

### 3.3 Advanced Chunking

#### Semantic Chunking
```python
# Group by semantic similarity
from sentence_transformers import SentenceTransformer

def semantic_chunk(text: str, threshold: float = 0.7):
    sentences = text.split('. ')
    embeddings = model.encode(sentences)
    
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(1, len(sentences)):
        similarity = cosine_similarity(embeddings[i-1], embeddings[i])
        if similarity > threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append('. '.join(current_chunk))
            current_chunk = [sentences[i]]
    
    return chunks
```

#### Hierarchical Chunking
```python
# Parent-child chunks for better context
def hierarchical_chunk(text: str):
    # Large parent chunks
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=4000)
    parents = parent_splitter.split_text(text)
    
    # Smaller child chunks
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
    
    hierarchy = []
    for parent in parents:
        children = child_splitter.split_text(parent)
        hierarchy.append({
            "parent": parent,
            "children": children
        })
    
    return hierarchy
```

#### Document-Aware Chunking
```python
# Respect document structure
separators = [
    "\n## ",      # H2 headers
    "\n### ",     # H3 headers
    "\n\n",       # Paragraphs
    "\n",         # Lines
    ". ",         # Sentences
    " "           # Words
]
```

---

## 4. Summarization Techniques

### 4.1 Chunk Summarization
```python
def summarize_chunk(chunk: str) -> str:
    prompt = f"""Summarize the following text in 2-3 sentences:
    
    {chunk}
    
    Summary:"""
    
    return llm.generate(prompt)
```

### 4.2 Map-Reduce Summarization
```python
# For very large documents
def map_reduce_summarize(chunks: list) -> str:
    # Map: Summarize each chunk
    summaries = [summarize_chunk(c) for c in chunks]
    
    # Reduce: Combine summaries
    combined = "\n".join(summaries)
    
    final_prompt = f"""Combine these summaries into one coherent summary:
    
    {combined}"""
    
    return llm.generate(final_prompt)
```

### 4.3 Extractive vs Abstractive

| Type | Method | Pros | Cons |
|------|--------|------|------|
| **Extractive** | Select key sentences | Fast, faithful | May miss context |
| **Abstractive** | Generate new text | More coherent | May hallucinate |
| **Hybrid** | Extract then refine | Best of both | More complex |

---

## 5. Retrieval Approaches

### 5.1 Dense Retrieval (Current)
```python
# Semantic search using embeddings
def search(query: str, limit: int = 5):
    query_embedding = embed_query(query)
    
    results = qdrant.search(
        collection_name="documents",
        query_vector=query_embedding,
        limit=limit
    )
    
    return results
```

### 5.2 Sparse Retrieval (BM25)
```python
from rank_bm25 import BM25Okapi

# Index documents
tokenized_corpus = [doc.split() for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)

# Search
def bm25_search(query: str, top_k: int = 5):
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [documents[i] for i in top_indices]
```

### 5.3 Hybrid Retrieval (Best Practice)
```python
def hybrid_search(query: str, alpha: float = 0.5, top_k: int = 5):
    # Dense retrieval
    dense_results = dense_search(query, top_k * 2)
    
    # Sparse retrieval
    sparse_results = bm25_search(query, top_k * 2)
    
    # Combine scores (Reciprocal Rank Fusion)
    combined = {}
    
    for rank, doc in enumerate(dense_results):
        doc_id = doc["id"]
        combined[doc_id] = combined.get(doc_id, 0) + alpha / (rank + 60)
    
    for rank, doc in enumerate(sparse_results):
        doc_id = doc["id"]
        combined[doc_id] = combined.get(doc_id, 0) + (1 - alpha) / (rank + 60)
    
    # Sort by combined score
    sorted_docs = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_docs[:top_k]
```

### 5.4 Retrieval Comparison

| Method | Best For | Speed | Accuracy |
|--------|----------|-------|----------|
| **Dense (Semantic)** | Conceptual queries | Medium | High |
| **Sparse (BM25)** | Keyword queries | Fast | Medium |
| **Hybrid** | General use | Medium | Highest |
| **Multi-Query** | Ambiguous queries | Slow | High |

### 5.5 Multi-Query Retrieval
```python
def multi_query_search(query: str, top_k: int = 5):
    # Generate query variations
    variations_prompt = f"""Generate 3 different versions of this question:
    {query}"""
    
    variations = llm.generate(variations_prompt).split('\n')
    
    # Search with each variation
    all_results = []
    for q in [query] + variations:
        results = dense_search(q, top_k)
        all_results.extend(results)
    
    # Deduplicate and rank
    unique = deduplicate(all_results)
    return unique[:top_k]
```

---

## 6. Re-ranking Strategies

### 6.1 Cross-Encoder Re-ranking
```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank(query: str, documents: list, top_k: int = 5):
    # Create query-document pairs
    pairs = [[query, doc["content"]] for doc in documents]
    
    # Score with cross-encoder
    scores = reranker.predict(pairs)
    
    # Sort by score
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    
    return [doc for doc, score in ranked[:top_k]]
```

### 6.2 Cohere Re-ranker
```python
import cohere

co = cohere.Client(api_key)

def cohere_rerank(query: str, documents: list, top_k: int = 5):
    results = co.rerank(
        model="rerank-english-v2.0",
        query=query,
        documents=[d["content"] for d in documents],
        top_n=top_k
    )
    
    return [documents[r.index] for r in results]
```

### 6.3 LLM-based Re-ranking
```python
def llm_rerank(query: str, documents: list, top_k: int = 3):
    prompt = f"""Given the question: "{query}"

Rank these documents from most to least relevant (return indices):

{chr(10).join([f'{i}: {doc[:200]}...' for i, doc in enumerate(documents)])}

Ranking (comma-separated indices):"""
    
    ranking = llm.generate(prompt)
    indices = [int(i.strip()) for i in ranking.split(',')]
    
    return [documents[i] for i in indices[:top_k]]
```

### 6.4 Re-ranking Comparison

| Method | Speed | Accuracy | Cost |
|--------|-------|----------|------|
| **Cross-Encoder** | Medium | High | Free |
| **Cohere Rerank** | Fast | Very High | Paid |
| **LLM Rerank** | Slow | High | Paid |
| **MMR (Diversity)** | Fast | Medium | Free |

### 6.5 Maximal Marginal Relevance (MMR)
```python
def mmr_rerank(query_embedding, doc_embeddings, documents, 
               lambda_param: float = 0.5, top_k: int = 5):
    """Balance relevance and diversity"""
    selected = []
    remaining = list(range(len(documents)))
    
    while len(selected) < top_k and remaining:
        mmr_scores = []
        
        for idx in remaining:
            # Relevance to query
            relevance = cosine_similarity(query_embedding, doc_embeddings[idx])
            
            # Similarity to already selected
            if selected:
                max_sim = max(cosine_similarity(doc_embeddings[idx], 
                             doc_embeddings[s]) for s in selected)
            else:
                max_sim = 0
            
            # MMR score
            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
            mmr_scores.append((idx, mmr))
        
        # Select highest MMR
        best_idx = max(mmr_scores, key=lambda x: x[1])[0]
        selected.append(best_idx)
        remaining.remove(best_idx)
    
    return [documents[i] for i in selected]
```

---

## 7. Current Implementation

### What This Project Uses

| Component | Approach | Why This Choice |
|-----------|----------|-----------------|
| **PDF Extraction** | PDFPlumber | Better table/layout extraction than PyPDF |
| **Embeddings** | Voyage AI (voyage-3) | High quality 1024-dim vectors, good accuracy |
| **LLM** | Groq (llama-3.3-70b) | Fast inference, good quality |
| **Vector Store** | Qdrant Cloud | Hosted, scalable, no infra management |
| **Chunking** | RecursiveCharacterTextSplitter | Respects semantic boundaries |
| **Retrieval** | Dense (Top-K) | Simple, effective for most queries |
| **Chat History** | Last 6 messages | Context for follow-up questions |

### Config Location
```python
# app/config.py
CHUNK_SIZE: int = 3000      # Large enough for context
CHUNK_OVERLAP: int = 300    # 10% overlap prevents losing context
MAX_CHUNKS: int = 100       # Limits API calls for large docs
TOP_K: int = 5              # Balance between coverage and noise
```

---

## 8. Why We Chose These Approaches

### 8.1 PDFPlumber over PyPDF

**Chosen:** PDFPlumber  
**Why:**
- **Better table extraction**: Preserves table structure as text
- **Layout awareness**: Respects multi-column layouts
- **Cleaner text**: Less garbage characters from complex PDFs
- **Trade-off**: Slightly slower, but worth it for quality

```python
# Our implementation
def extract_text_from_pdf(pdf_path: str):
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()  # Preserves layout
```

### 8.2 RecursiveCharacterTextSplitter (Best Chunking Strategy)

**Chosen:** RecursiveCharacterTextSplitter with semantic separators  
**Why:**

| Factor | Our Choice | Reasoning |
|--------|------------|-----------|
| **Chunk Size** | 3000 chars | Large enough for context, small enough for embedding |
| **Overlap** | 300 chars (10%) | Prevents losing context at boundaries |
| **Separators** | `["\n\n", "\n", ". ", ", ", " "]` | Respects document structure |

**Why Recursive over Fixed:**
```
Fixed Chunking:        Recursive Chunking:
┌─────────────────┐    ┌─────────────────┐
│ This is a para-│    │ This is a       │
│graph that gets │    │ paragraph.      │
│ cut in the mid-│    ├─────────────────┤
│dle randomly.   │    │ This is another │
└─────────────────┘    │ paragraph.      │
                       └─────────────────┘
```

**Key Benefits:**
1. **Semantic boundaries**: Splits at paragraphs first, then sentences
2. **Preserves meaning**: Doesn't cut mid-sentence
3. **Configurable**: Can adjust separators for different doc types

### 8.3 Voyage AI Embeddings

**Chosen:** voyage-3 (1024 dimensions)  
**Why:**
- **High quality**: Better semantic understanding than smaller models
- **Balanced size**: 1024 dims = good accuracy without excessive storage
- **API-based**: No local GPU needed, easy to deploy

**Trade-off:** Free tier has rate limits → We use batching + delays

### 8.4 Dense Retrieval (Top-K)

**Chosen:** Dense semantic search with k=5  
**Why:**
- **Simple & effective**: Works well for most natural language queries
- **No extra infrastructure**: Unlike hybrid, no BM25 index needed
- **Fast**: Single vector similarity search

**When to upgrade to Hybrid:**
- Keyword-heavy documents (legal, medical)
- Exact term matching needed
- Users search with specific terms

### 8.5 Chat History (Last 6 Messages)

**Chosen:** Include last 6 messages in LLM context  
**Why:**
- **Follow-up understanding**: "Tell me more about that"
- **Context continuity**: Remembers what was discussed
- **Token limit safe**: 6 messages fits within context window

```python
# Our implementation
if history:
    for msg in history[-6:]:  # Last 6 only
        messages.append({"role": msg["role"], "content": msg["content"]})
```

---

## 9. Quick Reference: When to Use What

| Scenario | Recommended Approach |
|----------|---------------------|
| **Small documents (<50 pages)** | Standard chunking, dense retrieval |
| **Large documents (>100 pages)** | Larger chunks, max chunk limit |
| **Technical/keyword-heavy** | Hybrid retrieval (dense + BM25) |
| **Conversational Q&A** | Chat history + dense retrieval |
| **High accuracy needed** | Re-ranking with cross-encoder |
| **Diverse results needed** | MMR re-ranking |
| **Production/scale** | Qdrant Cloud + batch processing |

---

## 10. Future Improvements

### Priority 1: Quick Wins
1. **Cross-Encoder Re-ranking**: Add `cross-encoder/ms-marco-MiniLM-L-6-v2` for better result quality
2. **Query Caching**: Cache frequent queries to reduce API calls

### Priority 2: Advanced
3. **Hybrid Retrieval**: Add BM25 for keyword matching alongside dense
4. **Semantic Chunking**: Use sentence embeddings to find natural break points
5. **Query Expansion**: Generate query variations for better recall

### Priority 3: Scale
6. **Async Processing**: Background job queue for large PDFs
7. **Multi-document Search**: Search across multiple uploaded PDFs
8. **Analytics**: Track query patterns and retrieval quality

---

## 11. Comparison: Why Not Other Approaches?

| Approach | Why We Didn't Choose |
|----------|---------------------|
| **sentence-transformers** | Heavy to deploy on Render, requires GPU for speed |
| **Fixed chunking** | Cuts mid-sentence, loses semantic meaning |
| **Very small chunks (500)** | Too granular, loses paragraph context |
| **No overlap** | Loses context at chunk boundaries |
| **BM25 only** | Misses semantic similarity (synonyms, paraphrasing) |
| **Cohere Rerank** | Paid service, adds latency |

---

*Last updated: February 2026*
