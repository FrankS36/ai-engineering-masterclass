# Chapter 5: RAG and Knowledge Systems

Large language models are trained on a snapshot of the world. The moment training ends, everything the model knows begins aging. Customers ask about last quarter's product changes, compliance teams need answers grounded in internal policy documents, and support agents need to reference tickets from this morning. The model has none of it. Retrieval-Augmented Generation (RAG) is the dominant pattern for closing that gap: instead of hoping the model memorized the right fact, you retrieve the relevant context at query time and hand it to the model alongside the user's question.

This chapter walks through every stage of building a RAG system that works in production, from ingestion to evaluation, including the failure modes that only surface once real data hits the pipeline.

---

**A practitioner's note:** The model doesn't know the difference between clean data and real data. When it received curated demo data, it produced clean outputs. When it receives what the warehouse actually produces, it will produce something else. The data infrastructure is where RAG systems live or die — not the retrieval algorithm.

---

## Why RAG Exists

Three problems motivate RAG:

**Knowledge cutoffs.** Models are frozen at training time. A model trained through April 2024 cannot answer questions about events, documents, or data created after that date. RAG lets you inject current information without retraining.

**Hallucination reduction.** When a model lacks knowledge, it fabricates plausible-sounding answers. By providing source documents in the prompt, you give the model something to cite rather than invent. This does not eliminate hallucination, but it reduces it substantially when the retrieval is accurate.

**Domain-specific knowledge.** Your internal knowledge base, proprietary data, customer records, and specialized documents were never in the training set. RAG makes them accessible without exposing them during model training.

### When to Use RAG vs. Fine-Tuning vs. Long Context

| Approach | Best For | Limitations |
|---|---|---|
| **RAG** | Large, changing knowledge bases; need for source attribution; data that must stay current | Adds retrieval latency; quality depends on chunking and retrieval |
| **Fine-tuning** | Teaching style, format, or domain-specific reasoning patterns | Does not reliably inject facts; expensive to update; no source citation |
| **Long context** | Small, stable document sets (under ~100K tokens); single-session analysis | Cost scales linearly with context size; attention degrades over very long contexts; no persistence |

In practice, these approaches combine. You might fine-tune a model to follow your output format while using RAG to supply the facts.

## The RAG Pipeline

Every RAG system follows the same core loop:

**Ingest** — Collect source documents (PDFs, HTML, database exports, API responses). Parse them into clean text. This is where most production systems break first: malformed PDFs, encoding issues, tables that lose structure, and headers that become noise.

**Chunk** — Split documents into smaller pieces. The model's context window is finite, and retrieval precision improves when chunks are focused on a single topic.

**Embed** — Convert each chunk into a dense vector (a list of floating-point numbers) using an embedding model. These vectors capture semantic meaning so that similar concepts land near each other in vector space.

**Store** — Write the vectors (and their associated text) into a vector database or index that supports fast nearest-neighbor search.

**Retrieve** — When a user query arrives, embed the query with the same model, then search the vector store for the K most similar chunks.

**Generate** — Pass the retrieved chunks into the LLM prompt alongside the user's question. The model synthesizes an answer grounded in the provided context.

```python
import openai

client = openai.OpenAI()

def rag_pipeline(query: str, chunks: list[str], k: int = 5) -> str:
    # Embed the query
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    # Retrieve top-k chunks (simplified — in production, use a vector DB)
    scored = []
    for chunk in chunks:
        chunk_emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        ).data[0].embedding
        score = sum(a * b for a, b in zip(query_embedding, chunk_emb))
        scored.append((score, chunk))
    scored.sort(reverse=True)
    top_chunks = [text for _, text in scored[:k]]

    # Generate
    context = "\n\n---\n\n".join(top_chunks)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": (
                "Answer the user's question using only the provided context. "
                "If the context does not contain the answer, say so."
            )},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    )
    return response.choices[0].message.content
```

In production, you would never embed every chunk on every query. You pre-compute and store chunk embeddings, then use a vector database for retrieval. The code above illustrates the logical flow.

## Chunking Strategies

Chunking determines what the retrieval system can find. If a fact is split across two chunks and neither chunk contains enough context to be useful alone, retrieval will fail silently — the system returns results, but none of them contain the complete answer.

**Fixed-size chunking.** Split text every N tokens with some overlap. Simple and predictable. Works well for uniform documents. Breaks badly when a logical section spans a split boundary.

**Sentence-level chunking.** Use sentence boundaries as split points, grouping sentences until you reach a target size. Preserves grammatical completeness. Can produce uneven chunk sizes.

**Paragraph-level chunking.** Split on paragraph breaks. Respects the author's original logical groupings. Paragraphs vary wildly in length, so some chunks will be too small to be useful and others too large for precise retrieval.

**Semantic chunking.** Embed sentences sequentially and split when the cosine similarity between adjacent sentences drops below a threshold. This detects topic shifts. More expensive to compute and sensitive to threshold tuning.

**Recursive chunking.** Try to split on the largest structural boundary first (sections, then paragraphs, then sentences, then tokens). Falls through to smaller boundaries only when a section exceeds the target size. This is the default in LangChain's `RecursiveCharacterTextSplitter` and works well as a starting point.

### Baseline Settings

Start with chunks of 200-1000 tokens with 10-20% overlap. Smaller chunks improve retrieval precision (each chunk is about one topic) but require retrieving more of them for complete answers. Larger chunks carry more context but dilute retrieval signal. Overlap prevents information loss at boundaries.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_text(document_text)
```

The right chunk size depends on your data. Run retrieval evals with different sizes before committing.

## Embedding Models

An embedding model converts text into a fixed-length vector. Two pieces of text that mean similar things will have vectors that are close together (high cosine similarity). Two unrelated pieces will be far apart.

This is the foundation of semantic search: you do not need exact keyword matches. A query about "employee termination policy" can match a chunk titled "Offboarding Procedures" because the embedding model learned that these concepts are related.

### Choosing a Model

| Model | Dimensions | Strengths | Considerations |
|---|---|---|---|
| **OpenAI text-embedding-3-small** | 1536 | Good quality, low cost, widely used | Proprietary; data sent to OpenAI API |
| **OpenAI text-embedding-3-large** | 3072 | Higher quality, supports dimension reduction | Higher cost per token |
| **Cohere embed-v3** | 1024 | Strong multilingual support; separate query/document modes | Proprietary |
| **BAAI/bge-large-en-v1.5** | 1024 | Open-source, self-hostable, no API dependency | Requires GPU for reasonable throughput |
| **sentence-transformers/all-MiniLM-L6-v2** | 384 | Lightweight, fast, good for prototyping | Lower quality on complex queries |

Key decision: if you cannot send data to an external API (regulatory, privacy), you need an open-source model you can host. Otherwise, start with `text-embedding-3-small` for its cost-to-quality ratio and switch only if retrieval evals show you need more.

You must use the same embedding model for indexing and querying. Switching models means re-embedding your entire corpus.

## Vector Databases

Once you have embeddings, you need somewhere to store and search them efficiently.

| Database | Type | Best For | Notes |
|---|---|---|---|
| **Pinecone** | Managed cloud | Teams that want zero ops; production workloads | Serverless tier available; scales automatically |
| **Weaviate** | Managed or self-hosted | Hybrid search (vector + keyword) built-in | Supports multiple vectorizers; GraphQL API |
| **Chroma** | Self-hosted / embedded | Prototyping; small-to-medium datasets | Runs in-process; simple Python API |
| **pgvector** | PostgreSQL extension | Teams already on Postgres; moderate scale | Keeps vectors alongside relational data; familiar SQL |
| **Qdrant** | Managed or self-hosted | High-performance filtering + vector search | Rust-based; strong payload filtering |
| **FAISS** | Library (in-memory) | Batch offline experiments; research | No persistence layer; you manage storage |

**Managed vs. self-hosted:** Managed services (Pinecone, Weaviate Cloud) eliminate operational burden — backups, scaling, index optimization. Self-hosted (Chroma, pgvector, Qdrant on your infra) gives you data locality and avoids vendor lock-in. If your team does not have dedicated infrastructure engineers, start managed.

```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("knowledge_base")

# Add pre-computed chunks and embeddings
collection.add(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=chunk_embeddings  # list of vectors
)

# Query
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)
retrieved_chunks = results["documents"][0]
```

## Retrieval Strategies

### Semantic Search

Embed the query, find nearest vectors. This is the default RAG retrieval method. It excels when the user's phrasing differs from the source document's phrasing but the meaning is the same.

**Where it fails:** Exact terms matter (product SKUs, error codes, proper names). A semantic search for "error code XJ-4012" may not rank the chunk containing that exact code highest if other chunks discuss errors in general.

### Keyword Search (BM25)

Classic term-frequency search. Ranks documents by how well their words match the query words, weighted by rarity. Strong for exact matches and specific terminology.

**Where it fails:** Synonyms and paraphrasing. A query about "cancellation policy" will miss a chunk that only uses the phrase "how to end your subscription."

### Hybrid Search

Combine semantic and keyword scores. The standard approach is Reciprocal Rank Fusion (RRF): run both searches independently, then merge the ranked lists using `1 / (k + rank)` where `k` is a constant (typically 60).

```python
def reciprocal_rank_fusion(
    semantic_results: list[str],
    keyword_results: list[str],
    k: int = 60
) -> list[str]:
    scores: dict[str, float] = {}
    for rank, doc_id in enumerate(semantic_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    for rank, doc_id in enumerate(keyword_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)
```

Hybrid search is usually the best default. It covers both semantic similarity and exact-match cases. Most production RAG systems end up here.

## Reranking

Initial retrieval (whether semantic, keyword, or hybrid) uses fast but approximate scoring. A reranker takes the top N candidates and re-scores them with a more powerful model, producing a more accurate final ranking.

**Bi-encoders** (used in initial retrieval) encode the query and document independently. Fast, but they never see the query and document together.

**Cross-encoders** (used in reranking) take the query and document as a single input and output a relevance score. Much more accurate, but too slow to run over the entire corpus. You run them over 20-50 candidates, not millions.

```python
import cohere

co = cohere.Client("your-api-key")

results = co.rerank(
    model="rerank-english-v3.0",
    query="What is our refund policy for enterprise contracts?",
    documents=retrieved_chunks,
    top_n=5
)

reranked_chunks = [r.document.text for r in results.results]
```

**When to add reranking:** When your retrieval evals show that relevant documents appear in the top 20 but not the top 5. Reranking is most valuable when initial retrieval has decent recall but poor precision at the top of the list.

## Advanced Patterns

### Query Expansion

The user's query is often too short or ambiguous for effective retrieval. Query expansion rewrites the query into multiple variants before searching.

```python
def expand_query(query: str) -> list[str]:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": (
                f"Generate 3 alternative phrasings of this search query. "
                f"Return each on a new line, no numbering.\n\nQuery: {query}"
            )
        }]
    )
    variants = response.choices[0].message.content.strip().split("\n")
    return [query] + [v.strip() for v in variants if v.strip()]
```

Run retrieval for each variant and merge results with RRF. This helps when user queries are terse ("refund policy") and the relevant chunks use different terminology.

### HyDE (Hypothetical Document Embeddings)

Instead of embedding the query directly, ask the LLM to generate a hypothetical answer, then embed that answer and use it for retrieval. The intuition: a hypothetical answer looks more like the stored documents than a short question does, so it lands closer in embedding space.

```python
def hyde_retrieve(query: str, collection, k: int = 5):
    # Generate hypothetical answer
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"Answer this question in a detailed paragraph: {query}"
        }]
    )
    hypothetical = response.choices[0].message.content

    # Embed the hypothetical answer, not the original query
    hyp_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=hypothetical
    ).data[0].embedding

    results = collection.query(query_embeddings=[hyp_embedding], n_results=k)
    return results["documents"][0]
```

HyDE works well for complex or abstract questions. It adds one LLM call of latency and can backfire if the hypothetical answer is off-topic, pulling retrieval in the wrong direction.

### Multi-Query RAG

Break a complex question into sub-questions, retrieve for each, then synthesize. Useful when a single question requires information from multiple unrelated sections of your knowledge base.

### Parent Document Retrieval

Index small chunks for precise retrieval, but when a chunk matches, return its parent (the larger section it came from) to the LLM. This gives you the precision of small chunks with the context completeness of large ones. Store a mapping from chunk ID to parent document ID and fetch the parent at generation time.

## RAG Evaluation

You need to evaluate two things independently: whether retrieval found the right documents, and whether the model generated a correct answer from those documents.

### Retrieval Quality Metrics

**Recall@K** — Of all relevant documents in the corpus, what fraction appeared in the top K results? High recall means you are not missing relevant information.

**MRR (Mean Reciprocal Rank)** — Average of `1 / rank_of_first_relevant_result` across queries. Measures how quickly a relevant result appears.

**NDCG (Normalized Discounted Cumulative Gain)** — Accounts for the position of all relevant documents, not just the first. Relevant documents ranked higher contribute more to the score.

### Generation Quality Metrics

**Faithfulness** — Does the generated answer only contain claims supported by the retrieved context? Unfaithful answers indicate the model is hallucinating despite having context.

**Relevance** — Does the answer actually address the user's question?

**Completeness** — Does the answer cover all aspects of the question that are addressed in the retrieved context?

### Building Eval Sets

Build a set of 50-200 question-answer-source triples manually. For each question, record which chunks contain the answer and what a correct answer looks like. This is tedious but essential. Without it, you are optimizing blind.

```python
eval_set = [
    {
        "question": "What is the SLA for enterprise support response time?",
        "relevant_chunk_ids": ["chunk_142", "chunk_143"],
        "expected_answer_contains": ["4 hours", "business hours", "Severity 1"]
    },
    # ... 50-200 more
]

def evaluate_retrieval(eval_set, retrieve_fn, k=5):
    recall_scores = []
    mrr_scores = []
    for item in eval_set:
        retrieved_ids = retrieve_fn(item["question"], k=k)
        relevant = set(item["relevant_chunk_ids"])
        hits = relevant.intersection(set(retrieved_ids))
        recall_scores.append(len(hits) / len(relevant))
        for rank, rid in enumerate(retrieved_ids, 1):
            if rid in relevant:
                mrr_scores.append(1 / rank)
                break
        else:
            mrr_scores.append(0)
    return {
        "recall@k": sum(recall_scores) / len(recall_scores),
        "mrr": sum(mrr_scores) / len(mrr_scores)
    }
```

Run this evaluation every time you change chunking, embedding models, retrieval strategy, or reranking. Without a quantitative baseline, you cannot tell if a change helped.

## Common Failure Modes

**Wrong chunks retrieved.** The retrieval returns documents that are topically adjacent but do not contain the actual answer. Usually caused by poor chunking (relevant information split across chunks) or an embedding model that does not distinguish fine-grained differences. Fix: smaller chunks, hybrid search, reranking.

**Model ignores context.** The retrieved chunks contain the answer, but the model generates from its parametric knowledge instead. This happens more with confident-sounding queries where the model "thinks" it knows the answer. Fix: stronger system prompts ("Use only the provided context"), lower temperature, or models specifically tuned for grounded generation.

**Stale data.** The vector store contains outdated documents. A user asks about current pricing and gets last year's numbers. Fix: implement document versioning, TTLs on indexed data, and re-indexing pipelines that run on your data update schedule.

**Chunking artifacts.** Tables, code blocks, or lists lose their structure when chunked naively. A table row without its header is meaningless. Fix: use format-aware parsers that preserve table structure, or include the table header with every row chunk.

**Over-retrieval.** Retrieving too many chunks floods the context with marginally relevant information, confusing the model. Fix: reduce K, add reranking, or use a summarization step for retrieved chunks.

### Systematic Debugging

When a RAG system produces a bad answer, diagnose in order:

1. **Check retrieval.** Did the correct chunks appear in the retrieved set? If not, the problem is retrieval (chunking, embedding, search strategy).
2. **Check ranking.** Were the correct chunks in the retrieved set but ranked low? If so, add reranking or tune hybrid search weights.
3. **Check generation.** Were the correct chunks ranked high but the model still produced a wrong answer? If so, the problem is the prompt, model behavior, or context window management.

This order matters. Most RAG failures are retrieval failures, not generation failures. Fix retrieval first.

## Cost and Latency

RAG adds overhead that does not exist in a plain LLM call. Understanding the budget helps you make informed trade-offs.

**Embedding the query.** Typically 0.5-2ms API time for a short query. Negligible cost (fractions of a cent).

**Vector search.** Managed databases return results in 10-50ms for millions of vectors. Self-hosted performance depends on your index configuration and hardware.

**Reranking.** A cross-encoder rerank call over 25 documents adds 200-500ms and costs roughly $0.001-0.003 per query with Cohere's API. Significant latency but often worth the quality gain.

**LLM generation with context.** The main cost driver. Retrieving K=10 chunks of 500 tokens each adds 5,000 input tokens to every call. At GPT-4o pricing, that is roughly $0.0125 per query just for the context tokens. Reducing K from 10 to 5 halves that cost.

| Lever | Effect on Quality | Effect on Latency | Effect on Cost |
|---|---|---|---|
| Increase K (more chunks) | More recall, but can dilute precision | +10-50ms retrieval, more generation tokens | Higher LLM cost |
| Add reranking | Better precision at top of list | +200-500ms | +$0.001-0.003/query |
| Use larger embedding model | Better retrieval accuracy | Negligible change | Slightly higher embedding cost |
| Use HyDE | Better for complex queries | +500-1500ms (extra LLM call) | +1 LLM call per query |
| Reduce chunk size | Better precision, worse context | Negligible | May need higher K to compensate |

The most common production configuration balances these factors: hybrid search with K=5-10, a reranker over the top 20-25 candidates narrowed to 5, and chunks in the 300-500 token range. This gives you sub-second total retrieval time, manageable cost, and good answer quality. Adjust from there based on your evaluation metrics and latency budget.

---

RAG is not a single algorithm. It is a pipeline, and every stage of that pipeline is a place where quality can degrade or improve. The teams that build effective RAG systems are the ones that instrument every stage, build evaluation sets early, and treat data quality with the same rigor they apply to model selection. Start with the simplest version that works, measure it, and add complexity only where the measurements tell you to.
