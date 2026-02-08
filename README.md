Intelligent RAG – Handling Conflicting Policy Documents
Overview

This project implements an “Intelligent RAG” system for an internal HR Q&A bot at NebulaCorp.
The knowledge base contains old, active, and draft versions of policy documents.
A naive RAG system retrieves documents purely by semantic similarity, which can surface outdated or draft policies. This system adds business-rule-aware retrieval to ensure answers are based on valid policies while still acknowledging drafts when the question is about future changes.

Part 1 – Coding Challenge: The Intelligent RAG
Problem

Standard RAG retrieves documents based on similarity, not validity.
In this dataset:

policy_v1_2020.txt → old (obsolete)

policy_v2_2023.txt → active (current)

policy_future_DRAFT_v3.txt → draft (not yet effective)

Without extra logic, the system may answer using old or draft documents.

Solution Design

Document Ingestion with Metadata
Each document is loaded with metadata:

status: old, active, or draft

version and year

source (filename for citation)

Vector Storage

Documents are embedded using sentence-transformers (all-MiniLM-L6-v2).

Stored in a local Chroma vector database.

This enables semantic retrieval by similarity.

Two-Stage Retrieval Logic

Stage 1: Retrieve top-k documents by semantic similarity.

Stage 2: Apply business rules:

If the question is about current policy → use only status == "active".

If the question is about plans / future / changes → allow status in ["active", "draft"].

Never use old documents for answers.

Answer Generation with Citation

The filtered documents are passed as context to Gemini.

The model is instructed:

To answer only from the given context.

To clearly state when a policy is a draft and not yet effective.

The system prints the answer and the source file(s) used.

Example Behaviors (As Required)

Q: “How many days can I work remotely?”
→ Uses only policy_v2_2023.txt → Answers: 3 days.

Q: “What is the meal allowance?”
→ Uses only policy_v2_2023.txt → Answers: $50 per day.

Q: “Are there any plans to change the vacation policy?”
→ Uses policy_v2_2023.txt + policy_future_DRAFT_v3.txt →
Mentions unlimited time off proposal and clearly states it is not yet effective.

This ensures the system prioritizes valid policy over mere semantic similarity.

Part 2 – System Design: “The Hospital Architect”
1) Clarification Questions

What exactly does “<100ms latency” measure: end-to-end or only retrieval time?

What accuracy expectations differ between the 70% keyword queries and 30% complex reasoning queries?

What is considered PHI in practice, and where is it allowed to exist transiently (memory, logs, etc.)?

2) Three Possible Architectures

A) Dual-Path Search (Speed-Optimized)

70% queries → fast lexical search (BM25 / Elasticsearch).

30% queries → vector search + lightweight re-ranking.

LLM used only for synthesis when needed.

Optimizes: Speed and Cost.

Trade-off: Limited deep reasoning.

B) Hybrid Retrieval + Small Re-ranker (Balanced) – Chosen

Parallel lexical + vector retrieval.

Merge and re-rank using a small local model.

Call larger LLM only when necessary.

Optimizes: Balance of speed, cost, and accuracy.

Trade-off: More system complexity.

C) Full RAG with Large LLM (Accuracy-Optimized)

Every query → vector retrieval + large LLM synthesis.

Optimizes: Accuracy.

Trade-off: Too slow and too expensive for <100ms and $5k/month budget.

Chosen: Architecture B, because it balances performance, cost, and quality under the given constraints.

3) Privacy Engineering: Handling PHI Without Storing It in Vector DB

Solution 1: Query-Time PHI Redaction

Strip or abstract PHI before retrieval.

Use sanitized query for search; keep PHI only in memory.

Failure mode: Over-redaction may remove important clinical context and reduce relevance.

Solution 2: Two-Stage Retrieval

Stage 1: Retrieve using only non-PHI medical concepts.

Stage 2: Re-rank/filter using PHI context in memory only.

Failure mode: If Stage 1 misses key documents, Stage 2 cannot recover.

4) RAG Strategies to Avoid for <100ms Latency

Multi-hop iterative retrieval with multiple LLM calls.

Large cross-encoder re-ranking over big candidate sets.

Always-on LLM synthesis for every query.

Cold-start embedding generation without caching.

These approaches introduce unpredictable and high tail latency.

5) 5-Step Latency Debugging Checklist

Measure timing for each stage: parsing, retrieval, re-ranking, LLM, response.

Inspect vector DB and lexical index latency (p50, p95, p99).

Profile re-ranking and post-processing for CPU or memory bottlenecks.

Measure external calls (LLM, services) and add timeouts/fallbacks.

Load test to identify tail-latency causes (cold caches, GC pauses, saturation).

How to Run:
pip install -r requirements.txt
python rag.py

Part 2 – System Design: “The Hospital Architect”
1) Clarification Questions

What exactly does “<100ms latency” measure: end-to-end (API → response) or only the retrieval stage? This affects whether any LLM call can be on the critical path.

What accuracy or recall is acceptable for the 70% keyword queries versus the 30% complex reasoning queries?

What is considered PHI in practice, and where is it allowed to exist transiently (e.g., in memory, logs, or caches)?

2) Three Possible Architectures

A) Dual-Path Search (Speed-Optimized)

70% simple queries go to a fast lexical index (e.g., BM25 / Elasticsearch).

30% complex queries go to vector search with lightweight re-ranking.

LLM is used only when synthesis is needed.

Optimizes: Speed and Cost.

Trade-off: Limited deep reasoning quality.

B) Hybrid Retrieval + Small Re-ranker (Balanced) – Chosen

Run lexical and vector retrieval in parallel.

Merge results and re-rank using a small local model.

Call a larger LLM only when necessary.

Optimizes: Balance of speed, cost, and accuracy.

Trade-off: More system complexity.

C) Full RAG with Large LLM (Accuracy-Optimized)

Every query goes through vector retrieval + large LLM synthesis.

Optimizes: Accuracy.

Trade-off: Too slow and too expensive to meet <100ms latency and the $5,000/month budget.

Chosen Architecture: B, because it balances performance, cost, and answer quality under the given constraints.

3) Privacy Engineering: Handling PHI Without Storing It in the Vector DB

Solution 1: Query-Time PHI Redaction

Detect and remove or abstract PHI from the query before retrieval.

Use the sanitized query for search; keep the original PHI only in memory for response formatting.

Failure mode: Over-redaction may remove important clinical context and reduce retrieval quality.

Solution 2: Two-Stage Retrieval

Stage 1: Retrieve documents using only non-PHI medical concepts (diseases, treatments, symptoms).

Stage 2: Re-rank or filter results using the PHI context in memory only, without storing it.

Failure mode: If Stage 1 fails to retrieve relevant documents, Stage 2 cannot recover.

4) RAG Strategies to Avoid for <100ms Latency

Multi-hop or iterative retrieval pipelines that require multiple LLM calls.

Large cross-encoder re-ranking over very large candidate sets.

Always-on LLM synthesis for every query.

Cold-start embedding generation without caching.

These approaches introduce high and unpredictable tail latency and make it difficult to meet the <100ms p95 requirement.

5) 5-Step Latency Debugging Checklist

Measure and log latency for each stage: request parsing, retrieval, re-ranking, LLM calls, and response serialization.

Inspect vector DB and lexical index performance (p50, p95, p99 latencies).

Profile re-ranking and post-processing to identify CPU or memory bottlenecks.

Measure external calls (LLM or other services) and add timeouts and fallbacks.

Load test the system to identify tail-latency causes such as cold caches, garbage collection pauses, or resource saturation.