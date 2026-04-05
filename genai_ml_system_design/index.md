---
layout: default
title: GenAI System Design
nav_order: 7
has_children: true
permalink: /genai_ml_system_design/
---

# GenAI System Design
{: .fs-9 }

Production system design for Generative AI — LLM chatbots, RAG pipelines, code assistants, content moderation, training platforms, and multi-modal search.
{: .fs-6 .fw-300 }

---

## Why GenAI System Design Is a Separate Category

Traditional ML system design focuses on **classification, ranking, and retrieval**. GenAI system design introduces fundamentally different challenges:

| Traditional ML | GenAI/LLM Systems |
|----------------|-------------------|
| Fixed output schema (class, score) | Open-ended text/image/code generation |
| Millisecond inference | Seconds-long autoregressive decoding |
| MBs per model | GBs–TBs per model (GPU clusters) |
| Feature engineering dominant | Prompt engineering + retrieval dominant |
| Train once, serve forever | Continuous alignment, RLHF, safety tuning |
| Deterministic evaluation | Subjective, multi-dimensional evaluation |

{: .warning }
> Google, Meta, and Anthropic interviews increasingly ask GenAI-specific designs. Saying "just call the OpenAI API" will not pass. You need to demonstrate understanding of **inference optimization, safety, grounding, cost control, and evaluation at scale**.

---

## Available Designs

### [LLM-Powered Chatbot]({{ site.baseurl }}/genai_ml_system_design/llm_chatbot)
{: .d-inline-block }

Conversational AI
{: .label .label-purple }

Design a production chatbot like Gemini/ChatGPT — multi-turn conversation, streaming, safety, and serving at Google scale.

**Key concepts:** KV-cache, PagedAttention, speculative decoding, RLHF, guardrails, conversation memory, streaming SSE, GPU auto-scaling

**Difficulty:** ⭐⭐⭐⭐⭐ Very Hard

---

### [Enterprise RAG System]({{ site.baseurl }}/genai_ml_system_design/enterprise_rag)
{: .d-inline-block }

Knowledge Grounding
{: .label .label-green }

Design a retrieval-augmented generation system for enterprise knowledge bases with citations, access control, and hallucination mitigation.

**Key concepts:** Chunking strategies, hybrid retrieval (BM25 + dense), re-ranking, citation grounding, ACL-aware retrieval, query routing, evaluation (faithfulness, relevance)

**Difficulty:** ⭐⭐⭐⭐ Hard

---

### [AI Code Assistant]({{ site.baseurl }}/genai_ml_system_design/ai_code_assistant)
{: .d-inline-block }

Developer Tools
{: .label .label-yellow }

Design an AI code completion and chat system like Gemini Code Assist / GitHub Copilot — IDE integration, repository-aware context, and low-latency suggestions.

**Key concepts:** Fill-in-the-middle (FIM), tree-sitter AST, repository-level context, speculative decoding, streaming, telemetry-driven evaluation, code safety

**Difficulty:** ⭐⭐⭐⭐⭐ Very Hard

---

### [LLM Content Moderation]({{ site.baseurl }}/genai_ml_system_design/content_moderation)
{: .d-inline-block }

Trust & Safety
{: .label .label-red }

Design a content moderation system using LLMs for text, image, and video — policy enforcement, appeals, and human-in-the-loop at scale.

**Key concepts:** Multi-modal classifiers, policy-as-code, cascade architecture (fast→accurate), adversarial robustness, human review queues, appeals, regulatory compliance

**Difficulty:** ⭐⭐⭐⭐ Hard

---

### [ML Training Platform]({{ site.baseurl }}/genai_ml_system_design/ml_training_platform)
{: .d-inline-block }

ML Infrastructure
{: .label .label-red }

Design an ML training platform like Vertex AI / SageMaker — job scheduling, distributed training, experiment tracking, and GPU cluster management.

**Key concepts:** GPU scheduling (gang scheduling), checkpointing, elastic training, experiment tracking, hyperparameter tuning, multi-tenancy, cost attribution

**Difficulty:** ⭐⭐⭐⭐⭐ Very Hard

---

### [Multi-Modal Search]({{ site.baseurl }}/genai_ml_system_design/multimodal_search)
{: .d-inline-block }

Multi-Modal AI
{: .label .label-blue }

Design a multi-modal search system like Google Lens — search across text, images, video, and audio with a unified embedding space.

**Key concepts:** CLIP/SigLIP embeddings, unified vector index, cross-modal retrieval, OCR integration, late-interaction models, query understanding, multi-modal fusion

**Difficulty:** ⭐⭐⭐⭐ Hard

---

## Quick Reference: System Comparison

| System | Latency Target | Key Challenge | Primary Metric |
|--------|---------------|---------------|----------------|
| **LLM Chatbot** | TTFT < 500ms | GPU cost, safety | User satisfaction, Helpfulness |
| **Enterprise RAG** | < 3s end-to-end | Hallucination, ACLs | Faithfulness, Recall@K |
| **AI Code Assistant** | < 200ms (completion) | Context window, accuracy | Acceptance rate, Keystroke savings |
| **Content Moderation** | < 500ms | Adversarial attacks, fairness | Precision, Recall, False positive rate |
| **ML Training Platform** | N/A (throughput) | GPU utilization, fault tolerance | MFU, Job completion rate |
| **Multi-Modal Search** | < 300ms | Cross-modal alignment | NDCG@K, Recall@K |

---

## GenAI System Design Framework

Use this adapted framework in your interviews:

### 1. Problem Setup (5 min)
- What modality? (text, image, video, code, multi-modal)
- What is the generation/retrieval task?
- Latency, throughput, cost constraints?
- Safety and compliance requirements?
- Success metrics (both ML and business)

### 2. Model & Data Strategy (10 min)
- Base model selection (size, architecture, open vs proprietary)
- Fine-tuning vs prompt engineering vs RAG
- Training data: collection, labeling, quality
- Alignment strategy (RLHF, DPO, Constitutional AI)
- Evaluation methodology

### 3. Serving Infrastructure (10 min)
- GPU cluster sizing and instance types
- Inference optimization (batching, quantization, KV-cache)
- Streaming architecture (SSE, WebSocket)
- Auto-scaling strategy
- Cost optimization (spot instances, model distillation)

### 4. Safety & Quality (5 min)
- Input/output guardrails
- Hallucination mitigation
- PII detection and filtering
- Adversarial robustness
- Human-in-the-loop workflows

### 5. Monitoring & Iteration (5 min)
- Online evaluation metrics
- A/B testing framework
- Feedback collection
- Retraining and model update pipeline
- Cost and latency dashboards

---

## Interview Tips for GenAI Design

{: .tip }
> **What differentiates a pass from a strong-hire at Google:**
> - **Pass:** Correct high-level architecture with RAG or fine-tuning
> - **Strong hire:** Deep discussion of inference optimization (PagedAttention, continuous batching), safety layering, evaluation methodology, and cost modeling

**Common mistakes to avoid:**
1. Treating LLM inference like traditional web service scaling (it's GPU-bound, not CPU-bound)
2. Ignoring safety and guardrails entirely
3. Not discussing how to evaluate generation quality
4. Assuming unlimited context windows solve all problems
5. Forgetting about cost — GPU inference is 100-1000x more expensive than traditional APIs
6. Not separating retrieval from generation in RAG systems

---

## How These Connect

```
GenAI/ML Fundamentals              GenAI System Design Questions
─────────────────────              ─────────────────────────────
Model Serving          ──────►     LLM Chatbot, Code Assistant
LLM Systems            ──────►     Enterprise RAG, Chatbot
Feature Stores         ──────►     Content Moderation
Distributed Training   ──────►     ML Training Platform
Data Pipelines         ──────►     All GenAI Systems

ML System Design                   GenAI System Design Questions
────────────────                   ─────────────────────────────
Image Search           ──────►     Multi-Modal Search
Search Ranking         ──────►     Enterprise RAG (retrieval)
Fraud Detection        ──────►     Content Moderation (cascade)
```

{: .note }
> Master the [GenAI/ML Fundamentals]({{ site.baseurl }}/genai_ml_basics/) building blocks first, then apply them here in full end-to-end system designs.
