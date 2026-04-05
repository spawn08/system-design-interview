---
layout: default
title: GenAI System Design
nav_order: 7
has_children: true
permalink: /genai_ml_system_design/
---

# GenAI System Design
{: .fs-9 }

Production system design for Generative AI — 9 designs covering chatbots, RAG, agents, code assistants, image generation, and more. Each includes a hypothetical Google-style interview transcript.
{: .fs-6 .fw-300 }

---

## Why GenAI System Design Is a Separate Category

Traditional ML system design focuses on **classification, ranking, and retrieval**. GenAI introduces fundamentally different challenges:

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

## Recommended Study Order

{: .tip }
> Follow this progression. Each design builds on concepts from earlier ones. All designs include a full hypothetical interview transcript.

### Phase 1: Core GenAI Patterns

| Order | Design | New Concepts Introduced | Why First |
|-------|--------|------------------------|-----------|
| 1 | [LLM-Powered Chatbot]({{ site.baseurl }}/genai_ml_system_design/llm_chatbot) | KV-cache, PagedAttention, streaming, safety | Foundation for all LLM serving |
| 2 | [Enterprise RAG System]({{ site.baseurl }}/genai_ml_system_design/enterprise_rag) | Chunking, hybrid retrieval, citations, ACLs | #1 production LLM pattern |
| 3 | [LLM Gateway]({{ site.baseurl }}/genai_ml_system_design/llm_gateway) | Multi-model routing, semantic caching, cost control | Infra layer used by all LLM apps |

### Phase 2: Specialized Applications

| Order | Design | New Concepts Introduced | Builds On |
|-------|--------|------------------------|-----------|
| 4 | [AI Code Assistant]({{ site.baseurl }}/genai_ml_system_design/ai_code_assistant) | FIM, repo context, speculative decoding | Chatbot (serving) + RAG (retrieval) |
| 5 | [AI Agent System]({{ site.baseurl }}/genai_ml_system_design/ai_agent_system) | ReAct, tool use, planning, memory, multi-agent | Chatbot + RAG + Gateway |
| 6 | [LLM Content Moderation]({{ site.baseurl }}/genai_ml_system_design/content_moderation) | Cascade, adversarial robustness, human-in-the-loop | Chatbot (safety pipeline) |

### Phase 3: Advanced GenAI

| Order | Design | New Concepts Introduced | Builds On |
|-------|--------|------------------------|-----------|
| 7 | [Text-to-Image Generation]({{ site.baseurl }}/genai_ml_system_design/text_to_image) | Diffusion models, CFG, latent space, safety | Chatbot (GPU serving) + Moderation |
| 8 | [Multi-Modal Search]({{ site.baseurl }}/genai_ml_system_design/multimodal_search) | CLIP/SigLIP, cross-modal retrieval, video | RAG (retrieval) + Image generation |
| 9 | [ML Training Platform]({{ site.baseurl }}/genai_ml_system_design/ml_training_platform) | Gang scheduling, checkpointing, GPU clusters | All (training infrastructure for all models) |

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

### [LLM Gateway]({{ site.baseurl }}/genai_ml_system_design/llm_gateway)
{: .d-inline-block }

AI Infrastructure
{: .label .label-yellow }

NEW
{: .label .label-blue }

Design an LLM gateway/proxy that handles routing, fallback, semantic caching, rate limiting, cost tracking, and observability across multiple LLM providers.

**Key concepts:** Semantic caching, intelligent model routing, token-based rate limiting, circuit breaker per provider, PII scrubbing, cost attribution, unified API normalization

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

### [AI Agent System]({{ site.baseurl }}/genai_ml_system_design/ai_agent_system)
{: .d-inline-block }

Autonomous AI
{: .label .label-purple }

NEW
{: .label .label-blue }

Design an autonomous AI agent system that can plan, use tools, maintain memory, and execute multi-step tasks — like Google's AI agents or Anthropic's computer use.

**Key concepts:** ReAct pattern, tool calling, task decomposition, working + semantic memory, multi-agent orchestration, sandboxed execution, human-in-the-loop

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

### [Text-to-Image Generation]({{ site.baseurl }}/genai_ml_system_design/text_to_image)
{: .d-inline-block }

Generative Media
{: .label .label-purple }

NEW
{: .label .label-blue }

Design a text-to-image generation system like Imagen / DALL-E 3 / Midjourney — from text prompts to high-quality images with safety and copyright controls.

**Key concepts:** Diffusion models, latent diffusion, classifier-free guidance, CLIP/T5 conditioning, ControlNet, LoRA, super-resolution cascades, content provenance (C2PA)

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

### [ML Training Platform]({{ site.baseurl }}/genai_ml_system_design/ml_training_platform)
{: .d-inline-block }

ML Infrastructure
{: .label .label-red }

Design an ML training platform like Vertex AI / SageMaker — job scheduling, distributed training, experiment tracking, and GPU cluster management.

**Key concepts:** GPU scheduling (gang scheduling), checkpointing, elastic training, experiment tracking, hyperparameter tuning, multi-tenancy, cost attribution

**Difficulty:** ⭐⭐⭐⭐⭐ Very Hard

---

### [Vector Database]({{ site.baseurl }}/genai_ml_system_design/vector_database)
{: .d-inline-block }

AI Infrastructure
{: .label .label-yellow }

NEW
{: .label .label-blue }

Design a vector database purpose-built for AI applications — HNSW, IVF-PQ indexing, hybrid search with metadata filtering, and billion-scale ANN at sub-10ms latency.

**Key concepts:** HNSW graph traversal, IVF-PQ quantization, distance metrics (cosine, L2, IP), hybrid pre/post-filtering, vector-space-aware sharding, memory-mapped indexes, tiered storage

**Difficulty:** ⭐⭐⭐⭐⭐ Very Hard

---

## Quick Reference: System Comparison

| System | Latency Target | Key Challenge | Primary Metric |
|--------|---------------|---------------|----------------|
| **LLM Chatbot** | TTFT < 500ms | GPU cost, safety | User satisfaction, Helpfulness |
| **Enterprise RAG** | < 3s end-to-end | Hallucination, ACLs | Faithfulness, Recall@K |
| **LLM Gateway** | < 20ms overhead | Multi-provider resilience | Availability, cost savings |
| **AI Code Assistant** | < 200ms (completion) | Context window, accuracy | Acceptance rate, Keystroke savings |
| **AI Agent System** | < 5min per task | Planning, tool reliability | Task completion rate |
| **Content Moderation** | < 500ms | Adversarial attacks, fairness | Precision, Recall, FPR |
| **Text-to-Image** | < 10s per image | Safety, quality | FID, CLIP score, Human preference |
| **Vector Database** | < 10ms P99 | Billion-scale ANN, recall | Recall@K, QPS |
| **Multi-Modal Search** | < 300ms | Cross-modal alignment | NDCG@K, Recall@K |
| **ML Training Platform** | N/A (throughput) | GPU utilization, fault tolerance | MFU, Job completion rate |

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
7. Ignoring multi-provider resilience (single provider = single point of failure)
8. Treating agents as simple chatbots (planning, memory, tool use are distinct subsystems)

---

## How These Connect

```
GenAI/ML Fundamentals              GenAI System Design Questions
─────────────────────              ─────────────────────────────
Model Serving          ──────►     LLM Chatbot, Code Assistant, LLM Gateway
LLM Systems            ──────►     Enterprise RAG, Chatbot, AI Agents
Feature Stores         ──────►     Content Moderation
Distributed Training   ──────►     ML Training Platform, Text-to-Image
Data Pipelines         ──────►     All GenAI Systems

ML System Design                   GenAI System Design Questions
────────────────                   ─────────────────────────────
Image Search           ──────►     Multi-Modal Search
Search Ranking         ──────►     Enterprise RAG (retrieval)
Fraud Detection        ──────►     Content Moderation (cascade)
Ads Ranking            ──────►     LLM Gateway (cost optimization)
```

{: .note }
> Master the [GenAI/ML Fundamentals]({{ site.baseurl }}/genai_ml_basics/) building blocks first, then apply them here in full end-to-end system designs.
