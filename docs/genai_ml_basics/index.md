# GenAI/ML Fundamentals

Core infrastructure topics for Generative AI and Machine Learning — the building blocks that appear in every ML system design interview.

---

## Why a Separate Section?

Just as **load balancing, caching, and databases** are fundamentals for software system design, topics like **model serving, feature stores, and distributed training** are fundamentals for ML/GenAI system design. You need to master these before tackling full ML system design questions.

| Software System Fundamentals | GenAI/ML Fundamentals |
|------------------------------|----------------------|
| Load balancing | Model serving |
| Caching | Feature stores |
| Databases | Vector databases |
| Data pipelines (ETL) | Data pipelines for ML |
| Networking | LLM APIs / RAG |
| Distributed systems | Distributed training |

!!! note
    These topics are particularly important for interviews at companies building AI-native products (OpenAI, Anthropic, Google DeepMind, Meta AI) and for ML Platform / MLOps roles at any company.

---

## Recommended Study Order

| Order | Topic | Time | Why This Order |
|-------|-------|------|---------------|
| 1 | [Data Pipelines for ML](data_pipelines.md) | 2-3 hours | Data comes first — can't train without it |
| 2 | [Feature Stores](feature_stores.md) | 2-3 hours | Organize features for training and serving |
| 3 | [Model Serving](model_serving.md) | 3-4 hours | Get models into production |
| 4 | [Distributed Training](distributed_training.md) | 3-4 hours | Scale training to large models |
| 5 | [Large Language Models](llm_systems.md) | 4-5 hours | The most complex — builds on everything above |
| 6 | [LLM Evaluation](llm_evaluation.md) | 2-3 hours | Benchmarks, LLM-as-judge, RAG eval, production metrics — needed for every GenAI design |
| 7 | [RLHF & Alignment](rlhf_alignment.md) | 3-4 hours | PPO, DPO, Constitutional AI, safety alignment — the most important GenAI topic after LLMs |

---

## What's Covered

### [Model Serving](model_serving.md)

Infrastructure

Design production model serving — REST/batch inference, versioning, A/B testing, drift detection, and GPU auto-scaling.

**Key concepts:** TorchServe, Triton, dynamic batching, canary deployments, ONNX quantization, model registry

**Difficulty:** ⭐⭐⭐ Medium-Hard

---

### [Feature Stores](feature_stores.md)

Data Platform

Design a centralized feature management platform for training and serving consistency.

**Key concepts:** Train-serve skew, point-in-time joins, online/offline stores, Feast, feature versioning, stream materialization

**Difficulty:** ⭐⭐⭐ Medium-Hard

---

### [Data Pipelines for ML](data_pipelines.md)

Data Engineering

Design end-to-end data pipelines for ML training — ingestion, transformation, validation, and orchestration.

**Key concepts:** Medallion architecture, Airflow DAGs, Kubeflow, data validation, dataset versioning, pipeline monitoring

**Difficulty:** ⭐⭐⭐ Medium-Hard

---

### [Large Language Models](llm_systems.md)

GenAI

Design production LLM systems — RAG, prompt engineering, fine-tuning, vector databases, and serving at scale.

**Key concepts:** RAG pipeline, chunking, vector DBs (Pinecone, Qdrant), LoRA fine-tuning, vLLM, guardrails, LLM evaluation

**Difficulty:** ⭐⭐⭐⭐ Hard

---

### [Distributed Training](distributed_training.md)

Training Infrastructure

Design training infrastructure that scales deep learning across hundreds of GPUs.

**Key concepts:** Data parallelism (DDP), ZeRO, tensor parallelism, pipeline parallelism, DeepSpeed, mixed precision, fault tolerance

**Difficulty:** ⭐⭐⭐⭐ Hard

---

### [LLM Evaluation](llm_evaluation.md)

Evaluation & Quality

Offline and online evaluation for LLMs — automatic metrics, LLM-as-judge, benchmarks (MMLU, Arena, HumanEval), RAGAS-style RAG metrics, production A/B and guardrails.

**Key concepts:** BLEU/ROUGE/BERTScore, human agreement, Elo/Arena, golden sets, faithfulness vs relevance, benchmark contamination

**Difficulty:** ⭐⭐⭐ Medium-Hard

---

### [RLHF & Alignment](rlhf_alignment.md)

Alignment & Safety

NEW

The complete alignment pipeline — from SFT through reward modeling to PPO/DPO. Constitutional AI, safety alignment, preference data collection, and production alignment loops.

**Key concepts:** SFT, reward model (Bradley-Terry), PPO with KL penalty, DPO loss, IPO/KTO/ORPO variants, Constitutional AI, red teaming, alignment tax, online RLHF

**Difficulty:** ⭐⭐⭐⭐ Hard

---

## Quick Reference

| Topic | Focus | Key Challenge | Primary Language |
|-------|-------|---------------|-----------------|
| **Model Serving** | Inference APIs | Latency, GPU utilization | Python |
| **Feature Stores** | Feature management | Train-serve consistency | Python |
| **Data Pipelines** | Data quality | Validation, lineage | Python |
| **LLM Systems** | GenAI applications | Hallucination, cost | Python |
| **Distributed Training** | Training at scale | Communication overhead | Python |
| **LLM Evaluation** | Quality & benchmarks | Subjective quality, RAG grounding | Python |
| **RLHF & Alignment** | Model alignment | Safety vs helpfulness trade-off | Python |

---

## How These Connect to System Design Questions

```
GenAI/ML Fundamentals              ML System Design Questions
─────────────────────              ─────────────────────────
Model Serving          ──────►     Image Caption Generator, all ML systems
Feature Stores         ──────►     Recommendation System, Fraud Detection, Ads Ranking
Data Pipelines         ──────►     Fraud Detection, Recommendation System

GenAI/ML Fundamentals              GenAI System Design Questions
─────────────────────              ─────────────────────────────
LLM Systems            ──────►     LLM Chatbot, Enterprise RAG, Code Assistant, AI Agents
LLM Evaluation         ──────►     All GenAI systems (offline gates + online KPIs)
RLHF & Alignment       ──────►     LLM Chatbot (safety), Content Moderation, Agents
Model Serving          ──────►     LLM Chatbot, Code Assistant, LLM Gateway
Distributed Training   ──────►     ML Training Platform, Text-to-Image
Feature Stores         ──────►     Content Moderation, Ads Ranking
Data Pipelines         ──────►     All GenAI Systems
```

!!! tip
    Master the fundamentals first, then apply them in the [ML System Design](../ml_system_design/index.md) (10 designs) and [GenAI System Design](../genai_ml_system_design/index.md) (10 designs with interview transcripts) sections.
