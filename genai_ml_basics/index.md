---
layout: default
title: GenAI/ML Fundamentals
nav_order: 5
has_children: true
permalink: /genai_ml_basics/
---

# GenAI/ML Fundamentals
{: .fs-9 }

Core infrastructure topics for Generative AI and Machine Learning — the building blocks that appear in every ML system design interview.
{: .fs-6 .fw-300 }

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

{: .note }
> These topics are particularly important for interviews at companies building AI-native products (OpenAI, Anthropic, Google DeepMind, Meta AI) and for ML Platform / MLOps roles at any company.

---

## What's Covered

### [Model Serving]({{ site.baseurl }}/genai_ml_basics/model_serving)
{: .d-inline-block }

Infrastructure
{: .label .label-red }

Design production model serving — REST/batch inference, versioning, A/B testing, drift detection, and GPU auto-scaling.

**Key concepts:** TorchServe, Triton, dynamic batching, canary deployments, ONNX quantization, model registry

**Difficulty:** ⭐⭐⭐ Medium-Hard

---

### [Feature Stores]({{ site.baseurl }}/genai_ml_basics/feature_stores)
{: .d-inline-block }

Data Platform
{: .label .label-green }

Design a centralized feature management platform for training and serving consistency.

**Key concepts:** Train-serve skew, point-in-time joins, online/offline stores, Feast, feature versioning, stream materialization

**Difficulty:** ⭐⭐⭐ Medium-Hard

---

### [Data Pipelines for ML]({{ site.baseurl }}/genai_ml_basics/data_pipelines)
{: .d-inline-block }

Data Engineering
{: .label .label-yellow }

Design end-to-end data pipelines for ML training — ingestion, transformation, validation, and orchestration.

**Key concepts:** Medallion architecture, Airflow DAGs, Kubeflow, data validation, dataset versioning, pipeline monitoring

**Difficulty:** ⭐⭐⭐ Medium-Hard

---

### [Large Language Models]({{ site.baseurl }}/genai_ml_basics/llm_systems)
{: .d-inline-block }

GenAI
{: .label .label-purple }

Design production LLM systems — RAG, prompt engineering, fine-tuning, vector databases, and serving at scale.

**Key concepts:** RAG pipeline, chunking, vector DBs (Pinecone, Qdrant), LoRA fine-tuning, vLLM, guardrails, LLM evaluation

**Difficulty:** ⭐⭐⭐⭐ Hard

---

### [Distributed Training]({{ site.baseurl }}/genai_ml_basics/distributed_training)
{: .d-inline-block }

Training Infrastructure
{: .label .label-red }

Design training infrastructure that scales deep learning across hundreds of GPUs.

**Key concepts:** Data parallelism (DDP), ZeRO, tensor parallelism, pipeline parallelism, DeepSpeed, mixed precision, fault tolerance

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

---

## How These Connect to ML System Design

```
GenAI/ML Fundamentals              ML System Design Questions
─────────────────────              ─────────────────────────
Model Serving          ──────►     Image Caption Generator
Feature Stores         ──────►     Recommendation System
Data Pipelines         ──────►     Fraud Detection System
LLM Systems            ──────►     LLM Chatbot / RAG App
Distributed Training   ──────►     LLM Pre-training Platform
```

{: .tip }
> Master the fundamentals first, then apply them in the [ML System Design]({{ site.baseurl }}/ml_system_design/) section where you'll design complete end-to-end systems.
