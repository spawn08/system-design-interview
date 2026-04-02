---
layout: default
title: ML System Design
nav_order: 4
has_children: true
permalink: /ml_system_design/
---

# ML System Design
{: .fs-9 }

Design machine learning systems for production - from data pipelines to model serving.
{: .fs-6 .fw-300 }

---

## What Makes ML Design Different?

ML system design interviews focus on the **full ML lifecycle**, not just the model. You need to think about:

| Stage | Key Considerations |
|-------|-------------------|
| **Data** | Collection, storage, labeling, versioning |
| **Training** | Distributed training, experiment tracking |
| **Serving** | Latency, throughput, model updates |
| **Monitoring** | Drift detection, performance metrics |

{: .warning }
> A common mistake: focusing only on the model architecture. Interviewers want to see you design the entire system around it.

---

## Available Designs

### [Image Caption Generator]({{ site.baseurl }}/ml_system_design/image_caption_generator)
{: .d-inline-block }

Computer Vision
{: .label .label-purple }

Design a system that generates descriptive captions for images using deep learning.

**Key concepts:** Encoder-decoder architecture, attention mechanisms, model serving (Triton), batching, caching, GPU optimization

**Difficulty:** ⭐⭐⭐ Medium-Hard

---

### [Recommendation System]({{ site.baseurl }}/ml_system_design/recommendation_system)
{: .d-inline-block }

Personalization
{: .label .label-green }

Design a recommendation system for e-commerce or content platforms like Netflix/Amazon.

**Key concepts:** Collaborative filtering, content-based filtering, Two-Tower models, ANN retrieval (FAISS), ranking models, cold start, A/B testing

**Difficulty:** ⭐⭐⭐⭐ Hard

---

### [Real-time Fraud Detection]({{ site.baseurl }}/ml_system_design/fraud_detection)
{: .d-inline-block }

Low-latency ML
{: .label .label-yellow }

Design a system that detects fraudulent transactions in real-time (<100ms).

**Key concepts:** Feature engineering (velocity features), class imbalance, ensemble models, rules engine, decision thresholds, case management, drift detection

**Difficulty:** ⭐⭐⭐⭐ Hard

---

### [Image Search System]({{ site.baseurl }}/ml_system_design/image_search)
{: .d-inline-block }

Vector Search
{: .label .label-blue }

Design a visual search system for finding similar images or searching by text description.

**Key concepts:** CLIP embeddings, vector databases (FAISS, Pinecone), ANN indexes (IVF, HNSW), multi-modal search, indexing pipelines, re-ranking

**Difficulty:** ⭐⭐⭐ Medium-Hard

---

## GenAI/ML Infrastructure Topics

### [Model Serving]({{ site.baseurl }}/ml_system_design/model_serving)
{: .d-inline-block }

Infrastructure
{: .label .label-red }

Design production model serving — REST/batch inference, versioning, A/B testing, drift detection, and GPU auto-scaling.

**Key concepts:** TorchServe, Triton, dynamic batching, canary deployments, ONNX quantization, model registry

**Difficulty:** ⭐⭐⭐ Medium-Hard

---

### [Feature Stores]({{ site.baseurl }}/ml_system_design/feature_stores)
{: .d-inline-block }

Data Platform
{: .label .label-green }

Design a centralized feature management platform for training and serving consistency.

**Key concepts:** Train-serve skew, point-in-time joins, online/offline stores, Feast, feature versioning, stream materialization

**Difficulty:** ⭐⭐⭐ Medium-Hard

---

### [Data Pipelines for ML]({{ site.baseurl }}/ml_system_design/data_pipelines)
{: .d-inline-block }

Data Engineering
{: .label .label-yellow }

Design end-to-end data pipelines for ML training — ingestion, transformation, validation, and orchestration.

**Key concepts:** Medallion architecture, Airflow DAGs, Kubeflow, data validation, dataset versioning, pipeline monitoring

**Difficulty:** ⭐⭐⭐ Medium-Hard

---

### [Large Language Models]({{ site.baseurl }}/ml_system_design/llm_systems)
{: .d-inline-block }

GenAI
{: .label .label-purple }

Design production LLM systems — RAG, prompt engineering, fine-tuning, vector databases, and serving at scale.

**Key concepts:** RAG pipeline, chunking, vector DBs (Pinecone, Qdrant), LoRA fine-tuning, vLLM, guardrails, LLM evaluation

**Difficulty:** ⭐⭐⭐⭐ Hard

---

### [Distributed Training]({{ site.baseurl }}/ml_system_design/distributed_training)
{: .d-inline-block }

Training Infrastructure
{: .label .label-red }

Design training infrastructure that scales deep learning across hundreds of GPUs.

**Key concepts:** Data parallelism (DDP), ZeRO, tensor parallelism, pipeline parallelism, DeepSpeed, mixed precision, fault tolerance

**Difficulty:** ⭐⭐⭐⭐ Hard

---

## Coming Soon

- **Search Ranking** - ML-powered search results with learning-to-rank
- **Real-time Personalization** - Session-based recommendations

---

## ML System Design Framework

Use this framework in your interviews:

### 1. Problem Setup (5 min)
- What are we predicting/generating?
- What data is available?
- What are the latency requirements?
- Online vs batch prediction?
- Success metrics (business + ML)

### 2. Data Pipeline (10 min)
- How is data collected and stored?
- Feature engineering approach
- Data validation and quality checks
- Training/serving data consistency
- Labeling strategy

### 3. Model Architecture (10 min)
- Model selection and justification
- Training strategy (pre-trained, fine-tuning)
- Evaluation metrics (offline)
- Handling edge cases (cold start, imbalance)

### 4. Serving Infrastructure (10 min)
- Model serving framework (TensorFlow Serving, TorchServe, Triton)
- Latency optimization (batching, caching, quantization)
- Scaling strategy (horizontal, GPU)
- A/B testing and gradual rollout

### 5. Monitoring & Iteration (5 min)
- Model performance monitoring
- Data drift detection
- Retraining triggers and pipelines
- Feedback loops

---

## Key ML Concepts to Know

```
MODEL SERVING
├── Batch Inference    → Process large datasets offline
├── Online Inference   → Real-time predictions, low latency
├── Model Versioning   → Track and rollback model versions
├── Dynamic Batching   → Group requests for GPU efficiency
├── Canary / Shadow    → Test new models safely before full rollout
└── A/B Testing        → Statistical comparison of model variants

LLM SYSTEMS
├── RAG Pipeline       → Retrieve context + generate grounded answers
├── Prompt Engineering → Zero-shot, few-shot, chain-of-thought
├── Fine-tuning (LoRA) → Adapt base models to domain tasks
├── Vector Databases   → Pinecone, Qdrant, Weaviate for similarity search
├── Guardrails         → PII detection, prompt injection, toxicity filtering
└── LLM Evaluation     → Groundedness, relevance, factuality

DISTRIBUTED TRAINING
├── Data Parallelism   → DDP: same model, different data shards
├── ZeRO Optimization  → Partition optimizer/gradients/params across GPUs
├── Tensor Parallelism → Split layers across GPUs (intra-node)
├── Pipeline Parallel  → Split sequential stages across nodes
├── Mixed Precision    → BF16/FP16 for compute, FP32 master weights
└── Fault Tolerance    → Checkpointing, signal handling, auto-resume

FEATURE ENGINEERING
├── Real-time Features → Computed on request (velocity, session)
├── Batch Features     → Precomputed (user history, aggregates)
├── Feature Store      → Feast, Tecton for train-serve consistency
├── Point-in-Time Join → Prevent label leakage in training data
└── Embeddings         → Dense representations from neural nets

DATA PIPELINES
├── Medallion Arch     → Bronze (raw) → Silver (clean) → Gold (features)
├── Orchestration      → Airflow, Kubeflow for DAG scheduling
├── Data Validation    → Schema, volume, distribution checks
├── Data Versioning    → Reproducible training with dataset lineage
└── Pipeline Monitoring → Freshness, volume, validation pass rate

MONITORING
├── Data Drift         → Input distribution changes (KS test, PSI)
├── Concept Drift      → Relationship between input/output changes
├── Model Degradation  → Performance decline over time
├── Latency Metrics    → P50, P95, P99 response times
└── Business Metrics   → CTR, conversion, revenue impact
```

---

## Pattern Recognition

As you study these designs, look for patterns that repeat:

| Pattern | Where You'll See It |
|---------|---------------------|
| **Two-stage retrieval+ranking** | Recommendations, Search, Fraud Detection, RAG |
| **Vector embeddings** | Image Search, Recommendations, LLM/RAG |
| **Feature stores** | Fraud Detection, Recommendations, Feature Store Design |
| **Dynamic batching** | Image Captioning, Model Serving, LLM Serving |
| **A/B testing** | All ML systems, Model Serving (canary) |
| **Ensemble models** | Fraud Detection, Recommendations |
| **Rules + ML hybrid** | Fraud Detection, Guardrails (LLM) |
| **Real-time aggregations** | Fraud Detection, Feature Store (streaming), Recommendations |
| **Distributed training** | LLM pre-training, Distributed Training |
| **Prompt engineering** | LLM Systems, RAG |
| **Data validation** | Data Pipelines, Feature Store |

{: .note }
> Master these patterns and you can apply them to any new ML system design problem.

---

## Quick Reference: System Comparison

### ML Application Systems

| System | Latency | Key Challenge | Primary Metric |
|--------|---------|---------------|----------------|
| **Image Captioning** | ~500ms | Model optimization | BLEU, CIDEr |
| **Recommendations** | <100ms | Cold start, scale | CTR, Conversion |
| **Fraud Detection** | <100ms | Class imbalance | Precision-Recall |
| **Image Search** | <200ms | Index at scale | Recall@K, Latency |
| **LLM Chatbot (RAG)** | <500ms TTFT | Hallucination, cost | Groundedness, Relevance |

### ML Infrastructure Systems

| System | Focus | Key Challenge | Success Metric |
|--------|-------|---------------|----------------|
| **Model Serving** | Inference | Latency, GPU utilization | P99 latency, throughput |
| **Feature Store** | Feature mgmt | Train-serve skew | Consistency, freshness |
| **Data Pipelines** | Data quality | Validation, lineage | Data freshness, coverage |
| **Distributed Training** | Training | Scaling efficiency | GPU utilization, time-to-train |

---

## Interview Tips for ML Design

{: .tip }
> **Don't just talk about the model!** Interviewers at top companies (Google, Meta, Netflix) want to see you design the full system:
> - Data pipelines and feature engineering
> - Training infrastructure and experiment tracking  
> - Serving architecture with latency considerations
> - Monitoring, A/B testing, and iteration

**Common mistakes to avoid:**
1. Jumping into model architecture without understanding requirements
2. Ignoring data quality and feature engineering
3. Forgetting about cold start problems
4. Not discussing how to measure success
5. Ignoring operational concerns (monitoring, retraining)

