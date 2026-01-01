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

## Coming Soon

- **LLM Serving System** - Deploy and scale large language models with RAG
- **Search Ranking** - ML-powered search results with learning-to-rank
- **Real-time Personalization** - Session-based recommendations
- **Model Training Platform** - Distributed training at scale

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
└── Shadow Mode        → Test new models without affecting users

RETRIEVAL & RANKING
├── Two-Stage Pipeline → Retrieval (fast, broad) + Ranking (accurate)
├── Vector Indexes     → FAISS, HNSW, IVF for ANN search
├── Feature Stores     → Consistent features for training/serving
└── Re-ranking         → Apply business rules, diversity

FEATURE ENGINEERING
├── Real-time Features → Computed on request (velocity, session)
├── Batch Features     → Precomputed (user history, aggregates)
├── Feature Store      → Feast, Tecton for consistency
└── Embeddings         → Dense representations from neural nets

MONITORING
├── Data Drift         → Input distribution changes
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
| **Two-stage retrieval+ranking** | Recommendations, Search, Fraud Detection |
| **Vector embeddings** | Image Search, Recommendations |
| **Feature stores** | Fraud Detection, Recommendations |
| **Dynamic batching** | Image Captioning, all GPU-based serving |
| **A/B testing** | All ML systems |
| **Ensemble models** | Fraud Detection, Recommendations |
| **Rules + ML hybrid** | Fraud Detection, Content Moderation |
| **Real-time aggregations** | Fraud Detection (velocity), Recommendations (session) |

{: .note }
> Master these patterns and you can apply them to any new ML system design problem.

---

## Quick Reference: System Comparison

| System | Latency | Key Challenge | Primary Metric |
|--------|---------|---------------|----------------|
| **Image Captioning** | ~500ms | Model optimization | BLEU, CIDEr |
| **Recommendations** | <100ms | Cold start, scale | CTR, Conversion |
| **Fraud Detection** | <100ms | Class imbalance | Precision-Recall |
| **Image Search** | <200ms | Index at scale | Recall@K, Latency |

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

