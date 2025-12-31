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

### [Image Caption Generator](image_caption_generator.md)
{: .d-inline-block }

ML Pipeline
{: .label .label-purple }

Design a system that generates descriptive captions for images using deep learning.

**Key concepts:** CNN/RNN architectures, model serving, batch vs real-time inference, caching

**Difficulty:** ⭐⭐⭐ Medium-Hard

---

## Coming Soon

- **Recommendation System** - Personalized content at scale
- **Fraud Detection** - Real-time ML for transactions
- **Search Ranking** - ML-powered search results
- **LLM Serving** - Deploy and scale large language models

---

## ML System Design Framework

Use this framework in your interviews:

### 1. Problem Setup (5 min)
- What are we predicting/generating?
- What data is available?
- What are the latency requirements?
- Online vs batch prediction?

### 2. Data Pipeline (10 min)
- How is data collected and stored?
- Feature engineering approach
- Data validation and quality checks
- Training/serving data consistency

### 3. Model Architecture (10 min)
- Model selection and justification
- Training strategy (pre-trained, fine-tuning)
- Evaluation metrics
- Offline vs online metrics

### 4. Serving Infrastructure (10 min)
- Model serving framework (TensorFlow Serving, TorchServe, Triton)
- Latency optimization (batching, caching, quantization)
- Scaling strategy
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
└── Shadow Mode        → Test new models without affecting users

FEATURE STORES
├── Offline Store      → Historical features for training
├── Online Store       → Low-latency features for serving
└── Feature Consistency → Same features in training & serving

MONITORING
├── Data Drift         → Input distribution changes
├── Concept Drift      → Relationship between input/output changes
├── Model Degradation  → Performance decline over time
└── Latency Metrics    → P50, P95, P99 response times
```

{: .tip }
> For ML interviews at top companies, be ready to discuss both the ML aspects AND the system design aspects. They're equally important!

