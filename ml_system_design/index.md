# ML System Design

Design machine learning systems for production — 10 designs covering ranking, retrieval, personalization, NLP, speech, and ML infrastructure.

---

## What Makes ML Design Different?

ML system design interviews focus on the **full ML lifecycle**, not just the model:

| Stage | Key Considerations |
|-------|-------------------|
| **Data** | Collection, storage, labeling, versioning |
| **Training** | Distributed training, experiment tracking |
| **Serving** | Latency, throughput, model updates |
| **Monitoring** | Drift detection, performance metrics |

!!! warning
    A common mistake: focusing only on the model architecture. Interviewers want to see you design the entire system around it.

---

## Recommended Study Order

!!! tip
    Follow this order — each design introduces new concepts that build on earlier ones.

| Order | Design | New Concepts Introduced | Prerequisite |
|-------|--------|------------------------|--------------|
| 1 | [Image Caption Generator](image_caption_generator.md) | Encoder-decoder, GPU serving, Triton | Model Serving fundamentals |
| 2 | [Image Search](image_search.md) | Embeddings, vector DBs, ANN indexes | Image Caption (embeddings) |
| 3 | [Recommendation System](recommendation_system.md) | Two-Tower, collaborative filtering, cold start | Image Search (ANN) |
| 4 | [Search Ranking](search_ranking.md) | BM25, LambdaMART, retrieval + ranking | Recommendation (two-stage) |
| 5 | [Fraud Detection](fraud_detection.md) | Real-time features, class imbalance, ensembles | Search Ranking (ranking) |
| 6 | [Real-time Personalization](realtime_personalization.md) | Session models, bandits, exploration | Recommendation + Fraud |
| 7 | [Ads Ranking System](ads_ranking.md) | CTR prediction, auctions, budget pacing | All of the above |
| 8 | [Real-time Feature Platform](feature_platform.md) | Streaming features, PIT joins, drift | Infra for all above |
| 9 | [Machine Translation](machine_translation.md) | Transformer NMT, multilingual, QE, low-resource | Seq2Seq fundamentals |
| 10 | [Speech Recognition](speech_recognition.md) | CTC/RNN-T, streaming ASR, diarization | Audio processing basics |

---

## Available Designs

### [Image Caption Generator](image_caption_generator.md)

Computer Vision

Design a system that generates descriptive captions for images using deep learning.

**Key concepts:** Encoder-decoder architecture, attention mechanisms, model serving (Triton), batching, caching, GPU optimization

**Difficulty:** ⭐⭐⭐ Medium-Hard

---

### [Image Search System](image_search.md)

Vector Search

Design a visual search system for finding similar images or searching by text description.

**Key concepts:** CLIP embeddings, vector databases (FAISS, Pinecone), ANN indexes (IVF, HNSW), multi-modal search, indexing pipelines, re-ranking

**Difficulty:** ⭐⭐⭐ Medium-Hard

---

### [Recommendation System](recommendation_system.md)

Personalization

Design a recommendation system for e-commerce or content platforms like Netflix/Amazon.

**Key concepts:** Collaborative filtering, content-based filtering, Two-Tower models, ANN retrieval (FAISS), ranking models, cold start, A/B testing

**Difficulty:** ⭐⭐⭐⭐ Hard

---

### [Search Ranking](search_ranking.md)

Information Retrieval

Design an ML-powered search ranking system (learning-to-rank, retrieval + ranking + re-ranking).

**Key concepts:** BM25, dense retrieval, hybrid fusion, LambdaMART, cross-encoder re-ranking, position bias, NDCG, serving at scale

**Difficulty:** ⭐⭐⭐⭐ Hard

---

### [Real-time Fraud Detection](fraud_detection.md)

Low-latency ML

Design a system that detects fraudulent transactions in real-time (<100ms).

**Key concepts:** Feature engineering (velocity features), class imbalance, ensemble models, rules engine, decision thresholds, case management, drift detection

**Difficulty:** ⭐⭐⭐⭐ Hard

---

### [Real-time Personalization](realtime_personalization.md)

Session-Based ML

Design a real-time personalization system that adapts to user behavior within a session.

**Key concepts:** Session models (GRU4Rec, SASRec), contextual bandits (Thompson Sampling, LinUCB), real-time feature engineering, multi-task ranking (MMoE), exploration vs exploitation, drift detection

**Difficulty:** ⭐⭐⭐⭐ Hard

---

### [Ads Ranking System](ads_ranking.md)

Revenue ML

NEW

Design an ads ranking system — the core revenue engine at Google, Meta, Amazon. Predict CTR/CVR, run auctions, manage advertiser budgets.

**Key concepts:** CTR prediction (DCN/DLRM), second-price/GSP auctions, budget pacing, position bias correction, calibration, exploration for new ads, near-real-time training

**Difficulty:** ⭐⭐⭐⭐⭐ Very Hard

---

### [Real-time Feature Platform](feature_platform.md)

ML Infrastructure

NEW

Design a real-time feature platform that computes, stores, and serves ML features with sub-millisecond latency — solving train-serve skew, point-in-time joins, and feature freshness at scale.

**Key concepts:** Batch vs streaming vs on-demand features, train-serve consistency, point-in-time joins, sliding window aggregations, feature drift monitoring, online/offline store architecture

**Difficulty:** ⭐⭐⭐⭐ Hard

---

### [Machine Translation](machine_translation.md)

NLP

NEW

Design a machine translation system like Google Translate — 100+ languages, text/image/speech, quality estimation, and low-resource language support.

**Key concepts:** Transformer encoder-decoder, multilingual NMT, BPE/SentencePiece, quality estimation, back-translation, pivot languages, beam search, model distillation for serving

**Difficulty:** ⭐⭐⭐⭐ Hard

---

### [Speech Recognition](speech_recognition.md)

Audio/Speech

NEW

Design a speech recognition (ASR) system like Google Speech-to-Text or Whisper — real-time streaming, speaker diarization, 100+ languages.

**Key concepts:** Mel spectrograms, CTC/RNN-T, Conformer, streaming inference, speaker diarization, language model fusion, SpecAugment, on-device vs cloud deployment

**Difficulty:** ⭐⭐⭐⭐ Hard

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

!!! tip
    For deep dives on Model Serving, Feature Stores, Data Pipelines, LLMs, and Distributed Training, see the [GenAI/ML Fundamentals](../genai_ml_basics/index.md) section.

---

## Pattern Recognition

| Pattern | Where You'll See It |
|---------|---------------------|
| **Two-stage retrieval+ranking** | Recommendations, Search, Ads Ranking, Fraud Detection |
| **Vector embeddings** | Image Search, Recommendations, Ads (DLRM) |
| **Feature stores** | Fraud Detection, Recommendations, Ads Ranking |
| **Dynamic batching** | Image Captioning, all GPU-based serving |
| **A/B testing** | All ML systems |
| **Ensemble models** | Fraud Detection, Recommendations |
| **Rules + ML hybrid** | Fraud Detection, Content Moderation, Ads |
| **Real-time aggregations** | Fraud Detection (velocity), Personalization, Ads |
| **Session modeling** | Real-time Personalization, Recommendations |
| **Multi-armed bandits** | Personalization, Ads (new ad exploration) |
| **Auction mechanics** | Ads Ranking (unique to ads) |
| **Encoder-decoder** | Machine Translation, Image Captioning, Speech Recognition |
| **Beam search/decoding** | Machine Translation, Speech Recognition |
| **Streaming inference** | Speech Recognition, Real-time Personalization |

!!! note
    Master these patterns and you can apply them to any new ML system design problem.

---

## Quick Reference: System Comparison

| System | Latency | Key Challenge | Primary Metric |
|--------|---------|---------------|----------------|
| **Image Captioning** | ~500ms | Model optimization | BLEU, CIDEr |
| **Recommendations** | <100ms | Cold start, scale | CTR, Conversion |
| **Fraud Detection** | <100ms | Class imbalance | Precision-Recall |
| **Image Search** | <200ms | Index at scale | Recall@K, Latency |
| **Search Ranking** | <200ms | Retrieval + rank budgets | NDCG@K, CTR |
| **Personalization** | <50ms | Session modeling, cold start | CTR, Session Depth |
| **Ads Ranking** | <50ms | Revenue × relevance | AUC, Revenue, Calibration |
| **Feature Platform** | <5ms (serving) | Train-serve consistency | Feature freshness, Drift rate |
| **Machine Translation** | <200ms | Low-resource, quality | BLEU, Human eval |
| **Speech Recognition** | <300ms RTF | Noise, streaming | WER, Latency |

---

## What's Next?

After mastering ML system design:

1. **Go deeper on GenAI** with [GenAI System Design](../genai_ml_system_design/index.md) — 10 LLM/GenAI systems with interview transcripts
2. **Review fundamentals** in [GenAI/ML Fundamentals](../genai_ml_basics/index.md) — 7 building blocks including LLM Evaluation and RLHF & Alignment
3. **Practice with transcripts** — the GenAI section includes full hypothetical interview walkthroughs

!!! note
    Looking for **LLM Chatbot, RAG, Code Assistant, AI Agents, or Text-to-Image** designs? These GenAI-specific system designs live in the dedicated [GenAI System Design](../genai_ml_system_design/index.md) section.
