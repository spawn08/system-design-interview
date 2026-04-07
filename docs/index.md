# System Design Interview Guide

Your comprehensive resource for mastering system design interviews — 75+ topics across software engineering, ML, and GenAI, with step-by-step walkthroughs, architecture diagrams, code examples, and interview transcripts.

---

## Who Is This For?

| You Are | Start Here | You'll Learn |
|---------|-----------|--------------|
| **Staff / L6 Engineer** targeting Google, Meta | [Staff Engineer Track](#staff-engineer-l6-interview-track) | Multi-region architecture, SLOs, consensus, leadership signals |
| **Senior SDE** preparing for FAANG | [Fundamentals](#fundamentals) → [System Design](#system-design-examples) | Core system design patterns and trade-offs |
| **ML/AI Engineer** designing production systems | [GenAI/ML Fundamentals](#genaiml-fundamentals) → [ML Design](#ml-system-design) | ML-specific architectures and serving patterns |
| **GenAI Engineer** building LLM-powered products | [GenAI/ML Fundamentals](#genaiml-fundamentals) → [GenAI Design](#genai-system-design) | LLM serving, RAG, agents, safety at scale |
| **Student / Junior Dev** wanting to level up | [Fundamentals](#fundamentals) | Foundational concepts with clear explanations |

---

## Recommended Learning Paths

!!! tip
    Follow these paths in order. Each builds on the previous. Skip sections you're already confident in.

### Path 1: Software System Design (SDE / Senior SDE)

```
Week 1-2: Fundamentals
  Interview Framework → Estimation → Networking → Databases → Caching → Load Balancing

Week 3-4: Core Design Problems
  URL Shortener → Rate Limiter → Key-Value Store → Distributed Cache → Chat System

Week 5-6: Complex Systems
  News Feed → Video Streaming → Collaborative Editor → Payment System → Message Queue

Week 7-8: Advanced Topics + Practice
  Consensus → Distributed Transactions → Sharding → Event Sourcing → Mock interviews
```

### Path 2: ML System Design (ML Engineer)

```
Week 1-2: Software Fundamentals (abbreviated)
  Databases → Caching → Distributed Systems → Message Queues

Week 3-4: ML Fundamentals
  Model Serving → Feature Stores → Data Pipelines → Distributed Training → LLM Evaluation → RLHF & Alignment

Week 5-6: ML Design Problems
  Recommendation System → Fraud Detection → Search Ranking → Ads Ranking → Machine Translation

Week 7-8: Practice
  Image Search → Real-time Personalization → Speech Recognition → Feature Platform → Mock interviews
```

### Path 3: GenAI System Design (GenAI / LLM Engineer)

```
Week 1-2: Foundations
  Distributed Systems → ML Fundamentals (Model Serving, Feature Stores)
  LLM Systems (RAG, fine-tuning, vector DBs) → Distributed Training → LLM Evaluation → RLHF & Alignment

Week 3-4: GenAI Design Problems
  LLM Chatbot → Enterprise RAG → Document Q&A → AI Code Assistant → LLM Gateway

Week 5-6: Advanced GenAI
  AI Agent System → Content Moderation → Hallucination Detection → Text-to-Image → Multi-Modal Search

Week 7-8: Infrastructure + LLM Ops + Practice
  ML Training Platform → Fine-Tuning Platform → LLM Evaluation Pipeline → Prompt Management
  Mock interviews with interview transcripts
```

### Path 4: Staff Engineer (L6) — All Domains

```
Week 1-2: Master Fundamentals + Advanced
  All Basics → Consensus → Distributed Transactions → Sharding → Observability

Week 3-4: Priority Designs (deep)
  Key-Value Store → Rate Limiter → Collaborative Editor → Task Scheduler → Payment System

Week 5-6: ML/GenAI Depth
  LLM Chatbot → Enterprise RAG → ML Training Platform → Ads Ranking

Week 7-8: Leadership + Practice
  Staff Engineer Guide → Behavioral & Leadership → Read interview transcripts → Mock interviews
```

---

## What's Inside

### [Fundamentals](basics/index.md)
Start here. The building blocks that appear in every system design interview.

- [**Interview Framework**](basics/interview_framework.md) - How to approach any system design question
- [**Estimation & Planning**](basics/estimation.md) - Back-of-the-envelope calculations
- [**Networking**](basics/networking.md) - TCP/UDP, HTTP, DNS, WebSockets
- [**Databases**](basics/databases.md) - SQL vs NoSQL, ACID, CAP theorem
- [**Caching**](basics/caching.md) - Speed up reads with in-memory storage
- [**Load Balancing**](basics/load_balancer.md) - Distribute traffic across servers
- [**API Design**](basics/api_design.md) - REST, GraphQL, versioning, authentication
- [**Concurrency**](basics/concurrency.md) - Threads, locks, async patterns
- [**Security**](basics/security.md) - Encryption, hashing, TLS, common vulnerabilities
- [**Scalability & Reliability**](basics/scalability.md) - Scaling, availability, disaster recovery
- [**Distributed Systems**](basics/distributed_systems.md) - CAP, consensus, message queues, DHTs

### [Advanced Topics](advanced/index.md)
Deep dives for Senior and Staff-level interviews.

- [**Message Queues & Streaming**](advanced/message_queues.md) - Kafka, RabbitMQ, Flink, event-driven patterns
- [**Search Systems**](advanced/search_systems.md) - Inverted indexes, Elasticsearch, BM25
- [**Consistency Patterns**](advanced/consistency_patterns.md) - CRDTs, sagas, transactional outbox, quorum
- [**Microservices Architecture**](advanced/microservices.md) - Service discovery, API gateways, Kubernetes
- [**Data Warehousing & Lakes**](advanced/data_warehousing.md) - ETL, star schema, Spark, lakehouse
- [**Object Storage & CDN**](advanced/object_storage_cdn.md) - S3, blob storage, edge caching
- [**Distributed Locking**](advanced/distributed_locking.md) - Redlock, fencing tokens, ZooKeeper
- [**Observability**](advanced/observability.md) - Logging, metrics, tracing, OpenTelemetry
- [**Event Sourcing & CQRS**](advanced/event_sourcing_cqrs.md) - Append-only logs, projections
- [**Consensus Algorithms**](advanced/consensus_algorithms.md) - Raft, Paxos, leader election
- [**Distributed Transactions**](advanced/distributed_transactions.md) - 2PC, Saga, transactional outbox
- [**Sharding & Partitioning**](advanced/sharding_partitioning.md) - Partition keys, hot spots, resharding
- [**Behavioral & Leadership (L6)**](advanced/behavioral_leadership.md) - STAR for Staff, conflict resolution

### Staff Engineer (L6) Interview Track

Targeting Google, Meta, or other top companies at the **Staff / Principal / L6** level?

- [**Staff Engineer Interview Guide**](software_system_design/staff_engineer_expectations.md) - L5 vs L6 expectations, 5 pillars, anti-patterns
- **Priority designs:** [Key-Value Store](software_system_design/key_value_store.md), [Rate Limiter](software_system_design/rate_limiter.md), [Collaborative Editor](software_system_design/collaborative_editor.md), [Task Scheduler](software_system_design/task_scheduler.md), [Notification System](software_system_design/notification_system.md)
- **Advanced foundations:** [Consensus](advanced/consensus_algorithms.md), [Distributed Transactions](advanced/distributed_transactions.md), [Sharding](advanced/sharding_partitioning.md)
- **Leadership round:** [Behavioral & Leadership Guide](advanced/behavioral_leadership.md)

### [System Design Examples](software_system_design/index.md)
Step-by-step walkthroughs of classic interview questions — 28 designs.

**Infrastructure & Data:**

- [**URL Shortener**](software_system_design/url_shortening.md) - Hashing, Base62, distributed IDs
- [**Rate Limiter**](software_system_design/rate_limiter.md) - Token bucket, sliding window, Redis
- [**Key-Value Store**](software_system_design/key_value_store.md) - Consistent hashing, quorum, vector clocks
- [**Distributed Cache**](software_system_design/distributed_cache.md) - LRU, hot keys, stampede mitigation
- [**Distributed Message Queue**](software_system_design/message_queue.md) - Append-only log, partitions, consumer groups
- [**Task Scheduler**](software_system_design/task_scheduler.md) - Priority queues, lease-based execution
- [**Metrics & Monitoring**](software_system_design/metrics_monitoring.md) - Time-series storage, alerting, Gorilla compression

**Communication & Social:**

- [**Chat System**](software_system_design/chat_system.md) - WebSockets, message ordering, presence
- [**Notification System**](software_system_design/notification_system.md) - Multi-channel, push vs pull
- [**News Feed / Timeline**](software_system_design/news_feed.md) - Fan-out strategies, ranking
- [**Voting System**](software_system_design/voting-system-design.md) - Consistency, duplicate prevention
- [**Email Delivery System**](software_system_design/email_delivery.md) - SMTP, DKIM/SPF, IP reputation, deliverability

**Media & Content:**

- [**Video Streaming (YouTube)**](software_system_design/video_streaming.md) - CDN, transcoding, adaptive bitrate
- [**Photo Sharing (Instagram)**](software_system_design/photo_sharing.md) - Object storage, feed, stories
- [**Cloud Storage (Google Drive)**](software_system_design/cloud_storage.md) - File sync, chunking, dedup

**Real-time & Geo:**

- [**Collaborative Editor (Google Docs)**](software_system_design/collaborative_editor.md) - OT/CRDTs, conflict resolution
- [**Ride Sharing (Uber/Lyft)**](software_system_design/ride_sharing.md) - Geospatial matching, tracking
- [**Proximity Service**](software_system_design/proximity_service.md) - Geohash, quadtree

**Commerce:**

- [**Event Booking (Ticketmaster)**](software_system_design/event_booking.md) - Inventory locking, flash crowds
- [**Payment System**](software_system_design/payment_system.md) - Idempotency, double-entry ledger

**Data Infrastructure:**

- [**Distributed File System (GFS)**](software_system_design/distributed_file_system.md) - Master-chunk architecture, replication, leases
- [**Ad Click Aggregator**](software_system_design/ad_click_aggregator.md) - Real-time aggregation, exactly-once, Flink/Kafka

**Search:**

- [**Web Crawler**](software_system_design/web_crawler.md) - Concurrency, politeness, dedup
- [**Search Autocomplete**](software_system_design/search_autocomplete.md) - Trie, ranking, caching

### [GenAI/ML Fundamentals](genai_ml_basics/index.md)
Core building blocks — master these before ML and GenAI design questions. 7 topics.

- [**Model Serving**](genai_ml_basics/model_serving.md) - Inference APIs, versioning, A/B testing, drift detection
- [**Feature Stores**](genai_ml_basics/feature_stores.md) - Train-serve consistency, point-in-time joins, Feast
- [**Data Pipelines for ML**](genai_ml_basics/data_pipelines.md) - Ingestion, validation, Airflow orchestration
- [**Large Language Models**](genai_ml_basics/llm_systems.md) - RAG, prompt engineering, fine-tuning, vector DBs
- [**Distributed Training**](genai_ml_basics/distributed_training.md) - Data/model/pipeline parallelism, DeepSpeed, ZeRO
- [**LLM Evaluation**](genai_ml_basics/llm_evaluation.md) - BLEU/ROUGE/BERTScore, LLM-as-judge, benchmarks, RAGAS
- [**RLHF & Alignment**](genai_ml_basics/rlhf_alignment.md) - PPO, DPO, Constitutional AI, safety alignment

### [ML System Design](ml_system_design/index.md)
Production ML systems — 10 designs covering ranking, retrieval, personalization, NLP, and feature infrastructure.

- [**Recommendation System**](ml_system_design/recommendation_system.md) - Collaborative filtering, Two-Tower, cold start
- [**Fraud Detection**](ml_system_design/fraud_detection.md) - Real-time ML, class imbalance, velocity features
- [**Image Search**](ml_system_design/image_search.md) - CLIP embeddings, FAISS, ANN indexes
- [**Image Caption Generator**](ml_system_design/image_caption_generator.md) - Encoder-decoder, attention, Triton
- [**Search Ranking**](ml_system_design/search_ranking.md) - BM25, LambdaMART, NDCG, retrieval + ranking
- [**Real-time Personalization**](ml_system_design/realtime_personalization.md) - Session models, contextual bandits
- [**Ads Ranking System**](ml_system_design/ads_ranking.md) - CTR prediction, auction mechanics, budget pacing
- [**Real-time Feature Platform**](ml_system_design/feature_platform.md) - Streaming features, PIT joins, train-serve consistency
- [**Machine Translation**](ml_system_design/machine_translation.md) - Transformer, multilingual NMT, quality estimation, low-resource
- [**Speech Recognition**](ml_system_design/speech_recognition.md) - CTC/RNN-T, streaming ASR, speaker diarization, Whisper

### [GenAI System Design](genai_ml_system_design/index.md)
Production GenAI systems — 15 designs with Google-style interview transcripts.

- [**LLM-Powered Chatbot**](genai_ml_system_design/llm_chatbot.md) - KV-cache, PagedAttention, streaming, safety
- [**Enterprise RAG System**](genai_ml_system_design/enterprise_rag.md) - Chunking, hybrid retrieval, ACLs, citations
- [**Document Q&A System**](genai_ml_system_design/document_qa_system.md) - PDF parsing, 10K+ docs, cross-encoder re-ranking, citations
- [**AI Code Assistant**](genai_ml_system_design/ai_code_assistant.md) - FIM, speculative decoding, repo context
- [**LLM Content Moderation**](genai_ml_system_design/content_moderation.md) - Cascade architecture, adversarial robustness
- [**Hallucination Detection**](genai_ml_system_design/hallucination_detection.md) - Claim extraction, NLI verification, confidence scoring
- [**ML Training Platform**](genai_ml_system_design/ml_training_platform.md) - Gang scheduling, checkpointing, GPU clusters
- [**LLM Fine-Tuning Platform**](genai_ml_system_design/llm_finetuning_platform.md) - LoRA/QLoRA, private data, differential privacy, blue-green deploy
- [**Multi-Modal Search**](genai_ml_system_design/multimodal_search.md) - CLIP embeddings, cross-modal retrieval
- [**AI Agent System**](genai_ml_system_design/ai_agent_system.md) - ReAct, tool use, planning, memory, multi-agent
- [**LLM Gateway**](genai_ml_system_design/llm_gateway.md) - Multi-model routing, semantic caching, cost control
- [**LLM Evaluation Pipeline**](genai_ml_system_design/llm_evaluation_pipeline.md) - LLM-as-judge, Elo rating, benchmarks, A/B testing
- [**Prompt Management & Versioning**](genai_ml_system_design/prompt_management.md) - Prompt registry, templating, A/B testing, environment promotion
- [**Text-to-Image Generation**](genai_ml_system_design/text_to_image.md) - Diffusion models, latent space, safety, CFG
- [**Vector Database**](genai_ml_system_design/vector_database.md) - HNSW, IVF-PQ, hybrid search, billion-scale ANN

---

## How to Use This Guide

### If You Have 1 Week

1. Read [Interview Framework](basics/interview_framework.md) — know the 4-step approach
2. Study [URL Shortener](software_system_design/url_shortening.md) and [Rate Limiter](software_system_design/rate_limiter.md)
3. For ML roles: add [Recommendation System](ml_system_design/recommendation_system.md)
4. For GenAI roles: add [LLM Chatbot](genai_ml_system_design/llm_chatbot.md) — read the interview transcript
5. Practice explaining designs out loud

### If You Have 1 Month

Follow the [Learning Path](#recommended-learning-paths) for your target role above.

### During Your Interview

1. **Clarify requirements** (5 min) — Don't assume!
2. **High-level design** (10 min) — Draw the big picture
3. **Deep dive** (15 min) — Focus on 2-3 components
4. **Wrap up** (5 min) — Discuss trade-offs and improvements

---

## Key Tips

!!! note
    **Always ask clarifying questions.** The interviewer wants to see how you think, not just what you know.

!!! tip
    **Draw diagrams.** A picture is worth a thousand words in system design.

!!! warning
    **Don't jump to solutions.** Understand the problem before proposing architecture.

---

## Content Overview

| Section | Topics | Difficulty |
|---------|--------|------------|
| **Fundamentals** | 11 essential topics | Beginner-Advanced |
| **Advanced Topics** | 13 deep-dive topics (incl. L6 track) | Advanced-Expert |
| **System Design Examples** | 28 classic problems (incl. Staff Guide) | Intermediate-Hard |
| **GenAI/ML Fundamentals** | 7 ML/GenAI building blocks | Medium-Hard |
| **ML System Design** | 10 ML systems | Hard |
| **GenAI System Design** | 15 GenAI systems (with interview transcripts) | Very Hard |
| **Total** | **84** | |

---

## Contributing

Found an error? Want to add a new design? Contributions are welcome! Open an issue or submit a pull request on GitHub.

---

<div align="center">

**Good luck with your interviews!**

*Remember: The goal isn't to give a "perfect" answer. It's to demonstrate clear thinking and good engineering judgment.*

</div>
