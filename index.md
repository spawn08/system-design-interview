---
layout: home
title: Home
nav_order: 1
description: "Master system design interviews with practical examples and clear explanations"
permalink: /
---

# System Design Interview Guide
{: .fs-9 }

Your comprehensive resource for mastering system design interviews — 62 topics across software engineering, ML, and GenAI, with step-by-step walkthroughs, architecture diagrams, code examples, and interview transcripts.
{: .fs-6 .fw-300 }

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

{: .tip }
> Follow these paths in order. Each builds on the previous. Skip sections you're already confident in.

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
  Model Serving → Feature Stores → Data Pipelines → Distributed Training

Week 5-6: ML Design Problems
  Recommendation System → Fraud Detection → Search Ranking → Ads Ranking

Week 7-8: Practice
  Image Search → Real-time Personalization → Mock interviews
```

### Path 3: GenAI System Design (GenAI / LLM Engineer)

```
Week 1-2: Foundations
  Distributed Systems → ML Fundamentals (Model Serving, Feature Stores)
  LLM Systems (RAG, fine-tuning, vector DBs) → Distributed Training

Week 3-4: GenAI Design Problems
  LLM Chatbot → Enterprise RAG → AI Code Assistant → LLM Gateway

Week 5-6: Advanced GenAI
  AI Agent System → Content Moderation → Text-to-Image → Multi-Modal Search

Week 7-8: Infrastructure + Practice
  ML Training Platform → Mock interviews with interview transcripts
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

### [Fundamentals]({{ site.baseurl }}/basics/)
Start here. The building blocks that appear in every system design interview.

- [**Interview Framework**]({{ site.baseurl }}/basics/interview_framework) - How to approach any system design question
- [**Estimation & Planning**]({{ site.baseurl }}/basics/estimation) - Back-of-the-envelope calculations
- [**Networking**]({{ site.baseurl }}/basics/networking) - TCP/UDP, HTTP, DNS, WebSockets
- [**Databases**]({{ site.baseurl }}/basics/databases) - SQL vs NoSQL, ACID, CAP theorem
- [**Caching**]({{ site.baseurl }}/basics/caching) - Speed up reads with in-memory storage
- [**Load Balancing**]({{ site.baseurl }}/basics/load_balancer) - Distribute traffic across servers
- [**API Design**]({{ site.baseurl }}/basics/api_design) - REST, GraphQL, versioning, authentication
- [**Concurrency**]({{ site.baseurl }}/basics/concurrency) - Threads, locks, async patterns
- [**Security**]({{ site.baseurl }}/basics/security) - Encryption, hashing, TLS, common vulnerabilities
- [**Scalability & Reliability**]({{ site.baseurl }}/basics/scalability) - Scaling, availability, disaster recovery
- [**Distributed Systems**]({{ site.baseurl }}/basics/distributed_systems) - CAP, consensus, message queues, DHTs

### [Advanced Topics]({{ site.baseurl }}/advanced/)
Deep dives for Senior and Staff-level interviews.

- [**Message Queues & Streaming**]({{ site.baseurl }}/advanced/message_queues) - Kafka, RabbitMQ, Flink, event-driven patterns
- [**Search Systems**]({{ site.baseurl }}/advanced/search_systems) - Inverted indexes, Elasticsearch, BM25
- [**Consistency Patterns**]({{ site.baseurl }}/advanced/consistency_patterns) - CRDTs, sagas, transactional outbox, quorum
- [**Microservices Architecture**]({{ site.baseurl }}/advanced/microservices) - Service discovery, API gateways, Kubernetes
- [**Data Warehousing & Lakes**]({{ site.baseurl }}/advanced/data_warehousing) - ETL, star schema, Spark, lakehouse
- [**Object Storage & CDN**]({{ site.baseurl }}/advanced/object_storage_cdn) - S3, blob storage, edge caching
- [**Distributed Locking**]({{ site.baseurl }}/advanced/distributed_locking) - Redlock, fencing tokens, ZooKeeper
- [**Observability**]({{ site.baseurl }}/advanced/observability) - Logging, metrics, tracing, OpenTelemetry
- [**Event Sourcing & CQRS**]({{ site.baseurl }}/advanced/event_sourcing_cqrs) - Append-only logs, projections
- [**Consensus Algorithms**]({{ site.baseurl }}/advanced/consensus_algorithms) - Raft, Paxos, leader election
- [**Distributed Transactions**]({{ site.baseurl }}/advanced/distributed_transactions) - 2PC, Saga, transactional outbox
- [**Sharding & Partitioning**]({{ site.baseurl }}/advanced/sharding_partitioning) - Partition keys, hot spots, resharding
- [**Behavioral & Leadership (L6)**]({{ site.baseurl }}/advanced/behavioral_leadership) - STAR for Staff, conflict resolution

### Staff Engineer (L6) Interview Track

Targeting Google, Meta, or other top companies at the **Staff / Principal / L6** level?

- [**Staff Engineer Interview Guide**]({{ site.baseurl }}/software_system_design/staff_engineer_expectations) - L5 vs L6 expectations, 5 pillars, anti-patterns
- **Priority designs:** [Key-Value Store]({{ site.baseurl }}/software_system_design/key_value_store), [Rate Limiter]({{ site.baseurl }}/software_system_design/rate_limiter), [Collaborative Editor]({{ site.baseurl }}/software_system_design/collaborative_editor), [Task Scheduler]({{ site.baseurl }}/software_system_design/task_scheduler), [Notification System]({{ site.baseurl }}/software_system_design/notification_system)
- **Advanced foundations:** [Consensus]({{ site.baseurl }}/advanced/consensus_algorithms), [Distributed Transactions]({{ site.baseurl }}/advanced/distributed_transactions), [Sharding]({{ site.baseurl }}/advanced/sharding_partitioning)
- **Leadership round:** [Behavioral & Leadership Guide]({{ site.baseurl }}/advanced/behavioral_leadership)

### [System Design Examples]({{ site.baseurl }}/software_system_design/)
Step-by-step walkthroughs of classic interview questions — 22 designs.

**Infrastructure & Data:**
- [**URL Shortener**]({{ site.baseurl }}/software_system_design/url_shortening) - Hashing, Base62, distributed IDs
- [**Rate Limiter**]({{ site.baseurl }}/software_system_design/rate_limiter) - Token bucket, sliding window, Redis
- [**Key-Value Store**]({{ site.baseurl }}/software_system_design/key_value_store) - Consistent hashing, quorum, vector clocks
- [**Distributed Cache**]({{ site.baseurl }}/software_system_design/distributed_cache) - LRU, hot keys, stampede mitigation
- [**Distributed Message Queue**]({{ site.baseurl }}/software_system_design/message_queue) - Append-only log, partitions, consumer groups
- [**Task Scheduler**]({{ site.baseurl }}/software_system_design/task_scheduler) - Priority queues, lease-based execution
- [**Metrics & Monitoring**]({{ site.baseurl }}/software_system_design/metrics_monitoring) - Time-series storage, alerting, Gorilla compression

**Communication & Social:**
- [**Chat System**]({{ site.baseurl }}/software_system_design/chat_system) - WebSockets, message ordering, presence
- [**Notification System**]({{ site.baseurl }}/software_system_design/notification_system) - Multi-channel, push vs pull
- [**News Feed / Timeline**]({{ site.baseurl }}/software_system_design/news_feed) - Fan-out strategies, ranking
- [**Voting System**]({{ site.baseurl }}/software_system_design/voting-system-design) - Consistency, duplicate prevention
- [**Email Delivery System**]({{ site.baseurl }}/software_system_design/email_delivery) - SMTP, DKIM/SPF, IP reputation, deliverability

**Media & Content:**
- [**Video Streaming (YouTube)**]({{ site.baseurl }}/software_system_design/video_streaming) - CDN, transcoding, adaptive bitrate
- [**Photo Sharing (Instagram)**]({{ site.baseurl }}/software_system_design/photo_sharing) - Object storage, feed, stories
- [**Cloud Storage (Google Drive)**]({{ site.baseurl }}/software_system_design/cloud_storage) - File sync, chunking, dedup

**Real-time & Geo:**
- [**Collaborative Editor (Google Docs)**]({{ site.baseurl }}/software_system_design/collaborative_editor) - OT/CRDTs, conflict resolution
- [**Ride Sharing (Uber/Lyft)**]({{ site.baseurl }}/software_system_design/ride_sharing) - Geospatial matching, tracking
- [**Proximity Service**]({{ site.baseurl }}/software_system_design/proximity_service) - Geohash, quadtree

**Commerce:**
- [**Event Booking (Ticketmaster)**]({{ site.baseurl }}/software_system_design/event_booking) - Inventory locking, flash crowds
- [**Payment System**]({{ site.baseurl }}/software_system_design/payment_system) - Idempotency, double-entry ledger

**Search:**
- [**Web Crawler**]({{ site.baseurl }}/software_system_design/web_crawler) - Concurrency, politeness, dedup
- [**Search Autocomplete**]({{ site.baseurl }}/software_system_design/search_autocomplete) - Trie, ranking, caching

### [GenAI/ML Fundamentals]({{ site.baseurl }}/genai_ml_basics/)
Core building blocks — master these before ML and GenAI design questions.

- [**Model Serving**]({{ site.baseurl }}/genai_ml_basics/model_serving) - Inference APIs, versioning, A/B testing, drift detection
- [**Feature Stores**]({{ site.baseurl }}/genai_ml_basics/feature_stores) - Train-serve consistency, point-in-time joins, Feast
- [**Data Pipelines for ML**]({{ site.baseurl }}/genai_ml_basics/data_pipelines) - Ingestion, validation, Airflow orchestration
- [**Large Language Models**]({{ site.baseurl }}/genai_ml_basics/llm_systems) - RAG, prompt engineering, fine-tuning, vector DBs
- [**Distributed Training**]({{ site.baseurl }}/genai_ml_basics/distributed_training) - Data/model/pipeline parallelism, DeepSpeed, ZeRO

### [ML System Design]({{ site.baseurl }}/ml_system_design/)
Production ML systems — 8 designs covering ranking, retrieval, personalization, and feature infrastructure.

- [**Recommendation System**]({{ site.baseurl }}/ml_system_design/recommendation_system) - Collaborative filtering, Two-Tower, cold start
- [**Fraud Detection**]({{ site.baseurl }}/ml_system_design/fraud_detection) - Real-time ML, class imbalance, velocity features
- [**Image Search**]({{ site.baseurl }}/ml_system_design/image_search) - CLIP embeddings, FAISS, ANN indexes
- [**Image Caption Generator**]({{ site.baseurl }}/ml_system_design/image_caption_generator) - Encoder-decoder, attention, Triton
- [**Search Ranking**]({{ site.baseurl }}/ml_system_design/search_ranking) - BM25, LambdaMART, NDCG, retrieval + ranking
- [**Real-time Personalization**]({{ site.baseurl }}/ml_system_design/realtime_personalization) - Session models, contextual bandits
- [**Ads Ranking System**]({{ site.baseurl }}/ml_system_design/ads_ranking) - CTR prediction, auction mechanics, budget pacing
- [**Real-time Feature Platform**]({{ site.baseurl }}/ml_system_design/feature_platform) - Streaming features, PIT joins, train-serve consistency

### [GenAI System Design]({{ site.baseurl }}/genai_ml_system_design/)
Production GenAI systems — 10 designs with Google-style interview transcripts.

- [**LLM-Powered Chatbot**]({{ site.baseurl }}/genai_ml_system_design/llm_chatbot) - KV-cache, PagedAttention, streaming, safety
- [**Enterprise RAG System**]({{ site.baseurl }}/genai_ml_system_design/enterprise_rag) - Chunking, hybrid retrieval, ACLs, citations
- [**AI Code Assistant**]({{ site.baseurl }}/genai_ml_system_design/ai_code_assistant) - FIM, speculative decoding, repo context
- [**LLM Content Moderation**]({{ site.baseurl }}/genai_ml_system_design/content_moderation) - Cascade architecture, adversarial robustness
- [**ML Training Platform**]({{ site.baseurl }}/genai_ml_system_design/ml_training_platform) - Gang scheduling, checkpointing, GPU clusters
- [**Multi-Modal Search**]({{ site.baseurl }}/genai_ml_system_design/multimodal_search) - CLIP embeddings, cross-modal retrieval
- [**AI Agent System**]({{ site.baseurl }}/genai_ml_system_design/ai_agent_system) - ReAct, tool use, planning, memory, multi-agent
- [**LLM Gateway**]({{ site.baseurl }}/genai_ml_system_design/llm_gateway) - Multi-model routing, semantic caching, cost control
- [**Text-to-Image Generation**]({{ site.baseurl }}/genai_ml_system_design/text_to_image) - Diffusion models, latent space, safety, CFG
- [**Vector Database**]({{ site.baseurl }}/genai_ml_system_design/vector_database) - HNSW, IVF-PQ, hybrid search, billion-scale ANN

---

## How to Use This Guide

### If You Have 1 Week

1. Read [Interview Framework]({{ site.baseurl }}/basics/interview_framework) — know the 4-step approach
2. Study [URL Shortener]({{ site.baseurl }}/software_system_design/url_shortening) and [Rate Limiter]({{ site.baseurl }}/software_system_design/rate_limiter)
3. For ML roles: add [Recommendation System]({{ site.baseurl }}/ml_system_design/recommendation_system)
4. For GenAI roles: add [LLM Chatbot]({{ site.baseurl }}/genai_ml_system_design/llm_chatbot) — read the interview transcript
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

{: .note }
> **Always ask clarifying questions.** The interviewer wants to see how you think, not just what you know.

{: .tip }
> **Draw diagrams.** A picture is worth a thousand words in system design.

{: .warning }
> **Don't jump to solutions.** Understand the problem before proposing architecture.

---

## Content Overview

| Section | Topics | Difficulty |
|---------|--------|------------|
| **Fundamentals** | 11 essential topics | Beginner-Advanced |
| **Advanced Topics** | 13 deep-dive topics (incl. L6 track) | Advanced-Expert |
| **System Design Examples** | 22 classic problems (incl. Staff Guide) | Intermediate-Hard |
| **GenAI/ML Fundamentals** | 5 ML/GenAI building blocks | Medium-Hard |
| **ML System Design** | 8 ML systems | Hard |
| **GenAI System Design** | 10 GenAI systems (with interview transcripts) | Very Hard |
| **Total** | **69** | |

---

## Contributing

Found an error? Want to add a new design? Contributions are welcome! Open an issue or submit a pull request on GitHub.

---

<div align="center">

**Good luck with your interviews!**

*Remember: The goal isn't to give a "perfect" answer. It's to demonstrate clear thinking and good engineering judgment.*

</div>
