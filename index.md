---
layout: home
title: Home
nav_order: 1
description: "Master system design interviews with practical examples and clear explanations"
permalink: /
---

# System Design Interview Guide
{: .fs-9 }

Your comprehensive resource for mastering system design interviews, whether you're targeting Senior Software Engineer roles or ML/GenAI positions.
{: .fs-6 .fw-300 }

---

## 👋 Welcome!

System design interviews can feel overwhelming. There's so much to know: databases, caching, load balancing, distributed systems... Where do you even start?

**This guide breaks it down for you.**

We take complex systems and explain them step-by-step, from "what is this?" all the way to "how do I design it at scale?"

---

## 🎯 Who Is This For?

| You Are | You'll Learn |
|---------|--------------|
| **Staff / L6 Engineer** targeting Google, Meta, etc. | Multi-region architecture, SLOs, consensus, leadership signals |
| **Senior Software Engineer** preparing for FAANG interviews | Core system design patterns and trade-offs |
| **ML/AI Engineer** designing production systems | ML-specific architectures and serving patterns |
| **Student or Junior Dev** wanting to level up | Foundational concepts with clear explanations |
| **Experienced Engineer** needing a refresher | Quick reference for common patterns |

---

## 📚 What's Inside

### [Fundamentals]({{ site.baseurl }}/basics/)
Start here if you're new to system design. Learn the building blocks that appear in every design.

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
Deep dives for Senior and Staff-level interviews — the systems behind the systems.

- [**Message Queues & Streaming**]({{ site.baseurl }}/advanced/message_queues) - Kafka, RabbitMQ, Flink, event-driven patterns
- [**Search Systems**]({{ site.baseurl }}/advanced/search_systems) - Inverted indexes, Elasticsearch, BM25, autocomplete
- [**Consistency Patterns**]({{ site.baseurl }}/advanced/consistency_patterns) - CRDTs, sagas, transactional outbox, quorum
- [**Microservices Architecture**]({{ site.baseurl }}/advanced/microservices) - Service discovery, API gateways, Docker, Kubernetes
- [**Data Warehousing & Lakes**]({{ site.baseurl }}/advanced/data_warehousing) - ETL, star schema, Spark, lakehouse architecture
- [**Object Storage & CDN**]({{ site.baseurl }}/advanced/object_storage_cdn) - S3, blob storage, CDN edge caching, pre-signed URLs
- [**Distributed Locking**]({{ site.baseurl }}/advanced/distributed_locking) - Redlock, fencing tokens, ZooKeeper, lease-based locks
- [**Observability**]({{ site.baseurl }}/advanced/observability) - Logging, metrics, tracing, OpenTelemetry, ELK stack
- [**Event Sourcing & CQRS**]({{ site.baseurl }}/advanced/event_sourcing_cqrs) - Append-only logs, read/write separation, projections
- [**Consensus Algorithms (Raft/Paxos)**]({{ site.baseurl }}/advanced/consensus_algorithms) - Leader election, log replication, when to use consensus
- [**Distributed Transactions**]({{ site.baseurl }}/advanced/distributed_transactions) - 2PC, Saga pattern, transactional outbox, Spanner
- [**Sharding & Partitioning**]({{ site.baseurl }}/advanced/sharding_partitioning) - Partition keys, hot spots, resharding strategies
- [**Behavioral & Leadership (L6)**]({{ site.baseurl }}/advanced/behavioral_leadership) - STAR for Staff, conflict resolution, technical vision

### Staff Engineer (L6) Interview Track

Targeting Google, Meta, or other top companies at the **Staff / Principal / L6** level? This curated track covers the elevated expectations:

- [**Staff Engineer Interview Guide**]({{ site.baseurl }}/software_system_design/staff_engineer_expectations) - L5 vs L6 expectations, 5 pillars, anti-patterns
- **Priority designs** with L6 deep dives: [Key-Value Store]({{ site.baseurl }}/software_system_design/key_value_store), [Rate Limiter]({{ site.baseurl }}/software_system_design/rate_limiter), [Collaborative Editor]({{ site.baseurl }}/software_system_design/collaborative_editor), [Task Scheduler]({{ site.baseurl }}/software_system_design/task_scheduler), [Notification System]({{ site.baseurl }}/software_system_design/notification_system)
- **Advanced foundations:** [Consensus]({{ site.baseurl }}/advanced/consensus_algorithms), [Distributed Transactions]({{ site.baseurl }}/advanced/distributed_transactions), [Sharding]({{ site.baseurl }}/advanced/sharding_partitioning)
- **Leadership round:** [Behavioral & Leadership Guide]({{ site.baseurl }}/advanced/behavioral_leadership)

### [System Design Examples]({{ site.baseurl }}/software_system_design/)
Step-by-step walkthroughs of classic interview questions.

- [**URL Shortener**]({{ site.baseurl }}/software_system_design/url_shortening) - Design TinyURL or Bitly
- [**Rate Limiter**]({{ site.baseurl }}/software_system_design/rate_limiter) - Protect APIs from abuse
- [**Key-Value Store**]({{ site.baseurl }}/software_system_design/key_value_store) - Design a distributed key-value store
- [**Distributed Cache**]({{ site.baseurl }}/software_system_design/distributed_cache) - Design Redis/Memcached
- [**Notification System**]({{ site.baseurl }}/software_system_design/notification_system) - Multi-channel notifications
- [**Web Crawler**]({{ site.baseurl }}/software_system_design/web_crawler) - Index the web at scale
- [**Chat System**]({{ site.baseurl }}/software_system_design/chat_system) - Real-time messaging platform
- [**News Feed / Timeline**]({{ site.baseurl }}/software_system_design/news_feed) - Social media feed design
- [**Search Autocomplete**]({{ site.baseurl }}/software_system_design/search_autocomplete) - Google-like typeahead
- [**Voting System**]({{ site.baseurl }}/software_system_design/voting-system-design) - Scalable polling platform
- [**Video Streaming (YouTube)**]({{ site.baseurl }}/software_system_design/video_streaming) - Video upload, transcoding, adaptive streaming
- [**Photo Sharing (Instagram)**]({{ site.baseurl }}/software_system_design/photo_sharing) - Photo upload, feed, stories
- [**Collaborative Editor (Google Docs)**]({{ site.baseurl }}/software_system_design/collaborative_editor) - Real-time collaborative editing
- [**Ride Sharing (Uber/Lyft)**]({{ site.baseurl }}/software_system_design/ride_sharing) - Geospatial matching, real-time tracking
- [**Cloud Storage (Google Drive)**]({{ site.baseurl }}/software_system_design/cloud_storage) - File sync, chunking, deduplication
- [**Event Booking (Ticketmaster)**]({{ site.baseurl }}/software_system_design/event_booking) - Inventory management, booking flow
- [**Task Scheduler**]({{ site.baseurl }}/software_system_design/task_scheduler) - Distributed task scheduling
- [**Payment System**]({{ site.baseurl }}/software_system_design/payment_system) - Idempotent payment processing
- [**Proximity Service**]({{ site.baseurl }}/software_system_design/proximity_service) - Geospatial search, nearby places

### [GenAI/ML Fundamentals]({{ site.baseurl }}/genai_ml_basics/)
Core building blocks for ML and GenAI system design — master these before tackling full ML designs.

- [**Model Serving**]({{ site.baseurl }}/genai_ml_basics/model_serving) - Inference APIs, versioning, A/B testing, drift detection
- [**Feature Stores**]({{ site.baseurl }}/genai_ml_basics/feature_stores) - Train-serve consistency, point-in-time joins, Feast
- [**Data Pipelines for ML**]({{ site.baseurl }}/genai_ml_basics/data_pipelines) - Ingestion, validation, Airflow orchestration
- [**Large Language Models**]({{ site.baseurl }}/genai_ml_basics/llm_systems) - RAG, prompt engineering, fine-tuning, vector DBs
- [**Distributed Training**]({{ site.baseurl }}/genai_ml_basics/distributed_training) - Data/model/pipeline parallelism, DeepSpeed, ZeRO

### [ML System Design]({{ site.baseurl }}/ml_system_design/)
Specialized designs for machine learning and AI systems.

- [**Recommendation System**]({{ site.baseurl }}/ml_system_design/recommendation_system) - Personalized content at scale (Netflix, Amazon)
- [**Fraud Detection**]({{ site.baseurl }}/ml_system_design/fraud_detection) - Real-time ML for transactions
- [**Image Search**]({{ site.baseurl }}/ml_system_design/image_search) - Visual search with embeddings
- [**Image Caption Generator**]({{ site.baseurl }}/ml_system_design/image_caption_generator) - ML pipeline from ingestion to inference
- [**Search Ranking**]({{ site.baseurl }}/ml_system_design/search_ranking) - Learning-to-rank, retrieval + ranking
- [**Real-time Personalization**]({{ site.baseurl }}/ml_system_design/realtime_personalization) - Session-based ML, contextual bandits

---

## 🚀 How to Use This Guide

### If You Have 1 Week

1. Read through [Load Balancing]({{ site.baseurl }}/basics/load_balancer) to understand fundamentals
2. Study [URL Shortener]({{ site.baseurl }}/software_system_design/url_shortening) - it's asked frequently
3. Practice explaining the design out loud

### If You Have 1 Month

1. Master all fundamentals in the Basics section
2. Work through each system design example
3. Draw diagrams from memory
4. Mock interview with a friend

### During Your Interview

1. **Clarify requirements** (5 min) - Don't assume!
2. **High-level design** (10 min) - Draw the big picture
3. **Deep dive** (15 min) - Focus on 2-3 components
4. **Wrap up** (5 min) - Discuss trade-offs and improvements

---

## 💡 Key Tips

{: .note }
> **Always ask clarifying questions.** The interviewer wants to see how you think, not just what you know.

{: .tip }
> **Draw diagrams.** A picture is worth a thousand words in system design.

{: .warning }
> **Don't jump to solutions.** Understand the problem before proposing architecture.

---

## 📖 Quick Reference: Essential Topics

Every system design interview touches on these concepts:

| Topic | What to Know |
|-------|--------------|
| **Interview Framework** | 4-step structure, time management, trade-off reasoning |
| **Estimation** | QPS, storage, bandwidth, capacity planning |
| **Networking** | TCP/UDP, HTTP/gRPC, DNS, WebSockets |
| **Databases** | SQL vs NoSQL, sharding, replication |
| **Caching** | Redis/Memcached, cache invalidation, TTL |
| **Load Balancing** | Round robin, least connections, health checks |
| **API Design** | REST, GraphQL, versioning, rate limiting, JWT/OAuth |
| **Concurrency** | Threads, locks, deadlocks, async patterns |
| **Security** | TLS, encryption, hashing, input validation |
| **Scalability** | Horizontal vs vertical, auto-scaling, circuit breakers |
| **Distributed Systems** | CAP theorem, Raft/Paxos, consistent hashing |

---

## 📊 Content Overview

| Section | Topics | Difficulty |
|---------|--------|------------|
| **Fundamentals** | 11 essential topics | ⭐⭐ Beginner-Advanced |
| **Advanced Topics** | 13 deep-dive topics (incl. L6 track) | ⭐⭐⭐ Advanced-Expert |
| **System Design Examples** | 20 classic problems (incl. Staff Guide) | ⭐⭐⭐ Intermediate-Hard |
| **GenAI/ML Fundamentals** | 5 ML/GenAI building blocks | ⭐⭐⭐ Medium-Hard |
| **ML System Design** | 6 ML systems | ⭐⭐⭐⭐ Hard |

---

## 🤝 Contributing

Found an error? Want to add a new design? Contributions are welcome! Open an issue or submit a pull request on GitHub.

---

<div align="center">

**Good luck with your interviews!** 🎉

*Remember: The goal isn't to give a "perfect" answer. It's to demonstrate clear thinking and good engineering judgment.*

</div>

