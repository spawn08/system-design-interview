---
layout: default
title: Fundamentals
nav_order: 2
has_children: true
permalink: /basics/
---

# System Design Fundamentals
{: .fs-9 }

Master the building blocks that appear in every system design interview.
{: .fs-6 .fw-300 }

---

## Why Fundamentals Matter

Before designing complex systems like Twitter or Uber, you need to understand the components that make them work. These fundamentals are the "vocabulary" of system design:

- **Interview Framework** teaches you HOW to approach any system design question
- **Estimation** sizes infrastructure before building
- **Networking** enables communication between services
- **Databases** persist your data reliably and at scale
- **Caches** store frequently accessed data for lightning-fast retrieval
- **Load Balancers** distribute traffic so no single server gets overwhelmed
- **API Design** defines how services communicate clearly and reliably
- **Concurrency** handles multiple operations simultaneously
- **Security** protects data and systems from threats
- **Scalability & Reliability** ensures systems handle growth and failures
- **Distributed Systems** coordinate work across multiple machines

---

## What's Covered

| Topic | Description | Difficulty |
|-------|-------------|------------|
| [Interview Framework]({{ site.baseurl }}/basics/interview_framework) | How to approach any system design question | ⭐ Beginner |
| [Estimation & Planning]({{ site.baseurl }}/basics/estimation) | Back-of-the-envelope calculations | ⭐⭐ Intermediate |
| [Networking]({{ site.baseurl }}/basics/networking) | TCP/UDP, HTTP, DNS, WebSockets | ⭐⭐ Intermediate |
| [Databases]({{ site.baseurl }}/basics/databases) | SQL vs NoSQL, ACID, CAP, scaling | ⭐⭐ Intermediate |
| [Caching]({{ site.baseurl }}/basics/caching) | Store data for faster access | ⭐ Beginner |
| [Load Balancing]({{ site.baseurl }}/basics/load_balancer) | Distribute requests across multiple servers | ⭐ Beginner |
| [API Design]({{ site.baseurl }}/basics/api_design) | REST, GraphQL, versioning, rate limiting | ⭐⭐ Intermediate |
| [Concurrency]({{ site.baseurl }}/basics/concurrency) | Threads, locks, async patterns | ⭐⭐ Intermediate |
| [Security]({{ site.baseurl }}/basics/security) | Encryption, hashing, TLS, vulnerabilities | ⭐⭐ Intermediate |
| [Scalability & Reliability]({{ site.baseurl }}/basics/scalability) | Scaling, availability, disaster recovery | ⭐⭐⭐ Advanced |
| [Distributed Systems]({{ site.baseurl }}/basics/distributed_systems) | CAP, consensus, DHTs, message queues | ⭐⭐⭐ Advanced |

---

## How to Study These

1. **Read through each topic** - Understand the concepts, not just the terms
2. **Know the trade-offs** - Every choice has pros and cons
3. **Be ready to apply them** - Interviewers will ask "which would you use here?"

{: .tip }
> When studying, ask yourself: "When would I use this? When would I NOT use this?"

---

## Quick Reference Card

Print this out or keep it handy:

```
LOAD BALANCING
├── Round Robin      → Simple, equal distribution
├── Least Connections → Best for varying request durations
├── IP Hash          → Session persistence (sticky sessions)
└── Health Checks    → Remove unhealthy servers automatically

CACHING
├── Cache-Aside      → App checks cache, then DB
├── Write-Through    → Write to cache AND DB together
├── Write-Behind     → Write cache, async to DB
└── Eviction: LRU, LFU, TTL

DATABASES
├── SQL              → ACID, relationships, complex queries
├── NoSQL            → Flexibility, horizontal scaling
├── Sharding         → Split data across multiple DBs
└── Replication      → Copy data for availability

NETWORKING
├── TCP              → Reliable, ordered (HTTP, databases)
├── UDP              → Fast, unreliable (video, gaming)
├── HTTP/REST        → Request-response, stateless
├── gRPC             → High-performance microservices
└── WebSocket        → Bidirectional, real-time

CONCURRENCY
├── Threads          → Shared memory, fast communication
├── Processes        → Isolated memory, crash safety
├── Mutex/Lock       → Exclusive access to resource
├── Thread Pool      → Reuse threads, bound resources
└── Async I/O        → Single thread, many I/O operations

DISTRIBUTED SYSTEMS
├── CAP Theorem      → Consistency vs Availability vs Partition Tolerance
├── Consensus        → Raft (understandable), Paxos (foundational)
├── Consistent Hash  → Minimal key remapping on node changes
└── Message Queues   → Kafka (log), RabbitMQ (broker), SQS (managed)

API DESIGN
├── REST             → Resource-oriented, HTTP semantics
├── GraphQL          → Client-specified queries, single endpoint
├── Versioning       → URI path (v1/v2), header, query param
└── Auth             → JWT (stateless), OAuth (delegated), API keys

SECURITY
├── Encryption       → AES (symmetric), RSA/ECDSA (asymmetric)
├── Hashing          → bcrypt/Argon2 (passwords), SHA-256 (integrity)
├── TLS              → Encrypt in transit, certificate chain
└── Input Validation → Parameterized queries, sanitization

SCALABILITY & RELIABILITY
├── Horizontal Scale → Stateless services + load balancer
├── Redundancy       → Active-active, active-passive
├── Circuit Breaker  → Stop cascading failures
└── Monitoring       → Latency, traffic, errors, saturation

ESTIMATION
├── Traffic          → DAU × actions/user ÷ 86,400 = QPS
├── Storage          → writes/day × size × 365 × replication
├── Bandwidth        → QPS × response_size
└── Rule of thumb    → 1M req/day ≈ 12 QPS
```

---

## Interview Cheat Sheet

| When They Ask About... | Think About... |
|------------------------|----------------|
| **Handling traffic** | Load balancing, horizontal scaling, auto-scaling |
| **Making it fast** | Caching, CDN, read replicas |
| **Storing data** | SQL vs NoSQL, consistency needs, sharding |
| **Real-time features** | WebSockets, message queues, pub/sub |
| **Handling failures** | Replication, circuit breakers, retries, failover |
| **Concurrent access** | Locks, optimistic concurrency, idempotency |
| **Multiple services** | Consensus protocols, leader election, DHTs |
| **API design** | REST vs GraphQL vs gRPC, versioning, pagination |
| **Protecting data** | TLS, encryption at rest, input validation, auth |
| **How big should it be?** | Back-of-envelope estimation, capacity planning |

---

## What's Next?

After mastering these fundamentals:

1. **Go deeper** with [Advanced Topics]({{ site.baseurl }}/advanced/) for Senior/Staff-level concepts
2. **Apply them** in [System Design Examples]({{ site.baseurl }}/software_system_design/) with 19 walkthroughs
3. **See ML-specific patterns** in [GenAI/ML Fundamentals]({{ site.baseurl }}/genai_ml_basics/) and [ML System Design]({{ site.baseurl }}/ml_system_design/)
4. **Practice** drawing architectures using these building blocks
