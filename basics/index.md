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

- **Load Balancers** distribute traffic so no single server gets overwhelmed
- **Caches** store frequently accessed data for lightning-fast retrieval  
- **Databases** persist your data reliably and at scale
- **Networking** enables communication between services
- **Concurrency** handles multiple operations simultaneously

---

## What's Covered

| Topic | Description | Difficulty |
|-------|-------------|------------|
| [Load Balancing](load_balancer.md) | Distribute requests across multiple servers | ⭐ Beginner |
| [Caching](caching.md) | Store data for faster access | ⭐ Beginner |
| [Databases](databases.md) | SQL vs NoSQL, ACID, CAP, scaling | ⭐⭐ Intermediate |
| [Networking](networking.md) | TCP/UDP, HTTP, DNS, WebSockets | ⭐⭐ Intermediate |
| [Concurrency](concurrency.md) | Threads, locks, async patterns | ⭐⭐ Intermediate |

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
```

---

## Interview Cheat Sheet

| When They Ask About... | Think About... |
|------------------------|----------------|
| **Handling traffic** | Load balancing, horizontal scaling |
| **Making it fast** | Caching, CDN, read replicas |
| **Storing data** | SQL vs NoSQL, consistency needs |
| **Real-time features** | WebSockets, message queues |
| **Handling failures** | Replication, health checks, retries |
| **Concurrent access** | Locks, optimistic concurrency, idempotency |

---

## What's Next?

After mastering these fundamentals:

1. **Apply them** in [System Design Examples](../software_system_design/)
2. **See ML-specific patterns** in [ML System Design](../ml_system_design/)
3. **Practice** drawing architectures using these building blocks
