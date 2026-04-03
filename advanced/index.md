---
layout: default
title: Advanced Topics
nav_order: 3
has_children: true
permalink: /advanced/
---

# Advanced Topics
{: .fs-9 }

Deep dives into the building blocks of large-scale, production-grade distributed systems — essential for Senior and Staff-level interviews.
{: .fs-6 .fw-300 }

---

## Why Advanced Topics?

Once you've mastered the fundamentals, interviewers at Senior/Staff level expect you to reason about the **systems behind the systems**. These aren't just theoretical — every modern tech company runs on message queues, search engines, data pipelines, and microservice architectures.

---

## What's Covered

| Topic | Description | Difficulty |
|-------|-------------|------------|
| [Message Queues & Stream Processing]({{ site.baseurl }}/advanced/message_queues) | Kafka, RabbitMQ, Flink, event-driven architecture | ⭐⭐⭐ Advanced |
| [Search Systems]({{ site.baseurl }}/advanced/search_systems) | Inverted indexes, Elasticsearch, ranking, autocomplete | ⭐⭐⭐ Advanced |
| [Data Warehousing & Data Lakes]({{ site.baseurl }}/advanced/data_warehousing) | ETL, star schema, Hadoop, Spark, lakehouse | ⭐⭐⭐ Advanced |
| [Microservices Architecture]({{ site.baseurl }}/advanced/microservices) | Service discovery, API gateways, Docker, Kubernetes | ⭐⭐⭐ Advanced |
| [Consistency Patterns]({{ site.baseurl }}/advanced/consistency_patterns) | Strong, eventual, causal consistency, CRDTs, sagas | ⭐⭐⭐⭐ Expert |
| [Object Storage & CDN]({{ site.baseurl }}/advanced/object_storage_cdn) | S3-style storage, edge caching, presigned URLs, streaming | ⭐⭐⭐ Advanced |
| [Distributed Locking]({{ site.baseurl }}/advanced/distributed_locking) | Redlock, fencing tokens, ZooKeeper, lease-based locks | ⭐⭐⭐ Advanced |
| [Observability]({{ site.baseurl }}/advanced/observability) | Logging, metrics, tracing, OpenTelemetry, alerting | ⭐⭐⭐ Advanced |
| [Event Sourcing & CQRS]({{ site.baseurl }}/advanced/event_sourcing_cqrs) | Append-only logs, projections, read/write separation | ⭐⭐⭐⭐ Expert |

---

## How These Relate to Interviews

| When They Ask... | You Need... |
|------------------|-------------|
| "How do services communicate asynchronously?" | Message Queues & Stream Processing |
| "How would you implement search/autocomplete?" | Search Systems |
| "How do you handle analytics at scale?" | Data Warehousing & Data Lakes |
| "How do you decompose a monolith?" | Microservices Architecture |
| "How do you keep data consistent across services?" | Consistency Patterns |
| "How do you serve images/video globally?" | Object Storage & CDN |
| "How do you prevent race conditions across services?" | Distributed Locking |
| "How do you monitor and debug distributed systems?" | Observability |
| "How do you maintain a complete audit trail?" | Event Sourcing & CQRS |

---

## Prerequisites

You should be comfortable with the [Fundamentals]({{ site.baseurl }}/basics/) before tackling these topics, especially:

- **Distributed Systems** — CAP theorem, consensus protocols
- **Databases** — SQL vs NoSQL, replication, sharding
- **Networking** — HTTP, gRPC, WebSockets
- **Scalability & Reliability** — Horizontal scaling, circuit breakers

{: .tip }
> At Senior/Staff level, interviewers care less about knowing the "right answer" and more about your ability to reason through trade-offs, articulate why you'd choose one approach over another, and identify failure modes before they become production incidents.

---

## Quick Reference Card

```
MESSAGE QUEUES & STREAMING
├── Kafka         → Distributed log, replay, partitions, consumer groups
├── RabbitMQ      → AMQP broker, routing, dead-letter queues
├── Flink         → Stateful stream processing, event time, windows
└── Patterns      → Event sourcing, CQRS, saga

SEARCH SYSTEMS
├── Inverted Index → Term → [doc1, doc5, doc9] mapping
├── Elasticsearch  → Distributed search, shards, relevance scoring
├── Ranking        → TF-IDF, BM25, learning-to-rank
└── Autocomplete   → Trie, prefix matching, popularity weighting

DATA WAREHOUSING & DATA LAKES
├── ETL/ELT       → Extract, Transform, Load pipelines
├── Star Schema   → Fact + dimension tables, denormalized
├── Data Lake     → Raw storage (S3/HDFS), schema-on-read
└── Lakehouse     → Best of warehouse + lake (Delta, Iceberg)

MICROSERVICES
├── Discovery     → Consul, Eureka, DNS-based
├── API Gateway   → Routing, auth, rate limiting, aggregation
├── Resilience    → Circuit breaker, bulkhead, retry, timeout
└── Orchestration → Docker containers, Kubernetes pods/services

CONSISTENCY PATTERNS
├── Strong        → Linearizability, Raft/Paxos, 2PC
├── Eventual      → Async replication, conflict resolution
├── Causal        → Vector clocks, session guarantees
├── CRDTs         → Conflict-free replicated data types
└── Sagas         → Distributed transactions via compensation

OBJECT STORAGE & CDN
├── Object store  → Buckets, keys, multipart, lifecycle tiers
├── CDN           → Edge cache, TTL, purge, origin pull
├── Access        → Presigned URLs, OAC, bucket policies, CORS
└── Patterns      → Static assets, HLS/DASH, API cache at edge

DISTRIBUTED LOCKING
├── Redis         → SET NX PX, Redlock (multi-node)
├── ZooKeeper     → Ephemeral sequential nodes, watches
├── Database      → SELECT FOR UPDATE, advisory locks
├── Fencing       → Monotonic tokens prevent stale locks
└── etcd          → Lease-based, compare-and-swap

OBSERVABILITY
├── Logging       → Structured (JSON), ELK, correlation IDs
├── Metrics       → RED (services), USE (resources), Prometheus
├── Tracing       → OpenTelemetry, Jaeger, span context propagation
└── Alerting      → Symptom-based, error budgets, runbooks

EVENT SOURCING & CQRS
├── Events        → Immutable, append-only, versioned schema
├── Projections   → Materialized views, async rebuild
├── CQRS          → Separate read/write models, eventual consistency
└── Use cases     → Financial ledger, order lifecycle, audit trail
```
