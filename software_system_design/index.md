---
layout: default
title: System Design Examples
nav_order: 4
has_children: true
permalink: /software_system_design/
---

# System Design Examples
{: .fs-9 }

Step-by-step walkthroughs of the most common system design interview questions.
{: .fs-6 .fw-300 }

---

## Staff Engineer (L6) Track

Preparing for a **Staff / Principal / L6** role? Start here. These resources are specifically designed for the elevated expectations at Staff level.

{: .important }
> At L6, the interviewer gives you a vague prompt and expects you to **define the problem, drive the whiteboard, and discuss multi-year evolution**. The designs below include dedicated "Staff Engineer Deep Dive" sections with multi-region strategies, operational excellence (SLOs), failure analysis, and system evolution.

### Must-Read First

- [**Staff Engineer Interview Guide**]({{ site.baseurl }}/software_system_design/staff_engineer_expectations) - L5 vs L6 expectations, the 5 pillars, anti-patterns that get you down-leveled

### Priority Design Problems (80/20 Rule)

These 5 designs cover 80% of distributed systems concepts tested at L6:

| Design | Staff-Level Concepts Covered |
|--------|-----------------------------|
| [**Key-Value Store**]({{ site.baseurl }}/software_system_design/key_value_store) | CAP theorem, consistent hashing, quorum, vector clocks, Spanner/TrueTime, multi-region replication |
| [**Rate Limiter**]({{ site.baseurl }}/software_system_design/rate_limiter) | Global rate limiting, race conditions, cascading failures, load shedding, adaptive limits |
| [**Collaborative Editor**]({{ site.baseurl }}/software_system_design/collaborative_editor) | OT vs CRDTs decision framework, WebSocket scaling, hot document problem, multi-region deployment |
| [**Task Scheduler**]({{ site.baseurl }}/software_system_design/task_scheduler) | Fencing tokens, zombie workers, multi-tenant fairness, cron correctness at scale |
| [**Notification System**]({{ site.baseurl }}/software_system_design/notification_system) | Exactly-once delivery chain, transactional outbox, load shedding, notification aggregation |

### Supporting Advanced Topics

| Topic | Why It Matters for L6 |
|-------|-----------------------|
| [**Consensus Algorithms (Raft/Paxos)**]({{ site.baseurl }}/advanced/consensus_algorithms) | Foundation for every strongly consistent system |
| [**Distributed Transactions (2PC/Saga/Outbox)**]({{ site.baseurl }}/advanced/distributed_transactions) | Cross-service consistency patterns |
| [**Sharding & Partitioning**]({{ site.baseurl }}/advanced/sharding_partitioning) | Partition key selection, hot spots, resharding |
| [**Behavioral & Leadership (L6)**]({{ site.baseurl }}/advanced/behavioral_leadership) | The dealbreaker round: STAR stories, conflict resolution, technical vision |

---

## How to Use These Examples

Each example follows a consistent structure that mirrors what interviewers expect:

| Phase | What You'll Learn | Time in Interview |
|-------|-------------------|-------------------|
| 1. Requirements | Clarifying questions to ask | 5 minutes |
| 2. Estimation | Back-of-envelope calculations | 5 minutes |
| 3. High-Level Design | Architecture overview | 10 minutes |
| 4. Deep Dive | Key components in detail | 15 minutes |
| 5. Scaling & Trade-offs | Production considerations | 5 minutes |

{: .tip }
> Practice drawing these designs on a whiteboard or paper. The physical act of drawing helps with memory and interview confidence.

---

## Available Designs

### [Staff Engineer Interview Guide]({{ site.baseurl }}/software_system_design/staff_engineer_expectations)
{: .d-inline-block }

Staff L6
{: .label .label-red }

The definitive guide for Staff / L6 system design interviews. Covers L5 vs L6 expectations, the 5 pillars of a Staff-level answer, and anti-patterns that get you down-leveled.

**Key concepts:** Multi-region architecture, SLOs/SLIs, system evolution, driving consensus, CAP positioning

**Difficulty:** N/A (Meta-guide)

---

### [URL Shortener (TinyURL)]({{ site.baseurl }}/software_system_design/url_shortening)
{: .d-inline-block }

Classic
{: .label .label-green }

Design a service that converts long URLs into short, shareable links.

**Key concepts:** Hashing, Base62 encoding, distributed ID generation (Snowflake), caching, database sharding

**Difficulty:** ⭐⭐ Medium

---

### [Voting System]({{ site.baseurl }}/software_system_design/voting-system-design)
{: .d-inline-block }

Real-world
{: .label .label-blue }

Design a scalable platform for creating and participating in polls/elections.

**Key concepts:** Consistency, duplicate prevention (3-layer defense), async processing with Kafka, idempotency, real-time results

**Difficulty:** ⭐⭐⭐ Medium-Hard

---

### [Rate Limiter]({{ site.baseurl }}/software_system_design/rate_limiter)
{: .d-inline-block }

Essential
{: .label .label-yellow }

Design a distributed rate limiter to protect APIs from abuse.

**Key concepts:** Token bucket, sliding window counter, Redis atomic operations, Lua scripts, distributed systems challenges

**Difficulty:** ⭐⭐ Medium

---

### [Web Crawler]({{ site.baseurl }}/software_system_design/web_crawler)
{: .d-inline-block }

Complex
{: .label .label-purple }

Design a web crawler to index the internet at scale.

**Key concepts:** URL frontier, politeness, robots.txt, deduplication (Bloom filter, SimHash), distributed crawling, freshness

**Difficulty:** ⭐⭐⭐⭐ Hard

---

### [Notification System]({{ site.baseurl }}/software_system_design/notification_system)
{: .d-inline-block }

Multi-channel
{: .label .label-blue }

Design a system to send push notifications, emails, SMS, and in-app messages.

**Key concepts:** Multi-channel routing, user preferences, template systems, provider integration (FCM, APNS, SES, Twilio), rate limiting, analytics

**Difficulty:** ⭐⭐⭐ Medium-Hard

---

### [Chat System]({{ site.baseurl }}/software_system_design/chat_system)
{: .d-inline-block }

Realtime
{: .label .label-green }

Design a real-time messaging platform for 1:1 and group chat with presence, delivery semantics, and push.

**Key concepts:** WebSockets, stateful gateways, Kafka ordering, Cassandra message storage, Redis presence, fan-out on write vs read, idempotency, FCM/APNs

**Difficulty:** ⭐⭐⭐⭐ Hard

---

### [News Feed / Timeline]({{ site.baseurl }}/software_system_design/news_feed)
{: .d-inline-block }

Social
{: .label .label-purple }

Design a social media feed: fan-out (push vs pull vs hybrid), ranking, Redis feed caches, celebrity problem, media and real-time updates.

**Key concepts:** Fan-out on write/read, Kafka workers, sorted-set feeds, ML ranking, CDN, WebSockets/SSE, sharding

**Difficulty:** ⭐⭐⭐⭐ Hard

---

### [Search Autocomplete]({{ site.baseurl }}/software_system_design/search_autocomplete)
{: .d-inline-block }

Search
{: .label .label-blue }

Design Google-like typeahead: trie serving, analytics pipelines, ranking, personalization, and multi-tier caching.

**Key concepts:** Prefix trie / radix tree, top-K heaps, Kafka + Spark/Flink aggregates, atomic trie snapshots, Redis/CDN caching, sharding, safety filters

**Difficulty:** ⭐⭐⭐ Medium-Hard

---

### [Distributed Cache (Redis / Memcached–style)]({{ site.baseurl }}/software_system_design/distributed_cache)
{: .d-inline-block }

Infrastructure
{: .label .label-yellow }

Design an in-memory distributed cache with sharding, replication, eviction, and persistence.

**Key concepts:** Consistent hashing, LRU/LFU, cache-aside vs write-through, Redis Cluster slots, replication, RDB/AOF, hot keys, stampede mitigation

**Difficulty:** ⭐⭐⭐ Medium-Hard

---

### [Video Streaming (YouTube)]({{ site.baseurl }}/software_system_design/video_streaming)
{: .d-inline-block }

Media
{: .label .label-purple }

Design a video streaming platform with upload, transcoding, and adaptive bitrate delivery.

**Key concepts:** Chunked upload, FFmpeg transcoding, HLS/DASH adaptive streaming, CDN, DAG-based processing pipeline, deduplication

**Difficulty:** ⭐⭐⭐⭐ Hard

---

### [Photo Sharing (Instagram)]({{ site.baseurl }}/software_system_design/photo_sharing)
{: .d-inline-block }

Social
{: .label .label-green }

Design a photo sharing platform with upload, feed generation, and stories.

**Key concepts:** Object storage (S3), CDN, image resizing/thumbnails, fan-out on write vs read, celebrity problem, feed ranking, sharding

**Difficulty:** ⭐⭐⭐ Medium-Hard

---

### [Collaborative Editor (Google Docs)]({{ site.baseurl }}/software_system_design/collaborative_editor)
{: .d-inline-block }

Realtime
{: .label .label-green }

Design a real-time collaborative editing system with conflict resolution.

**Key concepts:** OT vs CRDTs, WebSocket connection management, document versioning, cursor/presence tracking, offline support

**Difficulty:** ⭐⭐⭐⭐ Hard

---

### [Ride Sharing (Uber/Lyft)]({{ site.baseurl }}/software_system_design/ride_sharing)
{: .d-inline-block }

Geospatial
{: .label .label-blue }

Design a ride-sharing platform with real-time matching and tracking.

**Key concepts:** Geospatial indexing (geohash, quadtree, S2), driver-rider matching, ETA computation, trip state machine, surge pricing

**Difficulty:** ⭐⭐⭐⭐ Hard

---

### [Cloud Storage (Google Drive)]({{ site.baseurl }}/software_system_design/cloud_storage)
{: .d-inline-block }

Storage
{: .label .label-yellow }

Design a cloud file storage and sync service.

**Key concepts:** File chunking (Rabin fingerprint), content-addressable deduplication, sync protocol, metadata service, conflict resolution, versioning

**Difficulty:** ⭐⭐⭐⭐ Hard

---

### [Event Booking (Ticketmaster)]({{ site.baseurl }}/software_system_design/event_booking)
{: .d-inline-block }

E-commerce
{: .label .label-blue }

Design a ticket booking system that handles flash sales and prevents overselling.

**Key concepts:** Seat inventory locking (optimistic/pessimistic), virtual waiting room, payment timeout, idempotency, preventing double booking

**Difficulty:** ⭐⭐⭐ Medium-Hard

---

### [Task Scheduler]({{ site.baseurl }}/software_system_design/task_scheduler)
{: .d-inline-block }

Infrastructure
{: .label .label-yellow }

Design a distributed task scheduler for delayed and recurring tasks.

**Key concepts:** Priority queue (min-heap), timing wheel, lease-based execution, cron expressions, dead letter queue, idempotency

**Difficulty:** ⭐⭐⭐ Medium-Hard

---

### [Key-Value Store]({{ site.baseurl }}/software_system_design/key_value_store)
{: .d-inline-block }

Infrastructure
{: .label .label-yellow }

Design a distributed key-value store (Dynamo-style).

**Key concepts:** Consistent hashing, quorum (W+R>N), vector clocks, gossip protocol, LSM tree, Merkle trees, read repair

**Difficulty:** ⭐⭐⭐⭐ Hard

---

### [Payment System]({{ site.baseurl }}/software_system_design/payment_system)
{: .d-inline-block }

Financial
{: .label .label-red }

Design a payment processing system with idempotency and compliance.

**Key concepts:** Payment state machine, idempotency keys, double-entry ledger, PSP integration, reconciliation, PCI-DSS compliance

**Difficulty:** ⭐⭐⭐⭐ Hard

---

### [Proximity Service]({{ site.baseurl }}/software_system_design/proximity_service)
{: .d-inline-block }

Geospatial
{: .label .label-blue }

Design a service to find nearby places (restaurants, gas stations, etc.).

**Key concepts:** Geohash, quadtree, S2 geometry, PostGIS, Redis GEO, bounding box search, Haversine distance, caching by geohash prefix

**Difficulty:** ⭐⭐⭐ Medium-Hard

---

## Pattern Recognition

As you study these designs, look for patterns that repeat:

| Pattern | Where You'll See It |
|---------|---------------------|
| **Cache-aside** | URL Shortener, Rate Limiter, Distributed Cache, Proximity Service |
| **Message Queue** | Voting System, Notification System, Video Streaming, Task Scheduler |
| **Read Replicas** | URL Shortener, Voting System, News Feed |
| **Distributed IDs** | URL Shortener, Payment System - Snowflake algorithm |
| **Idempotency** | Voting System, Notification System, Payment System, Task Scheduler |
| **Rate Limiting** | Rate Limiter, Web Crawler, Event Booking (virtual queue) |
| **Consistent Hashing** | Key-Value Store, Distributed Cache - data partitioning |
| **WebSockets** | Chat System, Collaborative Editor, Ride Sharing - real-time updates |
| **Geospatial Indexing** | Ride Sharing, Proximity Service - geohash, quadtree |
| **Fan-out** | News Feed, Photo Sharing - push vs pull vs hybrid |
| **State Machine** | Payment System, Ride Sharing, Task Scheduler, Event Booking |
| **CDN** | Video Streaming, Photo Sharing, Cloud Storage - edge delivery |
| **Conflict Resolution** | Key-Value Store (vector clocks), Collaborative Editor (OT/CRDT), Cloud Storage |
| **Bloom Filters** | Web Crawler, Key-Value Store - memory-efficient set membership |

{: .note }
> Master these patterns and you can apply them to any new problem the interviewer throws at you.

---

## Quick Reference: Complexity

| Design | Read/Write Ratio | Scale Challenge | Core Trade-off |
|--------|------------------|-----------------|----------------|
| **URL Shortener** | 100:1 (read-heavy) | Billions of URLs | Consistency vs latency |
| **Rate Limiter** | N/A | Millions of clients | Precision vs memory |
| **Key-Value Store** | Varies | Partitioning, replication | Consistency vs availability |
| **Distributed Cache** | Read-heavy | Hot keys, memory pressure | Stale reads vs latency |
| **Notification System** | Write-heavy | Millions/minute | Reliability vs latency |
| **Web Crawler** | N/A | Billions of pages | Speed vs politeness |
| **Chat System** | Write-heavy + connections | Millions concurrent sockets | Ordering vs fan-out cost |
| **News Feed / Timeline** | Read-heavy | Billions of feed loads/day | Fan-out cost vs read latency |
| **Search Autocomplete** | Read-heavy | Peak QPS, trie memory | Freshness vs latency |
| **Voting System** | 1:1 | Thousands/sec spikes | Accuracy vs throughput |
| **Video Streaming** | Read-heavy | Petabytes of video | Storage cost vs quality |
| **Photo Sharing** | Read-heavy | Billions of images | Fan-out cost vs feed latency |
| **Collaborative Editor** | Write-heavy | Concurrent editors | Consistency vs responsiveness |
| **Ride Sharing** | Write-heavy (locations) | Millions of drivers | Matching speed vs optimality |
| **Cloud Storage** | Balanced | Petabytes, file sync | Sync speed vs bandwidth |
| **Event Booking** | Write-heavy (flash) | Thousands/sec spikes | Consistency vs throughput |
| **Task Scheduler** | Write-heavy | Millions of tasks | At-least-once vs exactly-once |
| **Payment System** | Write-heavy | Financial accuracy | Consistency vs availability |
| **Proximity Service** | Read-heavy | Millions of locations | Precision vs query speed |

