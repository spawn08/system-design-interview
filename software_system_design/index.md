---
layout: default
title: System Design Examples
nav_order: 4
has_children: true
permalink: /software_system_design/
---

# System Design Examples
{: .fs-9 }

Step-by-step walkthroughs of 27 system design interview questions — from classic URL shortener to distributed file systems.
{: .fs-6 .fw-300 }

---

## Recommended Study Order

{: .tip }
> Don't study randomly. Follow this progression — each design builds on concepts from earlier ones.

### Tier 1: Foundation (Start Here)

| # | Design | Why First | Prerequisites |
|---|--------|-----------|---------------|
| 1 | [URL Shortener]({{ site.baseurl }}/software_system_design/url_shortening) | Simplest end-to-end design | Databases, Caching |
| 2 | [Rate Limiter]({{ site.baseurl }}/software_system_design/rate_limiter) | Core API protection pattern | Redis, distributed systems |
| 3 | [Distributed Cache]({{ site.baseurl }}/software_system_design/distributed_cache) | Appears in every other design | Consistent hashing |
| 4 | [Key-Value Store]({{ site.baseurl }}/software_system_design/key_value_store) | Deepens distributed systems understanding | Replication, consensus |

### Tier 2: Communication & Social

| # | Design | Builds On | New Concepts |
|---|--------|-----------|--------------|
| 5 | [Notification System]({{ site.baseurl }}/software_system_design/notification_system) | Rate Limiter, Cache | Multi-channel routing, delivery guarantees |
| 6 | [Chat System]({{ site.baseurl }}/software_system_design/chat_system) | Notification System | WebSockets, message ordering |
| 7 | [News Feed / Timeline]({{ site.baseurl }}/software_system_design/news_feed) | Chat, Cache | Fan-out strategies, ranking |
| 8 | [Voting System]({{ site.baseurl }}/software_system_design/voting-system-design) | URL Shortener | Idempotency, consistency |

### Tier 3: Media & Content

| # | Design | Builds On | New Concepts |
|---|--------|-----------|--------------|
| 9 | [Photo Sharing (Instagram)]({{ site.baseurl }}/software_system_design/photo_sharing) | News Feed, Cache | Object storage, CDN |
| 10 | [Video Streaming (YouTube)]({{ site.baseurl }}/software_system_design/video_streaming) | Photo Sharing | Transcoding, adaptive bitrate |
| 11 | [Cloud Storage (Google Drive)]({{ site.baseurl }}/software_system_design/cloud_storage) | Key-Value Store | File chunking, sync protocol |

### Tier 4: Real-time & Geospatial

| # | Design | Builds On | New Concepts |
|---|--------|-----------|--------------|
| 12 | [Collaborative Editor]({{ site.baseurl }}/software_system_design/collaborative_editor) | Chat System | OT/CRDTs, conflict resolution |
| 13 | [Proximity Service]({{ site.baseurl }}/software_system_design/proximity_service) | Cache, Databases | Geohash, quadtree |
| 14 | [Ride Sharing (Uber/Lyft)]({{ site.baseurl }}/software_system_design/ride_sharing) | Proximity Service | Real-time matching, ETA |

### Tier 5: Infrastructure & Commerce

| # | Design | Builds On | New Concepts |
|---|--------|-----------|--------------|
| 15 | [Task Scheduler]({{ site.baseurl }}/software_system_design/task_scheduler) | Distributed Cache, KV Store | Lease-based execution, priority queues |
| 16 | [Distributed Message Queue]({{ site.baseurl }}/software_system_design/message_queue) | Task Scheduler, KV Store | Append-only log, consumer groups, zero-copy I/O |
| 17 | [Event Booking (Ticketmaster)]({{ site.baseurl }}/software_system_design/event_booking) | Rate Limiter | Inventory locking, flash sales |
| 18 | [Payment System]({{ site.baseurl }}/software_system_design/payment_system) | Event Booking | Double-entry ledger, idempotency |
| 19 | [Metrics & Monitoring]({{ site.baseurl }}/software_system_design/metrics_monitoring) | Message Queue | Time-series storage, alerting, Gorilla compression |
| 20 | [Email Delivery System]({{ site.baseurl }}/software_system_design/email_delivery) | Notification System | SMTP, DKIM/SPF, IP reputation, deliverability |

### Tier 6: Data Infrastructure

| # | Design | Builds On | New Concepts |
|---|--------|-----------|--------------|
| 21 | [Distributed File System (GFS)]({{ site.baseurl }}/software_system_design/distributed_file_system) | KV Store, Cache | Master-chunk, leases, replication, append-only writes |
| 22 | [Ad Click Aggregator]({{ site.baseurl }}/software_system_design/ad_click_aggregator) | Message Queue, Metrics | Real-time aggregation, exactly-once, click fraud, reconciliation |

### Tier 7: Search

| # | Design | Builds On | New Concepts |
|---|--------|-----------|--------------|
| 23 | [Web Crawler]({{ site.baseurl }}/software_system_design/web_crawler) | Message Queue, Cache | URL frontier, politeness, dedup |
| 24 | [Search Autocomplete]({{ site.baseurl }}/software_system_design/search_autocomplete) | Cache, Web Crawler | Trie, ranking, type-ahead |

### Tier 8: Modern & Trending

| # | Design | Builds On | New Concepts |
|---|--------|-----------|--------------|
| 25 | [Gaming Leaderboard]({{ site.baseurl }}/software_system_design/gaming_leaderboard) | Cache, Message Queue | Redis sorted sets, real-time ranking, anti-cheat |
| 26 | [API Gateway]({{ site.baseurl }}/software_system_design/api_gateway) | Rate Limiter, Load Balancer | Plugin architecture, circuit breaker, hot config reload |
| 27 | [Content Delivery Network]({{ site.baseurl }}/software_system_design/content_delivery_network) | Distributed Cache, Cloud Storage | Edge caching, PoP hierarchy, Anycast routing |

---

## Staff Engineer (L6) Track

Preparing for a **Staff / Principal / L6** role? Start here.

{: .important }
> At L6, the interviewer gives you a vague prompt and expects you to **define the problem, drive the whiteboard, and discuss multi-year evolution**. The designs below include dedicated "Staff Engineer Deep Dive" sections.

### Must-Read First

- [**Staff Engineer Interview Guide**]({{ site.baseurl }}/software_system_design/staff_engineer_expectations) - L5 vs L6 expectations, the 5 pillars, anti-patterns that get you down-leveled

### Priority Design Problems (80/20 Rule)

These 5 designs cover 80% of distributed systems concepts tested at L6:

| Design | Staff-Level Concepts Covered |
|--------|-----------------------------|
| [**Key-Value Store**]({{ site.baseurl }}/software_system_design/key_value_store) | CAP theorem, consistent hashing, quorum, vector clocks, Spanner/TrueTime, multi-region replication |
| [**Rate Limiter**]({{ site.baseurl }}/software_system_design/rate_limiter) | Global rate limiting, race conditions, cascading failures, load shedding, adaptive limits |
| [**Collaborative Editor**]({{ site.baseurl }}/software_system_design/collaborative_editor) | OT vs CRDTs decision framework, WebSocket scaling, hot document problem, multi-region |
| [**Task Scheduler**]({{ site.baseurl }}/software_system_design/task_scheduler) | Fencing tokens, zombie workers, multi-tenant fairness, cron correctness at scale |
| [**Notification System**]({{ site.baseurl }}/software_system_design/notification_system) | Exactly-once delivery chain, transactional outbox, load shedding |

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

## All Designs by Category

### Infrastructure & Data

| Design | Difficulty | Core Pattern |
|--------|-----------|--------------|
| [**URL Shortener**]({{ site.baseurl }}/software_system_design/url_shortening) | ⭐⭐ Medium | Hashing, distributed IDs |
| [**Rate Limiter**]({{ site.baseurl }}/software_system_design/rate_limiter) | ⭐⭐ Medium | Token bucket, sliding window |
| [**Key-Value Store**]({{ site.baseurl }}/software_system_design/key_value_store) | ⭐⭐⭐⭐ Hard | Consistent hashing, quorum |
| [**Distributed Cache**]({{ site.baseurl }}/software_system_design/distributed_cache) | ⭐⭐⭐ Medium-Hard | LRU, hot keys, stampede |
| [**Distributed Message Queue**]({{ site.baseurl }}/software_system_design/message_queue) | ⭐⭐⭐⭐ Hard | Append-only log, consumer groups |
| [**Task Scheduler**]({{ site.baseurl }}/software_system_design/task_scheduler) | ⭐⭐⭐ Medium-Hard | Priority queue, leases |
| [**Metrics & Monitoring**]({{ site.baseurl }}/software_system_design/metrics_monitoring) | ⭐⭐⭐⭐ Hard | Time-series, alerting |
| [**Distributed File System (GFS)**]({{ site.baseurl }}/software_system_design/distributed_file_system) | ⭐⭐⭐⭐ Hard | Master-chunk, leases, replication |
| [**Ad Click Aggregator**]({{ site.baseurl }}/software_system_design/ad_click_aggregator) | ⭐⭐⭐⭐ Hard | Real-time aggregation, exactly-once |

### Communication & Social

| Design | Difficulty | Core Pattern |
|--------|-----------|--------------|
| [**Chat System**]({{ site.baseurl }}/software_system_design/chat_system) | ⭐⭐⭐⭐ Hard | WebSockets, message ordering |
| [**Notification System**]({{ site.baseurl }}/software_system_design/notification_system) | ⭐⭐⭐ Medium-Hard | Multi-channel routing |
| [**News Feed / Timeline**]({{ site.baseurl }}/software_system_design/news_feed) | ⭐⭐⭐⭐ Hard | Fan-out strategies |
| [**Voting System**]({{ site.baseurl }}/software_system_design/voting-system-design) | ⭐⭐⭐ Medium-Hard | Idempotency, consistency |
| [**Email Delivery System**]({{ site.baseurl }}/software_system_design/email_delivery) | ⭐⭐⭐⭐ Hard | SMTP, DKIM/SPF, deliverability |

### Media & Content

| Design | Difficulty | Core Pattern |
|--------|-----------|--------------|
| [**Video Streaming (YouTube)**]({{ site.baseurl }}/software_system_design/video_streaming) | ⭐⭐⭐⭐ Hard | CDN, transcoding |
| [**Photo Sharing (Instagram)**]({{ site.baseurl }}/software_system_design/photo_sharing) | ⭐⭐⭐ Medium-Hard | Object storage, fan-out |
| [**Cloud Storage (Google Drive)**]({{ site.baseurl }}/software_system_design/cloud_storage) | ⭐⭐⭐⭐ Hard | File chunking, sync |

### Real-time & Geospatial

| Design | Difficulty | Core Pattern |
|--------|-----------|--------------|
| [**Collaborative Editor**]({{ site.baseurl }}/software_system_design/collaborative_editor) | ⭐⭐⭐⭐ Hard | OT/CRDTs, conflict resolution |
| [**Ride Sharing (Uber/Lyft)**]({{ site.baseurl }}/software_system_design/ride_sharing) | ⭐⭐⭐⭐ Hard | Geospatial matching |
| [**Proximity Service**]({{ site.baseurl }}/software_system_design/proximity_service) | ⭐⭐⭐ Medium-Hard | Geohash, quadtree |

### Commerce & Finance

| Design | Difficulty | Core Pattern |
|--------|-----------|--------------|
| [**Event Booking (Ticketmaster)**]({{ site.baseurl }}/software_system_design/event_booking) | ⭐⭐⭐ Medium-Hard | Inventory locking |
| [**Payment System**]({{ site.baseurl }}/software_system_design/payment_system) | ⭐⭐⭐⭐ Hard | Double-entry ledger |

### Search

| Design | Difficulty | Core Pattern |
|--------|-----------|--------------|
| [**Web Crawler**]({{ site.baseurl }}/software_system_design/web_crawler) | ⭐⭐⭐⭐ Hard | URL frontier, dedup |
| [**Search Autocomplete**]({{ site.baseurl }}/software_system_design/search_autocomplete) | ⭐⭐⭐ Medium-Hard | Trie, ranking |

### Modern & Trending

| Design | Difficulty | Core Pattern |
|--------|-----------|--------------|
| [**Gaming Leaderboard**]({{ site.baseurl }}/software_system_design/gaming_leaderboard) | ⭐⭐⭐ Medium-Hard | Redis sorted sets, real-time ranking |
| [**API Gateway**]({{ site.baseurl }}/software_system_design/api_gateway) | ⭐⭐⭐⭐ Hard | Plugin architecture, circuit breaker |
| [**Content Delivery Network**]({{ site.baseurl }}/software_system_design/content_delivery_network) | ⭐⭐⭐⭐ Hard | Edge caching, PoP hierarchy |

---

## Pattern Recognition

| Pattern | Where You'll See It |
|---------|---------------------|
| **Cache-aside** | URL Shortener, Rate Limiter, Distributed Cache, Proximity Service |
| **Message Queue** | Voting System, Notification, Video Streaming, Task Scheduler, Message Queue |
| **Read Replicas** | URL Shortener, Voting System, News Feed |
| **Distributed IDs** | URL Shortener, Payment System — Snowflake algorithm |
| **Idempotency** | Voting, Notification, Payment, Task Scheduler, Message Queue |
| **Rate Limiting** | Rate Limiter, Web Crawler, Event Booking (virtual queue) |
| **Consistent Hashing** | Key-Value Store, Distributed Cache — data partitioning |
| **WebSockets** | Chat System, Collaborative Editor, Ride Sharing |
| **Geospatial Indexing** | Ride Sharing, Proximity Service — geohash, quadtree |
| **Fan-out** | News Feed, Photo Sharing — push vs pull vs hybrid |
| **State Machine** | Payment System, Ride Sharing, Task Scheduler, Event Booking |
| **CDN** | Video Streaming, Photo Sharing, Cloud Storage |
| **Conflict Resolution** | Key-Value Store (vector clocks), Collaborative Editor (OT/CRDT) |
| **Append-Only Log** | Message Queue, Event Sourcing, Metrics & Monitoring, Distributed File System |
| **Stream Processing** | Ad Click Aggregator, Metrics & Monitoring — windowed aggregation |
| **Master-Worker** | Distributed File System, Task Scheduler — coordination patterns |
| **Sorted Sets / Skip Lists** | Gaming Leaderboard — real-time rank queries |
| **Circuit Breaker** | API Gateway, Notification System — fault isolation |
| **Edge Caching** | Content Delivery Network, Video Streaming — PoP hierarchy |
| **Plugin / Middleware** | API Gateway — extensible request processing |

{: .note }
> Master these patterns and you can apply them to any new problem the interviewer throws at you.

---

## Quick Reference: Complexity

| Design | Read/Write Ratio | Scale Challenge | Core Trade-off |
|--------|------------------|-----------------|----------------|
| **URL Shortener** | 100:1 (read-heavy) | Billions of URLs | Consistency vs latency |
| **Rate Limiter** | N/A | Millions of clients | Precision vs memory |
| **Key-Value Store** | Varies | Partitioning | Consistency vs availability |
| **Distributed Cache** | Read-heavy | Hot keys | Stale reads vs latency |
| **Message Queue** | Write-heavy | 1M+ msgs/sec | Ordering vs throughput |
| **Metrics & Monitoring** | Write-heavy | 100K metrics/sec | Granularity vs storage |
| **Notification System** | Write-heavy | Millions/minute | Reliability vs latency |
| **Web Crawler** | N/A | Billions of pages | Speed vs politeness |
| **Chat System** | Write-heavy | Millions sockets | Ordering vs fan-out |
| **News Feed** | Read-heavy | Billions loads/day | Fan-out cost vs read latency |
| **Search Autocomplete** | Read-heavy | Peak QPS | Freshness vs latency |
| **Voting System** | 1:1 | Spike traffic | Accuracy vs throughput |
| **Video Streaming** | Read-heavy | Petabytes of video | Storage cost vs quality |
| **Photo Sharing** | Read-heavy | Billions of images | Fan-out cost vs latency |
| **Collaborative Editor** | Write-heavy | Concurrent editors | Consistency vs responsiveness |
| **Ride Sharing** | Write-heavy | Millions of drivers | Speed vs optimality |
| **Cloud Storage** | Balanced | Petabytes | Sync speed vs bandwidth |
| **Event Booking** | Write-heavy (flash) | Spike traffic | Consistency vs throughput |
| **Task Scheduler** | Write-heavy | Millions of tasks | At-least-once vs exactly-once |
| **Payment System** | Write-heavy | Financial accuracy | Consistency vs availability |
| **Proximity Service** | Read-heavy | Millions locations | Precision vs query speed |
| **Distributed File System** | Write-heavy (append) | Petabytes | Single master vs availability |
| **Ad Click Aggregator** | Write-heavy | 1M events/sec | Exactness vs latency |
| **Gaming Leaderboard** | Read-heavy | 50K writes/sec, 500K reads/sec | Rank precision vs latency |
| **API Gateway** | Balanced | 100K RPS/node | Routing overhead vs feature richness |
| **Content Delivery Network** | Read-heavy | 10M RPS globally | Cache hit ratio vs freshness |

---

## What's Next?

After mastering these software system designs:

1. **Go deeper** with [Advanced Topics]({{ site.baseurl }}/advanced/) for Senior/Staff-level concepts
2. **Learn ML fundamentals** in [GenAI/ML Fundamentals]({{ site.baseurl }}/genai_ml_basics/) — 7 building blocks
3. **Tackle ML designs** in [ML System Design]({{ site.baseurl }}/ml_system_design/) — 10 production ML systems
4. **Master GenAI** in [GenAI System Design]({{ site.baseurl }}/genai_ml_system_design/) — 10 LLM/GenAI systems with interview transcripts
