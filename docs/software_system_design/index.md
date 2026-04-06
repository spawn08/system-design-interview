# System Design Examples

Step-by-step walkthroughs of 27 system design interview questions — from classic URL shortener to distributed file systems.

---

## Recommended Study Order

!!! tip
    Don't study randomly. Follow this progression — each design builds on concepts from earlier ones.

### Tier 1: Foundation (Start Here)

| # | Design | Why First | Prerequisites |
|---|--------|-----------|---------------|
| 1 | [URL Shortener](url_shortening.md) | Simplest end-to-end design | Databases, Caching |
| 2 | [Rate Limiter](rate_limiter.md) | Core API protection pattern | Redis, distributed systems |
| 3 | [Distributed Cache](distributed_cache.md) | Appears in every other design | Consistent hashing |
| 4 | [Key-Value Store](key_value_store.md) | Deepens distributed systems understanding | Replication, consensus |

### Tier 2: Communication & Social

| # | Design | Builds On | New Concepts |
|---|--------|-----------|--------------|
| 5 | [Notification System](notification_system.md) | Rate Limiter, Cache | Multi-channel routing, delivery guarantees |
| 6 | [Chat System](chat_system.md) | Notification System | WebSockets, message ordering |
| 7 | [News Feed / Timeline](news_feed.md) | Chat, Cache | Fan-out strategies, ranking |
| 8 | [Voting System](voting-system-design.md) | URL Shortener | Idempotency, consistency |

### Tier 3: Media & Content

| # | Design | Builds On | New Concepts |
|---|--------|-----------|--------------|
| 9 | [Photo Sharing (Instagram)](photo_sharing.md) | News Feed, Cache | Object storage, CDN |
| 10 | [Video Streaming (YouTube)](video_streaming.md) | Photo Sharing | Transcoding, adaptive bitrate |
| 11 | [Cloud Storage (Google Drive)](cloud_storage.md) | Key-Value Store | File chunking, sync protocol |

### Tier 4: Real-time & Geospatial

| # | Design | Builds On | New Concepts |
|---|--------|-----------|--------------|
| 12 | [Collaborative Editor](collaborative_editor.md) | Chat System | OT/CRDTs, conflict resolution |
| 13 | [Proximity Service](proximity_service.md) | Cache, Databases | Geohash, quadtree |
| 14 | [Ride Sharing (Uber/Lyft)](ride_sharing.md) | Proximity Service | Real-time matching, ETA |

### Tier 5: Infrastructure & Commerce

| # | Design | Builds On | New Concepts |
|---|--------|-----------|--------------|
| 15 | [Task Scheduler](task_scheduler.md) | Distributed Cache, KV Store | Lease-based execution, priority queues |
| 16 | [Distributed Message Queue](message_queue.md) | Task Scheduler, KV Store | Append-only log, consumer groups, zero-copy I/O |
| 17 | [Event Booking (Ticketmaster)](event_booking.md) | Rate Limiter | Inventory locking, flash sales |
| 18 | [Payment System](payment_system.md) | Event Booking | Double-entry ledger, idempotency |
| 19 | [Metrics & Monitoring](metrics_monitoring.md) | Message Queue | Time-series storage, alerting, Gorilla compression |
| 20 | [Email Delivery System](email_delivery.md) | Notification System | SMTP, DKIM/SPF, IP reputation, deliverability |

### Tier 6: Data Infrastructure

| # | Design | Builds On | New Concepts |
|---|--------|-----------|--------------|
| 21 | [Distributed File System (GFS)](distributed_file_system.md) | KV Store, Cache | Master-chunk, leases, replication, append-only writes |
| 22 | [Ad Click Aggregator](ad_click_aggregator.md) | Message Queue, Metrics | Real-time aggregation, exactly-once, click fraud, reconciliation |

### Tier 7: Search

| # | Design | Builds On | New Concepts |
|---|--------|-----------|--------------|
| 23 | [Web Crawler](web_crawler.md) | Message Queue, Cache | URL frontier, politeness, dedup |
| 24 | [Search Autocomplete](search_autocomplete.md) | Cache, Web Crawler | Trie, ranking, type-ahead |

### Tier 8: Modern & Trending

| # | Design | Builds On | New Concepts |
|---|--------|-----------|--------------|
| 25 | [Gaming Leaderboard](gaming_leaderboard.md) | Cache, Message Queue | Redis sorted sets, real-time ranking, anti-cheat |
| 26 | [API Gateway](api_gateway.md) | Rate Limiter, Load Balancer | Plugin architecture, circuit breaker, hot config reload |
| 27 | [Content Delivery Network](content_delivery_network.md) | Distributed Cache, Cloud Storage | Edge caching, PoP hierarchy, Anycast routing |

---

## Staff Engineer (L6) Track

Preparing for a **Staff / Principal / L6** role? Start here.

!!! important
    At L6, the interviewer gives you a vague prompt and expects you to **define the problem, drive the whiteboard, and discuss multi-year evolution**. The designs below include dedicated "Staff Engineer Deep Dive" sections.

### Must-Read First

- [**Staff Engineer Interview Guide**](staff_engineer_expectations.md) - L5 vs L6 expectations, the 5 pillars, anti-patterns that get you down-leveled

### Priority Design Problems (80/20 Rule)

These 5 designs cover 80% of distributed systems concepts tested at L6:

| Design | Staff-Level Concepts Covered |
|--------|-----------------------------|
| [**Key-Value Store**](key_value_store.md) | CAP theorem, consistent hashing, quorum, vector clocks, Spanner/TrueTime, multi-region replication |
| [**Rate Limiter**](rate_limiter.md) | Global rate limiting, race conditions, cascading failures, load shedding, adaptive limits |
| [**Collaborative Editor**](collaborative_editor.md) | OT vs CRDTs decision framework, WebSocket scaling, hot document problem, multi-region |
| [**Task Scheduler**](task_scheduler.md) | Fencing tokens, zombie workers, multi-tenant fairness, cron correctness at scale |
| [**Notification System**](notification_system.md) | Exactly-once delivery chain, transactional outbox, load shedding |

### Supporting Advanced Topics

| Topic | Why It Matters for L6 |
|-------|-----------------------|
| [**Consensus Algorithms (Raft/Paxos)**](../advanced/consensus_algorithms.md) | Foundation for every strongly consistent system |
| [**Distributed Transactions (2PC/Saga/Outbox)**](../advanced/distributed_transactions.md) | Cross-service consistency patterns |
| [**Sharding & Partitioning**](../advanced/sharding_partitioning.md) | Partition key selection, hot spots, resharding |
| [**Behavioral & Leadership (L6)**](../advanced/behavioral_leadership.md) | The dealbreaker round: STAR stories, conflict resolution, technical vision |

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

!!! tip
    Practice drawing these designs on a whiteboard or paper. The physical act of drawing helps with memory and interview confidence.

---

## All Designs by Category

### Infrastructure & Data

| Design | Difficulty | Core Pattern |
|--------|-----------|--------------|
| [**URL Shortener**](url_shortening.md) | ⭐⭐ Medium | Hashing, distributed IDs |
| [**Rate Limiter**](rate_limiter.md) | ⭐⭐ Medium | Token bucket, sliding window |
| [**Key-Value Store**](key_value_store.md) | ⭐⭐⭐⭐ Hard | Consistent hashing, quorum |
| [**Distributed Cache**](distributed_cache.md) | ⭐⭐⭐ Medium-Hard | LRU, hot keys, stampede |
| [**Distributed Message Queue**](message_queue.md) | ⭐⭐⭐⭐ Hard | Append-only log, consumer groups |
| [**Task Scheduler**](task_scheduler.md) | ⭐⭐⭐ Medium-Hard | Priority queue, leases |
| [**Metrics & Monitoring**](metrics_monitoring.md) | ⭐⭐⭐⭐ Hard | Time-series, alerting |
| [**Distributed File System (GFS)**](distributed_file_system.md) | ⭐⭐⭐⭐ Hard | Master-chunk, leases, replication |
| [**Ad Click Aggregator**](ad_click_aggregator.md) | ⭐⭐⭐⭐ Hard | Real-time aggregation, exactly-once |

### Communication & Social

| Design | Difficulty | Core Pattern |
|--------|-----------|--------------|
| [**Chat System**](chat_system.md) | ⭐⭐⭐⭐ Hard | WebSockets, message ordering |
| [**Notification System**](notification_system.md) | ⭐⭐⭐ Medium-Hard | Multi-channel routing |
| [**News Feed / Timeline**](news_feed.md) | ⭐⭐⭐⭐ Hard | Fan-out strategies |
| [**Voting System**](voting-system-design.md) | ⭐⭐⭐ Medium-Hard | Idempotency, consistency |
| [**Email Delivery System**](email_delivery.md) | ⭐⭐⭐⭐ Hard | SMTP, DKIM/SPF, deliverability |

### Media & Content

| Design | Difficulty | Core Pattern |
|--------|-----------|--------------|
| [**Video Streaming (YouTube)**](video_streaming.md) | ⭐⭐⭐⭐ Hard | CDN, transcoding |
| [**Photo Sharing (Instagram)**](photo_sharing.md) | ⭐⭐⭐ Medium-Hard | Object storage, fan-out |
| [**Cloud Storage (Google Drive)**](cloud_storage.md) | ⭐⭐⭐⭐ Hard | File chunking, sync |

### Real-time & Geospatial

| Design | Difficulty | Core Pattern |
|--------|-----------|--------------|
| [**Collaborative Editor**](collaborative_editor.md) | ⭐⭐⭐⭐ Hard | OT/CRDTs, conflict resolution |
| [**Ride Sharing (Uber/Lyft)**](ride_sharing.md) | ⭐⭐⭐⭐ Hard | Geospatial matching |
| [**Proximity Service**](proximity_service.md) | ⭐⭐⭐ Medium-Hard | Geohash, quadtree |

### Commerce & Finance

| Design | Difficulty | Core Pattern |
|--------|-----------|--------------|
| [**Event Booking (Ticketmaster)**](event_booking.md) | ⭐⭐⭐ Medium-Hard | Inventory locking |
| [**Payment System**](payment_system.md) | ⭐⭐⭐⭐ Hard | Double-entry ledger |

### Search

| Design | Difficulty | Core Pattern |
|--------|-----------|--------------|
| [**Web Crawler**](web_crawler.md) | ⭐⭐⭐⭐ Hard | URL frontier, dedup |
| [**Search Autocomplete**](search_autocomplete.md) | ⭐⭐⭐ Medium-Hard | Trie, ranking |

### Modern & Trending

| Design | Difficulty | Core Pattern |
|--------|-----------|--------------|
| [**Gaming Leaderboard**](gaming_leaderboard.md) | ⭐⭐⭐ Medium-Hard | Redis sorted sets, real-time ranking |
| [**API Gateway**](api_gateway.md) | ⭐⭐⭐⭐ Hard | Plugin architecture, circuit breaker |
| [**Content Delivery Network**](content_delivery_network.md) | ⭐⭐⭐⭐ Hard | Edge caching, PoP hierarchy |

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

!!! note
    Master these patterns and you can apply them to any new problem the interviewer throws at you.

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

1. **Go deeper** with [Advanced Topics](../advanced/index.md) for Senior/Staff-level concepts
2. **Learn ML fundamentals** in [GenAI/ML Fundamentals](../genai_ml_basics/index.md) — 7 building blocks
3. **Tackle ML designs** in [ML System Design](../ml_system_design/index.md) — 10 production ML systems
4. **Master GenAI** in [GenAI System Design](../genai_ml_system_design/index.md) — 10 LLM/GenAI systems with interview transcripts
