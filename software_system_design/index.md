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

## Pattern Recognition

As you study these designs, look for patterns that repeat:

| Pattern | Where You'll See It |
|---------|---------------------|
| **Cache-aside** | URL Shortener, Rate Limiter, Distributed Cache - Redis for fast lookups |
| **Message Queue** | Voting System, Notification System - async processing |
| **Read Replicas** | URL Shortener, Voting System - separate read/write load |
| **Distributed IDs** | URL Shortener - Snowflake algorithm |
| **Idempotency** | Voting System, Notification System - prevent duplicates |
| **Rate Limiting** | Rate Limiter, Web Crawler - control request flow |
| **Priority Queues** | Notification System, Web Crawler - prioritize work |
| **Bloom Filters** | Web Crawler - memory-efficient set membership |

{: .note }
> Master these patterns and you can apply them to any new problem the interviewer throws at you.

---

## Quick Reference: Complexity

| Design | Read/Write Ratio | Scale Challenge | Core Trade-off |
|--------|------------------|-----------------|----------------|
| **URL Shortener** | 100:1 (read-heavy) | Billions of URLs | Consistency vs latency |
| **Voting System** | 1:1 | Thousands/sec spikes | Accuracy vs throughput |
| **Rate Limiter** | N/A | Millions of clients | Precision vs memory |
| **Web Crawler** | N/A | Billions of pages | Speed vs politeness |
| **Notification System** | Write-heavy | Millions/minute | Reliability vs latency |
| **News Feed / Timeline** | Read-heavy | Billions of feed loads/day | Fan-out cost vs read latency |
| **Chat System** | Write-heavy + connections | Millions concurrent sockets | Ordering vs fan-out cost |
| **Distributed Cache** | Read-heavy | Hot keys, memory pressure | Stale reads vs latency |
| **Search Autocomplete** | Read-heavy | Peak QPS, trie memory | Freshness vs latency |

