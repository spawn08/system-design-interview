---
layout: default
title: System Design Examples
nav_order: 3
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
| 2. High-Level Design | Architecture overview | 10 minutes |
| 3. Deep Dive | Key components in detail | 15 minutes |
| 4. Scaling & Trade-offs | Production considerations | 5 minutes |

{: .tip }
> Practice drawing these designs on a whiteboard or paper. The physical act of drawing helps with memory and interview confidence.

---

## Available Designs

### [URL Shortener (TinyURL)](url_shortening.md)
{: .d-inline-block }

Classic
{: .label .label-green }

Design a service that converts long URLs into short, shareable links.

**Key concepts:** Hashing, Base62 encoding, distributed ID generation, caching

**Difficulty:** ⭐⭐ Medium

---

### [Voting System](voting-system-design.md)
{: .d-inline-block }

Real-world
{: .label .label-blue }

Design a scalable platform for creating and participating in polls/elections.

**Key concepts:** Consistency, duplicate prevention, async processing, security

**Difficulty:** ⭐⭐⭐ Medium-Hard

---

## Coming Soon

- **Rate Limiter** - Protect your APIs from abuse
- **Web Crawler** - Index the internet at scale
- **Notification System** - Push notifications to millions
- **Chat System** - Real-time messaging architecture

---

## Pattern Recognition

As you study these designs, look for patterns that repeat:

| Pattern | Where You'll See It |
|---------|---------------------|
| **Cache-aside** | URL Shortener, most read-heavy systems |
| **Message Queue** | Voting System, any async processing |
| **Read Replicas** | Both designs - separating read/write load |
| **Distributed IDs** | URL Shortener - Snowflake algorithm |
| **Idempotency** | Voting System - prevent duplicate votes |

{: .note }
> Master these patterns and you can apply them to any new problem the interviewer throws at you.

