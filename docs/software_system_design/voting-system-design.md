# Design a Voting/Polling System

---

## What We're Building

A voting system lets users create polls and cast votes. While the concept is simple, building a reliable voting system at scale presents fascinating distributed systems challenges.

### Types of Voting Systems

| Type | Examples | Key Challenges |
|------|----------|----------------|
| **Casual polls** | Twitter polls, Strawpoll | High throughput, real-time results |
| **Content voting** | Reddit upvotes, Stack Overflow | Preventing gaming, vote velocity |
| **Corporate elections** | Board votes, shareholder meetings | Auditability, strict access control |
| **Government elections** | Presidential elections | Security, anonymity, auditing |

The technical requirements vary dramatically:
- A Twitter poll needs to handle millions of votes per second with real-time results
- A corporate election needs ironclad auditability with restricted access
- A government election needs cryptographic verification while preserving anonymity

### Core Challenge

The fundamental problem in voting systems is: **How do you ensure each person votes exactly once while handling massive concurrent traffic?**

This seems simple but becomes complex when:
- Thousands of people vote in the same second
- Network requests can be duplicated (retries, timeouts)
- Databases can have race conditions
- Malicious users try to vote multiple times

---

## Step 1: Requirements Clarification

### Questions to Ask

| Question | Impact on Design |
|----------|------------------|
| What type of voting? (polls, elections, upvotes) | Determines security and anonymity needs |
| Who can vote? (public, authenticated, specific groups) | Authentication complexity |
| How strict is "one vote per person"? | Duplicate prevention strategy |
| Real-time results or only after voting ends? | Caching and consistency approach |
| Can votes be changed after casting? | Complicates uniqueness guarantees |
| What's the expected scale? | Infrastructure sizing |

### Our Design: General-Purpose Polling Platform

Let's design a system similar to **Strawpoll** or **Twitter Polls**:

**Functional Requirements:**

| Feature | Priority | Description |
|---------|----------|-------------|
| User registration/login | Must have | Identify voters |
| Create polls | Must have | Title, options, duration |
| Cast one vote per poll | Must have | Core functionality |
| View results | Must have | After poll closes or real-time |
| Custom poll settings | Nice to have | Anonymous, allow result viewing |
| Poll expiration | Nice to have | Auto-close after deadline |

**Non-Functional Requirements:**

| Requirement | Target | Rationale |
|-------------|--------|-----------|
| **Availability** | 99.99% | People expect voting to work |
| **Latency** | < 500ms for voting | Responsive UX |
| **Accuracy** | Zero lost/duplicate votes | Data integrity |
| **Throughput** | 10,000+ votes/sec (peak) | Handle viral polls |
| **Consistency** | No duplicate votes ever | System correctness |

---

## Technology Selection & Tradeoffs

Choosing storage, aggregation, deduplication, and rate limiting is not “pick the fastest”—each option trades correctness, ops cost, and failure behavior. Interviewers want to hear **why** a choice fits *this* workload (burst writes, exactly-once semantics per user/poll).

### Vote storage (PostgreSQL vs Redis vs DynamoDB vs Cassandra)

| Store | Strengths | Weaknesses | Fit for high-burst vote writes |
|-------|-----------|------------|--------------------------------|
| **PostgreSQL** | ACID, `UNIQUE` constraints, rich queries, familiar ops | Single-primary write path; needs pooling/sharding at extreme scale | **Strong default** when dedup must be authoritative; use async ingestion + batching to smooth bursts |
| **Redis** | Extremely fast writes | Not a durable system of record; persistence modes add complexity; cluster split-brain risks | **Cache + fast dedup flag**, not sole source of truth for “did this user vote?” in production |
| **DynamoDB** | Managed, horizontal scale, predictable at high TPS | Conditional writes / idempotency patterns needed; hot partitions on `poll_id` | **Good** if you partition keys carefully (e.g. `user_id`+`poll_id` composite, avoid mega-hot polls on one key without sharding strategy) |
| **Cassandra** | Write-optimized, wide-column | No cheap cross-partition uniqueness—**duplicate prevention is harder**; LWT adds latency | **Risky** for strict one-vote-per-user unless you model uniqueness in application layer or use strong patterns per partition |

**Why it matters:** Duplicate votes are a **consistency** problem. The durable store must enforce `(user_id, poll_id)` uniqueness. Redis/Dynamo/Cassandra can participate, but the *guarantee* usually lives in PostgreSQL (or Dynamo conditional writes with careful key design).

### Count aggregation (real-time counter vs batch vs CRDT)

| Approach | Pros | Cons | When to use |
|----------|------|------|-------------|
| **Real-time counter** (Redis `HINCRBY`, DB column++) | Low read latency; great UX for live results | Lost increments if not paired with durable write/replay; ordering issues under retries | Live polls **after** vote is durably accepted or idempotently applied |
| **Batch aggregation** (Spark/Flink/scheduled SQL) | Simple mental model; good for analytics | High **lag**; poor for “live” leaderboard | Reporting, reconciliation, historical analytics |
| **CRDT / distributed counters** | Nice for highly partitioned eventually consistent counts | Overkill for most polls; still need **authoritative** dedup elsewhere | Specialized multi-region *display* layers—not for replacing vote integrity |

**Why it matters:** Counters can be **eventually consistent** for *display*; **deduplication cannot**. Often: Kafka + worker updates Redis for speed, DB remains source of truth, periodic reconciliation fixes drift.

### Deduplication (DB constraint vs Redis set vs Bloom filter)

| Mechanism | Behavior | False negatives? | False positives? | Notes |
|-----------|----------|-------------------|-------------------|--------|
| **DB `UNIQUE (user_id, poll_id)`** | Strong guarantee | No | No | **Source of truth**; handles races via transaction/constraint violation |
| **Redis `SET` / `SETNX`** | Fast “already voted?” | No (if hit) | No | **Cache**; can be wrong on eviction/crash unless TTL and recovery sync from DB |
| **Bloom filter** | Memory-efficient membership test | No | **Yes** (may say “maybe voted”) | Use only as **prefilter**—“probably voted, check DB”—never sole reject path |

**Why it matters:** Retries and concurrent tabs make **idempotency keys** and **unique constraints** non-negotiable; Redis accelerates happy path; Bloom filters save RAM at cost of rare extra DB lookups.

### Rate limiting (token bucket vs sliding window; in-memory vs distributed)

| Algorithm | Burst behavior | Steady-state fairness | Implementation notes |
|-----------|----------------|----------------------|----------------------|
| **Token bucket** | Allows controlled bursts (tokens refill) | Good for APIs that tolerate short spikes | Common in gateways; bucket per user/IP |
| **Sliding window** | Stricter in a rolling interval | Fairer for “max N per minute” abuse prevention | Often implemented with Redis sorted sets or approximate counters |
| **Fixed window** | Simple | Spike at window boundaries | Easier but uneven; mention as naive option |

| Deployment | Pros | Cons |
|------------|------|------|
| **In-memory (per gateway instance)** | Zero extra RTT; very fast | Wrong under load balancing unless sticky sessions or sync—**counts split across nodes** |
| **Distributed (Redis/Redis Cluster, Envoy rate limit service)** | Consistent limits across instances | Extra hop; must handle Redis failure (fail open vs closed) |

**Why it matters:** Rate limits protect **availability** and reduce fraud; they are **not** a substitute for vote dedup (different problem).

### Our choice

- **Vote durability + dedup:** **PostgreSQL** with `UNIQUE (user_id, poll_id)` as the final arbiter; **Redis** for fast duplicate checks and live counts; **Kafka** (or similar) to absorb bursts and decouple API from DB write pressure.
- **Counts:** **Real-time Redis counters** updated after successful durable/idempotent processing, plus **batch or periodic reconciliation** against PostgreSQL for accuracy.
- **Dedup layers:** **Redis `SETNX` / key per (poll, user)** + **idempotent consumer** + **DB constraint**; optional **Bloom** only if memory is constrained and extra DB reads are acceptable.
- **Rate limiting:** **Distributed sliding window or token bucket** in **Redis** (or API gateway plugin backed by Redis) for consistent limits; tune **fail-open** vs **fail-closed** for voting (often fail-open for availability with abuse handled elsewhere).

**Rationale in one line:** Optimize the **write path** for bursts without sacrificing **exactly-once vote semantics**; treat **display counts** as eventually consistent with a reconciling truth in PostgreSQL.

---

## CAP Theorem Analysis

CAP is a lens, not a prescription: in practice we choose **per operation** consistency vs availability, often with **latency** (PACELC) as the real tie-breaker.

### The tension

- **Vote deduplication** needs **strong consistency**: a user must not record two successful votes for the same poll.
- **Peak voting** demands **high availability**: users expect the button to work during spikes; regional outages should degrade gracefully.
- **Leaderboards / live counts** can often tolerate **eventual consistency** if bounds are stated (seconds of lag).

### Per-operation posture

| Surface | Typical choice | Rationale |
|---------|----------------|-----------|
| **Vote submission (acceptance)** | **CP** on the *decision* “have we recorded this vote?” | Duplicate prevention is a correctness invariant; prefer failing or retrying over double-counting |
| **Vote counts / bar charts** | **AP** / **eventual** | Reads scale with replicas and caches; small lag is acceptable if disclosed |
| **Leaderboard (if ranked by votes)** | **AP** | Same as counts; tie-breaks may use snapshot timestamps |

### Why not “all AP” or “all CP”?

- **All AP:** You might show snappy results while **risking duplicate acceptance** unless another layer enforces uniqueness—dangerous.
- **All CP:** You maximize correctness but **sacrifice availability** under partitions (e.g. strict quorum everywhere), hurting peak traffic.

### Diagram: logical partitions of consistency

```mermaid
flowchart TB
    subgraph CP_Path["CP-oriented: vote integrity"]
        VS[Vote submission path]
        UC[(DB UNIQUE / conditional write)]
        VS --> UC
    end

    subgraph AP_Path["AP-oriented: read scaling"]
        RC[Results / counts read path]
        CACHE[(Redis / read replicas)]
        RC --> CACHE
    end

    subgraph Eventual["Eventual: display freshness"]
        LB[Leaderboard / live ticker]
        Q[Stream / worker updates]
        LB --> Q
        Q --> CACHE
    end

    UC -.->|"reconcile / async projection"| CACHE
```

**Interview takeaway:** Say clearly: **dedup is a consistency problem**; **fan-out reads and UI aggregates are availability/throughput problems**. Separate paths, reconcile.

---

## SLA and SLO Definitions

SLAs are **contracts** with users or customers; SLOs are **internal targets** used to steer engineering; error budgets quantify **how much unreliability** you can spend before slowing feature work.

### SLOs (example targets for a viral-ready poll platform)

| SLO | Target | Measurement window | Notes |
|-----|--------|--------------------|--------|
| **Vote submission latency** | **p99 < 500 ms**, **p50 < 150 ms** | Rolling 30 days | End-to-end from edge to “accepted” response (202 or 200 per design) |
| **Vote acceptance rate** | **≥ 99.95%** of *valid* authenticated requests succeed | Rolling 30 days | Exclude client errors (4xx); include 5xx and timeouts |
| **Count accuracy (lag)** | **≤ 5 s** behind authoritative processing for live results (p99) | Rolling 7 days | Compare Redis/display vs DB reconciliation job |
| **Availability during peak** | **≥ 99.99%** for vote API during declared peak events | Per event window | May use stricter internal target than global monthly SLA |

### SLI (what we measure)

- **Latency:** API gateway or service histograms for `POST /polls/{id}/votes`.
- **Success:** ratio of `2xx` to total minus `4xx` validation failures.
- **Accuracy:** scheduled job comparing `SUM(votes)` vs `poll_options.vote_count` / Redis hashes.
- **Availability:** synthetic probes + success rate from real traffic.

### Error budget policy

| Monthly budget (example) | If burn is fast… |
|--------------------------|------------------|
| **99.9% availability** ≈ **43 minutes** downtime/month | Freeze non-critical releases; prioritize hotfix, scaling, load tests |
| **99.95%** ≈ **22 minutes** | Same, plus incident review for vote path only |
| **Latency SLO miss** (p99 > 500 ms sustained) | Treat like availability burn: scale workers, Kafka consumers, DB pool; review sync hot paths |

**Policy:** Consuming **>50% of error budget** in **<25% of the window** triggers a **freeze** on risky changes and an **incident review**. Voting systems: prioritize **acceptance rate** and **dedup correctness** over cosmetic features when budgets burn.

---

## API Design

REST-shaped JSON APIs are easy to reason about in interviews; emphasize **idempotency**, **clear errors**, and **async acceptance** where queues are used.

### Conventions

- **Auth:** `Authorization: Bearer <JWT>` for authenticated routes.
- **Idempotency:** `Idempotency-Key` header on `POST` vote (and optional on poll create) for safe retries.
- **Async vote path:** `202 Accepted` + `operation_id` or `vote_id` when processing is queued; `GET` status endpoint for polling clients that need it.

### Endpoints overview

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/v1/polls` | Create poll |
| `GET` | `/v1/polls/{poll_id}` | Poll metadata + options |
| `PATCH` | `/v1/polls/{poll_id}` | Update poll (owner; restricted fields) |
| `POST` | `/v1/polls/{poll_id}/votes` | Submit a vote |
| `GET` | `/v1/polls/{poll_id}/results` | Aggregated counts / percentages |
| `GET` | `/v1/polls/{poll_id}/votes/me` | Check whether current user voted (and option if allowed) |
| `GET` | `/v1/polls/{poll_id}/votes/{request_id}` | Optional: status of async vote by client request id |

---

### Submit a vote

`POST /v1/polls/{poll_id}/votes`

**Request headers:** `Authorization`, `Idempotency-Key: <uuid>`

**Request body:**

```json
{
  "option_id": "550e8400-e29b-41d4-a716-446655440000",
  "client_meta": {
    "app_version": "1.4.2",
    "platform": "ios"
  }
}
```

**Response `202 Accepted` (queued processing):**

```json
{
  "status": "accepted",
  "poll_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
  "request_id": "a3bb189e-8bf9-3888-b889-317e3f6b8c4d",
  "message": "Vote is being processed"
}
```

**Response `200 OK` (synchronous path, if implemented):**

```json
{
  "status": "recorded",
  "vote_id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
  "poll_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
  "option_id": "550e8400-e29b-41d4-a716-446655440000",
  "recorded_at": "2026-04-05T12:00:01.234Z"
}
```

**Errors (examples):**

```json
{
  "error": {
    "code": "ALREADY_VOTED",
    "message": "You have already voted on this poll",
    "poll_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7"
  }
}
```

---

### Get results / counts

`GET /v1/polls/{poll_id}/results`

Optional query: `?tally=final|live` (if product distinguishes).

**Response `200 OK`:**

```json
{
  "poll_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
  "status": "active",
  "total_votes": 12847,
  "as_of": "2026-04-05T12:00:05.000Z",
  "options": [
    {
      "option_id": "550e8400-e29b-41d4-a716-446655440000",
      "label": "Option A",
      "votes": 7201,
      "percentage": 56.1
    },
    {
      "option_id": "6f9619ff-8b86-d011-b42d-00cf4fc964ff",
      "label": "Option B",
      "votes": 5646,
      "percentage": 43.9
    }
  ]
}
```

**Why `as_of`:** Signals **eventual** freshness for interview discussion (cached vs authoritative).

---

### Create / manage polls

**Create:** `POST /v1/polls`

```json
{
  "title": "Best programming language in 2026?",
  "description": "For backend services at scale.",
  "options": [
    { "text": "Go", "display_order": 0 },
    { "text": "Rust", "display_order": 1 },
    { "text": "TypeScript", "display_order": 2 }
  ],
  "start_time": "2026-04-05T10:00:00Z",
  "end_time": "2026-04-12T10:00:00Z",
  "settings": {
    "show_results_during_voting": true,
    "allow_anonymous": false
  }
}
```

**Response `201 Created`:**

```json
{
  "poll_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
  "status": "draft",
  "created_at": "2026-04-05T09:55:00.000Z",
  "manage_url": "https://api.example.com/v1/polls/7c9e6679-7425-40de-944b-e07fc1f90ae7"
}
```

**Update (owner):** `PATCH /v1/polls/{poll_id}`

```json
{
  "status": "active",
  "title": "Best programming language in 2026? (edited)"
}
```

**Response `200 OK`:** updated poll envelope (omit for brevity in interview if time-boxed).

---

### Check vote status

`GET /v1/polls/{poll_id}/votes/me`

**Response `200 OK` (has voted):**

```json
{
  "has_voted": true,
  "option_id": "550e8400-e29b-41d4-a716-446655440000",
  "voted_at": "2026-04-05T11:59:58.000Z"
}
```

**Response `200 OK` (not voted):**

```json
{
  "has_voted": false
}
```

**Optional async status:** `GET /v1/polls/{poll_id}/votes/requests/{request_id}`

```json
{
  "request_id": "a3bb189e-8bf9-3888-b889-317e3f6b8c4d",
  "state": "completed",
  "vote_id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
}
```

`state` may be `pending`, `completed`, or `failed` with `error` object—useful when linking **API design** to **Kafka** backpressure and retries in discussion.

---

## Step 2: Back-of-Envelope Estimation

Let's size for a Twitter-scale polling system.

### Traffic Estimation

```
Assumptions:
- 10 million daily active users
- 1% create polls daily = 100,000 polls/day
- Average poll gets 100 votes = 10 million votes/day
- Viral poll gets 10 million votes

Average load:
- 10M votes / 86,400 seconds ≈ 115 votes/second

Peak load (viral poll):
- 10 million votes in 1 hour = 2,800 votes/second
- With burst: 10,000 votes/second
```

### Storage Estimation

```
Per vote:
- vote_id: 8 bytes
- poll_id: 8 bytes  
- user_id: 8 bytes
- option_id: 8 bytes
- timestamp: 8 bytes
- metadata: 20 bytes
Total: ~60 bytes

Daily: 10M votes × 60 bytes = 600 MB
Yearly: 220 GB
5 years: ~1 TB

Per poll:
- Metadata: ~500 bytes
- Options: 100 bytes × 5 = 500 bytes
Daily: 100K polls × 1KB = 100 MB
```

### Database Operations

```
Writes:
- Peak: 10,000 votes/sec (INSERT)
- Each vote requires unique constraint check

Reads:
- Poll viewing: 100x vote rate = 1 million reads/sec (cached)
- Results aggregation: Heavy queries, should be pre-computed
```

---

## Step 3: High-Level Architecture

### System Overview

```mermaid
flowchart TB
    subgraph Clients["Client Layer"]
        Web[Web App]
        Mobile[Mobile App]
        API[API Clients]
    end
    
    subgraph Edge["Edge Layer"]
        CDN[CDN]
        LB[Load Balancer]
    end
    
    subgraph Gateway["API Layer"]
        APIGw[API Gateway<br/>Rate Limiting + Auth]
    end
    
    subgraph Services["Application Layer"]
        UserSvc[User Service]
        PollSvc[Poll Service]
        VoteSvc[Vote Service]
        ResultSvc[Results Service]
    end
    
    subgraph Async["Async Processing"]
        Queue[Kafka]
        VoteWorker[Vote Processor]
        ResultWorker[Results Aggregator]
    end
    
    subgraph Data["Data Layer"]
        Cache[(Redis)]
        PollDB[(Poll DB)]
        VoteDB[(Vote DB)]
        AuditLog[(Audit Log)]
    end
    
    Web --> CDN
    Mobile --> LB
    API --> LB
    CDN --> LB
    LB --> APIGw
    
    APIGw --> UserSvc
    APIGw --> PollSvc
    APIGw --> VoteSvc
    APIGw --> ResultSvc
    
    VoteSvc --> Queue
    VoteSvc --> Cache
    Queue --> VoteWorker
    VoteWorker --> VoteDB
    VoteWorker --> Cache
    VoteWorker --> AuditLog
    
    ResultSvc --> Cache
    
    PollSvc --> PollDB
    PollSvc --> Cache
```

### Why Async Vote Processing?

At 10,000 votes/second, synchronous database writes would overwhelm PostgreSQL.

**Synchronous approach (won't scale):**
```
User → API → Database → Response
       └── 10,000 writes/sec to one DB
```

**Asynchronous approach (scales):**
```
User → API → Message Queue → Response (202 Accepted)
             └── Workers → Database (batched, controlled)
```

Benefits:
- **Backpressure handling:** Queue buffers traffic spikes
- **Reliability:** Messages persist even if workers crash
- **Scalability:** Add more workers for more throughput
- **Latency:** User gets fast response, actual processing happens async

---

## Step 4: Deep Dive

### 4.1 The Core Challenge - Preventing Duplicate Votes

This is the most critical part of the design. A user must never be able to vote twice on the same poll.

### Why It's Hard

Consider this scenario:
1. User clicks "Vote" button
2. Request times out (network issue)
3. Client retries
4. Original request eventually succeeds
5. Retry also succeeds
6. **User has voted twice!**

Or this race condition:
```
Thread 1: Check if user voted → No
Thread 2: Check if user voted → No
Thread 1: Insert vote → Success
Thread 2: Insert vote → Success (duplicate!)
```

### Defense in Depth Strategy

We use multiple layers of protection:

```mermaid
flowchart TD
    Vote[Vote Request] --> L1{"Layer 1:<br/>Redis Check"}
    L1 -->|Already voted| Reject1[Reject]
    L1 -->|Not found| L2["Layer 2:<br/>Kafka Dedup"]
    L2 --> L3{"Layer 3:<br/>DB Constraint"}
    L3 -->|Constraint violation| Reject2[Log & Ignore]
    L3 -->|Success| Store[Vote Stored]
    Store --> UpdateCache[Update Redis]
```

### Layer 1: Fast Redis Check (API Level)

Before accepting the vote, check if the user already voted:

```python
class VoteService:
    def __init__(self, redis_client, kafka_producer):
        self.redis = redis_client
        self.kafka = kafka_producer
    
    async def cast_vote(self, user_id: str, poll_id: str, option_id: str) -> dict:
        # Validate poll is active
        poll = await self.get_poll(poll_id)
        if not poll or poll.status != "active":
            raise PollNotActiveError()
        
        if not self.is_valid_option(poll, option_id):
            raise InvalidOptionError()
        
        # Layer 1: Fast duplicate check
        vote_key = f"voted:{poll_id}:{user_id}"
        if self.redis.exists(vote_key):
            raise AlreadyVotedError("You have already voted on this poll")
        
        # Optimistic: Set Redis flag immediately
        # This provides fast rejection for subsequent requests
        # TTL should be longer than poll duration
        self.redis.setex(vote_key, poll.ttl_seconds, "pending")
        
        # Enqueue for processing
        await self.kafka.send("votes", {
            "user_id": user_id,
            "poll_id": poll_id,
            "option_id": option_id,
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": generate_request_id()  # For idempotency
        })
        
        return {"status": "accepted", "message": "Vote is being processed"}
```

**Why this isn't enough:**

- Redis SET + Kafka SEND isn't atomic
- If the server crashes between them, state is inconsistent
- Redis might not have the data (cache miss)

### Layer 2: Idempotent Message Processing

Each vote message has a unique `request_id`. The worker deduplicates:

```python
class VoteProcessor:
    def __init__(self):
        self.processed_requests = {}  # In-memory for hot path
    
    async def process_vote(self, message: dict):
        request_id = message["request_id"]
        user_id = message["user_id"]
        poll_id = message["poll_id"]
        option_id = message["option_id"]
        
        # Idempotency check (in-memory for speed)
        if request_id in self.processed_requests:
            log.info(f"Duplicate request {request_id}, skipping")
            return
        
        try:
            # Layer 3: Database insert with constraint
            await self.db.execute("""
                INSERT INTO votes (user_id, poll_id, option_id, voted_at)
                VALUES ($1, $2, $3, NOW())
            """, user_id, poll_id, option_id)
            
            # Mark as processed
            self.processed_requests[request_id] = True
            
            # Update Redis for fast future checks
            vote_key = f"voted:{poll_id}:{user_id}"
            self.redis.setex(vote_key, 86400 * 7, "confirmed")
            
            # Update vote count
            await self.increment_vote_count(poll_id, option_id)
            
        except UniqueViolationError:
            # Already voted - this is expected for retries
            log.info(f"User {user_id} already voted on poll {poll_id}")
            
        except Exception as e:
            # Unexpected error - let Kafka retry
            log.error(f"Failed to process vote: {e}")
            raise
```

### Layer 3: Database Unique Constraint (The Last Line of Defense)

The database constraint guarantees correctness even if all other layers fail:

```sql
CREATE TABLE votes (
    vote_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    poll_id UUID NOT NULL REFERENCES polls(poll_id),
    user_id UUID NOT NULL REFERENCES users(user_id),
    option_id UUID NOT NULL REFERENCES poll_options(option_id),
    voted_at TIMESTAMP DEFAULT NOW(),
    
    -- THE CRITICAL CONSTRAINT
    CONSTRAINT uq_one_vote_per_user_per_poll UNIQUE (user_id, poll_id)
);

-- This index also speeds up "has user voted?" queries
CREATE INDEX idx_votes_user_poll ON votes(user_id, poll_id);
```

**Why database constraints are non-negotiable:**

1. **Atomicity:** The INSERT either fully succeeds or fully fails
2. **Durability:** Once committed, the vote is permanent
3. **Correctness:** Even with race conditions, only one row can exist

```python
# This is safe even with concurrent requests:
try:
    db.execute("INSERT INTO votes ...")
except UniqueViolationError:
    # Another thread/process already inserted - that's fine
    pass
```

---

### 4.2 Database Design

### Schema Design

```sql
-- Users table
CREATE TABLE users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    is_verified BOOLEAN DEFAULT FALSE
);

-- Polls table
CREATE TABLE polls (
    poll_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    creator_id UUID NOT NULL REFERENCES users(user_id),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    status VARCHAR(20) DEFAULT 'draft', -- draft, active, closed
    
    -- Settings
    allow_anonymous BOOLEAN DEFAULT FALSE,
    show_results_during_voting BOOLEAN DEFAULT TRUE,
    allow_multiple_options BOOLEAN DEFAULT FALSE,
    max_votes_per_user INT DEFAULT 1,
    
    -- Denormalized for quick access
    total_votes INT DEFAULT 0,
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_polls_status ON polls(status, end_time);
CREATE INDEX idx_polls_creator ON polls(creator_id);

-- Poll options
CREATE TABLE poll_options (
    option_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    poll_id UUID NOT NULL REFERENCES polls(poll_id) ON DELETE CASCADE,
    option_text VARCHAR(500) NOT NULL,
    display_order INT NOT NULL,
    vote_count INT DEFAULT 0,  -- Denormalized for fast results
    
    UNIQUE (poll_id, display_order)
);

CREATE INDEX idx_options_poll ON poll_options(poll_id);

-- Votes table (critical for integrity)
CREATE TABLE votes (
    vote_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    poll_id UUID NOT NULL,
    user_id UUID NOT NULL,
    option_id UUID NOT NULL,
    voted_at TIMESTAMP DEFAULT NOW(),
    ip_address INET,  -- For abuse detection
    user_agent TEXT,
    
    CONSTRAINT fk_poll FOREIGN KEY (poll_id) REFERENCES polls(poll_id),
    CONSTRAINT fk_user FOREIGN KEY (user_id) REFERENCES users(user_id),
    CONSTRAINT fk_option FOREIGN KEY (option_id) REFERENCES poll_options(option_id),
    
    -- One vote per user per poll
    CONSTRAINT uq_user_poll UNIQUE (user_id, poll_id)
);

CREATE INDEX idx_votes_poll_option ON votes(poll_id, option_id);
CREATE INDEX idx_votes_user ON votes(user_id);

-- Audit log for compliance
CREATE TABLE vote_audit_log (
    log_id BIGSERIAL PRIMARY KEY,
    vote_id UUID NOT NULL,
    poll_id UUID NOT NULL,
    user_id UUID NOT NULL,
    action VARCHAR(20) NOT NULL,  -- 'cast', 'changed', 'revoked'
    old_option_id UUID,
    new_option_id UUID,
    timestamp TIMESTAMP DEFAULT NOW(),
    ip_address INET,
    request_metadata JSONB
);

CREATE INDEX idx_audit_poll ON vote_audit_log(poll_id, timestamp);
```

### Denormalization for Performance

Instead of counting votes on every request:

```sql
-- Slow: COUNT on every read
SELECT option_id, COUNT(*) 
FROM votes 
WHERE poll_id = 'xxx' 
GROUP BY option_id;
```

We maintain counters:

```sql
-- Fast: Read denormalized count
SELECT option_id, option_text, vote_count 
FROM poll_options 
WHERE poll_id = 'xxx' 
ORDER BY display_order;
```

The vote processor updates counters atomically:

```python
async def increment_vote_count(poll_id: str, option_id: str):
    # Atomic increment
    await db.execute("""
        UPDATE poll_options 
        SET vote_count = vote_count + 1 
        WHERE option_id = $1
    """, option_id)
    
    await db.execute("""
        UPDATE polls 
        SET total_votes = total_votes + 1 
        WHERE poll_id = $1
    """, poll_id)
    
    # Invalidate cache
    await redis.delete(f"results:{poll_id}")
```

---

### 4.3 Real-Time Results with Redis

For polls that show live results, we need sub-second updates.

### Redis Data Model

```bash
# Vote counts per option (Hash)
HSET results:poll123 option_a 1542
HSET results:poll123 option_b 3201
HSET results:poll123 option_c 876

# Increment on vote
HINCRBY results:poll123 option_a 1

# Get all results
HGETALL results:poll123
# Returns: {"option_a": "1542", "option_b": "3201", "option_c": "876"}

# Track who voted (Set)
SADD voters:poll123 user_abc
SISMEMBER voters:poll123 user_abc  # Returns 1 (true)
```

### Results Service

```python
class ResultsService:
    def __init__(self, redis_client, db):
        self.redis = redis_client
        self.db = db
    
    async def get_results(self, poll_id: str, user_id: str = None) -> dict:
        poll = await self.get_poll(poll_id)
        
        # Check if results are visible
        if poll.status == "active" and not poll.show_results_during_voting:
            if user_id:
                has_voted = await self.has_user_voted(poll_id, user_id)
                if not has_voted:
                    raise ResultsNotAvailableError("Vote to see results")
        
        # Try Redis first
        results_key = f"results:{poll_id}"
        cached = self.redis.hgetall(results_key)
        
        if cached:
            return self.format_results(poll, cached)
        
        # Cache miss - compute from database
        results = await self.db.fetch("""
            SELECT option_id, vote_count
            FROM poll_options
            WHERE poll_id = $1
            ORDER BY display_order
        """, poll_id)
        
        # Cache for next time
        for row in results:
            self.redis.hset(results_key, str(row['option_id']), row['vote_count'])
        self.redis.expire(results_key, 3600)  # 1 hour TTL
        
        return self.format_results(poll, results)
    
    def format_results(self, poll, results) -> dict:
        total = sum(int(v) for v in results.values())
        
        return {
            "poll_id": poll.poll_id,
            "title": poll.title,
            "status": poll.status,
            "total_votes": total,
            "options": [
                {
                    "option_id": opt_id,
                    "votes": int(count),
                    "percentage": round(int(count) / total * 100, 1) if total > 0 else 0
                }
                for opt_id, count in results.items()
            ]
        }
```

### Real-Time Updates with WebSockets

For live-updating results:

```python
from fastapi import WebSocket
import asyncio

class ResultsWebSocket:
    def __init__(self):
        self.connections: dict[str, list[WebSocket]] = {}
    
    async def connect(self, poll_id: str, websocket: WebSocket):
        await websocket.accept()
        
        if poll_id not in self.connections:
            self.connections[poll_id] = []
        self.connections[poll_id].append(websocket)
        
        # Send current results immediately
        results = await results_service.get_results(poll_id)
        await websocket.send_json(results)
    
    async def broadcast_update(self, poll_id: str, results: dict):
        if poll_id not in self.connections:
            return
        
        for ws in self.connections[poll_id]:
            try:
                await ws.send_json(results)
            except:
                self.connections[poll_id].remove(ws)

# In vote processor, after updating counts:
async def on_vote_processed(poll_id: str, option_id: str):
    results = await results_service.get_results(poll_id)
    await ws_manager.broadcast_update(poll_id, results)
```

---

### 4.4 Message Queue Design with Kafka

### Why Kafka?

| Requirement | Kafka Feature |
|-------------|---------------|
| High throughput | Partitioned, batched writes |
| Durability | Replication, disk persistence |
| Ordering | Per-partition ordering |
| Replay | Consumer offsets, retention |
| Scalability | Add partitions, consumers |

### Topic Configuration

```yaml
topic: votes
partitions: 12  # Allows 12 parallel consumers
replication_factor: 3  # Survives 2 broker failures
retention.ms: 604800000  # 7 days for replay/audit
min.insync.replicas: 2  # Require 2 brokers to ack
```

### Partitioning Strategy

**Option 1: Partition by user_id**
```python
partition = hash(user_id) % num_partitions
```
- Same user's votes go to same partition
- Helps with per-user deduplication
- Good load distribution

**Option 2: Partition by poll_id**
```python
partition = hash(poll_id) % num_partitions
```
- All votes for a poll in one partition
- Easier to maintain poll-level ordering
- Risk: Hot polls overload one partition

**Recommendation:** Partition by `user_id` for better load distribution.

### Consumer Group

```python
from kafka import KafkaConsumer

consumer = KafkaConsumer(
    'votes',
    bootstrap_servers=['kafka1:9092', 'kafka2:9092', 'kafka3:9092'],
    group_id='vote-processors',
    auto_offset_reset='earliest',
    enable_auto_commit=False,  # Manual commit for reliability
    value_deserializer=lambda m: json.loads(m.decode())
)

for message in consumer:
    try:
        await process_vote(message.value)
        consumer.commit()  # Only commit after successful processing
    except Exception as e:
        log.error(f"Failed to process vote: {e}")
        # Don't commit - Kafka will redeliver
```

### Dead Letter Queue

Votes that fail repeatedly need investigation:

```mermaid
flowchart LR
    Main[votes topic] --> Consumer[Vote Processor]
    Consumer -->|success| Commit[Commit Offset]
    Consumer -->|failure x3| DLQ[votes-dlq topic]
    DLQ --> Alert[Alert Team]
    DLQ --> Manual[Manual Processing]
```

```python
MAX_RETRIES = 3

async def process_with_retry(message):
    retries = message.headers.get('x-retry-count', 0)
    
    try:
        await process_vote(message.value)
    except Exception as e:
        if retries >= MAX_RETRIES:
            # Send to dead letter queue
            await kafka.send('votes-dlq', 
                value=message.value,
                headers={'x-error': str(e), 'x-original-offset': message.offset}
            )
        else:
            # Re-enqueue with retry count
            await kafka.send('votes',
                value=message.value,
                headers={'x-retry-count': retries + 1}
            )
```

---

## Step 5: Scaling & Production

### 5.1 Scaling Strategies

### Scaling Components

| Component | Scaling Strategy | Trigger |
|-----------|------------------|---------|
| **API Gateway** | Horizontal (add instances) | CPU > 70%, latency > 200ms |
| **Vote Service** | Horizontal | Queue depth, latency |
| **Kafka** | Add partitions, brokers | Throughput limits |
| **Vote Processors** | Add consumers | Queue lag > 10,000 |
| **PostgreSQL** | Read replicas → Sharding | Query latency, connections |
| **Redis** | Cluster mode | Memory usage, ops/sec |

### Handling Viral Polls

When a celebrity creates a poll and 10 million people vote:

```mermaid
flowchart LR
    subgraph Normal["Normal: 100 votes/sec"]
        N1[3 API pods]
        N2[3 workers]
    end
    
    subgraph Viral["Viral: 10K votes/sec"]
        V1[30 API pods]
        V2[30 workers]
    end
    
    Monitor[Metrics] --> HPA[HPA / Auto-scaler]
    HPA -->|scale up| Viral
```

**Auto-scaling rules:**
```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  scaleTargetRef:
    kind: Deployment
    name: vote-service
  minReplicas: 3
  maxReplicas: 50
  metrics:
  - type: External
    external:
      metric:
        name: kafka_consumer_lag
      target:
        type: Value
        value: "5000"  # Scale up if lag exceeds 5K
```

### Database Scaling

**Stage 1: Read replicas**
```
Writes → Primary
Reads → Read Replicas (round-robin)
```

**Stage 2: Connection pooling (PgBouncer)**
```
App (1000 connections) → PgBouncer (100 connections) → PostgreSQL
```

**Stage 3: Sharding by poll_id**
```python
def get_shard(poll_id: str) -> str:
    shard_id = hash(poll_id) % NUM_SHARDS
    return f"shard_{shard_id}"

# Votes for poll ABC → shard_2
# Votes for poll XYZ → shard_0
```

---

### 5.2 Failure Handling

### Failure Modes and Recovery

| Failure | Impact | Detection | Recovery |
|---------|--------|-----------|----------|
| **API pod crash** | Some requests fail | Health check | LB routes to healthy pods |
| **Kafka broker down** | Reduced capacity | Broker health | Failover to replicas |
| **Vote processor crash** | Processing delays | Consumer lag | Kafka redelivers, other workers continue |
| **Primary DB down** | Writes fail | Connection errors | Promote replica |
| **Redis down** | Slower checks | Connection timeout | Fall back to DB |

### Graceful Degradation

When components fail, degrade gracefully instead of hard failing:

```python
async def cast_vote_with_fallback(user_id, poll_id, option_id):
    # Try Redis for duplicate check
    try:
        if await redis.exists(f"voted:{poll_id}:{user_id}"):
            raise AlreadyVotedError()
    except RedisConnectionError:
        # Redis down - skip fast check, rely on DB constraint
        log.warning("Redis unavailable, proceeding without cache check")
    
    # Try to enqueue vote
    try:
        await kafka.send("votes", {...})
        return {"status": "accepted"}
    except KafkaError:
        # Kafka down - try direct DB write (slower but works)
        log.warning("Kafka unavailable, writing directly to DB")
        try:
            await db.execute("INSERT INTO votes ...")
            return {"status": "confirmed"}
        except UniqueViolationError:
            raise AlreadyVotedError()
```

### Data Consistency Checks

Periodically verify Redis counts match database:

```python
async def reconcile_vote_counts():
    """Run hourly to catch any drift."""
    
    polls = await db.fetch("SELECT poll_id FROM polls WHERE status = 'active'")
    
    for poll in polls:
        poll_id = poll['poll_id']
        
        # Get DB truth
        db_counts = await db.fetch("""
            SELECT option_id, COUNT(*) as count
            FROM votes WHERE poll_id = $1
            GROUP BY option_id
        """, poll_id)
        
        # Get Redis counts
        redis_counts = redis.hgetall(f"results:{poll_id}")
        
        # Compare and fix
        for row in db_counts:
            option_id = str(row['option_id'])
            db_count = row['count']
            redis_count = int(redis_counts.get(option_id, 0))
            
            if db_count != redis_count:
                log.warning(f"Count mismatch for {poll_id}/{option_id}: "
                           f"DB={db_count}, Redis={redis_count}")
                # Fix Redis
                redis.hset(f"results:{poll_id}", option_id, db_count)
```

---

### 5.3 Security Considerations

### Authentication and Authorization

```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def get_current_user(token: str = Depends(security)) -> User:
    try:
        payload = jwt.decode(token.credentials, SECRET_KEY, algorithms=["HS256"])
        user_id = payload.get("sub")
        return await user_service.get_user(user_id)
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
    except jwt.JWTError:
        raise HTTPException(401, "Invalid token")

@app.post("/polls/{poll_id}/vote")
async def vote(
    poll_id: str,
    option_id: str,
    user: User = Depends(get_current_user)
):
    # User is authenticated
    return await vote_service.cast_vote(user.user_id, poll_id, option_id)
```

### Rate Limiting

Prevent abuse:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/polls/{poll_id}/vote")
@limiter.limit("10/minute")  # 10 votes per minute per IP
async def vote(poll_id: str, request: Request):
    ...

@app.post("/polls")
@limiter.limit("5/hour")  # 5 polls per hour per user
async def create_poll(poll: PollCreate, user: User = Depends(get_current_user)):
    ...
```

### Bot Prevention

| Technique | Implementation | Effectiveness |
|-----------|----------------|---------------|
| **CAPTCHA** | reCAPTCHA on vote | High, but friction |
| **Rate limiting** | Per IP, per user | Medium |
| **Device fingerprinting** | FingerprintJS | Medium-High |
| **Behavioral analysis** | ML on voting patterns | High |
| **Email verification** | Require verified email | High |

```python
async def validate_vote_request(request: Request, user: User):
    # Check for suspicious patterns
    recent_votes = await get_recent_votes(user.user_id, minutes=5)
    
    if len(recent_votes) > 20:
        raise SuspiciousActivityError("Too many votes in short time")
    
    # Check device fingerprint
    fingerprint = request.headers.get("X-Device-Fingerprint")
    if fingerprint:
        devices_for_user = await get_user_devices(user.user_id)
        if len(devices_for_user) > 5:
            raise SuspiciousActivityError("Too many devices")
```

### Anonymity vs Auditability Trade-off

**Challenge:** You want votes to be anonymous (no one knows how I voted), but you also need to verify one-vote-per-person.

**Our approach:**
- System knows *who* voted (required for duplicate prevention)
- System stores *what* they voted for (for results)
- Other users cannot see individual votes
- Admins can audit if legally required

**For truly anonymous voting (e.g., government elections):**
- Use cryptographic techniques (blind signatures, homomorphic encryption)
- Voter gets a receipt they can verify without revealing their vote
- This is complex and outside typical interview scope

---

### 5.4 Multi-Language Implementations

### Vote service with idempotency and layered deduplication

=== "Java"

    ```java
    import java.time.Instant;
    import java.util.UUID;
    
    public class VoteService {
    
        public record VoteRequest(String pollId, String optionId, String userId, String idempotencyKey) {}
        public record VoteResult(String voteId, boolean accepted, String message) {}
    
        private final VoteRepository voteRepository;
        private final RedisTemplate<String, String> redis;
        private final KafkaTemplate<String, VoteEvent> kafka;
    
        public VoteService(VoteRepository voteRepository, RedisTemplate<String, String> redis,
                            KafkaTemplate<String, VoteEvent> kafka) {
            this.voteRepository = voteRepository;
            this.redis = redis;
            this.kafka = kafka;
        }
    
        public VoteResult castVote(VoteRequest request) {
            String voteId = UUID.randomUUID().toString();
    
            // Layer 1: Redis deduplication (fast path)
            String dedupeKey = "vote:" + request.pollId() + ":" + request.userId();
            Boolean isNew = redis.opsForValue().setIfAbsent(dedupeKey, voteId);
            if (Boolean.FALSE.equals(isNew)) {
                return new VoteResult(voteId, false, "Duplicate vote detected (cache)");
            }
    
            // Layer 2: Idempotency check
            if (voteRepository.existsByIdempotencyKey(request.idempotencyKey())) {
                return new VoteResult(voteId, false, "Duplicate vote detected (idempotency)");
            }
    
            // Layer 3: Database unique constraint (final safety net)
            try {
                Vote vote = new Vote(voteId, request.pollId(), request.optionId(),
                    request.userId(), request.idempotencyKey(), Instant.now());
                voteRepository.save(vote);
            } catch (DuplicateKeyException e) {
                return new VoteResult(voteId, false, "Duplicate vote detected (database)");
            }
    
            // Publish to Kafka for async result aggregation
            kafka.send("votes", request.pollId(),
                new VoteEvent(voteId, request.pollId(), request.optionId(), Instant.now()));
    
            // Increment Redis counter for real-time results
            redis.opsForHash().increment("poll:" + request.pollId() + ":results",
                request.optionId(), 1);
    
            return new VoteResult(voteId, true, "Vote recorded");
        }
    }
    
    /**
     * Kafka consumer that aggregates votes and reconciles with the database.
     */
    public class VoteAggregationConsumer {
        
        @KafkaListener(topics = "votes", groupId = "vote-aggregator")
        public void onVoteEvent(VoteEvent event) {
            // periodic reconciliation between Redis counters and DB counts
            // runs less frequently, ensures eventual consistency
            long dbCount = voteRepository.countByPollIdAndOptionId(
                event.pollId(), event.optionId());
            long redisCount = Long.parseLong(
                redis.opsForHash().get("poll:" + event.pollId() + ":results",
                    event.optionId()).toString());
    
            if (Math.abs(dbCount - redisCount) > 10) {
                redis.opsForHash().put("poll:" + event.pollId() + ":results",
                    event.optionId(), String.valueOf(dbCount));
            }
        }
    }
    ```

=== "Go"

    ```go
    package voting
    
    import (
    	"context"
    	"fmt"
    	"time"
    
    	"github.com/google/uuid"
    	"github.com/redis/go-redis/v9"
    )
    
    type VoteRequest struct {
    	PollID         string
    	OptionID       string
    	UserID         string
    	IdempotencyKey string
    }
    
    type VoteResult struct {
    	VoteID   string
    	Accepted bool
    	Message  string
    }
    
    type VoteRepository interface {
    	Save(ctx context.Context, vote *Vote) error
    	ExistsByIdempotencyKey(ctx context.Context, key string) (bool, error)
    	CountByPollAndOption(ctx context.Context, pollID, optionID string) (int64, error)
    }
    
    type Vote struct {
    	ID             string
    	PollID         string
    	OptionID       string
    	UserID         string
    	IdempotencyKey string
    	CreatedAt      time.Time
    }
    
    type VoteService struct {
    	repo   VoteRepository
    	redis  *redis.Client
    	events chan<- VoteEvent
    }
    
    type VoteEvent struct {
    	VoteID   string `json:"vote_id"`
    	PollID   string `json:"poll_id"`
    	OptionID string `json:"option_id"`
    }
    
    func NewVoteService(repo VoteRepository, redisClient *redis.Client,
    	events chan<- VoteEvent) *VoteService {
    	return &VoteService{repo: repo, redis: redisClient, events: events}
    }
    
    func (s *VoteService) CastVote(ctx context.Context, req VoteRequest) VoteResult {
    	voteID := uuid.New().String()
    
    	// Layer 1: Redis deduplication
    	dedupeKey := fmt.Sprintf("vote:%s:%s", req.PollID, req.UserID)
    	set, err := s.redis.SetNX(ctx, dedupeKey, voteID, 24*time.Hour).Result()
    	if err != nil || !set {
    		return VoteResult{voteID, false, "Duplicate vote (cache)"}
    	}
    
    	// Layer 2: Idempotency check
    	exists, _ := s.repo.ExistsByIdempotencyKey(ctx, req.IdempotencyKey)
    	if exists {
    		return VoteResult{voteID, false, "Duplicate vote (idempotency)"}
    	}
    
    	// Layer 3: Database save with unique constraint
    	vote := &Vote{
    		ID:             voteID,
    		PollID:         req.PollID,
    		OptionID:       req.OptionID,
    		UserID:         req.UserID,
    		IdempotencyKey: req.IdempotencyKey,
    		CreatedAt:      time.Now(),
    	}
    	if err := s.repo.Save(ctx, vote); err != nil {
    		return VoteResult{voteID, false, "Duplicate vote (database)"}
    	}
    
    	// Update real-time counter in Redis
    	resultsKey := fmt.Sprintf("poll:%s:results", req.PollID)
    	s.redis.HIncrBy(ctx, resultsKey, req.OptionID, 1)
    
    	// Publish event for async processing
    	s.events <- VoteEvent{VoteID: voteID, PollID: req.PollID, OptionID: req.OptionID}
    
    	return VoteResult{voteID, true, "Vote recorded"}
    }
    
    func (s *VoteService) GetResults(ctx context.Context, pollID string) (map[string]int64, error) {
    	resultsKey := fmt.Sprintf("poll:%s:results", pollID)
    	results, err := s.redis.HGetAll(ctx, resultsKey).Result()
    	if err != nil {
    		return nil, err
    	}
    
    	counts := make(map[string]int64)
    	for optionID, countStr := range results {
    		var count int64
    		fmt.Sscanf(countStr, "%d", &count)
    		counts[optionID] = count
    	}
    	return counts, nil
    }
    ```

---

### 5.5 Monitoring and Observability

### Key Metrics

| Category | Metric | Alert Threshold |
|----------|--------|-----------------|
| **Availability** | Vote success rate | < 99.9% |
| **Latency** | Vote API P99 | > 500ms |
| **Throughput** | Votes processed/sec | Trend down |
| **Queue** | Kafka consumer lag | > 10,000 |
| **Duplicates** | Duplicate vote attempts | Spike (> 10%) |
| **Errors** | 5xx error rate | > 0.1% |

### Logging

Structured logs for easy querying:

```json
{
    "timestamp": "2024-01-15T10:30:00Z",
    "level": "INFO",
    "service": "vote-processor",
    "event": "vote_processed",
    "poll_id": "uuid",
    "user_id": "uuid",
    "option_id": "uuid",
    "processing_time_ms": 15,
    "was_duplicate": false,
    "cache_hit": true
}
```

### Dashboard Panels

1. **Real-time metrics:**
   - Votes per second
   - Active polls count
   - Error rate

2. **Queue health:**
   - Consumer lag
   - Partition distribution
   - Processing latency

3. **Database health:**
   - Query latency
   - Connection pool usage
   - Replication lag

---

## Interview Checklist

- [ ] **Clarified requirements** (type of voting, scale, anonymity)
- [ ] **Technology tradeoffs** (storage, aggregation, dedup, rate limits—and **our choice**)
- [ ] **CAP / per-operation consistency** (vote path vs counts vs leaderboard)
- [ ] **SLA/SLO and error budget** (latency, acceptance rate, count lag, peak availability)
- [ ] **API design** (vote, results, poll CRUD, vote status, idempotency)
- [ ] **Estimated capacity** (votes/sec, storage)
- [ ] **Drew high-level architecture** (async processing)
- [ ] **Explained duplicate prevention** (Redis + DB constraint)
- [ ] **Designed database schema** (with unique constraint)
- [ ] **Covered real-time results** (Redis counters)
- [ ] **Discussed message queue design** (Kafka partitioning)
- [ ] **Addressed scaling strategies** (horizontal, sharding)
- [ ] **Covered failure handling** (graceful degradation)
- [ ] **Mentioned security** (auth, rate limiting, bots)

---

## Sample Interview Dialogue

**Interviewer:** "Design a voting system."

**You:** "Great! Let me clarify a few things first. What type of voting system? A casual polling app like Strawpoll, Reddit-style upvotes, or something more formal like corporate elections?"

**Interviewer:** "Let's go with a general polling platform. Users create polls, others vote."

**You:** "Got it. How strict is duplicate vote prevention, and what scale are we targeting?"

**Interviewer:** "Strict—exactly one vote per user per poll. And let's say it could go viral—maybe 10,000 votes per second peak."

**You:** "Perfect. So we need strong consistency for duplicate prevention and high throughput. The core challenge is ensuring exactly-once voting while handling massive concurrency.

I'd use a three-layer defense:
1. Fast Redis check to reject obvious duplicates
2. Idempotent message processing in Kafka
3. Database unique constraint as the final guarantee

For the architecture, I'd process votes asynchronously through Kafka to handle the 10K/sec throughput. The API accepts votes quickly, returns 202 Accepted, and workers process them in controlled batches to the database.

Let me draw the architecture..."

---

## Summary

| Challenge | Solution |
|-----------|----------|
| **Tech stack choices** | PostgreSQL + Redis + Kafka; CP for vote integrity, AP for reads; distributed rate limits |
| **CAP tension** | Strong consistency on dedup; eventual counts/leaderboard with reconciliation |
| **SLA/SLO** | p99 latency, acceptance rate, count lag SLOs; error budget gates releases |
| **API surface** | REST + idempotency keys; async `202` + status; results with `as_of` freshness |
| **Duplicate prevention** | Three layers: Redis check + Kafka dedup + DB constraint |
| **High throughput** | Async processing via Kafka, horizontal scaling |
| **Real-time results** | Redis counters, WebSocket updates |
| **Data accuracy** | Database unique constraint, periodic reconciliation |
| **Scalability** | Kafka partitioning, worker scaling, DB sharding |
| **Reliability** | Graceful degradation, dead letter queue, retry logic |
| **Security** | JWT auth, rate limiting, bot detection |

The voting system demonstrates critical distributed systems concepts: **consistency** (one vote per user), **idempotency** (safe retries), and **async processing** (handling bursts). Master this pattern, and you'll be equipped to design many similar systems.
