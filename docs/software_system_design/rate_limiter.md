# Design a Rate Limiter

---

## What We're Building

A rate limiter controls how many requests a client can make to an API within a given time window. When a client exceeds the limit, the rate limiter rejects additional requests—typically returning HTTP 429 (Too Many Requests).

**Example:** "Each user can make 100 API calls per minute. After 100 calls, further requests are rejected until the next minute."

### Why Rate Limiting Matters

| Problem | How Rate Limiting Helps |
|---------|------------------------|
| **DDoS attacks** | Prevents malicious actors from overwhelming your servers |
| **Runaway clients** | Stops buggy clients from accidentally hammering your API |
| **Resource fairness** | Ensures no single user hogs all capacity |
| **Cost control** | Prevents expensive operations from spiraling |
| **Service stability** | Protects downstream services from cascading failures |

### Real-World Examples

| Service | Rate Limit | Purpose |
|---------|------------|---------|
| **Twitter API** | 450 requests/15 min | Prevent abuse, protect infrastructure |
| **GitHub API** | 5,000 requests/hour (authenticated) | Fair usage across millions of developers |
| **Stripe API** | 100 requests/sec | Protect payment processing systems |
| **OpenAI API** | Tokens per minute (TPM) | Manage GPU capacity, billing |

### Types of Rate Limiting

| Type | Description | Example |
|------|-------------|---------|
| **User-based** | Limit per user/API key | 1000 requests/hour per user |
| **IP-based** | Limit per IP address | 100 requests/minute per IP |
| **Endpoint-based** | Different limits per endpoint | GET: 1000/min, POST: 100/min |
| **Global** | Total requests to the system | 1M requests/minute total |
| **Concurrent** | Active requests at once | Max 10 concurrent requests |

---

## Step 1: Requirements Clarification

### Questions to Ask

| Question | Why It Matters |
|----------|----------------|
| What are we limiting? | Users, IPs, API keys, endpoints? |
| What are the rate limits? | 100/min? 1000/hour? Multiple tiers? |
| Hard limit or soft limit? | Reject immediately vs. queue/throttle? |
| Distributed or single-server? | Major architectural difference |
| How do we identify clients? | API key, JWT, IP address? |
| What happens when rate limited? | 429 response? Retry-After header? |

### Functional Requirements

| Requirement | Priority | Description |
|-------------|----------|-------------|
| Accurately limit request rate | Must have | Core functionality |
| Low latency | Must have | < 1ms overhead |
| Distributed support | Must have | Work across multiple servers |
| Multiple rate limit rules | Nice to have | Different limits per endpoint/tier |
| Graceful handling | Nice to have | Retry-After headers, queuing |

### Non-Functional Requirements

| Requirement | Target | Rationale |
|-------------|--------|-----------|
| **Latency** | < 1ms | Rate limiting shouldn't slow down requests |
| **Accuracy** | Near-perfect | Small variance acceptable (1-2%) |
| **Availability** | 99.99% | Failing open is usually preferred |
| **Scalability** | Millions of users | Must handle high cardinality |

---

### Technology Selection & Tradeoffs

Production rate limiters are built from **state store + algorithm + deployment shape + synchronization primitives**. The right combination depends on latency budget, accuracy needs, operational maturity, and whether limits are global or regional.

#### State store

| Option | Strengths | Weaknesses | When to choose |
|--------|-----------|------------|----------------|
| **Redis** | Sub-ms reads/writes, Lua for atomic multi-key logic, TTL, clustering, mature ops story | Another dependency; single-region strong consistency is natural; cross-region needs a strategy | Default for distributed APIs at scale |
| **Memcached** | Very fast, simple GET/INCR | No Lua, no rich data structures, weaker TTL/story for complex windows; harder to do atomic sliding-window math | High-QPS simple fixed-window or INCR-only limits |
| **In-process memory** | Zero network hop, simplest | Not shared across instances → **per-server** limits unless sticky sessions; lost on restart | Edge-only or canary; local burst protection layered on top of Redis |
| **DynamoDB** | Durable, regional, pay-per-use; conditional writes for counters | Higher p99 than Redis; need idempotent design; cost at extreme QPS | Control plane, audit, rule config; or when you already standardize on Dynamo for hot path |

!!! tip
    **Interview angle:** Redis wins the **default** because you need **shared mutable state** with **atomic updates** across many app servers. Memcached is acceptable only when your algorithm maps cleanly to simple counters. In-process is never enough alone for a horizontally scaled API unless you explicitly partition traffic (sticky) or accept approximate per-host limits.

#### Algorithm — production choice and tradeoffs

The file already compares algorithms; in interviews, explain **why** you pick one for production:

| Production goal | Prefer | Why it wins | What you give up |
|-----------------|--------|-------------|------------------|
| Smooth per-minute/hour limits without boundary spikes | **Sliding window counter** | O(1) state, good accuracy, fits Redis INCR + two windows | Small approximation vs sliding log |
| Product needs **burst allowance** (e.g. “100/min but burst 20 in a second”) | **Token bucket** | Natural burst semantics; one hash per client in Redis | Refill tuning; not identical to “N per calendar minute” |
| Strict **calendar windows** and simplicity | **Fixed window** | Cheapest operations | Boundary burst (classic interview trap) |
| Billing or compliance needs **exact** request counts in a window | **Sliding window log** | Exact | Memory and Redis memory proportional to request rate |

**Why sliding window counter is often “the” production default:** it balances **memory**, **accuracy**, and **implementability** in Redis without storing every timestamp. **Why token bucket** when the PM says “users can spike”: bursts are a product requirement, not a bug—token bucket encodes that explicitly.

#### Deployment model

| Model | Pros | Cons | Typical use |
|-------|------|------|-------------|
| **API gateway** (Kong, AWS API Gateway, Envoy rate limit filter) | Central policy, one place to audit, consistent headers | Gateway becomes critical path; vendor limits on expressiveness | Multi-team APIs, edge-first security |
| **Middleware** in app (framework filter/interceptor) | Full request context (user, route, body size); fast iteration | Duplicated across services if not shared as library | Single service or shared internal library |
| **Sidecar** (e.g. Envoy + ext_authz / local RL) | Consistent per-pod behavior, language-agnostic | More moving parts; two hops if calling Redis from sidecar | Kubernetes mesh, uniform limits per workload |
| **Standalone rate limiter service** | Independent scaling, clear SLO, reusable across many callers | Extra network hop; must stay on critical path budget | Large orgs, gRPC “check limit” used by many backends |

#### Synchronization

| Approach | Behavior | Tradeoff |
|----------|----------|----------|
| **Redis Lua** | Read-modify-write in one atomic script | Best correctness/perf balance; keep scripts short |
| **Single-key atomic ops** (`INCR`, `DECR`) | Simple, fast | Multi-key windows need careful ordering or Lua |
| **Distributed locks** (Redlock, etc.) | Serializes updates | Higher latency, complexity, partition edge cases—**usually avoid** for hot-path limiting |
| **Eventually consistent replicas** | Lower read latency | Stale reads → occasional under- or over-counting; sometimes acceptable for soft limits |

**Our choice (typical interview answer):** **Redis** (clustered as needed) with **sliding window counter** or **token bucket** implemented via **Lua** (or carefully ordered `INCR` + rollback) for atomicity; **middleware or API gateway** for enforcement so product teams share one policy engine; **fail-open** on Redis errors unless compliance mandates otherwise. Add **PostgreSQL** (or similar) only for **rules, audit, and exemptions**—not for per-request hot path.

---

### CAP Theorem Analysis

Rate limiters sit in an interesting CAP position: the **counter store** is effectively a **low-latency coordination service**. Interviewers want to hear you connect **consistency of counts**, **availability of the API**, and **partition tolerance** of the infrastructure.

| Dimension | Typical stance | Why |
|-----------|----------------|-----|
| **Consistency** | Strong per-key updates on the **Redis primary** (Lua/INCR) | You want a single authoritative count per `(client, rule, window)` on the hot path |
| **Availability (of limiting)** | Often **degraded**: **fail open** (allow traffic) when Redis is unavailable | Protecting the business from total outage usually beats strict enforcement |
| **Partition tolerance** | Infrastructure partitions happen; **you cannot** have both strict global limits and full availability without compromise | Cross-region splits duplicate or miss counts unless you add coordination |

**Multi-region:** a **single global Redis** optimizes consistency of one number but hurts **latency and partition resilience**. **Per-region Redis** improves availability and RTT but **violates a strict global limit** unless you add async reconciliation (eventually consistent) or a **partitioned budget** (e.g. 200/min per region).

**During a Redis partition:** clients on the minority side may see **stale** or **split-brain** counts if reads go to replicas or split primaries—designs usually **route writes to one primary** per key and accept **unavailability of enforcement** (fail open) rather than wrong hard enforcement. **Split clients** (some to each partition) can **exceed** the intended limit until the partition heals—acknowledge this as **bounded over-admission**.

```mermaid
flowchart TB
  subgraph CP["Consistency vs availability under partition"]
    W[Writers to single Redis primary per shard]
    R[Reads from primary for limit checks]
    P[Network partition]
    W --> P
    R --> P
    P -->|"minority partition"| FO[Fail open OR queue checks]
    P -->|healed| REC["Reconcile counters / accept over-count"]
  end
```

!!! tip
    **Sound bite:** *“We’re not choosing CAP for the whole company—we’re choosing it for the rate limit path: **strong consistency per key on the primary** for correctness, **partition tolerance** via Redis clustering, and **availability of the edge** via fail-open when the store is unreachable, trading strict enforcement for site uptime.”*

---

### Database Schema / State Model

Separate **hot-path state** (ephemeral counters) from **cold-path configuration** (durable rules and audit).

#### Redis key patterns and values (hot path)

| Concept | Key pattern (example) | Value / structure | TTL |
|---------|-------------------------|-------------------|-----|
| Sliding window counter | `rl:{rule_id}:{subject_hash}:w:{window_id}` | String integer count | ~2× window |
| Token bucket | `rl:tb:{rule_id}:{subject_hash}` | Hash: `tokens`, `last_refill` (ms) | derived from refill |
| Composite subject | `subject` = hash of `(tenant_id, api_key_id)` or `ip` | — | — |
| Burst / cost | Same key family; **cost** as weighted tokens | Lua subtracts `cost` | Same as bucket |

**Subject keying:** prefer **stable IDs** (API key id, user id) over raw secrets in keys; hash if keys are sensitive.

#### Rule definition (logical model)

| Field | Purpose |
|-------|---------|
| `rule_id` | Stable identifier |
| `scope` | global / tenant / endpoint / method |
| `limit`, `window_sec` | Policy |
| `algorithm` | `sliding_counter` \| `token_bucket` \| … |
| `priority` | Which rule wins when multiple apply |

#### PostgreSQL schema (rules, audit, exemptions)

Rules and exemptions change rarely; use a relational store for **versioning**, **admin UI**, and **compliance audit**. Caches push active rules to Redis or app memory.

```sql
-- Rate limit rules (configuration)
CREATE TABLE rate_limit_rules (
  id              BIGSERIAL PRIMARY KEY,
  rule_uuid       UUID NOT NULL UNIQUE DEFAULT gen_random_uuid(),
  name            TEXT NOT NULL,
  scope           TEXT NOT NULL CHECK (scope IN ('global','tenant','endpoint')),
  tenant_id       BIGINT,  -- NULL = global scope; optional FK to tenants(id) in your schema
  http_method     TEXT,
  path_pattern    TEXT,
  algorithm       TEXT NOT NULL CHECK (algorithm IN ('sliding_window_counter','token_bucket','fixed_window')),
  max_requests    INTEGER NOT NULL CHECK (max_requests > 0),
  window_seconds  INTEGER NOT NULL CHECK (window_seconds > 0),
  burst_capacity  INTEGER,
  priority        INTEGER NOT NULL DEFAULT 0,
  enabled         BOOLEAN NOT NULL DEFAULT TRUE,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_rate_rules_tenant ON rate_limit_rules(tenant_id) WHERE tenant_id IS NOT NULL;

-- Exemptions (allowlists)
CREATE TABLE rate_limit_exemptions (
  id           BIGSERIAL PRIMARY KEY,
  rule_id      BIGINT NOT NULL REFERENCES rate_limit_rules(id) ON DELETE CASCADE,
  subject_type TEXT NOT NULL CHECK (subject_type IN ('api_key','user','ip_cidr','service_account')),
  subject_value TEXT NOT NULL,
  reason       TEXT,
  expires_at   TIMESTAMPTZ,
  created_by   TEXT NOT NULL,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (rule_id, subject_type, subject_value)
);

-- Audit trail for config changes
CREATE TABLE rate_limit_rule_audit (
  id          BIGSERIAL PRIMARY KEY,
  rule_id     BIGINT REFERENCES rate_limit_rules(id),
  action      TEXT NOT NULL CHECK (action IN ('insert','update','delete','enable','disable')),
  old_row     JSONB,
  new_row     JSONB,
  actor       TEXT NOT NULL,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

**Optional usage snapshots:** batch **aggregate** usage into `usage_daily` for billing—not for synchronous enforcement.

---

### API Design

Expose **machine-facing** headers on every checked response, and **operator-facing** HTTP APIs for configuration and observability.

#### Check path (middleware / gateway) — response headers

Clients should receive consistent headers on **200** and **429** (and often on errors). Align with common conventions (GitHub, Stripe-style); `RateLimit-*` is newer IETF draft style—mention both in interviews.

| Header | Meaning | Example |
|--------|---------|---------|
| `X-RateLimit-Limit` | Max requests in the policy window | `100` |
| `X-RateLimit-Remaining` | Estimated remaining in current window | `42` |
| `X-RateLimit-Reset` | Unix time (seconds) when the window resets | `1743877200` |
| `Retry-After` | Seconds (or HTTP-date) to wait after **429** | `30` |
| `RateLimit-Limit` / `RateLimit-Remaining` / `RateLimit-Reset` | Draft standard naming (optional dual-publish) | same values |

**429 example:**

```http
HTTP/1.1 429 Too Many Requests
Content-Type: application/json
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1743877260
Retry-After: 45

{"error":"rate_limit_exceeded","message":"Too many requests","rule":"per_api_key_per_minute"}
```

**200 example (successful check):**

```http
HTTP/1.1 200 OK
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 41
X-RateLimit-Reset: 1743877260
```

!!! tip
    **Why headers matter:** they let clients **back off** without guessing, reduce **retry storms**, and support **SDK** throttling—say this explicitly in interviews.

#### Configuration API (operators / tenants)

| Method & path | Purpose |
|---------------|---------|
| `GET /v1/rate-limit/rules` | List rules (filter by tenant, scope) |
| `POST /v1/rate-limit/rules` | Create rule |
| `PUT /v1/rate-limit/rules/{ruleId}` | Update rule |
| `DELETE /v1/rate-limit/rules/{ruleId}` | Soft-delete or disable |
| `POST /v1/rate-limit/rules/{ruleId}/exemptions` | Add exemption subject |
| `DELETE /v1/rate-limit/exemptions/{exemptionId}` | Remove exemption |

**Example create rule:**

```http
POST /v1/rate-limit/rules HTTP/1.1
Content-Type: application/json
Authorization: Bearer <admin_token>

{
  "name": "default_per_key",
  "scope": "global",
  "algorithm": "sliding_window_counter",
  "maxRequests": 100,
  "windowSeconds": 60,
  "priority": 10
}
```

#### Usage query API

| Method & path | Purpose |
|---------------|---------|
| `GET /v1/rate-limit/usage?subjectType=api_key&subjectId=...&ruleId=...` | Current count / remaining for a subject (read from Redis or replica with **staleness note**) |
| `GET /v1/rate-limit/usage/me` | Caller’s own usage (scoped token) |

**Response sketch:**

```json
{
  "ruleId": "rl_default_per_key",
  "limit": 100,
  "remaining": 73,
  "resetAt": "2026-04-05T12:01:00Z",
  "windowSeconds": 60
}
```

#### Internal check API (optional)

Microservices may call a small **gRPC/HTTP** service: `CheckLimit(subject, rule_id) → { allowed, remaining, reset }` so enforcement stays consistent outside a single framework.

---

## Step 2: Back-of-Envelope Estimation

Before choosing an algorithm or architecture, estimate the scale to understand memory, bandwidth, and infrastructure needs.

### Assumptions

```
- 10 million unique API clients (identified by API key)
- Average client makes 500 requests/day
- Rate limit: 100 requests per minute per client
- Rate limit check must be < 1ms latency
```

### Traffic Estimation

```
Total requests/day  = 10M × 500 = 5 billion requests/day
Average QPS         = 5B / 86,400 ≈ 58,000 QPS
Peak QPS (3x)       = ~175,000 QPS
```

### Memory Estimation (Token Bucket)

```
Per-client state:
  - Client ID (key): ~50 bytes (API key hash)
  - Tokens remaining: 8 bytes (double)
  - Last refill timestamp: 8 bytes (long)
  - Total per client: ~66 bytes → round to 100 bytes (with overhead)

Total memory for all clients:
  10M × 100 bytes = 1 GB

Active clients (20% at peak):
  2M × 100 bytes = 200 MB → fits in a single Redis instance
```

### Memory Estimation (Sliding Window Log)

```
Per-request entry: ~20 bytes (timestamp + overhead)
Worst case: each client at max rate = 100 entries/minute
Active clients: 2M × 100 × 20 bytes = 4 GB → requires Redis cluster
```

### Infrastructure Decision

| Metric | Value | Implication |
|--------|-------|-------------|
| Peak QPS | 175K | Single Redis handles 100K+; may need 2-3 nodes |
| Memory (token bucket) | ~200 MB active | Single Redis instance is sufficient |
| Memory (sliding window) | ~4 GB active | Redis cluster or sharding needed |
| Latency target | < 1ms | Redis in-memory is well within target |
| Availability | 99.99% | Redis Sentinel or Cluster for failover |

!!! tip
    Token bucket is clearly more memory-efficient. This estimation helps justify the algorithm choice — not just on correctness but on infrastructure cost.

---

## Step 3: High-Level Design

### 3.1 Rate Limiting Algorithms

There are several algorithms for rate limiting, each with different characteristics.

### Algorithm 1: Token Bucket

The most popular algorithm. Imagine a bucket that holds tokens:
- Tokens are added at a constant rate (e.g., 10 tokens/second)
- Each request consumes one token
- If the bucket is empty, the request is rejected
- Bucket has a maximum capacity (allows bursts)

```
┌─────────────────────────────┐
│         Token Bucket        │
│                             │
│  ┌─────┐                    │
│  │ + │ ← Tokens added at    │
│  └──┬──┘   constant rate    │
│     ▼                       │
│  ┌─────────────────────┐    │
│  │ 🪙 🪙 🪙 🪙 🪙       │    │  ← Bucket (max capacity)
│  └─────────────────────┘    │
│     │                       │
│     ▼                       │
│  Request takes 1 token      │
│                             │
└─────────────────────────────┘
```

**Implementation:**

```python
import time

class TokenBucket:
    def __init__(self, capacity: int, refill_rate: float):
        """
        Args:
            capacity: Maximum tokens the bucket can hold
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
    
    def _refill(self):
        """Add tokens based on time elapsed."""
        now = time.time()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate
        
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def allow_request(self, tokens: int = 1) -> bool:
        """Check if request is allowed and consume tokens."""
        self._refill()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def get_wait_time(self, tokens: int = 1) -> float:
        """How long to wait before tokens are available."""
        self._refill()
        
        if self.tokens >= tokens:
            return 0
        
        tokens_needed = tokens - self.tokens
        return tokens_needed / self.refill_rate
```

**Characteristics:**

| Aspect | Token Bucket |
|--------|-------------|
| **Burst handling** | ✅ Allows bursts up to bucket capacity |
| **Smooth rate** | ✅ Averages to refill rate over time |
| **Memory** | O(1) per client |
| **Complexity** | Simple |

**Use case:** When you want to allow bursts but limit average rate.

### Algorithm 2: Leaky Bucket

Requests enter a queue (bucket) and are processed at a constant rate. If the bucket is full, new requests are rejected.

```
┌─────────────────────────────┐
│         Leaky Bucket        │
│                             │
│  Requests → ┌─────────────┐ │
│             │ Queue       │ │  ← Bucket (limited size)
│             │ ░░░░░░░░░   │ │
│             └──────┬──────┘ │
│                    │        │
│                    ▼        │
│              ┌─────────┐    │
│              │ Process │ ← Fixed rate output
│              └─────────┘    │
└─────────────────────────────┘
```

**Implementation:**

```python
import time
from collections import deque
import threading

class LeakyBucket:
    def __init__(self, capacity: int, leak_rate: float):
        """
        Args:
            capacity: Maximum requests in queue
            leak_rate: Requests processed per second
        """
        self.capacity = capacity
        self.leak_rate = leak_rate
        self.queue = deque()
        self.last_leak = time.time()
        self.lock = threading.Lock()
    
    def _leak(self):
        """Process requests based on time elapsed."""
        now = time.time()
        elapsed = now - self.last_leak
        requests_to_leak = int(elapsed * self.leak_rate)
        
        for _ in range(min(requests_to_leak, len(self.queue))):
            self.queue.popleft()
        
        if requests_to_leak > 0:
            self.last_leak = now
    
    def allow_request(self) -> bool:
        """Add request to queue if there's space."""
        with self.lock:
            self._leak()
            
            if len(self.queue) < self.capacity:
                self.queue.append(time.time())
                return True
            return False
```

**Characteristics:**

| Aspect | Leaky Bucket |
|--------|-------------|
| **Burst handling** | ❌ No bursts—constant output rate |
| **Smooth rate** | ✅ Very smooth, predictable |
| **Memory** | O(n) where n = queue size |
| **Complexity** | Moderate |

**Use case:** When you need a perfectly smooth, constant rate (e.g., video streaming).

### Algorithm 3: Fixed Window Counter

Divide time into fixed windows (e.g., 1-minute intervals). Count requests in each window.

```
┌─────────────────────────────────────────────────────────────┐
│                    Fixed Window Counter                      │
│                                                              │
│  Window 1        Window 2        Window 3        Window 4   │
│  [00:00-01:00]   [01:00-02:00]   [02:00-03:00]  [03:00-04:00]│
│  ┌──────────┐    ┌──────────┐    ┌──────────┐   ┌──────────┐│
│  │ 95/100   │    │ 100/100  │    │ 45/100   │   │ 0/100    ││
│  │ ✓        │    │ FULL     │    │ ✓        │   │ ✓        ││
│  └──────────┘    └──────────┘    └──────────┘   └──────────┘│
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Implementation:**

```python
import time

class FixedWindowCounter:
    def __init__(self, limit: int, window_size: int):
        """
        Args:
            limit: Max requests per window
            window_size: Window size in seconds
        """
        self.limit = limit
        self.window_size = window_size
        self.counts = {}  # window_key -> count
    
    def _get_window_key(self, timestamp: float) -> int:
        """Get the window this timestamp belongs to."""
        return int(timestamp // self.window_size)
    
    def allow_request(self, client_id: str) -> bool:
        """Check if request is allowed."""
        now = time.time()
        window_key = self._get_window_key(now)
        key = f"{client_id}:{window_key}"
        
        current_count = self.counts.get(key, 0)
        
        if current_count < self.limit:
            self.counts[key] = current_count + 1
            return True
        return False
    
    def cleanup_old_windows(self, client_id: str):
        """Remove old window data to prevent memory bloat."""
        now = time.time()
        current_window = self._get_window_key(now)
        
        keys_to_delete = [
            k for k in self.counts 
            if k.startswith(f"{client_id}:") and 
               int(k.split(":")[1]) < current_window - 1
        ]
        for k in keys_to_delete:
            del self.counts[k]
```

**Problem: Boundary Burst**

A client could make 100 requests at 00:59 and 100 more at 01:01—200 requests in 2 seconds!

```
Window 1 [00:00-01:00]    Window 2 [01:00-02:00]
         │ 100 requests    │ 100 requests
         ▼                 ▼
    ─────┼────────────────┼─────
      00:59            01:01
         └── 2 seconds ──┘
         └── 200 requests! ──┘
```

**Use case:** Simple scenarios where occasional bursts at boundaries are acceptable.

### Algorithm 4: Sliding Window Log

Track the timestamp of each request. Count requests within the sliding window.

**Implementation:**

```python
import time
from collections import deque
import threading

class SlidingWindowLog:
    def __init__(self, limit: int, window_size: int):
        """
        Args:
            limit: Max requests per window
            window_size: Window size in seconds
        """
        self.limit = limit
        self.window_size = window_size
        self.logs = {}  # client_id -> deque of timestamps
        self.lock = threading.Lock()
    
    def allow_request(self, client_id: str) -> bool:
        """Check if request is allowed."""
        now = time.time()
        window_start = now - self.window_size
        
        with self.lock:
            if client_id not in self.logs:
                self.logs[client_id] = deque()
            
            log = self.logs[client_id]
            
            # Remove timestamps outside the window
            while log and log[0] <= window_start:
                log.popleft()
            
            if len(log) < self.limit:
                log.append(now)
                return True
            return False
```

**Characteristics:**

| Aspect | Sliding Window Log |
|--------|-------------------|
| **Accuracy** | ✅ Perfect—no boundary issues |
| **Memory** | ❌ O(limit) per client—stores every timestamp |
| **Complexity** | Higher |

**Use case:** When you need perfect accuracy and can afford the memory.

### Algorithm 5: Sliding Window Counter (Recommended)

Hybrid of fixed window and sliding window. Uses weighted average of current and previous window.

```
┌─────────────────────────────────────────────────────────────┐
│                  Sliding Window Counter                      │
│                                                              │
│  Previous Window [00:00-01:00]   Current Window [01:00-02:00]│
│  ┌──────────────────────────┐    ┌──────────────────────────┐│
│  │         70 requests      │    │    30 requests          ││
│  └──────────────────────────┘    └──────────────────────────┘│
│                              │                               │
│                     Current time: 01:15                      │
│                              │                               │
│  Sliding window: 01:00-01:15 is 25% into current window     │
│                                                              │
│  Weight of previous: 75% (45 minutes overlap)               │
│  Weight of current: 100%                                     │
│                                                              │
│  Weighted count = 70 × 0.75 + 30 × 1.0 = 52.5 + 30 = 82.5   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Implementation:**

```python
import time

class SlidingWindowCounter:
    def __init__(self, limit: int, window_size: int):
        """
        Args:
            limit: Max requests per window
            window_size: Window size in seconds
        """
        self.limit = limit
        self.window_size = window_size
        self.windows = {}  # client_id -> {window_key: count}
    
    def allow_request(self, client_id: str) -> bool:
        """Check if request is allowed using weighted average."""
        now = time.time()
        
        current_window = int(now // self.window_size)
        previous_window = current_window - 1
        
        # Position within current window (0.0 to 1.0)
        window_position = (now % self.window_size) / self.window_size
        
        if client_id not in self.windows:
            self.windows[client_id] = {}
        
        client_windows = self.windows[client_id]
        
        # Get counts for current and previous windows
        current_count = client_windows.get(current_window, 0)
        previous_count = client_windows.get(previous_window, 0)
        
        # Calculate weighted average
        # Previous window contributes (1 - window_position) weight
        # because that's how much of it overlaps with our sliding window
        weighted_count = (
            previous_count * (1 - window_position) + 
            current_count
        )
        
        if weighted_count < self.limit:
            client_windows[current_window] = current_count + 1
            # Cleanup old windows
            self._cleanup(client_id, current_window - 2)
            return True
        return False
    
    def _cleanup(self, client_id: str, before_window: int):
        """Remove windows older than needed."""
        if client_id in self.windows:
            keys_to_delete = [
                k for k in self.windows[client_id] 
                if k < before_window
            ]
            for k in keys_to_delete:
                del self.windows[client_id][k]
```

**Characteristics:**

| Aspect | Sliding Window Counter |
|--------|----------------------|
| **Accuracy** | ✅ Very good (slight approximation) |
| **Memory** | ✅ O(1) per client—just 2 counters |
| **Boundary handling** | ✅ Smooth, no burst at boundaries |
| **Complexity** | Simple |

**This is the recommended algorithm** for most use cases.

### Algorithm Comparison

| Algorithm | Memory | Accuracy | Burst Handling | Complexity |
|-----------|--------|----------|----------------|------------|
| Token Bucket | O(1) | Good | Allows controlled bursts | Simple |
| Leaky Bucket | O(n) | Perfect | No bursts | Moderate |
| Fixed Window | O(1) | Poor (boundaries) | Boundary issues | Simple |
| Sliding Log | O(n) | Perfect | Smooth | Higher |
| Sliding Window Counter | O(1) | Very Good | Smooth | Simple |

**Recommendations:**
- **Token Bucket:** When you want to allow bursts (API rate limiting)
- **Sliding Window Counter:** When you want smooth limiting (general purpose)
- **Leaky Bucket:** When you need constant output rate (streaming)

---

### 3.2 System Architecture

For a distributed system, we need a centralized store for rate limit state.

### Architecture Overview

```mermaid
flowchart TB
    subgraph Clients["Clients"]
        C1[Client 1]
        C2[Client 2]
        C3[Client N]
    end
    
    subgraph Edge["Edge Layer"]
        LB[Load Balancer]
    end
    
    subgraph App["Application Layer"]
        API1[API Server 1]
        API2[API Server 2]
        API3[API Server N]
    end
    
    subgraph RateLimit["Rate Limiting Layer"]
        RL[Rate Limiter Service]
        Redis[(Redis Cluster)]
    end
    
    subgraph Backend["Backend Services"]
        Svc[Backend Services]
    end
    
    C1 --> LB
    C2 --> LB
    C3 --> LB
    
    LB --> API1
    LB --> API2
    LB --> API3
    
    API1 --> RL
    API2 --> RL
    API3 --> RL
    
    RL --> Redis
    
    API1 --> Svc
    API2 --> Svc
    API3 --> Svc
```

### Where to Place the Rate Limiter?

| Location | Pros | Cons |
|----------|------|------|
| **Client-side** | Reduces server load | Can be bypassed, unreliable |
| **Load balancer** | Centralized, early rejection | Limited customization |
| **API Gateway** | Flexible, centralized | Single point of failure |
| **Application code** | Full control, context-aware | Code duplication |
| **Dedicated service** | Clean separation | Additional latency |

**Recommendation:** API Gateway or dedicated middleware is usually best.

### Request Flow

```mermaid
sequenceDiagram
    participant Client
    participant LB as Load Balancer
    participant API as API Server
    participant RL as Rate Limiter
    participant Redis
    participant Backend
    
    Client->>LB: Request
    LB->>API: Forward
    API->>RL: Check rate limit
    RL->>Redis: INCR key, GET count
    Redis-->>RL: Current count
    
    alt Under limit
        RL-->>API: Allowed
        API->>Backend: Process request
        Backend-->>API: Response
        API-->>Client: 200 OK + X-RateLimit headers
    else Over limit
        RL-->>API: Rejected
        API-->>Client: 429 Too Many Requests + Retry-After
    end
```

---

## Step 4: Deep Dive

### 4.1 Distributed Rate Limiting with Redis

For a distributed system, we need atomic operations. Redis provides these.

### Redis Implementation: Sliding Window Counter

```python
import redis
import time

class DistributedRateLimiter:
    def __init__(self, redis_client: redis.Redis, limit: int, window_size: int):
        """
        Args:
            redis_client: Redis connection
            limit: Max requests per window
            window_size: Window size in seconds
        """
        self.redis = redis_client
        self.limit = limit
        self.window_size = window_size
    
    def is_allowed(self, client_id: str) -> tuple[bool, dict]:
        """
        Check if request is allowed.
        Returns (is_allowed, rate_limit_info)
        """
        now = time.time()
        current_window = int(now // self.window_size)
        previous_window = current_window - 1
        window_position = (now % self.window_size) / self.window_size
        
        current_key = f"ratelimit:{client_id}:{current_window}"
        previous_key = f"ratelimit:{client_id}:{previous_window}"
        
        # Use pipeline for atomic operations
        pipe = self.redis.pipeline()
        pipe.get(previous_key)
        pipe.incr(current_key)
        pipe.expire(current_key, self.window_size * 2)
        results = pipe.execute()
        
        previous_count = int(results[0] or 0)
        current_count = int(results[1])
        
        # Calculate weighted count (excluding this request)
        weighted_count = (
            previous_count * (1 - window_position) + 
            (current_count - 1)  # Exclude current request
        )
        
        allowed = weighted_count < self.limit
        
        # If not allowed, decrement the counter we just incremented
        if not allowed:
            self.redis.decr(current_key)
            current_count -= 1
        
        remaining = max(0, self.limit - int(weighted_count) - 1)
        reset_time = (current_window + 1) * self.window_size
        
        rate_limit_info = {
            "limit": self.limit,
            "remaining": remaining,
            "reset": int(reset_time),
            "retry_after": None if allowed else int(reset_time - now)
        }
        
        return allowed, rate_limit_info
```

### Redis Implementation: Token Bucket with Lua Script

For perfect atomicity, use a Lua script:

```python
import redis
import time

TOKEN_BUCKET_SCRIPT = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local requested = tonumber(ARGV[4])

-- Get current bucket state
local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
local tokens = tonumber(bucket[1]) or capacity
local last_refill = tonumber(bucket[2]) or now

-- Calculate tokens to add based on time elapsed
local elapsed = now - last_refill
local tokens_to_add = elapsed * refill_rate
tokens = math.min(capacity, tokens + tokens_to_add)

-- Check if request can be allowed
local allowed = 0
if tokens >= requested then
    tokens = tokens - requested
    allowed = 1
end

-- Save state
redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
redis.call('EXPIRE', key, math.ceil(capacity / refill_rate) * 2)

return {allowed, tokens}
"""

class TokenBucketLimiter:
    def __init__(self, redis_client: redis.Redis, capacity: int, refill_rate: float):
        self.redis = redis_client
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.script = self.redis.register_script(TOKEN_BUCKET_SCRIPT)
    
    def is_allowed(self, client_id: str, tokens: int = 1) -> tuple[bool, float]:
        """Check if request is allowed."""
        key = f"tokenbucket:{client_id}"
        now = time.time()
        
        result = self.script(
            keys=[key],
            args=[self.capacity, self.refill_rate, now, tokens]
        )
        
        allowed = bool(result[0])
        remaining_tokens = float(result[1])
        
        return allowed, remaining_tokens
```

### Why Lua Scripts?

Redis Lua scripts are **atomic**—they execute without interruption. This prevents race conditions:

```
Without Lua (race condition possible):
Thread 1: GET count → 99
Thread 2: GET count → 99
Thread 1: SET count = 100
Thread 2: SET count = 100  ← Both think they got the 100th request

With Lua:
Thread 1: [GET and SET atomically] → 100
Thread 2: [GET and SET atomically] → 101, rejected
```

---

### 4.2 Handling Distributed Challenges

### Challenge 1: Race Conditions

**Problem:** Multiple API servers check rate limits simultaneously.

**Solution:** Use atomic Redis operations (INCR, Lua scripts).

```python
# Bad: Race condition
count = redis.get(key)  # Thread 1: 99, Thread 2: 99
if count < limit:
    redis.incr(key)      # Both increment: 100, 101
    return True          # Both return True!

# Good: Atomic
count = redis.incr(key)  # Atomic: returns 100, then 101
if count <= limit:
    return True
else:
    redis.decr(key)      # Rollback
    return False
```

### Challenge 2: Clock Synchronization

**Problem:** Different servers have different clocks, causing inconsistent window calculations.

**Solutions:**
1. **Use Redis time:** `redis.time()` returns server time
2. **NTP synchronization:** Keep all servers synced
3. **Use sliding window:** Less sensitive to exact timing

```python
def get_current_window(self) -> int:
    # Use Redis server time for consistency
    redis_time = self.redis.time()
    timestamp = redis_time[0] + redis_time[1] / 1_000_000
    return int(timestamp // self.window_size)
```

### Challenge 3: Redis Failures

**Problem:** What happens when Redis is down?

**Options:**

| Strategy | Behavior | Risk |
|----------|----------|------|
| **Fail open** | Allow all requests | System overload possible |
| **Fail closed** | Reject all requests | Service unavailable |
| **Local fallback** | Use in-memory limiter | Inconsistent limits |

**Recommendation:** Fail open with alerts. Rate limiting is protection, not core functionality.

```python
async def check_rate_limit(self, client_id: str) -> bool:
    try:
        allowed, info = self.limiter.is_allowed(client_id)
        return allowed
    except redis.RedisError as e:
        # Log and alert
        logger.error(f"Redis error in rate limiter: {e}")
        metrics.incr("rate_limiter.redis_failure")
        
        # Fail open—allow the request
        return True
```

### Challenge 4: Memory and Performance

**Problem:** Millions of clients = millions of Redis keys.

**Solutions:**

1. **Aggressive TTL:** Set short expiration on keys
```python
# Key expires after 2 windows
redis.expire(key, window_size * 2)
```

2. **Key compression:** Use shorter key names
```python
# Instead of: "ratelimit:user:12345:endpoint:/api/v1/users:window:1234567890"
# Use: "rl:12345:/api/v1/users:1234567890"
```

3. **Redis Cluster:** Shard across multiple Redis nodes

4. **Local caching:** Cache rate limit decisions briefly (risky)

---

### 4.3 HTTP Response Headers

Communicate rate limit status to clients via headers:

| Header | Description | Example |
|--------|-------------|---------|
| `X-RateLimit-Limit` | Max requests in window | `100` |
| `X-RateLimit-Remaining` | Requests left in window | `45` |
| `X-RateLimit-Reset` | Unix timestamp when limit resets | `1705320000` |
| `Retry-After` | Seconds to wait (on 429) | `30` |

**Implementation:**

```python
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_id = get_client_id(request)  # API key, IP, etc.
    
    allowed, info = rate_limiter.is_allowed(client_id)
    
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded"},
            headers={
                "X-RateLimit-Limit": str(info["limit"]),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(info["reset"]),
                "Retry-After": str(info["retry_after"])
            }
        )
    
    response = await call_next(request)
    
    # Add rate limit headers to successful responses
    response.headers["X-RateLimit-Limit"] = str(info["limit"])
    response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
    response.headers["X-RateLimit-Reset"] = str(info["reset"])
    
    return response

def get_client_id(request: Request) -> str:
    """Extract client identifier from request."""
    # Prefer API key
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"api:{api_key}"
    
    # Fall back to IP
    return f"ip:{request.client.host}"
```

---

### 4.4 Advanced Features

### Multiple Rate Limit Rules

Apply different limits to different endpoints or user tiers:

```python
from dataclasses import dataclass
from typing import List

@dataclass
class RateLimitRule:
    name: str
    limit: int
    window_size: int  # seconds
    key_prefix: str

class MultiRuleRateLimiter:
    def __init__(self, redis_client: redis.Redis, rules: List[RateLimitRule]):
        self.redis = redis_client
        self.rules = rules
    
    def check_all_rules(self, client_id: str, endpoint: str) -> tuple[bool, dict]:
        """Check all applicable rules."""
        results = []
        
        for rule in self.get_applicable_rules(endpoint):
            allowed, info = self.check_rule(client_id, rule)
            results.append((rule, allowed, info))
            
            if not allowed:
                return False, info
        
        # All rules passed—return most restrictive remaining
        min_remaining = min(r[2]["remaining"] for r in results)
        return True, {"remaining": min_remaining}
    
    def get_applicable_rules(self, endpoint: str) -> List[RateLimitRule]:
        """Get rules that apply to this endpoint."""
        return [r for r in self.rules if self.rule_matches(r, endpoint)]

# Example configuration
rules = [
    RateLimitRule("global", limit=1000, window_size=60, key_prefix="global"),
    RateLimitRule("per_endpoint", limit=100, window_size=60, key_prefix="endpoint"),
    RateLimitRule("burst", limit=10, window_size=1, key_prefix="burst"),
]
```

### User Tiers

Different limits for different user types:

```python
TIER_LIMITS = {
    "free": {"limit": 100, "window": 3600},       # 100/hour
    "basic": {"limit": 1000, "window": 3600},     # 1000/hour
    "premium": {"limit": 10000, "window": 3600},  # 10000/hour
    "enterprise": {"limit": 100000, "window": 3600},
}

async def get_rate_limit_for_user(user: User) -> RateLimitConfig:
    tier = user.subscription_tier
    config = TIER_LIMITS.get(tier, TIER_LIMITS["free"])
    return RateLimitConfig(**config)
```

### Request Costing

Some requests cost more than others:

```python
ENDPOINT_COSTS = {
    "GET /users": 1,
    "POST /users": 5,
    "GET /reports/generate": 100,  # Expensive operation
    "POST /ai/generate": 50,       # GPU-intensive
}

def get_request_cost(method: str, path: str) -> int:
    key = f"{method} {path}"
    return ENDPOINT_COSTS.get(key, 1)

async def rate_limit_with_cost(request: Request):
    cost = get_request_cost(request.method, request.url.path)
    allowed, remaining = token_bucket.is_allowed(client_id, tokens=cost)
    return allowed
```

---

### 4.5 Multi-Language Implementations

### Token Bucket Rate Limiter with Redis

Same algorithm as **Redis Implementation: Token Bucket with Lua Script** in section 4.1 (full walkthrough and prose above).

=== "Python"

    ```python
    import redis
    import time
    
    TOKEN_BUCKET_SCRIPT = """
    local key = KEYS[1]
    local capacity = tonumber(ARGV[1])
    local refill_rate = tonumber(ARGV[2])
    local now = tonumber(ARGV[3])
    local requested = tonumber(ARGV[4])
    
    -- Get current bucket state
    local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
    local tokens = tonumber(bucket[1]) or capacity
    local last_refill = tonumber(bucket[2]) or now
    
    -- Calculate tokens to add based on time elapsed
    local elapsed = now - last_refill
    local tokens_to_add = elapsed * refill_rate
    tokens = math.min(capacity, tokens + tokens_to_add)
    
    -- Check if request can be allowed
    local allowed = 0
    if tokens >= requested then
        tokens = tokens - requested
        allowed = 1
    end
    
    -- Save state
    redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
    redis.call('EXPIRE', key, math.ceil(capacity / refill_rate) * 2)
    
    return {allowed, tokens}
    """
    
    class TokenBucketLimiter:
        def __init__(self, redis_client: redis.Redis, capacity: int, refill_rate: float):
            self.redis = redis_client
            self.capacity = capacity
            self.refill_rate = refill_rate
            self.script = self.redis.register_script(TOKEN_BUCKET_SCRIPT)
    
        def is_allowed(self, client_id: str, tokens: int = 1) -> tuple[bool, float]:
            """Check if request is allowed."""
            key = f"tokenbucket:{client_id}"
            now = time.time()
    
            result = self.script(
                keys=[key],
                args=[self.capacity, self.refill_rate, now, tokens]
            )
    
            allowed = bool(result[0])
            remaining_tokens = float(result[1])
    
            return allowed, remaining_tokens
    ```

=== "Java"

    ```java
    import redis.clients.jedis.JedisPooled;
    import java.util.List;
    
    public class RedisTokenBucketLimiter {
        private final JedisPooled jedis;
        private final String luaScript;
        private String scriptSha;
    
        public RedisTokenBucketLimiter(JedisPooled jedis) {
            this.jedis = jedis;
            this.luaScript = """
                local key = KEYS[1]
                local capacity = tonumber(ARGV[1])
                local refill_rate = tonumber(ARGV[2])
                local now = tonumber(ARGV[3])
                local requested = tonumber(ARGV[4])
    
                local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
                local tokens = tonumber(bucket[1]) or capacity
                local last_refill = tonumber(bucket[2]) or now
    
                local elapsed = now - last_refill
                tokens = math.min(capacity, tokens + elapsed * refill_rate)
    
                local allowed = 0
                if tokens >= requested then
                    tokens = tokens - requested
                    allowed = 1
                end
    
                redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
                redis.call('EXPIRE', key, math.ceil(capacity / refill_rate) * 2)
    
                return {allowed, math.floor(tokens)}
                """;
            this.scriptSha = jedis.scriptLoad(luaScript);
        }
    
        public record RateLimitResult(boolean allowed, int remainingTokens) {}
    
        public RateLimitResult isAllowed(String clientId, int capacity, double refillRate) {
            String key = "ratelimit:" + clientId;
            double now = System.currentTimeMillis() / 1000.0;
    
            Object result = jedis.evalsha(scriptSha,
                List.of(key),
                List.of(
                    String.valueOf(capacity),
                    String.valueOf(refillRate),
                    String.valueOf(now),
                    "1"
                ));
    
            @SuppressWarnings("unchecked")
            List<Long> values = (List<Long>) result;
            return new RateLimitResult(values.get(0) == 1, values.get(1).intValue());
        }
    }
    ```

=== "Go"

    ```go
    package ratelimit
    
    import (
    	"context"
    	"fmt"
    	"strconv"
    	"time"
    
    	"github.com/redis/go-redis/v9"
    )
    
    var tokenBucketScript = redis.NewScript(`
    local key = KEYS[1]
    local capacity = tonumber(ARGV[1])
    local refill_rate = tonumber(ARGV[2])
    local now = tonumber(ARGV[3])
    local requested = tonumber(ARGV[4])
    
    local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
    local tokens = tonumber(bucket[1]) or capacity
    local last_refill = tonumber(bucket[2]) or now
    
    local elapsed = now - last_refill
    tokens = math.min(capacity, tokens + elapsed * refill_rate)
    
    local allowed = 0
    if tokens >= requested then
        tokens = tokens - requested
        allowed = 1
    end
    
    redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
    redis.call('EXPIRE', key, math.ceil(capacity / refill_rate) * 2)
    
    return {allowed, math.floor(tokens)}
    `)
    
    type RateLimiter struct {
    	client     *redis.Client
    	capacity   int
    	refillRate float64
    }
    
    type Result struct {
    	Allowed   bool
    	Remaining int
    }
    
    func NewRateLimiter(client *redis.Client, capacity int, refillRate float64) *RateLimiter {
    	return &RateLimiter{client: client, capacity: capacity, refillRate: refillRate}
    }
    
    func (rl *RateLimiter) IsAllowed(ctx context.Context, clientID string) (Result, error) {
    	key := fmt.Sprintf("ratelimit:%s", clientID)
    	now := float64(time.Now().UnixMilli()) / 1000.0
    
    	vals, err := tokenBucketScript.Run(ctx, rl.client, []string{key},
    		rl.capacity, rl.refillRate, now, 1).Int64Slice()
    	if err != nil {
    		return Result{Allowed: true, Remaining: rl.capacity}, err // fail open
    	}
    
    	return Result{
    		Allowed:   vals[0] == 1,
    		Remaining: int(vals[1]),
    	}, nil
    }
    ```

### Rate Limiting HTTP Middleware

Same pattern as **HTTP Response Headers** in section 4.3 (`rate_limit_middleware` / `get_client_id`).

=== "Python"

    ```python
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.responses import JSONResponse
    
    app = FastAPI()
    
    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        client_id = get_client_id(request)  # API key, IP, etc.
    
        allowed, info = rate_limiter.is_allowed(client_id)
    
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded"},
                headers={
                    "X-RateLimit-Limit": str(info["limit"]),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(info["reset"]),
                    "Retry-After": str(info["retry_after"])
                }
            )
    
        response = await call_next(request)
    
        # Add rate limit headers to successful responses
        response.headers["X-RateLimit-Limit"] = str(info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(info["reset"])
    
        return response
    
    def get_client_id(request: Request) -> str:
        """Extract client identifier from request."""
        # Prefer API key
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api:{api_key}"
    
        # Fall back to IP
        return f"ip:{request.client.host}"
    ```

=== "Java"

    ```java
    import jakarta.servlet.*;
    import jakarta.servlet.http.*;
    import java.io.IOException;
    
    public class RateLimitFilter implements Filter {
        private final RedisTokenBucketLimiter limiter;
        private final int capacity;
        private final double refillRate;
    
        public RateLimitFilter(RedisTokenBucketLimiter limiter, int capacity, double refillRate) {
            this.limiter = limiter;
            this.capacity = capacity;
            this.refillRate = refillRate;
        }
    
        @Override
        public void doFilter(ServletRequest req, ServletResponse resp, FilterChain chain)
                throws IOException, ServletException {
            HttpServletRequest httpReq = (HttpServletRequest) req;
            HttpServletResponse httpResp = (HttpServletResponse) resp;
    
            String clientId = resolveClientId(httpReq);
            var result = limiter.isAllowed(clientId, capacity, refillRate);
    
            httpResp.setHeader("X-RateLimit-Limit", String.valueOf(capacity));
            httpResp.setHeader("X-RateLimit-Remaining", String.valueOf(result.remainingTokens()));
    
            if (!result.allowed()) {
                httpResp.setStatus(429);
                httpResp.setHeader("Retry-After", String.valueOf((int) (1.0 / refillRate)));
                httpResp.getWriter().write("""
                    {"error": "rate_limit_exceeded", "message": "Too many requests"}""");
                return;
            }
            chain.doFilter(req, resp);
        }
    
        private String resolveClientId(HttpServletRequest req) {
            String apiKey = req.getHeader("X-API-Key");
            if (apiKey != null) return "api:" + apiKey;
            String userId = req.getHeader("X-User-Id");
            if (userId != null) return "user:" + userId;
            return "ip:" + req.getRemoteAddr();
        }
    }
    ```

=== "Go"

    ```go
    package ratelimit
    
    import (
    	"encoding/json"
    	"net/http"
    	"strconv"
    )
    
    func Middleware(limiter *RateLimiter) func(http.Handler) http.Handler {
    	return func(next http.Handler) http.Handler {
    		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    			clientID := resolveClientID(r)
    			result, err := limiter.IsAllowed(r.Context(), clientID)
    			if err != nil {
    				next.ServeHTTP(w, r) // fail open on Redis error
    				return
    			}
    
    			w.Header().Set("X-RateLimit-Limit", strconv.Itoa(limiter.capacity))
    			w.Header().Set("X-RateLimit-Remaining", strconv.Itoa(result.Remaining))
    
    			if !result.Allowed {
    				w.Header().Set("Retry-After", "60")
    				w.WriteHeader(http.StatusTooManyRequests)
    				json.NewEncoder(w).Encode(map[string]string{
    					"error":   "rate_limit_exceeded",
    					"message": "Too many requests",
    				})
    				return
    			}
    			next.ServeHTTP(w, r)
    		})
    	}
    }
    
    func resolveClientID(r *http.Request) string {
    	if key := r.Header.Get("X-API-Key"); key != "" {
    		return "api:" + key
    	}
    	if uid := r.Header.Get("X-User-Id"); uid != "" {
    		return "user:" + uid
    	}
    	return "ip:" + r.RemoteAddr
    }
    ```

---

## Step 5: Scaling & Production

### Redis Cluster for High Availability

```mermaid
flowchart TB
    subgraph Cluster["Redis Cluster"]
        subgraph Shard1["Shard 1"]
            M1[Master 1]
            S1[Replica 1]
        end
        
        subgraph Shard2["Shard 2"]
            M2[Master 2]
            S2[Replica 2]
        end
        
        subgraph Shard3["Shard 3"]
            M3[Master 3]
            S3[Replica 3]
        end
    end
    
    API[API Servers] --> Cluster
```

**Configuration:**
- 3+ master nodes for fault tolerance
- Replica for each master
- Keys distributed across shards by hash

### Monitoring

Track these metrics:

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `rate_limit.allowed` | Allowed requests | - |
| `rate_limit.rejected` | Rejected requests | Spike |
| `rate_limit.latency_ms` | Rate limit check latency | > 5ms |
| `rate_limit.redis_errors` | Redis failures | Any |
| `rate_limit.by_client` | Top rate-limited clients | - |

```python
import prometheus_client as prom

rate_limit_requests = prom.Counter(
    "rate_limit_requests_total",
    "Total rate limit checks",
    ["client_tier", "result"]  # result: allowed/rejected
)

rate_limit_latency = prom.Histogram(
    "rate_limit_latency_seconds",
    "Rate limit check latency"
)

@rate_limit_latency.time()
def check_rate_limit(client_id: str, tier: str) -> bool:
    allowed, info = limiter.is_allowed(client_id)
    rate_limit_requests.labels(tier=tier, result="allowed" if allowed else "rejected").inc()
    return allowed
```

---

## Interview Checklist

- [ ] **Clarified requirements** (what to limit, limits, behavior)
- [ ] **Explained algorithms** (token bucket vs sliding window)
- [ ] **Drew architecture** (where rate limiter sits)
- [ ] **Discussed Redis implementation** (atomic operations)
- [ ] **Handled distributed challenges** (race conditions, clock sync)
- [ ] **Covered failure modes** (Redis down → fail open/closed)
- [ ] **Mentioned HTTP headers** (429, Retry-After)
- [ ] **Discussed advanced features** (tiers, multiple rules)
- [ ] **Addressed monitoring** (metrics, alerting)

---

## Sample Interview Dialogue

**Interviewer:** "Design a rate limiter for an API."

**You:** "Great question! Let me clarify a few things. What are we limiting—per user, per IP, or both? And what's the scale—how many requests per second are we handling?"

**Interviewer:** "Per API key, and we're handling about 10,000 requests per second across all clients."

**You:** "Got it. For the algorithm, I'd recommend a sliding window counter. It's memory-efficient—just O(1) per client—and handles window boundaries smoothly, unlike fixed window counters that can allow bursts at boundaries.

For a distributed system with 10K requests/sec, we need a centralized store for rate limit state. I'd use Redis—it's fast, supports atomic operations, and can handle this throughput easily.

The key insight is using Redis INCR for atomicity. Each request increments a counter atomically, so even with concurrent requests from multiple API servers, we get accurate counts.

Let me draw the architecture..."

---

## Summary

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Algorithm** | Sliding Window Counter | Memory efficient, no boundary issues |
| **Storage** | Redis | Fast, atomic operations, distributed |
| **Atomicity** | Lua scripts or INCR | Prevent race conditions |
| **Failure mode** | Fail open | Rate limiting is protection, not core |
| **Headers** | X-RateLimit-* | Client visibility |

Rate limiting is deceptively simple in concept but requires careful handling of distributed systems challenges. The key is choosing the right algorithm for your use case and ensuring atomicity in a distributed environment.

---

## Staff Engineer (L6) Deep Dive

The sections above cover the fundamentals. The sections below cover the **Staff-level depth** that distinguishes an L6 answer from an L5 one. See the [Staff Engineer Interview Guide](staff_engineer_expectations.md) for what L6 interviewers look for.

### Global Rate Limiting Across Regions

At L5, candidates design a rate limiter backed by a single Redis cluster. At L6, you must address: **what happens when your API serves users from 5 regions?**

| Strategy | How It Works | Trade-off |
|----------|-------------|-----------|
| **Centralized Redis** | All regions call a single Redis cluster | High latency for distant regions (100–300ms cross-region RTT); single point of failure |
| **Local counters + async sync** | Each region has local Redis; periodically sync counts via gossip or Kafka | Allows temporary over-limit (burst tolerance); eventually consistent |
| **Partitioned global limit** | Divide the global limit (e.g., 1000/min) across N regions proportionally (e.g., 200/min each) | Simple but wastes capacity in quiet regions; requires dynamic rebalancing |
| **Token bucket with central refill** | Local bucket per region; a central service periodically refills tokens | Good balance; central service is a dependency but not on hot path |

!!! tip
    **Staff-level answer:** *"I'd start with partitioned limits per region with a central rebalancing loop that runs every 30 seconds. This avoids cross-region latency on the hot path and handles the 95% case. For the top-tier enterprise clients who need a strict global limit, I'd route them to a single authoritative region with a slightly higher latency budget."*

### Race Conditions and Atomicity Deep Dive

The naive `GET → check → INCR` pattern has a well-known TOCTOU race. The Lua script approach solves this, but Staff candidates must also address:

| Problem | Impact | Solution |
|---------|--------|----------|
| **Redis pipeline non-atomicity** | `INCR` and `GET` in a pipeline aren't transactional across keys | Use single-key Lua scripts for atomicity |
| **Clock skew across API servers** | Different servers compute different window keys | Use `REDIS TIME` command for canonical time, or accept bounded skew |
| **Redis replica lag** | Read from replica returns stale count; request bypasses limit | Write and read to the same master; or accept eventual consistency for rate limiting |
| **Thundering herd on window rollover** | All counters reset simultaneously; burst at window boundary | Sliding window counter eliminates this; or stagger windows with client-specific offsets |

### Cascading Failure and Load Shedding

At L6, connect rate limiting to broader system resilience:

```mermaid
flowchart TD
  RL[Rate Limiter] -->|"Rejects excess"| Client
  RL -->|"Admits requests"| API[API Server]
  API -->|Overloaded| CB[Circuit Breaker]
  CB -->|Open| Fallback[Degraded Response]
  CB -->|Closed| Backend[Backend Service]
  Backend -->|"Backpressure signal"| API
  API -->|"Adaptive limit"| RL
```

| Concept | How It Applies |
|---------|----------------|
| **Adaptive rate limiting** | Reduce limits dynamically when backend health degrades (CPU > 80%, p99 > threshold) |
| **Load shedding** | When rate limiter Redis is down, shed low-priority traffic; admit only critical endpoints |
| **Priority-aware limiting** | Separate limits for `/health` (unlimited), `/v1/payments` (high limit), `/v1/search` (lower limit) |
| **Retry amplification** | Clients retrying 429s amplify load; add `Retry-After` with jitter to spread retry storms |

### Multi-Tenant Fairness (Noisy Neighbor Problem)

| Approach | Description |
|----------|-------------|
| **Per-tenant quotas** | Each tenant has an independent limit; simple but wastes capacity |
| **Weighted fair queuing** | Tenants share a pool; weights prevent any single tenant from consuming > X% |
| **Token bucket per tenant + global bucket** | Tenant bucket limits individual consumption; global bucket caps total system load |
| **Hierarchical rate limiting** | Global → Organization → User → Endpoint: each layer has its own budget |

### SLO and Observability for Rate Limiters

| SLI | Target | Alert |
|-----|--------|-------|
| Rate limit check latency (p99) | < 1ms | > 5ms for 5 minutes |
| False rejection rate | < 0.1% | Any spike correlates with Redis partition |
| Redis availability | 99.99% | Failover triggered; check fallback mode |
| Limit accuracy (actual vs. configured) | Within 2% | Cross-region sync lag > 30s |

### System Evolution (Year 0 → Year 3)

| Year | State | Action |
|------|-------|--------|
| **Year 0** | Single-region, single Redis | Ship fast; manual tuning per client |
| **Year 1** | Multi-region with per-region Redis | Add async sync; build admin UI for limit management |
| **Year 2** | Self-service rate limit policies via API | Tenants configure their own sub-limits; metering for billing |
| **Year 3** | ML-driven adaptive limits | Anomaly detection adjusts limits based on traffic patterns; auto-scale Redis fleet |
