# Design a URL Shortener (TinyURL/Bitly)

---

## What We're Building

A URL shortener takes a long, unwieldy URL like:

```
https://example.com/articles/2024/how-to-prepare-for-system-design-interviews?ref=social&utm_source=twitter&utm_medium=post&campaign_id=12345
```

And converts it to something short and shareable:

```
https://tiny.co/aB3xYz1
```

When someone clicks the short link, they get redirected to the original URL.

### Why URL Shorteners Exist

1. **Character limits:** Twitter (now X) famously had 140-character limits. Long URLs consumed precious space.
2. **Tracking:** Short URLs enable click analytics—who clicked, when, from where.
3. **Aesthetics:** `tiny.co/sale` looks better in marketing than a 200-character URL.
4. **Memorability:** Short, custom URLs (`bit.ly/my-resume`) are easier to remember and share verbally.

**Real-world examples:** TinyURL (2002), Bitly, t.co (Twitter), goo.gl (Google, discontinued), rb.gy

This is one of the most common system design interview questions because it covers core concepts: hashing, databases, caching, and scaling—while being simple enough to discuss in 45 minutes.

---

## Step 1: Requirements Clarification

Before designing anything, clarify what you're building. This demonstrates maturity and prevents wasted effort.

### Questions to Ask the Interviewer

| Question | Why It Matters | Typical Answer |
|----------|----------------|----------------|
| How many URLs shortened per day? | Determines database size, write throughput | 100M/day (Twitter-scale) to 1K/day (startup) |
| What's the read-to-write ratio? | Caching strategy, read replica needs | Usually 100:1 or higher (many more clicks than creates) |
| Should short URLs expire? | Storage growth, cleanup jobs | Sometimes. Enterprise: yes. Consumer: often no. |
| Do we need analytics? | Adds complexity, async processing | Usually yes (that's how Bitly makes money) |
| Can users create custom short URLs? | Collision handling, validation | Nice-to-have, but common |
| What's the maximum URL length? | Database schema, validation | Typically 2,048 characters (browser limit) |

### Functional Requirements

| Requirement | Priority | Notes |
|-------------|----------|-------|
| Shorten a long URL | Must have | Core feature |
| Redirect short URL to original | Must have | Core feature |
| Custom aliases (vanity URLs) | Nice to have | `tiny.co/my-resume` |
| Click analytics | Nice to have | Count, referrer, geography |
| URL expiration | Nice to have | Auto-delete after N days |
| API access | Nice to have | Programmatic creation |

### Non-Functional Requirements

| Requirement | Target | Rationale |
|-------------|--------|-----------|
| **Availability** | 99.99% for redirects | If redirects fail, all shared links break |
| **Redirect latency** | < 50ms | Users expect instant redirect |
| **Shortening latency** | < 200ms | Can be slightly slower than reads |
| **Durability** | Never lose a mapping | Losing URLs breaks all links |
| **Scalability** | Handle 100K+ redirects/sec | Popular links go viral |

### API Design

Let's define the core APIs:

**Create Short URL:**
```http
POST /api/v1/urls
Content-Type: application/json
Authorization: Bearer <token>

{
    "long_url": "https://example.com/very/long/path?with=params",
    "custom_alias": "my-link",     // Optional
    "expires_at": "2025-12-31"     // Optional
}
```

**Response:**
```json
{
    "short_url": "https://tiny.co/aB3xYz1",
    "long_url": "https://example.com/very/long/path?with=params",
    "short_code": "aB3xYz1",
    "created_at": "2024-01-15T10:00:00Z",
    "expires_at": "2025-12-31T00:00:00Z"
}
```

**Redirect (handled by browser):**
```http
GET /aB3xYz1
→ 302 Found
→ Location: https://example.com/very/long/path?with=params
```

**Get Analytics:**
```http
GET /api/v1/urls/aB3xYz1/stats
```

### Technology Selection & Tradeoffs

Interviewers often ask *why* you picked specific technologies. The URL shortener is **read-heavy** (redirects ≫ creates), needs **strong uniqueness** on write (no duplicate short codes), and benefits from **predictable hot-key caching**. Below are structured comparisons you can cite in an interview; section 4.2 revisits SQL vs NoSQL in the context of schema design.

#### Database (read-heavy workload)

| Criterion | PostgreSQL | MySQL (InnoDB) | DynamoDB | Cassandra |
|-----------|------------|-----------------|----------|-----------|
| **Read path fit** | Excellent with indexes + replicas; mature query planner | Excellent; similar to PG for simple key lookups | Excellent at key-value get by partition key; single-digit ms at scale | Excellent for wide-partition reads; tunable consistency |
| **Strong uniqueness (`short_code`)** | Native `UNIQUE`, transactional | Native `UNIQUE`, transactional | Conditional writes / idempotent keys; app must handle races carefully | Lightweight transactions / compare-and-set; more app logic |
| **Horizontal write scale** | Sharding is manual (or Citus); vertical + replicas first | Same pattern | Native partitioning; on-demand throughput | Native; best for multi-DC active-active |
| **Ops & ecosystem** | Rich extensions, JSON, great HA story (Patroni, etc.) | Ubiquitous; slightly fewer advanced types | Fully managed on AWS; vendor lock-in | Complex ops unless managed; excellent for huge scale |
| **Consistency model** | Serializable / repeatable read (tunable) | InnoDB ACID | Per-item strong reads available; global secondary indexes eventual | Tunable per read (ONE, QUORUM, ALL) |
| **Best when** | Default for most designs; strong consistency + SQL | Team already standardized on MySQL | AWS-native, huge partition-key QPS, minimal joins | Extreme write scale, multi-region, time-series style patterns |

**Read-heavy nuance:** For redirects you mostly do `SELECT long_url WHERE short_code = ?`—any of these can work if **cache hit rate is high**. The harder part is **write correctness** (one canonical mapping per code) and **operational simplicity** at your target scale.

#### Cache layer

| Criterion | Redis | Memcached | CDN / edge cache |
|-----------|-------|-----------|------------------|
| **Data structures** | Strings, hashes, sorted sets, streams | Simple key-value | Cached HTTP responses / edge KV |
| **Persistence & HA** | Optional AOF/RDB; Redis Cluster / Sentinel | RAM-only; client-side sharding | POP replication; origin on miss |
| **Eviction & TTL** | Rich TTL, LRU policies | LRU | HTTP `Cache-Control`; URL-specific policies |
| **Best for** | **Redirect lookup cache**, rate limits, locks, analytics buffers | Pure microsecond KV when you need simplicity | **Static or cacheable redirect responses**, geo latency reduction; careful with personalized/analytics accuracy |
| **Tradeoff** | Slightly more features = slightly more complexity | No persistence; fewer features | Stale mappings until purge; harder to invalidate on URL update/delete |

**Interview tip:** Say **Redis (or compatible)** for application cache (cache-aside), optional **CDN** in front for popular short links to shave RTT globally—pair CDN with **short TTLs** or **purge APIs** when mappings change.

#### ID generation for `short_code`

| Approach | Uniqueness | Coordination | Predictability | Notes |
|----------|------------|--------------|----------------|-------|
| **Hash-based** (truncate MD5/SHA) | Risk of collisions at short length; must detect & retry | Low | Same URL → same code (dedupe) or varies with salt | Simple; collision handling adds latency and complexity |
| **Counter + Base62** | Strong if counter is single source of truth | High if one DB sequence; **low** with ranges / Snowflake | Sequential unless obfuscated | Industry standard; combine with Snowflake or range allocator |
| **UUID (v4 / v7)** | Practically unique | None | Not URL-short without encoding | Longer as string; usually encoded → longer than 7 chars unless truncated (collision risk) |
| **Snowflake-style** | Strong per machine + time | None per ID (clock sync matters) | Roughly time-ordered | Scales horizontally; encode to Base62 for short codes |

#### Our choice

| Layer | Choice | Rationale |
|-------|--------|-----------|
| **Primary store** | **PostgreSQL** (or MySQL if org-standard) | ACID `UNIQUE` on `short_code`, straightforward HA, great read replica story for redirect path; sufficient until sharding is truly required. |
| **Cache** | **Redis** (cluster if needed) | Rich TTL, HA options, rate limiting & locks for thundering herd; Memcached is fine if you only need dumb KV and have other solutions for limits. |
| **Edge** | **CDN optional** | Cuts latency for hot links; use conservative caching or active purge when edits matter. |
| **IDs** | **Snowflake (or DB sequence / range allocation)** for production scale; **hash** only with explicit collision handling | Snowflake avoids a single DB bottleneck while keeping uniqueness; counter+Base62 from one DB is acceptable at smaller scale. |

This aligns with the architecture diagram in Step 3 (PostgreSQL + Redis + async analytics).

---

### CAP Theorem Analysis

CAP (**Consistency**, **Availability**, **Partition tolerance**) applies to *distributed* stores. In practice: **network partitions happen**, so systems choose between **CP** (sacrifice availability during partition) and **AP** (sacrifice strong consistency during partition). The URL shortener is interesting because **different operations want different tradeoffs**.

| Operation | Desired behavior | CAP-style reading | Why |
|-----------|------------------|-------------------|-----|
| **Redirect (read)** | Always answer quickly; stale cache occasionally OK | **AP** for the serving path | Users tolerate rare slightly stale reads better than widespread failed redirects; cache + async replication favor availability and latency. |
| **Create short URL (write)** | **Never** hand out the same `short_code` to two URLs | **CP** at the persistence boundary | Uniqueness is a correctness property: you need atomic insert or conditional write against an authoritative source. |
| **Analytics** | Complete eventually | **AP** | Counts can lag; use async pipelines (Kafka → ClickHouse). |

**Per-operation narrative:**

1. **Redirects:** Load balancer + app + Redis strive for **high availability**. If Redis is empty after a failover, you **degrade to PostgreSQL**—slightly higher latency, still **available** if the DB is up. During a partition, you might serve from cache (**A**) while replicas converge (**eventual C** for non-critical fields).
2. **Creates:** The **insert into `urls` with UNIQUE(short_code)** (or conditional put in DynamoDB) is the **consistency choke point**: two concurrent creators must not succeed with the same code. That is a **CP** moment: you wait for the authoritative node / quorum—not “best effort duplicate.”
3. **Analytics:** **AP**—duplicate or lost events can be bounded with idempotent consumers; SLAs are softer (see below).

```mermaid
flowchart TB
    subgraph AP_path["AP-leaning: redirect read path"]
        R[Client] --> Edge[CDN / LB]
        Edge --> App[Redirect service]
        App --> Cache[(Redis)]
        App --> DB[(Primary / replicas)]
    end

    subgraph CP_moment["CP moment: create mapping"]
        C[Create API] --> Auth[Transactional insert UNIQUE short_code]
        Auth --> Primary[(Authoritative store)]
    end

    subgraph Analytics_AP["AP: analytics pipeline"]
        App --> Q[Queue]
        Q --> OLAP[(Analytics store)]
    end
```

**One-liner for interviews:** *We bias AP on the hot read path for availability and latency, but we enforce CP semantics at write time so short codes are never duplicated.*

---

### SLA and SLO Definitions

**SLA** = contract with users (often includes credits). **SLO** = internal target you measure against. **Error budget** = allowable bad events (e.g., 0.01% unavailability per month) before you freeze features and fix reliability.

#### Service-level objectives (examples)

| Area | SLO | Measurement window | Notes |
|------|-----|-------------------|-------|
| **Redirect latency** | **p99 < 50 ms** (server-side, excluding client RTT) | Rolling 30 days | Aligns with NFR; track cache hit ratio correlation. |
| **Redirect availability** | **99.99%** successful HTTP 302/301 (excluding 404 for bad codes) | Monthly | Matches “links must work” expectation; exclude abuse-driven 4xx if policy-defined. |
| **URL creation latency** | **p99 < 200 ms** | Rolling 30 days | Includes ID gen + DB commit; stricter if synchronous custom alias checks. |
| **Data durability** | **RPO ≤ 1 min**, **RTO ≤ 15 min** for mapping data | Per disaster scenario | Sync replica → lower RPO; quantify last mapping loss risk. |
| **Analytics accuracy** | **≤ 0.1%** discrepancy vs raw log reconciliation | Daily | Async pipeline; bounded delay (e.g., 5 min freshness) stated separately. |

#### Error budget policy

- **Budget:** For **99.99%** availability, allowed unavailability ≈ **4.38 minutes/month**. Spend budget on **planned maintenance** only with explicit approval; **redirect** burns budget fastest—prioritize alerts on redirect SLO.
- **If budget is exhausted:** Stop non-critical releases; focus on HA, load tests, cache failover; consider **tighter redirect-only on-call** until burn rate recovers.
- **Latency SLOs:** Treat **p99 redirect** regressions as **launch-blocking** if sustained; they often precede availability incidents.

These SLOs connect directly to monitoring in Step 5 (p99 redirect, success rate, cache hit ratio).

---

## Step 2: Back-of-Envelope Estimation

Capacity planning is crucial. Let's do the math for a Twitter-scale service.

### Traffic Estimation

```
Given:
- 100 million URLs shortened per month
- 100:1 read-to-write ratio (for every URL created, 100 clicks)

Writes:
- 100M / (30 days × 24 hours × 3600 seconds) = ~40 URLs/second
- Peak (5× average): 200 URLs/second

Reads:
- 40 × 100 = 4,000 redirects/second
- Peak: 20,000 redirects/second
```

### Storage Estimation

```
Per URL record:
- Short code: 7 bytes
- Long URL: ~500 bytes (average)
- Timestamps: 16 bytes
- User ID: 8 bytes
- Other metadata: ~50 bytes
- Total: ~600 bytes

5 years of data:
- 100M/month × 12 months × 5 years = 6 billion URLs
- 6B × 600 bytes = 3.6 TB

With indexes and overhead: ~10 TB
```

### Short Code Capacity

How many unique short codes can we generate?

```
Using Base62 (a-z, A-Z, 0-9):
- 6 characters: 62^6 = 56.8 billion combinations
- 7 characters: 62^7 = 3.5 trillion combinations

At 100M URLs/month:
- 6 chars last: 56.8B / (100M × 12) = ~47 years
- 7 chars last: Effectively unlimited

Decision: Use 7 characters for ample headroom
```

### Bandwidth Estimation

```
Reads:
- 4,000 requests/sec × (200 bytes request + 500 bytes response) = 2.8 MB/sec
- Peak: 14 MB/sec

This is easily handled by modern networks.
```

---

## Step 3: High-Level Architecture

Now let's design the system architecture.

### Core Components

```mermaid
flowchart TB
    subgraph Clients [Client Layer]
        Web[Web Browser]
        Mobile[Mobile App]
        API[API Client]
    end
    
    subgraph Edge [Edge Layer]
        CDN[CDN / Edge Cache]
        LB[Load Balancer]
    end
    
    subgraph App [Application Layer]
        APIGw[API Gateway]
        ShortenSvc[Shortening Service]
        RedirectSvc[Redirect Service]
        AnalyticsSvc[Analytics Service]
    end
    
    subgraph Data [Data Layer]
        Cache[(Redis Cache)]
        DB[(PostgreSQL)]
        Analytics[(ClickHouse)]
    end
    
    subgraph Async [Async Processing]
        Queue[Kafka]
        Worker[Analytics Worker]
    end
    
    Web --> CDN
    Mobile --> LB
    API --> LB
    CDN --> LB
    LB --> APIGw
    
    APIGw --> ShortenSvc
    APIGw --> RedirectSvc
    APIGw --> AnalyticsSvc
    
    ShortenSvc --> DB
    ShortenSvc --> Cache
    
    RedirectSvc --> Cache
    RedirectSvc --> DB
    RedirectSvc --> Queue
    
    Queue --> Worker
    Worker --> Analytics
    
    AnalyticsSvc --> Analytics
```

### Two Core Flows

The system has two main operations:

**Write Flow (URL Shortening):**
1. Client sends long URL to API
2. Service generates unique short code
3. Mapping stored in database
4. Short URL returned to client

**Read Flow (Redirect):**
1. User clicks short URL
2. Service looks up mapping (cache first, then DB)
3. Returns HTTP redirect to original URL
4. Logs click event asynchronously

Let's dive deep into each component.

---

## Step 4: Deep Dive

### 4.1 Short Code Generation

This is the heart of the system. How do we create unique, short, URL-safe codes?

### Approach 1: Hash the URL

The obvious approach—hash the long URL and take the first N characters.

```python
import hashlib
import base64

def hash_url(long_url: str) -> str:
    # MD5 hash (128 bits)
    hash_bytes = hashlib.md5(long_url.encode()).digest()
    # Base64 encode and take first 7 chars
    encoded = base64.urlsafe_b64encode(hash_bytes).decode()
    return encoded[:7].replace('-', 'a').replace('_', 'b')

# Example
hash_url("https://example.com/page")  # "rL0Y7vC"
```

**Problems with hashing:**

1. **Collisions:** Two different URLs might hash to the same 7 characters. MD5 is 128 bits, but we're only using 7 Base62 characters (~41 bits). Collisions are mathematically certain at scale.

2. **Collision resolution is complex:**
```python
def shorten_with_collision_handling(long_url: str) -> str:
    for attempt in range(10):
        # Add attempt number to create different hashes
        hash_input = long_url if attempt == 0 else f"{long_url}:{attempt}"
        short_code = hash_url(hash_input)
        
        if not database.exists(short_code):
            database.insert(short_code, long_url)
            return short_code
    
    raise Exception("Too many collisions")
```

3. **Same URL always gets same code:** This might be desirable (deduplication) or not (each user wants their own tracking link).

**Verdict:** Hashing works for small scale but becomes problematic at billions of URLs.

### Approach 2: Counter with Base62 Encoding (Recommended)

Use an auto-incrementing counter and convert to Base62.

```python
ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

def to_base62(num: int) -> str:
    """Convert a number to base62 string."""
    if num == 0:
        return ALPHABET[0]
    
    result = []
    while num > 0:
        result.append(ALPHABET[num % 62])
        num //= 62
    
    return ''.join(reversed(result))

def from_base62(s: str) -> int:
    """Convert base62 string back to number."""
    num = 0
    for char in s:
        num = num * 62 + ALPHABET.index(char)
    return num

# Examples
to_base62(1)           # "1"
to_base62(62)          # "10"
to_base62(1000000)     # "4c92"
to_base62(999999999)   # "15FTGf"
```

**Advantages:**
- **No collisions:** Every number maps to a unique string
- **Predictable length:** Easy to calculate when you'll need more characters
- **Simple implementation:** Just arithmetic

**Challenge:** Where does the counter come from in a distributed system?

### Distributed ID Generation

When you have multiple servers, they all need unique IDs without coordination delays.

**Option A: Database Auto-Increment**

```sql
CREATE TABLE urls (
    id BIGSERIAL PRIMARY KEY,
    short_code VARCHAR(10) UNIQUE,
    long_url TEXT NOT NULL
);

-- Insert returns the auto-generated ID
INSERT INTO urls (long_url) VALUES ('https://example.com') RETURNING id;
```

*Pros:* Simple, guaranteed unique
*Cons:* Database becomes bottleneck, single point of failure

**Option B: Range Allocation**

Each server pre-allocates a range of IDs:
- Server 1: IDs 1 - 1,000,000
- Server 2: IDs 1,000,001 - 2,000,000
- etc.

When a server exhausts its range, it requests a new one from a coordinator.

```python
class IDGenerator:
    def __init__(self, coordinator_url):
        self.coordinator = coordinator_url
        self.current_range = None
        self.current_id = 0
    
    def get_next_id(self):
        if self.current_range is None or self.current_id >= self.current_range['end']:
            self.current_range = self.request_new_range()
            self.current_id = self.current_range['start']
        
        id = self.current_id
        self.current_id += 1
        return id
    
    def request_new_range(self):
        # Atomically get next range from coordinator
        response = requests.post(f"{self.coordinator}/allocate-range")
        return response.json()  # {"start": 5000001, "end": 6000000}
```

*Pros:* Reduces coordination to rare events
*Cons:* Gaps in IDs if server crashes, need coordinator service

**Option C: Snowflake IDs (Twitter's Approach)**

Generate 64-bit IDs that embed timestamp, machine ID, and sequence number.

```
┌─────────────────────────────────────────────────────────────────┐
│                        64-bit Snowflake ID                      │
├────────┬───────────────────────┬──────────┬────────────────────┤
│ 1 bit  │      41 bits          │ 10 bits  │      12 bits       │
│ unused │     timestamp         │machine ID│      sequence      │
│        │  (milliseconds)       │          │   (per millisecond)│
└────────┴───────────────────────┴──────────┴────────────────────┘
```

**Breaking it down:**

- **41 bits for timestamp:** Milliseconds since custom epoch. Lasts 69 years.
- **10 bits for machine ID:** Supports 1,024 machines
- **12 bits for sequence:** 4,096 IDs per machine per millisecond

```python
import time

class SnowflakeGenerator:
    def __init__(self, machine_id: int):
        self.machine_id = machine_id
        self.sequence = 0
        self.last_timestamp = -1
        self.epoch = 1609459200000  # Custom epoch: 2021-01-01
    
    def generate(self) -> int:
        timestamp = int(time.time() * 1000) - self.epoch
        
        if timestamp == self.last_timestamp:
            self.sequence = (self.sequence + 1) & 0xFFF  # 12 bits
            if self.sequence == 0:
                # Wait for next millisecond
                while timestamp <= self.last_timestamp:
                    timestamp = int(time.time() * 1000) - self.epoch
        else:
            self.sequence = 0
        
        self.last_timestamp = timestamp
        
        # Construct 64-bit ID
        id = ((timestamp << 22) |
              (self.machine_id << 12) |
              self.sequence)
        
        return id

# Usage
generator = SnowflakeGenerator(machine_id=1)
id = generator.generate()  # e.g., 6904298052141748224
short_code = to_base62(id)  # e.g., "aBc123def"
```

*Pros:* 
- No coordination needed
- IDs are roughly time-sorted
- Each machine can generate 4M IDs/second

*Cons:*
- Requires careful clock synchronization
- Machine ID management needed

**Recommendation:** Use Snowflake for large-scale systems. Use database auto-increment for simpler deployments.

### Security: Predictable vs Random

Counter-based IDs are sequential. Someone could enumerate all short codes:
- `tiny.co/1`, `tiny.co/2`, `tiny.co/3`, ...

**If this is a concern:**

1. **Add randomness:** XOR the ID with a secret key or shuffle bits
2. **Use longer codes:** 7+ characters make enumeration impractical
3. **Rate limit:** Detect and block enumeration attempts

```python
SECRET = 0xDEADBEEF12345678

def obfuscate_id(id: int) -> int:
    """Make IDs non-sequential while remaining reversible."""
    return id ^ SECRET

def deobfuscate_id(obfuscated: int) -> int:
    return obfuscated ^ SECRET
```

---

### 4.2 Database Design

### Schema Design

```sql
-- Main URL mappings table
CREATE TABLE urls (
    id BIGINT PRIMARY KEY,                    -- Snowflake ID
    short_code VARCHAR(10) UNIQUE NOT NULL,   -- Base62 encoded
    long_url TEXT NOT NULL,                   -- Original URL
    long_url_hash VARCHAR(64),                -- For duplicate detection
    user_id UUID,                             -- NULL for anonymous
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP,                     -- NULL = never expires
    is_active BOOLEAN DEFAULT TRUE,
    click_count BIGINT DEFAULT 0              -- Denormalized for quick access
);

-- Critical indexes
CREATE INDEX idx_short_code ON urls(short_code);
CREATE INDEX idx_long_url_hash ON urls(long_url_hash);
CREATE INDEX idx_user_urls ON urls(user_id, created_at DESC);
CREATE INDEX idx_expires ON urls(expires_at) WHERE expires_at IS NOT NULL;

-- Click events for analytics (consider separate database)
CREATE TABLE clicks (
    id BIGSERIAL PRIMARY KEY,
    short_code VARCHAR(10) NOT NULL,
    clicked_at TIMESTAMP DEFAULT NOW(),
    ip_address INET,
    user_agent TEXT,
    referrer TEXT,
    country_code VARCHAR(2)
);

CREATE INDEX idx_clicks_code_time ON clicks(short_code, clicked_at);
```

### SQL vs NoSQL Decision

| Factor | SQL (PostgreSQL) | NoSQL (DynamoDB/Cassandra) |
|--------|------------------|---------------------------|
| **Consistency** | Strong (ACID) | Eventual (tunable) |
| **Query flexibility** | Complex queries, JOINs | Limited (key-value) |
| **Scaling writes** | Harder (vertical first) | Easy (horizontal) |
| **Unique constraints** | Built-in | Application-enforced |
| **Operational complexity** | Lower (familiar) | Higher |

The table above contrasts **SQL vs NoSQL families**. For **engine-level** tradeoffs (PostgreSQL vs MySQL vs DynamoDB vs Cassandra) and how they relate to a read-heavy workload, see **Technology Selection & Tradeoffs** in Step 1.

**Recommendation:** Start with PostgreSQL. It handles millions of URLs easily. Migrate to NoSQL only if you hit scaling limits (billions of URLs, 100K+ writes/sec).

### Handling Duplicate URLs

Should the same long URL always map to the same short code?

**Option 1: Allow duplicates (different short codes for same URL)**
- Each user can have their own tracking link
- Simpler implementation
- Uses more storage

**Option 2: Deduplicate (same short code for same URL)**
- Saves storage
- Requires hash lookup before insert
- Might not fit all use cases (users want individual analytics)

```python
def shorten_url(long_url: str, user_id: str, deduplicate: bool = True) -> str:
    url_hash = hashlib.sha256(long_url.encode()).hexdigest()
    
    if deduplicate:
        # Check if URL already exists
        existing = db.query(
            "SELECT short_code FROM urls WHERE long_url_hash = %s AND user_id = %s",
            (url_hash, user_id)
        )
        if existing:
            return existing.short_code
    
    # Generate new short code
    id = id_generator.generate()
    short_code = to_base62(id)
    
    db.execute(
        """INSERT INTO urls (id, short_code, long_url, long_url_hash, user_id)
           VALUES (%s, %s, %s, %s, %s)""",
        (id, short_code, long_url, url_hash, user_id)
    )
    
    return short_code
```

---

### 4.3 Caching Strategy

With a 100:1 read-to-write ratio, caching is essential. Most URLs follow the 80/20 rule: 20% of URLs get 80% of traffic.

### Cache Architecture

```mermaid
flowchart LR
    Request[Request] --> Cache{Redis}
    Cache -->|HIT| Return[Return URL]
    Cache -->|MISS| DB[(PostgreSQL)]
    DB --> Store[Store in Cache]
    Store --> Return
```

### Redis Data Model

```bash
# Store URL mappings
SET url:aB3xYz1 "https://example.com/long/url" EX 86400

# Hash for multiple fields
HSET url:aB3xYz1 long_url "https://example.com/..." expires_at "2025-12-31"
```

### Cache Implementation

```python
import redis
import json

class URLCache:
    def __init__(self, redis_client, ttl_seconds=86400):
        self.redis = redis_client
        self.ttl = ttl_seconds
    
    def get(self, short_code: str) -> str | None:
        """Get long URL from cache."""
        key = f"url:{short_code}"
        result = self.redis.get(key)
        return result.decode() if result else None
    
    def set(self, short_code: str, long_url: str, expires_at: int = None):
        """Cache URL mapping."""
        key = f"url:{short_code}"
        
        if expires_at:
            # Use URL's expiration as TTL
            ttl = max(expires_at - int(time.time()), 1)
        else:
            ttl = self.ttl
        
        self.redis.setex(key, ttl, long_url)
    
    def delete(self, short_code: str):
        """Remove from cache (on delete or update)."""
        self.redis.delete(f"url:{short_code}")

# In redirect service
def get_long_url(short_code: str) -> str:
    # Try cache first
    long_url = cache.get(short_code)
    
    if long_url:
        metrics.incr("cache.hit")
        return long_url
    
    # Cache miss - query database
    metrics.incr("cache.miss")
    result = db.query("SELECT long_url FROM urls WHERE short_code = %s", short_code)
    
    if not result:
        raise NotFoundError("Short URL not found")
    
    # Cache for next time
    cache.set(short_code, result.long_url)
    
    return result.long_url
```

### Cache Hit Ratio Target

Aim for **95%+ cache hit ratio**. With proper TTLs and sufficient memory:
- 95% of redirects served from cache (~1ms)
- 5% hit database (~10-50ms)
- Average latency: ~3ms

### Cache Sizing

```
Cache size = Active URLs × Average URL size

If 10 million URLs are "active" (clicked in last 24h):
10M × 600 bytes ≈ 6 GB

With Redis overhead: ~10-15 GB
```

This easily fits in a single Redis instance. For larger scale, use Redis Cluster.

---

### 4.4 Redirect Flow Deep Dive

The redirect is the most critical path. It must be fast and reliable.

### 301 vs 302 Redirects

| Status Code | Type | Browser Caches? | Analytics Impact |
|-------------|------|-----------------|------------------|
| **301** | Permanent | Yes (forever) | Loses repeat clicks |
| **302** | Temporary | Usually no | Captures all clicks |
| **307** | Temporary | No | Captures all clicks |

**Trade-off:**
- **301:** Better for SEO (passes link equity), but browsers cache it, so you lose analytics for repeat visits
- **302:** Every click hits your server, enabling accurate analytics

**Recommendation:** Use **302** for most use cases. Use **301** only if SEO is critical and analytics aren't.

### Complete Redirect Flow

```mermaid
sequenceDiagram
    participant Browser
    participant LB as Load Balancer
    participant App as Redirect Service
    participant Cache as Redis
    participant DB as PostgreSQL
    participant Queue as Kafka
    
    Browser->>LB: GET /aB3xYz1
    LB->>App: Forward request
    
    App->>Cache: GET url:aB3xYz1
    
    alt Cache Hit
        Cache-->>App: "https://example.com/..."
    else Cache Miss
        App->>DB: SELECT long_url WHERE short_code = 'aB3xYz1'
        DB-->>App: "https://example.com/..."
        App->>Cache: SET url:aB3xYz1 (async)
    end
    
    App->>Queue: Log click event (async, non-blocking)
    App-->>Browser: 302 Redirect → Location: https://example.com/...
    
    Browser->>Browser: Follow redirect to destination
```

### Implementation

```python
from fastapi import FastAPI, Response, HTTPException
from fastapi.responses import RedirectResponse
import asyncio

app = FastAPI()

@app.get("/{short_code}")
async def redirect(short_code: str, request: Request):
    # Validate short code format
    if not is_valid_short_code(short_code):
        raise HTTPException(status_code=400, detail="Invalid short code")
    
    # Get long URL
    try:
        long_url = await get_long_url(short_code)
    except NotFoundError:
        raise HTTPException(status_code=404, detail="URL not found")
    
    # Log click asynchronously (don't wait)
    asyncio.create_task(log_click(
        short_code=short_code,
        ip=request.client.host,
        user_agent=request.headers.get("user-agent"),
        referrer=request.headers.get("referer")
    ))
    
    # Return redirect
    return RedirectResponse(
        url=long_url,
        status_code=302
    )

async def log_click(short_code: str, ip: str, user_agent: str, referrer: str):
    """Send click event to Kafka for async processing."""
    event = {
        "short_code": short_code,
        "timestamp": datetime.utcnow().isoformat(),
        "ip": ip,
        "user_agent": user_agent,
        "referrer": referrer
    }
    await kafka_producer.send("click-events", value=event)
```

---

### 4.5 Analytics System

Analytics must not slow down redirects. Process them asynchronously.

### Architecture

```mermaid
flowchart LR
    Redirect[Redirect Service] -->|async| Kafka[Kafka]
    Kafka --> Worker1[Analytics Worker]
    Kafka --> Worker2[Analytics Worker]
    Worker1 --> ClickHouse[(ClickHouse)]
    Worker2 --> ClickHouse
    ClickHouse --> Dashboard[Analytics Dashboard]
```

### Click Event Schema

```json
{
    "short_code": "aB3xYz1",
    "timestamp": "2024-01-15T10:30:00Z",
    "ip": "203.0.113.50",
    "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0...",
    "referrer": "https://twitter.com/user/status/123",
    "country": "US",
    "city": "San Francisco",
    "device_type": "mobile",
    "browser": "Safari",
    "os": "iOS"
}
```

### Analytics Worker

```python
from kafka import KafkaConsumer
import geoip2.database

class AnalyticsWorker:
    def __init__(self):
        self.consumer = KafkaConsumer(
            'click-events',
            group_id='analytics-workers',
            bootstrap_servers=['kafka:9092'],
            value_deserializer=lambda m: json.loads(m.decode())
        )
        self.geoip = geoip2.database.Reader('GeoLite2-City.mmdb')
        self.batch = []
        self.batch_size = 1000
    
    def run(self):
        for message in self.consumer:
            event = message.value
            
            # Enrich with geo data
            try:
                geo = self.geoip.city(event['ip'])
                event['country'] = geo.country.iso_code
                event['city'] = geo.city.name
            except:
                event['country'] = 'UNKNOWN'
            
            # Parse user agent
            ua = user_agents.parse(event['user_agent'])
            event['device_type'] = 'mobile' if ua.is_mobile else 'desktop'
            event['browser'] = ua.browser.family
            event['os'] = ua.os.family
            
            self.batch.append(event)
            
            if len(self.batch) >= self.batch_size:
                self.flush_batch()
    
    def flush_batch(self):
        # Batch insert to ClickHouse
        clickhouse.execute(
            "INSERT INTO clicks VALUES",
            self.batch
        )
        self.batch = []
```

### Why ClickHouse for Analytics?

ClickHouse is a column-oriented database optimized for analytical queries.

**Traditional DB (PostgreSQL):**
```sql
SELECT COUNT(*) FROM clicks WHERE short_code = 'aB3xYz1' AND clicked_at > '2024-01-01';
-- Scans all columns of matching rows
-- Slow for billions of rows
```

**ClickHouse:**
- Stores columns separately
- Compresses similar data efficiently
- Processes analytical queries 100-1000x faster

```sql
-- Fast aggregations
SELECT 
    toDate(clicked_at) as day,
    country,
    COUNT(*) as clicks
FROM clicks
WHERE short_code = 'aB3xYz1'
GROUP BY day, country
ORDER BY day;
```

---

## Step 5: Scaling & Production

### 5.1 Scaling Strategies

### Scaling the Read Path (Redirects)

Redirects are the hot path. Here's how to scale:

| Bottleneck | Solution |
|------------|----------|
| **Single server** | Add more app servers behind load balancer |
| **Database reads** | Add caching (95%+ hit rate) |
| **Cache size** | Use Redis Cluster (shard data) |
| **Single DB** | Add read replicas |
| **Geographic latency** | Deploy in multiple regions |

### Scaling the Write Path (Shortening)

| Bottleneck | Solution |
|------------|----------|
| **ID generation** | Distributed ID generator (Snowflake) |
| **Database writes** | Connection pooling, write batching |
| **Single DB** | Shard by short_code hash |

### Database Sharding

At extreme scale (billions of URLs), shard the database:

```python
def get_shard(short_code: str) -> int:
    """Determine which shard holds this short code."""
    hash_value = hash(short_code)
    return hash_value % NUM_SHARDS

# Example with 4 shards
# short_code "aB3xYz1" → Shard 2
# short_code "xY7zT9q" → Shard 0
```

**Sharding architecture:**

```mermaid
flowchart TD
    App[Application] --> Router[Shard Router]
    Router --> S0[(Shard 0)]
    Router --> S1[(Shard 1)]
    Router --> S2[(Shard 2)]
    Router --> S3[(Shard 3)]
```

**Challenges:**
- Cross-shard queries are complex
- Rebalancing shards is painful
- Need a shard mapping service

**When to shard:** Only after you've exhausted vertical scaling, read replicas, and caching. Most URL shorteners never need sharding.

---

### 5.2 High Availability and Fault Tolerance

### Failure Scenarios

| Component | Failure Impact | Mitigation |
|-----------|---------------|------------|
| **App server** | Some requests fail | Load balancer health checks, multiple instances |
| **Load balancer** | Total outage | Redundant LBs, DNS failover |
| **Redis cache** | Slower redirects | Fall back to DB, cache warmup on recovery |
| **PostgreSQL** | Can't read/write | Multi-AZ deployment, automatic failover |
| **Kafka** | Analytics delayed | Multi-broker cluster, data replicated |

### Database High Availability

```mermaid
flowchart LR
    App[Application] --> Primary[(Primary DB)]
    Primary -->|sync replication| Standby[(Standby DB)]
    Primary -->|async replication| Replica1[(Read Replica 1)]
    Primary -->|async replication| Replica2[(Read Replica 2)]
    
    App -.->|reads| Replica1
    App -.->|reads| Replica2
```

**PostgreSQL HA setup:**
- Primary handles writes
- Synchronous standby for failover (RPO = 0)
- Async replicas for read scaling
- Use PgBouncer for connection pooling

### Cache Thundering Herd

When a popular URL's cache expires, thousands of requests simultaneously hit the database.

**Solution 1: Distributed Locking**

```python
async def get_long_url_with_lock(short_code: str) -> str:
    # Try cache
    cached = cache.get(short_code)
    if cached:
        return cached
    
    lock_key = f"lock:{short_code}"
    
    # Try to acquire lock
    if redis.set(lock_key, "1", nx=True, ex=10):
        try:
            # We have the lock - fetch from DB
            url = db.query("SELECT long_url FROM urls WHERE short_code = %s", short_code)
            cache.set(short_code, url)
            return url
        finally:
            redis.delete(lock_key)
    else:
        # Someone else is fetching - wait and retry
        await asyncio.sleep(0.1)
        return await get_long_url_with_lock(short_code)
```

**Solution 2: Probabilistic Early Refresh**

Refresh cache *before* it expires, randomly:

```python
def get_with_early_refresh(short_code: str) -> str:
    cached, ttl = cache.get_with_ttl(short_code)
    
    if cached:
        # Probabilistically refresh if TTL is low
        if ttl < 300 and random.random() < 0.1:  # 10% chance
            asyncio.create_task(refresh_cache(short_code))
        return cached
    
    return fetch_and_cache(short_code)
```

---

### 5.3 Security Considerations

### URL Validation

Don't blindly redirect to any URL:

```python
from urllib.parse import urlparse
import requests

BLOCKED_SCHEMES = {'javascript', 'data', 'vbscript'}
BLOCKED_DOMAINS = {'malware.example.com', 'phishing.example.com'}

def validate_url(url: str) -> bool:
    """Validate URL is safe to redirect to."""
    try:
        parsed = urlparse(url)
        
        # Check scheme
        if parsed.scheme not in ('http', 'https'):
            return False
        
        if parsed.scheme in BLOCKED_SCHEMES:
            return False
        
        # Check against blocklist
        if parsed.netloc in BLOCKED_DOMAINS:
            return False
        
        # Check against Google Safe Browsing API (optional)
        if is_malicious(url):
            return False
        
        return True
    except:
        return False
```

### Rate Limiting

Prevent abuse:

```python
from redis import Redis
import time

class RateLimiter:
    def __init__(self, redis: Redis, max_requests: int, window_seconds: int):
        self.redis = redis
        self.max_requests = max_requests
        self.window = window_seconds
    
    def is_allowed(self, key: str) -> bool:
        """Token bucket rate limiting."""
        now = int(time.time())
        window_key = f"ratelimit:{key}:{now // self.window}"
        
        count = self.redis.incr(window_key)
        if count == 1:
            self.redis.expire(window_key, self.window)
        
        return count <= self.max_requests

# Usage
limiter = RateLimiter(redis, max_requests=100, window_seconds=3600)

@app.post("/api/v1/urls")
async def create_url(request: Request):
    user_id = get_user_id(request)
    
    if not limiter.is_allowed(f"create:{user_id}"):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # ... create URL
```

### Preventing Enumeration

If short codes are sequential (`1`, `2`, `3`...), attackers can enumerate all URLs.

**Mitigations:**
1. Use longer codes (7+ chars)
2. Obfuscate IDs (XOR with secret)
3. Monitor and block bulk access patterns
4. Add CAPTCHA for suspicious requests

---

### 5.4 Multi-Language Implementations

### Java: URL Shortening Service

```java
import java.security.SecureRandom;
import java.time.Instant;
import java.util.Optional;

public class UrlShortenerService {

    private static final String BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    private static final int SHORT_CODE_LENGTH = 7;
    private static final SecureRandom RANDOM = new SecureRandom();

    private final UrlRepository urlRepository;
    private final CacheService cache;
    private final SnowflakeIdGenerator idGenerator;

    public UrlShortenerService(UrlRepository urlRepository, CacheService cache,
                                SnowflakeIdGenerator idGenerator) {
        this.urlRepository = urlRepository;
        this.cache = cache;
        this.idGenerator = idGenerator;
    }

    public record ShortenedUrl(String shortCode, String longUrl, Instant createdAt, Instant expiresAt) {}

    public ShortenedUrl shorten(String longUrl, String userId, Instant expiresAt) {
        // check if URL already shortened
        Optional<ShortenedUrl> existing = urlRepository.findByLongUrl(longUrl);
        if (existing.isPresent()) return existing.get();

        // generate short code via Snowflake ID → Base62
        long id = idGenerator.nextId();
        String shortCode = toBase62(id);

        ShortenedUrl result = new ShortenedUrl(shortCode, longUrl, Instant.now(), expiresAt);
        urlRepository.save(result, userId);
        cache.put("url:" + shortCode, longUrl, expiresAt);

        return result;
    }

    public Optional<String> resolve(String shortCode) {
        // check cache first
        String cached = cache.get("url:" + shortCode);
        if (cached != null) return Optional.of(cached);

        // cache miss → query database
        Optional<ShortenedUrl> fromDb = urlRepository.findByShortCode(shortCode);
        fromDb.ifPresent(url -> cache.put("url:" + shortCode, url.longUrl(), url.expiresAt()));

        return fromDb.map(ShortenedUrl::longUrl);
    }

    private String toBase62(long value) {
        StringBuilder sb = new StringBuilder();
        value = Math.abs(value);
        while (sb.length() < SHORT_CODE_LENGTH) {
            sb.append(BASE62.charAt((int) (value % 62)));
            value /= 62;
        }
        return sb.reverse().toString();
    }
}
```

### Java: Redirect Controller

```java
@RestController
public class RedirectController {
    private final UrlShortenerService service;
    private final AnalyticsPublisher analytics;

    @GetMapping("/{shortCode}")
    public ResponseEntity<Void> redirect(@PathVariable String shortCode,
                                          HttpServletRequest request) {
        return service.resolve(shortCode)
            .map(longUrl -> {
                analytics.publishClickEvent(shortCode, request.getRemoteAddr(),
                    request.getHeader("User-Agent"), Instant.now());
                return ResponseEntity.status(HttpStatus.FOUND)
                    .header("Location", longUrl)
                    .header("Cache-Control", "private, max-age=90")
                    .<Void>build();
            })
            .orElse(ResponseEntity.notFound().build());
    }

    @PostMapping("/api/v1/shorten")
    public ResponseEntity<Map<String, String>> shorten(@RequestBody ShortenRequest req) {
        var result = service.shorten(req.url(), req.userId(), req.expiresAt());
        return ResponseEntity.status(HttpStatus.CREATED).body(Map.of(
            "short_url", "https://short.ly/" + result.shortCode(),
            "short_code", result.shortCode(),
            "expires_at", result.expiresAt().toString()
        ));
    }
}
```

### Go: URL Shortener Service

```go
package shortener

import (
	"context"
	"crypto/rand"
	"fmt"
	"math/big"
	"time"
)

const base62Chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
const codeLength = 7

type ShortenedURL struct {
	ShortCode string
	LongURL   string
	UserID    string
	CreatedAt time.Time
	ExpiresAt time.Time
}

type URLRepository interface {
	FindByShortCode(ctx context.Context, code string) (*ShortenedURL, error)
	FindByLongURL(ctx context.Context, longURL string) (*ShortenedURL, error)
	Save(ctx context.Context, url *ShortenedURL) error
}

type Cache interface {
	Get(ctx context.Context, key string) (string, error)
	Set(ctx context.Context, key, value string, ttl time.Duration) error
}

type Service struct {
	repo  URLRepository
	cache Cache
}

func NewService(repo URLRepository, cache Cache) *Service {
	return &Service{repo: repo, cache: cache}
}

func (s *Service) Shorten(ctx context.Context, longURL, userID string,
	expiresAt time.Time) (*ShortenedURL, error) {

	// check if already shortened
	existing, err := s.repo.FindByLongURL(ctx, longURL)
	if err == nil && existing != nil {
		return existing, nil
	}

	code, err := generateBase62Code(codeLength)
	if err != nil {
		return nil, fmt.Errorf("generating code: %w", err)
	}

	url := &ShortenedURL{
		ShortCode: code,
		LongURL:   longURL,
		UserID:    userID,
		CreatedAt: time.Now(),
		ExpiresAt: expiresAt,
	}

	if err := s.repo.Save(ctx, url); err != nil {
		return nil, fmt.Errorf("saving URL: %w", err)
	}

	ttl := time.Until(expiresAt)
	_ = s.cache.Set(ctx, "url:"+code, longURL, ttl)

	return url, nil
}

func (s *Service) Resolve(ctx context.Context, shortCode string) (string, error) {
	// cache first
	if cached, err := s.cache.Get(ctx, "url:"+shortCode); err == nil && cached != "" {
		return cached, nil
	}

	url, err := s.repo.FindByShortCode(ctx, shortCode)
	if err != nil {
		return "", err
	}
	if url == nil {
		return "", fmt.Errorf("short code not found: %s", shortCode)
	}

	ttl := time.Until(url.ExpiresAt)
	_ = s.cache.Set(ctx, "url:"+shortCode, url.LongURL, ttl)

	return url.LongURL, nil
}

func generateBase62Code(length int) (string, error) {
	result := make([]byte, length)
	max := big.NewInt(int64(len(base62Chars)))
	for i := range result {
		n, err := rand.Int(rand.Reader, max)
		if err != nil {
			return "", err
		}
		result[i] = base62Chars[n.Int64()]
	}
	return string(result), nil
}
```

---

### 5.5 Monitoring and Alerting

### Key Metrics

| Category | Metric | Alert Threshold |
|----------|--------|-----------------|
| **Latency** | P99 redirect time | > 100ms |
| **Availability** | Redirect success rate | < 99.9% |
| **Throughput** | Redirects per second | Depends on baseline |
| **Cache** | Hit ratio | < 90% |
| **Database** | Connection pool usage | > 80% |
| **Errors** | 5xx error rate | > 0.1% |

### Logging

```json
{
    "timestamp": "2024-01-15T10:30:00Z",
    "level": "INFO",
    "service": "redirect",
    "short_code": "aB3xYz1",
    "long_url": "https://example.com/...",
    "latency_ms": 3.5,
    "cache_hit": true,
    "user_agent": "...",
    "ip": "..."
}
```

---

## Interview Checklist

Use this to structure your answer:

- [ ] **Clarified requirements** (scale, features, read/write ratio)
- [ ] **Technology tradeoffs** (DB, cache, ID generation—with comparison tables)
- [ ] **CAP analysis** per operation (redirect AP vs create CP; partition behavior)
- [ ] **SLA/SLO and error budget** (redirect latency/availability, durability, analytics)
- [ ] **Estimated capacity** (storage, bandwidth, QPS)
- [ ] **Drew high-level architecture** (clients, LB, app, cache, DB)
- [ ] **Explained short code generation** (Base62 + distributed IDs)
- [ ] **Designed database schema** (with appropriate indexes)
- [ ] **Added caching layer** (Redis, cache-aside pattern)
- [ ] **Discussed redirect status codes** (301 vs 302)
- [ ] **Explained analytics** (async via message queue)
- [ ] **Covered scaling strategies** (horizontal scaling, sharding)
- [ ] **Addressed failure handling** (HA, thundering herd)
- [ ] **Mentioned security** (validation, rate limiting)

---

## Sample Interview Dialogue

**Interviewer:** "Design a URL shortener like TinyURL."

**You:** "Great question! Before I dive in, let me ask a few questions to understand the scope.

First, what's the expected scale? Are we building for a startup with thousands of URLs or a Twitter-scale system with billions?

Second, what's the read-to-write ratio? I'd guess most URL shorteners see many more clicks than creates.

Third, do we need analytics—click counting, referrer tracking, that sort of thing?"

**Interviewer:** "Let's say 100 million URLs created per month, 100:1 read/write ratio, and yes, we want basic analytics."

**You:** "Got it. So we're looking at about 40 writes per second, 4,000 reads per second, and we need an analytics pipeline that doesn't slow down redirects.

Let me start with the high-level architecture. We have two main flows: shortening URLs and redirecting. For shortening, a request comes in, we generate a unique short code—I'll explain how in a moment—store the mapping in a database, and return the short URL.

For redirects, we want to be extremely fast. I'd put a Redis cache in front of the database. With a 100:1 read ratio and proper caching, we should achieve 95%+ cache hit rates, meaning most redirects complete in under 5 milliseconds.

For short code generation, the key challenge is generating unique IDs across multiple servers without coordination delays. I'd recommend a Snowflake-style ID generator..."

---

## Summary

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **ID Generation** | Snowflake IDs | Distributed, no coordination, time-sorted |
| **Encoding** | Base62 (7 chars) | URL-safe, 3.5T combinations |
| **Database** | PostgreSQL | Strong consistency, familiar, scales well |
| **Cache** | Redis | Sub-millisecond lookups, simple KV model |
| **Analytics Queue** | Kafka | High throughput, durable, replayable |
| **Analytics DB** | ClickHouse | Optimized for aggregations |
| **Redirect Code** | 302 | Enables accurate analytics |

The URL shortener demonstrates core distributed systems concepts: unique ID generation, caching, database design, and async processing. Master this design, and you'll have a solid foundation for more complex systems.
