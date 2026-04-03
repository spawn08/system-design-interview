---
layout: default
title: News Feed / Timeline
parent: System Design Examples
nav_order: 7
---

# Design a News Feed / Timeline System
{: .no_toc }

<details open markdown="block">
  <summary>Table of Contents</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

---

## What We're Building

A **news feed** (or **timeline**) is the continuously updated stream of posts users see when they open a social app. It aggregates content from accounts they follow (and sometimes recommended content), ranks it for relevance, and delivers it with low latency on web and mobile clients.

**Examples in the wild:**
- **Facebook** — Home feed blends friend posts, Groups, Pages, and ads; ranked by ML with billions of daily active users.
- **Twitter / X** — Reverse-chronological or algorithmic “For You” timeline; very high write and read rates on hot accounts.
- **Instagram** — Photo/video feed with Stories and Reels interleaved; heavy media and CDN usage.
- **LinkedIn** — Professional feed mixing network updates, articles, and job-related content.

### Core Problems This Design Solves

| Problem | Why it matters |
|---------|----------------|
| **Feed generation** | Combine posts from many followees into one ordered list per user |
| **Ranking** | Not everything fits on screen—pick the “best” subset |
| **Scale** | Celebrities have millions of followers; naive fan-out breaks |
| **Consistency vs latency** | Fresh enough posts without blocking reads on global writes |
| **Media** | Images/video need upload, processing, and CDN delivery |

### Real-World Scale (Order-of-Magnitude References)

| Platform | Scale hint | Notes |
|----------|------------|--------|
| **Facebook** | 2B+ monthly active people; feed is one of the largest distributed systems | Heavy ML ranking, hybrid storage |
| **Twitter / X** | Order of **hundreds of millions** of tweets per day (public figures vary by year) | Fan-out and timeline read paths are classic interview topics |
| **Instagram** | Billions of accounts; feed is media-centric | CDN + transcoding critical |

{: .note }
> Interview numbers are **approximate**. Cite ranges and explain *how* you’d validate (metrics, load tests) rather than memorizing exact press stats.

### Feed Generation and Ranking Pipeline (Conceptual)

```mermaid
flowchart LR
    subgraph Write [Write path]
        P[Post created] --> FS[Fan-out / graph]
        FS --> Cache[Per-user feed cache]
        FS --> Idx[Search / index optional]
    end
    subgraph Read [Read path]
        R[GET /feed] --> Rank[Ranking]
        Cache --> Rank
        Rank --> Resp[JSON + cursors]
    end
```

---

## Step 1: Requirements Clarification

### Questions to Ask

| Question | Why it matters |
|----------|----------------|
| **Who sees what?** | Follow graph only vs recommendations/ads |
| **Ordering** | Strict reverse-chronological vs ranked feed |
| **Content types** | Text, images, video, links—different storage and latency |
| **Consistency** | Is slightly stale feed OK? (usually yes) |
| **Celebrity / viral behavior** | Drives fan-out vs read trade-offs |
| **Private accounts / blocking** | Extra filters on read and write paths |
| **Real-time updates** | Polling vs push; WebSocket for live counts? |
| **Regions** | Multi-region users and compliance (data residency) |

### Functional Requirements

| Requirement | Priority | Description |
|-------------|----------|-------------|
| Post creation (text, images, video) | Must have | Users create posts with optional media |
| News feed generation | Must have | Paginated feed for a logged-in user |
| Follow / unfollow | Must have | Maintain social graph |
| Like, comment, share | Must have | Engagement signals + social proof |
| Feed ranking (non-trivial order) | Must have | Beyond pure time if product requires relevance |
| Search / profile timelines | Nice to have | Often separate services |
| Mute / block | Nice to have | Filters feed inputs |
| Notifications for new activity | Nice to have | Often separate notification system |

### Non-Functional Requirements

| Requirement | Target | Rationale |
|-------------|--------|-----------|
| **Feed read latency (p99)** | &lt; 500 ms | Mobile UX; excludes client rendering |
| **Availability** | 99.99% for read path | Degrade gracefully vs hard outage |
| **Consistency** | Eventual consistency OK | Cross-user ordering can be approximate |
| **Durability** | Posts must not be lost | Persist before acknowledging publish |
| **Scale** | Horizontal | Sharding by user_id / tenant |

### API Design

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/v1/posts` | Create post (metadata + media descriptors) |
| `GET` | `/v1/feed` | Get ranked feed (`?limit=&cursor=`) |
| `POST` | `/v1/follow` | Follow user `{ "target_user_id": "..." }` |
| `DELETE` | `/v1/follow/{userId}` | Unfollow |
| `GET` | `/v1/users/{id}/posts` | Profile timeline (may share backend) |
| `POST` | `/v1/posts/{id}/like` | Like |
| `POST` | `/v1/media/upload-url` | Pre-signed URL for direct client → object storage |
| `GET` | `/v1/ws` or SSE | Optional real-time updates |

{: .tip }
> Use **cursor-based pagination** for feeds (opaque token), not offset, for stable pages under concurrent writes.

**Example request/response sketches:**

```json
// POST /v1/posts
{
  "text": "Hello world",
  "media": [{ "object_key": "u/1/a.jpg", "type": "image" }]
}

// GET /v1/feed?limit=20&cursor=eyJjIjoifQ
{
  "items": [
    {
      "post_id": "p_123",
      "author_id": "u_9",
      "text": "...",
      "created_at_ms": 1712000000000,
      "engagement": { "likes": 42, "comments": 3 }
    }
  ],
  "next_cursor": "eyJjIjoifQ"
}
```

---

## Step 2: Back-of-Envelope Estimation

### Assumptions

```
- DAU: 500 million
- Avg following (friends / follows): 300 per user
- Posts per active user per day: 5
- Read: each DAU loads feed 10 times/day, 20 posts per load (not all unique writes)
```

### Feed Generation QPS

```
Writes (new posts):
  500M DAU × 5 posts/day = 2.5B posts/day
  Average QPS = 2.5e9 / 86,400 ≈ 28,900 posts/sec
  Peak (e.g., 3× average) ≈ 87,000 posts/sec

Reads (feed requests):
  500M × 10 loads/day = 5B feed requests/day
  Average ≈ 57,900 feed API calls/sec
  Peak ≈ 170,000 feed QPS
```

{: .warning }
> Peak factors depend on time zones and events; in interviews, state assumptions explicitly (e.g., 2–5× average).

### Storage (Posts + Metadata)

```
Assume average post metadata record = 500 bytes (text, ids, timestamps)
2.5B posts/day × 500 B ≈ 1.25 TB/day raw post rows
Add indexes, replication, media pointers → multi-PB at year scale in aggregate (media in object store dominates)

Media: average 2 MB per post with media (mixed images/video)
If 50% of posts have media: 1.25B media objects/day × 2 MB ≈ 2.5 PB/day — 
in practice not every DAU posts media-heavy; tune fractions in interview.
```

### Bandwidth (Illustrative)

```
One feed response ~ 20 KB (compressed JSON with URLs, not full images)
57,900 RPS × 20 KB ≈ 1.1 GB/s average egress from API layer
Images/video served from CDN, not API JSON — API returns links only.
```

---

## Step 3: High-Level Design

### Architecture Overview

```mermaid
flowchart TB
    Client[Client apps]
    LB[Load balancer]
    PostAPI[Post Service]
    Fanout[Fan-out Service]
    FeedSvc[Feed Service]
    Kafka{{Kafka}}
    Redis[(Feed cache Redis)]
    DB[(Relational / NoSQL)]
    Obj[(Object storage)]
    CDN[CDN]

    Client --> LB
    LB --> PostAPI
    LB --> FeedSvc
    PostAPI --> DB
    PostAPI --> Obj
    PostAPI --> Kafka
    Kafka --> Fanout
    Fanout --> Redis
    Fanout --> DB
    FeedSvc --> Redis
    FeedSvc --> DB
    Obj --> CDN
    Client --> CDN
```

**Flow:**
1. **Post Service** — Validates post, stores metadata, enqueues fan-out work, returns post id.
2. **Fan-out Service** — Consumes events; pushes post ids into followers’ feed caches (or schedules read-time aggregation for celebrities).
3. **Feed Cache (Redis)** — Fast per-user candidate lists (often sorted sets).
4. **Feed Service** — Merges cache + ranking, applies pagination, returns response.

### Fan-out on Write vs Fan-out on Read

| Approach | Mechanism | Pros | Cons |
|----------|-----------|------|------|
| **Fan-out on write (push)** | On post, write into each follower’s feed bucket | Read is cheap and fast | Hot users cause huge write amplification |
| **Fan-out on read (pull)** | On read, gather posts from followees | No per-follower writes on post | Read is heavier; harder to hit latency at scale |
| **Hybrid** | Push for normal users; pull (or partial push) for celebrities | Balances cost and latency | More complex; needs “celebrity” detection |

```mermaid
flowchart LR
    subgraph Push [Fan-out on write]
        A1[New post] --> W1[Write N follower caches]
        R1[Read feed] --> F1[Read 1 key]
    end
    subgraph Pull [Fan-out on read]
        A2[New post] --> W2[Write 1 row]
        R2[Read feed] --> F2[Query K users + merge]
    end
```

### Hybrid Approach (Production Pattern)

- **Regular users** (followers &lt; threshold): **push** fan-out into Redis sorted sets (cap list length).
- **Celebrities** (followers &gt; threshold): **skip** full push or push only to a “recent subset”; at read time **merge** in-memory from celebrity shards / recent posts cache.

{: .note }
> Twitter and similar systems have described hybrid models publicly at a high level; exact thresholds are tunable and dynamic.

---

## Step 4: Deep Dive

### 4.1 Post Storage & Publishing

#### Database Schema (Illustrative)

**Relational style (PostgreSQL):**

| Table | Key columns |
|-------|-------------|
| `users` | `id`, `username`, `created_at`, `follower_count` (denormalized, updated async) |
| `posts` | `id`, `author_id`, `text`, `created_at`, `visibility`, `media_count` |
| `post_media` | `id`, `post_id`, `storage_key`, `width`, `height`, `duration_sec`, `mime` |
| `follows` | `follower_id`, `followee_id`, `created_at` (PK: pair) |

**Graph / wide-column alternatives:** `follows` can live in a graph DB or as adjacency in Dynamo/Cassandra with `followee_id` as partition for “who follows me” queries.

#### Post Creation Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant API as Post Service
    participant O as Object Storage
    participant DB as Database
    participant Q as Kafka

    C->>API: POST /posts (metadata + media keys)
    API->>DB: INSERT posts, post_media
    API->>Q: publish PostCreated
    API-->>C: 201 + post_id
    Note over C,O: Client already uploaded blobs via pre-signed URLs
```

#### Media Upload Pipeline

1. Client calls `POST /media/upload-url` → receives **pre-signed URL** + `object_key`.
2. Client **PUT**s bytes directly to **S3-compatible** storage.
3. Optional **async pipeline**: virus scan, image resize, video transcoding → updates `post_media` status.

{: .tip }
> Never stream large uploads through your stateless API tier; use **direct-to-storage** uploads with short-lived credentials.

#### Code Examples: Post Service

**Java (Spring-style)**

```java
package com.example.feed.post;

import java.time.Instant;
import java.util.List;
import java.util.UUID;

public final class CreatePostRequest {
    public String text;
    public List<MediaRef> media;

    public static final class MediaRef {
        public String objectKey;
        public String mimeType;
    }
}

public final class PostRecord {
    public final String id;
    public final String authorId;
    public final String text;
    public final Instant createdAt;

    public PostRecord(String id, String authorId, String text, Instant createdAt) {
        this.id = id;
        this.authorId = authorId;
        this.text = text;
        this.createdAt = createdAt;
    }
}

public interface PostRepository {
    PostRecord insert(String authorId, String text, List<CreatePostRequest.MediaRef> media);
}

public interface EventPublisher {
    void publishPostCreated(PostRecord post, List<CreatePostRequest.MediaRef> media);
}

public final class PostService {
    private final PostRepository posts;
    private final EventPublisher events;

    public PostService(PostRepository posts, EventPublisher events) {
        this.posts = posts;
        this.events = events;
    }

    public PostRecord createPost(String authorId, CreatePostRequest req) {
        validate(req);
        PostRecord saved = posts.insert(authorId, req.text, req.media);
        events.publishPostCreated(saved, req.media);
        return saved;
    }

    private static void validate(CreatePostRequest req) {
        if (req.text == null || req.text.isBlank()) {
            if (req.media == null || req.media.isEmpty()) {
                throw new IllegalArgumentException("empty post");
            }
        }
        if (req.text != null && req.text.length() > 10_000) {
            throw new IllegalArgumentException("text too long");
        }
    }

    public static String newId() {
        return "p_" + UUID.randomUUID().toString().replace("-", "");
    }
}
```

**Python (FastAPI-style)**

```python
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class MediaRef(BaseModel):
    object_key: str
    mime_type: str


class CreatePostRequest(BaseModel):
    text: Optional[str] = None
    media: List[MediaRef] = Field(default_factory=list)

    @field_validator("text")
    @classmethod
    def strip_text(cls, v: Optional[str]) -> Optional[str]:
        return v.strip() if isinstance(v, str) else v

    def validate_nonempty(self) -> None:
        if (not self.text) and not self.media:
            raise ValueError("empty post")
        if self.text and len(self.text) > 10_000:
            raise ValueError("text too long")


class PostRecord(BaseModel):
    id: str
    author_id: str
    text: Optional[str]
    created_at: datetime


class PostRepository:
    def insert(
        self, author_id: str, req: CreatePostRequest
    ) -> PostRecord:  # pragma: no cover - interface
        raise NotImplementedError


class EventPublisher:
    def publish_post_created(self, post: PostRecord, media: List[MediaRef]) -> None:
        raise NotImplementedError


class PostService:
    def __init__(self, posts: PostRepository, events: EventPublisher) -> None:
        self._posts = posts
        self._events = events

    def create_post(self, author_id: str, req: CreatePostRequest) -> PostRecord:
        req.validate_nonempty()
        saved = self._posts.insert(author_id, req)
        self._events.publish_post_created(saved, req.media)
        return saved

    @staticmethod
    def new_id() -> str:
        return f"p_{uuid.uuid4().hex}"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
```

**Go**

```go
package post

import (
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
)

type MediaRef struct {
	ObjectKey string
	MimeType  string
}

type CreatePostRequest struct {
	Text  string
	Media []MediaRef
}

type PostRecord struct {
	ID        string
	AuthorID  string
	Text      string
	CreatedAt time.Time
}

type PostRepository interface {
	Insert(authorID string, req CreatePostRequest) (PostRecord, error)
}

type EventPublisher interface {
	PublishPostCreated(PostRecord, []MediaRef) error
}

type Service struct {
	Posts  PostRepository
	Events EventPublisher
}

func (s *Service) CreatePost(authorID string, req CreatePostRequest) (PostRecord, error) {
	if err := validateCreate(req); err != nil {
		return PostRecord{}, err
	}
	saved, err := s.Posts.Insert(authorID, req)
	if err != nil {
		return PostRecord{}, err
	}
	if err := s.Events.PublishPostCreated(saved, req.Media); err != nil {
		return PostRecord{}, fmt.Errorf("publish: %w", err)
	}
	return saved, nil
}

func validateCreate(req CreatePostRequest) error {
	text := strings.TrimSpace(req.Text)
	if text == "" && len(req.Media) == 0 {
		return errors.New("empty post")
	}
	if len(text) > 10_000 {
		return errors.New("text too long")
	}
	return nil
}

func NewID() string {
	return "p_" + strings.ReplaceAll(uuid.New().String(), "-", "")
}
```

---

### 4.2 Fan-out Service

#### Fan-out on Write (Push Model)

- When user **U** posts, enqueue work: for each follower **V**, insert `post_id` into **V**’s feed structure.
- **Pros:** `GET /feed` reads a precomputed structure — fast.
- **Cons:** User with 50M followers triggers 50M writes (mitigated by hybrid).

#### Fan-out on Read (Pull Model)

- Store posts by author; on feed read, **query recent posts** from each followee (or batched), **merge** by time, truncate.
- **Pros:** Posting stays O(1) in follower count.
- **Cons:** Read cost grows with follow count; caching and indexing are critical.

#### Hybrid

- **Threshold** `T` (e.g., 10k–100k followers): above `T`, do not push to all followers; use pull + partial materialization.

```mermaid
flowchart TB
    E[PostCreated event] --> D{Celebrity?}
    D -->|no| P[Push to Redis ZSET per follower]
    D -->|yes| S[Write to author's recent list only]
    S --> R[Followers merge at read]
    P --> R
```

```mermaid
flowchart LR
    subgraph OnRead [Fan-out on read]
        L[Load followee ids] --> Q[Parallel fetch recent posts]
        Q --> M[Merge + truncate]
    end
```

#### Java / Python / Go: Fan-out Worker with Kafka

**Java**

```java
package com.example.feed.fanout;

import java.util.List;

public record PostCreatedEvent(
        String postId,
        String authorId,
        long createdAtEpochMs,
        List<String> mediaKeys
) {}

public interface FollowGraphClient {
    /** Returns follower ids in pages; fan-out worker iterates. */
    Iterable<List<String>> followersPaged(String authorId, int pageSize);
}

public interface FeedCacheClient {
    void addToFeed(String followerUserId, String postId, long score);
}

public interface CelebrityPolicy {
    boolean shouldPushToAllFollowers(String authorId, long followerCount);
}

public final class FanOutWorker {
    private final FollowGraphClient graph;
    private final FeedCacheClient cache;
    private final CelebrityPolicy policy;

    public FanOutWorker(FollowGraphClient graph, FeedCacheClient cache, CelebrityPolicy policy) {
        this.graph = graph;
        this.cache = cache;
        this.policy = policy;
    }

    public void onPostCreated(PostCreatedEvent e, long followerCount) {
        if (!policy.shouldPushToAllFollowers(e.authorId(), followerCount)) {
            return;
        }
        long score = e.createdAtEpochMs();
        for (List<String> page : graph.followersPaged(e.authorId(), 5_000)) {
            for (String followerId : page) {
                cache.addToFeed(followerId, e.postId(), score);
            }
        }
    }
}
```

**Python**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Protocol


@dataclass(frozen=True)
class PostCreatedEvent:
    post_id: str
    author_id: str
    created_at_epoch_ms: int
    media_keys: List[str]


class FollowGraphClient(Protocol):
    def followers_paged(self, author_id: str, page_size: int) -> Iterable[List[str]]:
        ...


class FeedCacheClient(Protocol):
    def add_to_feed(self, follower_user_id: str, post_id: str, score: float) -> None:
        ...


class CelebrityPolicy(Protocol):
    def should_push_to_all_followers(self, author_id: str, follower_count: int) -> bool:
        ...


class FanOutWorker:
    def __init__(
        self,
        graph: FollowGraphClient,
        cache: FeedCacheClient,
        policy: CelebrityPolicy,
    ) -> None:
        self._graph = graph
        self._cache = cache
        self._policy = policy

    def on_post_created(self, e: PostCreatedEvent, follower_count: int) -> None:
        if not self._policy.should_push_to_all_followers(e.author_id, follower_count):
            return
        score = float(e.created_at_epoch_ms)
        for page in self._graph.followers_paged(e.author_id, 5000):
            for follower_id in page:
                self._cache.add_to_feed(follower_id, e.post_id, score)
```

**Go**

```go
package fanout

type PostCreatedEvent struct {
	PostID            string
	AuthorID          string
	CreatedAtEpochMs  int64
	MediaKeys         []string
}

type FollowGraphClient interface {
	FollowersPaged(authorID string, pageSize int) (pages [][]string)
}

type FeedCacheClient interface {
	AddToFeed(followerUserID, postID string, score float64) error
}

type CelebrityPolicy interface {
	ShouldPushToAllFollowers(authorID string, followerCount int64) bool
}

type Worker struct {
	Graph   FollowGraphClient
	Cache   FeedCacheClient
	Policy  CelebrityPolicy
}

func (w *Worker) OnPostCreated(e PostCreatedEvent, followerCount int64) error {
	if !w.Policy.ShouldPushToAllFollowers(e.AuthorID, followerCount) {
		return nil
	}
	score := float64(e.CreatedAtEpochMs)
	for _, page := range w.Graph.FollowersPaged(e.AuthorID, 5000) {
		for _, fid := range page {
			if err := w.Cache.AddToFeed(fid, e.PostID, score); err != nil {
				return err
			}
		}
	}
	return nil
}
```

{: .note }
> In production, **fan-out is batched**, **idempotent**, and **back-pressure aware**; Kafka partitions might be keyed by `author_id` to preserve ordering per author while scaling consumers.

---

### 4.3 Feed Generation & Ranking

#### Ranking Signals

| Signal | Example | Notes |
|--------|---------|--------|
| **Recency** | `created_at` | Baseline; decay over time |
| **Engagement** | likes, comments, shares, dwell | Requires logging; cold-start for new posts |
| **Relationship strength** | DMs, frequent interaction | Privacy-sensitive; opt-in in some products |
| **Content type** | video vs text | Product goals (e.g., promote Reels) |
| **Quality / safety** | policy scores | Demote harmful or spam |

#### Scoring Function (Illustrative)

\[
\text{score}(p) = w_1 \cdot f_{\text{recency}}(t_p) + w_2 \cdot g(\text{engagement}_p) + w_3 \cdot h(\text{affinity}_{u,a}) - w_4 \cdot \text{penalty}(p)
\]

Where \(f_{\text{recency}}\) might be exponential decay; \(g\) could be log-scaled counts to dampen virality.

#### ML-Based Ranking Pipeline

```mermaid
flowchart LR
    C[Candidates from Redis + retrieval] --> F[Feature joiner]
    F --> M[Model ensemble]
    M --> R[Re-rank top K]
    R --> Out[Response]
    L[Logging] --> T[Offline training]
    T --> M
```

- **Retrieval:** cheap candidate generation (time + social + light ML).
- **Ranking:** heavier models on smaller sets (hundreds to thousands).
- **Exploration:** inject some random eligible posts to gather engagement labels.

#### Feed Assembly

- Merge **in-network** posts with **out-of-network** recommendations if required.
- **Dedupe** by `post_id` and by **story clusters** (same URL / same media).

#### Python: Ranking Service (Sketch)

```python
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass(frozen=True)
class PostFeatures:
    post_id: str
    author_id: str
    created_at_ms: int
    likes: int
    comments: int
    affinity: float  # 0..1


def recency_score(now_ms: int, created_ms: int, half_life_ms: int = 86_400_000) -> float:
    age = max(0, now_ms - created_ms)
    return math.exp(-age / half_life_ms)


def engagement_score(likes: int, comments: int) -> float:
    raw = 1.0 + math.log1p(likes) + 0.5 * math.log1p(comments)
    return min(raw, 50.0)


def linear_score(f: PostFeatures, now_ms: int, weights: Dict[str, float]) -> float:
    r = recency_score(now_ms, f.created_at_ms)
    e = engagement_score(f.likes, f.comments)
    return (
        weights.get("recency", 1.0) * r
        + weights.get("engagement", 0.5) * e
        + weights.get("affinity", 0.8) * f.affinity
    )


class RankingService:
    def __init__(self, weights: Dict[str, float]) -> None:
        self._w = weights

    def rank(self, user_id: str, posts: Sequence[PostFeatures], now_ms: int) -> List[PostFeatures]:
        scored = [(linear_score(p, now_ms, self._w), p) for p in posts]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored]


# Example: later swap linear_score with a model server call returning P(engage)
```

---

### 4.4 Feed Cache Architecture

- **Redis sorted sets:** key `feed:{user_id}` → members `post_id`, score = `created_at_ms` or blended pre-score.
- **Cap** list length (e.g., keep latest 2k ids); trim after each push.
- **Invalidation:** on unfollow / delete / privacy change — remove entries or rebuild slice; often **lazy repair** + **TTL** on keys.

| Strategy | When to use |
|----------|-------------|
| **Pre-computed feed** | Push fan-out + Redis; best for read latency |
| **On-demand merge** | Pull path for celebrities or cold users |
| **Write-through** | Update cache when post deleted (best effort + compensating job) |

#### Java: Feed Cache Manager

```java
package com.example.feed.cache;

import redis.clients.jedis.Jedis;
import redis.clients.jedis.params.ZAddParams;

public final class FeedCacheManager {
    private final Jedis jedis;
    private final int maxEntries;

    public FeedCacheManager(Jedis jedis, int maxEntries) {
        this.jedis = jedis;
        this.maxEntries = maxEntries;
    }

    public void addPost(String userId, String postId, double score) {
        String key = "feed:" + userId;
        jedis.zadd(key, score, postId, ZAddParams.zAddParams().nx());
        jedis.zremrangeByRank(key, 0, -maxEntries - 1);
    }

    public java.util.List<String> getPage(String userId, int limit, double maxScoreExclusive) {
        String key = "feed:" + userId;
        return jedis.zrevrangeByScore(key, "(" + maxScoreExclusive, "-inf", 0, limit);
    }
}
```

#### Go: Feed Cache Manager

```go
package cache

import (
	"context"
	"fmt"

	"github.com/redis/go-redis/v9"
)

type FeedCacheManager struct {
	Rdb        *redis.Client
	MaxEntries int64
}

func (m *FeedCacheManager) AddPost(ctx context.Context, userID, postID string, score float64) error {
	key := fmt.Sprintf("feed:%s", userID)
	pipe := m.Rdb.TxPipeline()
	pipe.ZAdd(ctx, key, redis.Z{Score: score, Member: postID})
	pipe.ZRemRangeByRank(ctx, key, 0, -(m.MaxEntries + 1))
	_, err := pipe.Exec(ctx)
	return err
}

func (m *FeedCacheManager) GetPage(ctx context.Context, userID string, limit int, maxExclusive float64) ([]string, error) {
	key := fmt.Sprintf("feed:%s", userID)
	opt := &redis.ZRangeBy{
		Min:   "-inf",
		Max:   fmt.Sprintf("(%f", maxExclusive),
		Count: int64(limit),
	}
	return m.Rdb.ZRevRangeByScore(ctx, key, opt).Result()
}
```

---

### 4.5 Social Graph Storage

| Approach | Pros | Cons |
|----------|------|------|
| **Adjacency in RDBMS** | Simple, transactional follow/unfollow | Heavy fan-out queries need careful indexing |
| **Graph DB** | Rich traversals | Ops + cost; not always needed for follow-only |
| **Sharded key-value** | Massive scale | Application-level consistency patterns |

**Follow / unfollow:** transactional insert/delete on `(follower_id, followee_id)`; async job updates **follower counts** and **celebrity flags**.

**Celebrity detection:** e.g., `followee.follower_count > 100_000` → mark as **high fan-out**; store in profile cache.

#### Go: Graph Service

```go
package graph

import (
	"context"
	"database/sql"
	"errors"
)

type Store struct {
	DB *sql.DB
}

func (s *Store) Follow(ctx context.Context, followerID, followeeID string) error {
	if followerID == followeeID {
		return errors.New("cannot follow self")
	}
	_, err := s.DB.ExecContext(ctx,
		`INSERT INTO follows (follower_id, followee_id) VALUES ($1, $2)
		 ON CONFLICT DO NOTHING`,
		followerID, followeeID,
	)
	return err
}

func (s *Store) Unfollow(ctx context.Context, followerID, followeeID string) error {
	_, err := s.DB.ExecContext(ctx,
		`DELETE FROM follows WHERE follower_id = $1 AND followee_id = $2`,
		followerID, followeeID,
	)
	return err
}

func (s *Store) FollowerCount(ctx context.Context, userID string) (int64, error) {
	var c int64
	err := s.DB.QueryRowContext(ctx,
		`SELECT COUNT(*) FROM follows WHERE followee_id = $1`, userID,
	).Scan(&c)
	return c, err
}

func (s *Store) IsCelebrity(ctx context.Context, userID string, threshold int64) (bool, error) {
	c, err := s.FollowerCount(ctx, userID)
	if err != nil {
		return false, err
	}
	return c >= threshold, nil
}
```

---

### 4.6 Media Handling

| Piece | Role |
|-------|------|
| **Object storage (S3)** | Durable blobs; lifecycle policies |
| **CDN** | Edge caching of images/video segments |
| **Image pipeline** | Thumbnails, responsive sizes |
| **Video** | Transcoding ladders (1080p, 720p, …), captions |

**Pre-signed upload URL:** short TTL, `Content-Type` constraint, max size in policy.

**Image resizing pipeline (typical):** object created → **S3 event** / queue message → **worker** generates variants (`thumb`, `medium`, `large`, WebP) → writes keys back to `post_media` → clients get **srcset** URLs.

**Video transcoding:** ingest high-res upload → **transcoder** produces HLS/DASH renditions + poster frame → store manifests in object storage; **player** fetches adaptive bitrate from CDN.

```mermaid
flowchart LR
    U[Client upload] --> S3[(Raw object)]
    S3 --> Q[Queue]
    Q --> Img[Image workers]
    Q --> Vid[Video transcoder]
    Img --> V[(Variants + CDN)]
    Vid --> V
```

{: .tip }
> **Cache keys** at CDN should include **content hash** or version in the path so updates propagate predictably (`/media/{id}/v2/thumb.jpg`).

---

### 4.7 Real-time Updates

| Mechanism | Pros | Cons |
|-----------|------|------|
| **Long polling** | Simple through HTTP | Many connections; higher latency than push |
| **WebSockets** | Bidirectional, low latency | Stateful gateways; reconnect logic |
| **SSE** | One-way server → client over HTTP | Not for binary-heavy protocols alone |

**New post to active viewers:** optional **presence** service + **pub/sub** channel per user session; push lightweight “refresh” or incremental post payload.

**Live engagement counts:** **CRDT / periodic aggregation** or **Redis INCR** with batched writes to DB; clients may see slightly stale counts.

```mermaid
sequenceDiagram
    participant A as Author
    participant API as Post API
    participant PS as Pub/Sub
    participant GW as WS Gateway
    participant F as Follower client

    A->>API: POST /posts
    API->>PS: notify followers rooms (batched)
    PS->>GW: events
    GW-->>F: {type: "new_post", post_id}
```

### 4.8 Pagination, Cursors, and Idempotency

**Why cursors:** Offset pagination (`?offset=40`) performs poorly when new rows arrive while the user pages—items **shift** and users see skips or duplicates. **Cursor** pagination ties each page to a stable position in the ordering key (time or rank).

| Approach | Encode | Pros | Cons |
|----------|--------|------|------|
| **Opaque cursor** | Base64(JSON) with `score`, `post_id` tie-breaker | Simple to evolve | Tamperable unless signed |
| **Signed cursor** | HMAC(cursor_payload, secret) | Client cannot forge | Key rotation story |
| **Seek method** | `WHERE (score, id) < (?, ?)` | DB-friendly | Needs composite index |

**Idempotency for writes:** `POST /posts` should accept **`Idempotency-Key`** header; store `(user_id, key) → post_id` in Redis with TTL to dedupe retries.

**Fan-out idempotency:** Kafka messages carry **`event_id`**; workers record processed ids in Redis `SET` or DB to avoid duplicate ZSET inserts.

#### Java: Cursor encoding and feed page

```java
package com.example.feed.api;

import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.List;
import java.util.Optional;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;

public final class FeedCursor {
    public final double maxScoreExclusive;
    public final String tiePostId;

    @JsonCreator
    public FeedCursor(
            @JsonProperty("s") double maxScoreExclusive,
            @JsonProperty("t") String tiePostId) {
        this.maxScoreExclusive = maxScoreExclusive;
        this.tiePostId = tiePostId;
    }

    public static Optional<FeedCursor> decode(String token, ObjectMapper mapper) {
        if (token == null || token.isBlank()) {
            return Optional.empty();
        }
        try {
            byte[] raw = Base64.getUrlDecoder().decode(token);
            return Optional.of(mapper.readValue(raw, FeedCursor.class));
        } catch (Exception e) {
            return Optional.empty();
        }
    }

    public String encode(ObjectMapper mapper) throws Exception {
        byte[] raw = mapper.writeValueAsBytes(this);
        return Base64.getUrlEncoder().withoutPadding().encodeToString(raw);
    }
}

public record FeedItemDto(
        String postId,
        String authorId,
        String text,
        long createdAtMs,
        EngagementDto engagement) {}

public record EngagementDto(long likes, long comments) {}

public record FeedPageResponse(List<FeedItemDto> items, String nextCursor) {}

public final class FeedReadService {
    private final FeedCacheManager cache;
    private final ObjectMapper mapper;

    public FeedReadService(FeedCacheManager cache, ObjectMapper mapper) {
        this.cache = cache;
        this.mapper = mapper;
    }

    public FeedPageResponse getFeed(String userId, int limit, String cursorToken) throws Exception {
        double maxScore = Double.POSITIVE_INFINITY;
        if (cursorToken != null) {
            var c = FeedCursor.decode(cursorToken, mapper);
            if (c.isPresent()) {
                maxScore = c.get().maxScoreExclusive;
            }
        }
        List<String> ids = cache.getPage(userId, limit + 1, maxScore);
        boolean hasMore = ids.size() > limit;
        if (hasMore) {
            ids = ids.subList(0, limit);
        }
        String next = null;
        if (hasMore && !ids.isEmpty()) {
            // In production hydrate scores from Redis or DB for accurate cursor
            FeedCursor nc = new FeedCursor(Double.NEGATIVE_INFINITY, ids.get(ids.size() - 1));
            next = nc.encode(mapper);
        }
        return new FeedPageResponse(List.of(), next);
    }
}
```

#### Python: Idempotent post handler (sketch)

```python
from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Optional

import redis


def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


@dataclass
class FeedCursor:
    max_score_exclusive: float
    tie_post_id: Optional[str] = None

    @staticmethod
    def decode(token: Optional[str]) -> Optional["FeedCursor"]:
        if not token:
            return None
        try:
            obj = json.loads(_b64url_decode(token))
            return FeedCursor(float(obj["s"]), obj.get("t"))
        except Exception:
            return None

    def encode(self) -> str:
        payload = {"s": self.max_score_exclusive, "t": self.tie_post_id}
        return _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))


class IdempotencyStore:
    def __init__(self, r: redis.Redis, ttl_sec: int = 86400) -> None:
        self._r = r
        self._ttl = ttl_sec

    def key(self, user_id: str, idempotency_key: str) -> str:
        h = hashlib.sha256(f"{user_id}:{idempotency_key}".encode()).hexdigest()
        return f"idemp:post:{h}"

    def get_or_set(self, user_id: str, idempotency_key: str, post_id: str) -> str:
        k = self.key(user_id, idempotency_key)
        prev = self._r.get(k)
        if prev:
            return prev.decode("utf-8")
        self._r.set(k, post_id, ex=self._ttl, nx=True)
        return post_id
```

#### Go: Kafka fan-out consumer loop

```go
package fanout

import (
	"context"
	"encoding/json"
	"log"
	"time"

	"github.com/segmentio/kafka-go"
)

type Consumer struct {
	Reader *kafka.Reader
	Worker *Worker
	Graph  FollowGraphClient
}

func (c *Consumer) Run(ctx context.Context) {
	for {
		m, err := c.Reader.FetchMessage(ctx)
		if err != nil {
			log.Printf("fetch: %v", err)
			time.Sleep(time.Second)
			continue
		}
		var e PostCreatedEvent
		if err := json.Unmarshal(m.Value, &e); err != nil {
			_ = c.Reader.CommitMessages(ctx, m)
			continue
		}
		fc, err := followerCount(ctx, c.Graph, e.AuthorID)
		if err != nil {
			log.Printf("followers: %v", err)
			continue
		}
		if err := c.Worker.OnPostCreated(e, fc); err != nil {
			log.Printf("fanout: %v", err)
			continue
		}
		if err := c.Reader.CommitMessages(ctx, m); err != nil {
			log.Printf("commit: %v", err)
		}
	}
}

func followerCount(ctx context.Context, g FollowGraphClient, author string) (int64, error) {
	// Delegate to cached count service in production
	var total int64
	for _, page := range g.FollowersPaged(author, 5000) {
		total += int64(len(page))
	}
	return total, nil
}
```

{: .note }
> The Go snippet’s `followerCount` is illustrative — production reads **denormalized** `follower_count` from the profile store to avoid scanning followers.

---

### 4.9 Media Pipeline — Pre-signed URLs (Java / Python / Go)

**Java (AWS SDK v2 style, illustrative)**

```java
package com.example.feed.media;

import java.net.URL;
import java.time.Duration;
import java.util.UUID;

import software.amazon.awssdk.services.s3.model.PutObjectRequest;
import software.amazon.awssdk.services.s3.presigner.S3Presigner;
import software.amazon.awssdk.services.s3.presigner.model.PutObjectPresignRequest;

public final class PresignedUploadService {
    private final S3Presigner presigner;
    private final String bucket;

    public PresignedUploadService(S3Presigner presigner, String bucket) {
        this.presigner = presigner;
        this.bucket = bucket;
    }

    public UploadUrlResponse createUploadUrl(String userId, String contentType, long maxBytes) {
        String key = "u/" + userId + "/" + UUID.randomUUID();
        PutObjectRequest put = PutObjectRequest.builder()
                .bucket(bucket)
                .key(key)
                .contentType(contentType)
                .contentLength(maxBytes)
                .build();
        PutObjectPresignRequest pre = PutObjectPresignRequest.builder()
                .signatureDuration(Duration.ofMinutes(10))
                .putObjectRequest(put)
                .build();
        URL url = presigner.presignPutObject(pre).url();
        return new UploadUrlResponse(url.toString(), key);
    }

    public record UploadUrlResponse(String uploadUrl, String objectKey) {}
}
```

**Python (boto3)**

```python
import uuid
from dataclasses import dataclass

import boto3


@dataclass(frozen=True)
class UploadUrlResponse:
    upload_url: str
    object_key: str


class PresignedUploadService:
    def __init__(self, bucket: str, region: str) -> None:
        self._bucket = bucket
        self._client = boto3.client("s3", region_name=region)

    def create_upload_url(self, user_id: str, content_type: str, max_bytes: int) -> UploadUrlResponse:
        key = f"u/{user_id}/{uuid.uuid4().hex}"
        url = self._client.generate_presigned_url(
            ClientMethod="put_object",
            Params={
                "Bucket": self._bucket,
                "Key": key,
                "ContentType": content_type,
                "ContentLength": max_bytes,
            },
            ExpiresIn=600,
        )
        return UploadUrlResponse(upload_url=url, object_key=key)
```

**Go (aws-sdk-go-v2)**

```go
package media

import (
	"context"
	"fmt"
	"time"

	v4 "github.com/aws/aws-sdk-go-v2/aws/signer/v4"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/google/uuid"
)

type Presigner struct {
	Client    *s3.Client
	Presigner *s3.PresignClient
	Bucket    string
}

type UploadURLResponse struct {
	UploadURL string
	ObjectKey string
}

func (p *Presigner) CreateUploadURL(ctx context.Context, userID, contentType string, maxBytes int64) (UploadURLResponse, error) {
	key := fmt.Sprintf("u/%s/%s", userID, uuid.NewString())
	in := &s3.PutObjectInput{
		Bucket:        &p.Bucket,
		Key:           &key,
		ContentType:   &contentType,
		ContentLength: &maxBytes,
	}
	out, err := p.Presigner.PresignPutObject(ctx, in, s3.WithPresignExpires(10*time.Minute))
	if err != nil {
		return UploadURLResponse{}, err
	}
	return UploadURLResponse{UploadURL: out.URL, ObjectKey: key}, nil
}

// v4.Signer used implicitly by PresignClient — keep import if custom signing needed
var _ = v4.Signer{}
```

{: .warning }
> Tune **CORS** on the bucket for browser direct uploads; validate **MIME** server-side before attaching media to a post.

---

### 4.10 Pull-Path Feed Merge (Read-Time Aggregation)

When a user follows celebrities not fully pushed, the **Feed Service** merges:
1. Redis ZSET candidates (normal follows).
2. Recent posts from **celebrity shards** / **author timelines** (bounded K authors × M posts).

```mermaid
flowchart TB
    R[GET /feed] --> A[Load ZSET ids]
    R --> B[Load celebrity recent]
    A --> M[Merge + dedupe]
    B --> M
    M --> Rank[Rank / truncate]
```

**Java: merge helper**

```java
package com.example.feed.merge;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public final class FeedMerger {
    public static List<String> mergeByScore(
            List<ScoredId> a,
            List<ScoredId> b,
            int limit) {
        PriorityQueue<ScoredId> pq = new PriorityQueue<>(Comparator.comparingDouble(ScoredId::score).reversed());
        pq.addAll(a);
        pq.addAll(b);
        LinkedHashSet<String> seen = new LinkedHashSet<>();
        List<String> out = new ArrayList<>();
        while (!pq.isEmpty() && out.size() < limit) {
            ScoredId x = pq.poll();
            if (seen.add(x.id())) {
                out.add(x.id());
            }
        }
        return out;
    }

    public record ScoredId(String id, double score) {}
}
```

**Python**

```python
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Iterable, List, Set


@dataclass(order=True)
class ScoredId:
    neg_score: float
    post_id: str

    @staticmethod
    def from_pair(post_id: str, score: float) -> "ScoredId":
        return ScoredId(neg_score=-score, post_id=post_id)


def merge_by_score(streams: Iterable[List[tuple[str, float]]], limit: int) -> List[str]:
    heap: List[tuple[float, int, str, int]] = []
    arrays = [list(s) for s in streams]
    for i, arr in enumerate(arrays):
        if arr:
            pid, sc = arr.pop()
            heapq.heappush(heap, (-sc, i, pid, 0))
    out: List[str] = []
    seen: Set[str] = set()
    while heap and len(out) < limit:
        neg_sc, idx, pid, _ = heapq.heappop(heap)
        if pid not in seen:
            seen.add(pid)
            out.append(pid)
        if arrays[idx]:
            nxt, sc = arrays[idx].pop()
            heapq.heappush(heap, (-sc, idx, nxt, 0))
    return out
```

**Go**

```go
package merge

import "sort"

type ScoredID struct {
	ID    string
	Score float64
}

func MergeByScore(a, b []ScoredID, limit int) []string {
	all := append(append([]ScoredID{}, a...), b...)
	sort.Slice(all, func(i, j int) bool {
		if all[i].Score == all[j].Score {
			return all[i].ID < all[j].ID
		}
		return all[i].Score > all[j].Score
	})
	seen := map[string]struct{}{}
	var out []string
	for _, x := range all {
		if len(out) >= limit {
			break
		}
		if _, ok := seen[x.ID]; ok {
			continue
		}
		seen[x.ID] = struct{}{}
		out = append(out, x.ID)
	}
	return out
}
```

---

## Step 5: Scaling & Production

### Scaling Strategy

| Layer | Technique |
|-------|-----------|
| **Database** | Shard by `user_id` or `post_id`; separate read replicas for timelines |
| **Feed cache** | Redis Cluster; hash tags per `user_id` |
| **Fan-out workers** | Horizontal Kafka consumers; partition by author |
| **CDN + multi-region** | Static media at edge; API in regional stacks with replication |

### Celebrity Problem

- **Do not** fan-out millions of writes synchronously.
- **Combine:** materialized recent posts for star authors + merge at read; **cache** hot author timelines separately.

### Failure Handling

| Failure | Degradation |
|---------|-------------|
| **Redis miss** | Rebuild slice from origin DB / pull path (slower) |
| **Ranking timeout** | Fall back to reverse-chronological candidates |
| **Partial fan-out lag** | Serve stale feed + background catch-up |

{: .warning }
> Always define **SLIs**: p99 feed latency, fan-out lag, cache hit ratio — and **error budgets** for releases.

### Monitoring

| Metric | Why |
|--------|-----|
| **Feed p50/p99 latency** | User-visible |
| **Cache hit ratio** | Capacity and correctness |
| **Fan-out lag** | Kafka consumer health |
| **Posts / sec** | Traffic spikes |
| **Per-author fan-out time** | Celebrity incidents |

---

## Interview Tips

| Topic | Common follow-up |
|-------|------------------|
| **Fan-out** | Draw push vs pull; when hybrid wins |
| **Ordering** | Global vs per-user ordering; clock skew |
| **Ranking** | Exploration vs exploitation; bias |
| **Consistency** | What if follower list changes mid-fan-out? |
| **Storage** | Hot keys, partitioning, MySQL vs Cassandra |
| **Media** | CDN cache keys, signed URLs, abuse |

{: .tip }
> End with **trade-offs**: e.g., “We chose Redis sorted sets for O(log N) inserts with caps; alternative is Cassandra wide rows with TTL — higher ops complexity.”

---

## Quick Reference Tables

### Comparison: Fan-out Models

| Dimension | Push | Pull | Hybrid |
|-----------|------|------|--------|
| Post cost | High for many followers | Low | Medium |
| Read cost | Low | High | Medium |
| Complexity | Medium | Medium | High |

### API Summary

| Endpoint | Purpose |
|----------|---------|
| `POST /v1/posts` | Create |
| `GET /v1/feed` | Paginated feed |
| `POST /v1/follow` | Follow |

---

_Last updated: system design interview prep — News Feed / Timeline._
