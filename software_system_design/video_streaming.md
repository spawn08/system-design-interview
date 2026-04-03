---
layout: default
title: Video Streaming (YouTube)
parent: System Design Examples
nav_order: 11
---

# Video Streaming (YouTube)
{: .no_toc }

<details open markdown="block">
  <summary>Table of contents</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

## What We're Building

A **video streaming platform** lets creators upload long-form and short-form video, processes those files into multiple formats and bitrates, stores and indexes metadata, and delivers playback to viewers worldwide with adaptive quality, low startup latency, and high availability. The canonical reference product is **YouTube**; similar patterns appear in **Vimeo**, **Twitch** (live differs but shares CDN and transcoding ideas), and enterprise **VOD** stacks.

**Core capabilities we will cover:**

| Capability | Brief description |
|------------|-------------------|
| **Resumable upload** | Chunked uploads with checksums so large files survive flaky networks |
| **Transcoding** | FFmpeg (or cloud encoder APIs) to produce H.264/VP9/AV1 ladders |
| **Adaptive streaming** | HLS or DASH segments + manifests for ABR clients |
| **CDN delivery** | Edge caching of segments and manifests close to viewers |
| **Metadata and search** | Titles, tags, channels, inverted indexes, sometimes ASR captions |
| **Thumbnails** | Sprites or keyframe extractions for previews and seek bars |
| **View analytics** | Approximate counts, aggregation pipelines, fraud resistance |
| **Recommendations** | Candidate generation + ranking (often separate ML stack) |

### Why This Problem Is Hard

| Challenge | Consequence |
|-----------|-------------|
| **Egress cost** | Video bytes dwarf API JSON; CDN and encoding are the main bills |
| **CPU-heavy transcoding** | Long videos need distributed workers and queue backpressure |
| **Global latency** | Playback must start quickly; manifests and first segments must be hot at edge |
| **Consistency vs cost** | Strong consistency everywhere is expensive; many paths are eventual |
| **Abuse and compliance** | Copyright, CSAM, and spam require detection pipelines and policy |

{: .note }
> In interviews, **clarify live vs VOD** early. Live streaming uses RTMP/WebRTC ingest and LL-HLS/WebRTC egress; this page focuses on **VOD** (upload then watch), which matches “Design YouTube” unless the interviewer specifies live.

### High-Level System Context (Preview)

```mermaid
flowchart TB
    subgraph Clients
        U[Uploader client]
        V[Viewer client]
    end
    subgraph Edge
        CDN[CDN / Edge POPs]
    end
    subgraph API
        GW[API Gateway]
        UP[Upload service]
        META[Metadata service]
        SRCH[Search service]
    end
    subgraph Data
        OBJ[(Object storage)]
        DB[(Metadata DB)]
        IDX[(Search index)]
        CACHE[(Redis cache)]
    end
    subgraph Processing
        Q[Job queue]
        DAG[Transcode DAG workers]
        TH[Thumbnail workers]
        COPY[Copyright / safety pipeline]
    end
    U --> GW
    V --> CDN
    GW --> UP
    GW --> META
    GW --> SRCH
    UP --> OBJ
    UP --> Q
    Q --> DAG
    DAG --> OBJ
    TH --> OBJ
    META --> DB
    META --> CACHE
    SRCH --> IDX
    COPY --> Q
```

---

## Step 1: Requirements

### Functional Requirements

| ID | Requirement | Notes |
|----|-------------|--------|
| FR-1 | Users can **upload** video files (large, resumable) | Chunked multipart or gRPC streams |
| FR-2 | System **transcodes** uploads into multiple renditions | Resolution + bitrate ladder |
| FR-3 | Viewers **play** video in browser/app with **adaptive bitrate** | HLS and/or DASH |
| FR-4 | **Metadata**: title, description, tags, channel, visibility | CRUD + validation |
| FR-5 | **Search** videos by keyword (and optionally filters) | Inverted index; ranking later |
| FR-6 | **Thumbnails** for grid and seek preview | Generated post-upload |
| FR-7 | **View counts** and basic analytics for creators | Often approximate at scale |
| FR-8 | **Home / feed** surfaces recommended videos | ML ranking; can be scoped “out of band” |

{: .tip }
> Mark **recommendations** as a separate sub-system if time is short: ingest signals (views, watches, subs) and serve ranked lists from an offline + online stack.

### Non-Functional Requirements

| Category | Target | Rationale |
|----------|--------|-----------|
| **Playback start (p95)** | 2–5 s on good networks | Industry expectation; CDN + small first segments help |
| **Upload reliability** | Resume after disconnect | Mobile networks drop often |
| **Availability** | 99.9%+ for read path | Writes and processing can degrade gracefully with status UI |
| **Durability** | No loss of committed uploads | Replicated object storage + job idempotency |
| **Scale** | Horizontal workers, sharded metadata | Avoid single-node transcode |
| **Compliance** | Takedowns, copyright, regional restrictions | Legal and policy hooks |

### API Design

Typical REST (or gRPC internal) surface:

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/v1/uploads` | Start upload session: returns `upload_id`, chunk size, object key prefix |
| `PUT` | `/v1/uploads/{upload_id}/parts/{part_number}` | Upload one chunk (with Content-Range or fixed part size) |
| `POST` | `/v1/uploads/{upload_id}/complete` | Finalize multipart upload, trigger processing |
| `GET` | `/v1/videos/{video_id}` | Metadata + processing status + playback URLs |
| `GET` | `/v1/videos/{video_id}/manifest.m3u8` | HLS master or redirect to CDN URL |
| `GET` | `/v1/search?q=` | Keyword search |
| `POST` | `/v1/videos/{video_id}/views` | Client beacons for views (often batched) |

**Example: start upload response**

```json
{
  "upload_id": "up_7f91c2",
  "chunk_bytes": 8388608,
  "max_parts": 10000,
  "storage": { "backend": "s3", "bucket": "uploads-prod", "key_prefix": "raw/up_7f91c2/" }
}
```

{: .warning }
> Never expose **raw storage credentials** to clients. Use **pre-signed URLs** or short-lived upload tokens so the browser talks directly to object storage where appropriate.

---

## Step 2: Back-of-the-Envelope Estimation

Assume a simplified global VOD service (not YouTube’s exact numbers):

| Assumption | Value |
|------------|--------|
| Daily video uploads | 10 million |
| Average raw size per upload | 500 MB |
| Average views per uploaded video (first month) | 200 |
| Average watch: 40% of duration, 1080p equivalent egress | blended bitrate ~ 5 Mbps |

**Ingest storage per day (raw):**

```
10^7 uploads × 500 MB = 5 × 10^15 bytes ≈ 5 PB/day raw
```

In practice, not all stay forever; lifecycle policies and deduplication reduce **net** growth. Transcoded ladder might add **2–4×** raw for all renditions and segments.

**Transcoding compute (order of magnitude):**

```
Assume 1 CPU-hour encodes 20 minutes of 1080p (varies by codec and preset)
10^7 uploads/day × 20 min avg length = 2×10^8 minutes/day
CPU-hours ≈ (2×10^8 / 20) = 10^7 CPU-hours/day
→ hundreds of thousands of parallel cores at peak if not spread evenly
```

**Read path egress (very rough):**

```
10^7 uploads × 200 views × 20 min × 40% watched × 5 Mbps
= 10^7 × 200 × 1200 s × 0.4 × 625 KB/s
≈ huge PB-scale/month; CDN caches reduce origin hits
```

{: .note }
> Estimation **validates architecture**: you need **queues**, **worker autoscaling**, **CDN**, and **tiered storage**. Exact digits matter less than magnitude.

| Resource | Bottleneck signal | Mitigation |
|----------|-------------------|------------|
| Object storage | PB/month growth | Lifecycle to cold tier; dedupe |
| Transcode workers | Queue lag | Horizontal scale; spot instances for batch |
| CDN | Egress bill | Cache hit ratio, regional POPs |
| Metadata DB | Hot rows on viral video | Cache + read replicas; avoid contended counters |

---

## Step 3: High-Level Design

### Components

| Component | Responsibility |
|-----------|----------------|
| **Upload service** | Sessions, chunk validation, multipart completion, dedupe hints |
| **Object storage** | Raw uploads, transcoded segments, thumbnails (S3-compatible) |
| **Job queue / orchestrator** | Kafka, SQS, or Temporal for durable work items |
| **Transcode DAG engine** | Dependency graph: demux → decode → multiple encodes → package |
| **CDN** | Cache manifests and `.ts` / `.m4s` segments at edge |
| **Metadata service** | Postgres / Dynamo-style store for video and channel rows |
| **Search** | Elasticsearch/OpenSearch or managed search |
| **View pipeline** | Ingest beacons → stream processing → aggregated counts |
| **Recommendation** | Offline features + online serving (two-tower, etc.) |

### High-Level Architecture (Mermaid)

```mermaid
flowchart LR
    subgraph UploadPath[Upload path]
        C[Client] --> APIGW[API Gateway]
        APIGW --> US[Upload service]
        US -->|presigned| S3[(Object storage)]
        US -->|enqueue| K[Kafka topics]
    end
    subgraph ProcessPath[Processing]
        K --> ORC[Orchestrator / DAG]
        ORC --> TC[Transcode pool]
        TC --> S3
        ORC --> TH2[Thumbnail service]
        TH2 --> S3
        ORC --> META2[Metadata indexer]
    end
    subgraph ReadPath[Read path]
        V2[Viewer] --> CDN2[CDN]
        CDN2 -->|miss| S3
        V2 --> APIGW2[API Gateway]
        APIGW2 --> VID[Video API]
        VID --> DB[(DB)]
    end
```

### Data Flow Summary

1. **Upload**: Client obtains `upload_id`, sends chunks to storage, completes session; upload service emits **VideoReceived** event.
2. **Process**: Orchestrator builds a **DAG** (see deep dive); workers write renditions and segments to object storage; metadata updated to `READY`.
3. **Play**: Client fetches metadata API for **CDN URLs** of HLS/DASH; player downloads manifest then segments **from CDN**.

{: .tip }
> Keep **control plane** (APIs, auth) separate from **data plane** (bytes to CDN). Mixing them complicates scaling and security reviews.

---

## Step 4: Deep Dive

### 4.1 Video Upload and Processing Pipeline

**Chunked / resumable upload** reduces failure rates. Common pattern:

1. `POST /uploads` → receive `upload_id` and per-part size.
2. For each part: `PUT` with part number and **MD5 or SHA-256** (for multipart ETags in S3).
3. `POST /complete` with ordered **part list**; storage merges multipart object.
4. Upload service publishes message `{ upload_id, object_key, user_id, content_hash }`.

**Deduplication:** compute **perceptual hash** or cryptographic hash of source (after normalize) to skip re-transcoding identical uploads, or to map to a single canonical asset (policy-dependent).

```mermaid
sequenceDiagram
    participant CL as Client
    participant API as Upload API
    participant OS as Object Storage
    participant Q as Queue
    participant OR as Orchestrator
    CL->>API: POST /uploads (metadata)
    API->>CL: upload_id, presigned URLs
    loop chunks
        CL->>OS: PUT part n
        OS->>CL: ETag / checksum
    end
    CL->>API: POST /complete (parts, hashes)
    API->>OS: CompleteMultipartUpload
    API->>Q: VideoUploaded event
    Q->>OR: consume
    OR->>OR: build DAG, schedule jobs
```

{: .warning }
> **Copyright and policy** checks often run in parallel or after ingest: fingerprinting (e.g. Content ID–style), hash lists, and ML classifiers. Mention **latency vs safety** trade-off: block publish until cleared vs publish then remove.

### 4.2 Video Transcoding (Adaptive Bitrate)

**Transcoding** converts mezzanine (uploaded) video into:

- Multiple **resolutions** (e.g. 2160p, 1440p, 1080p, 720p, 480p, 360p)
- Multiple **bitrates** per resolution (ladder)
- Packaged as **HLS** (`.m3u8` + `.ts` or fMP4) or **DASH** (`.mpd` + `.m4s`)

**FFmpeg** is the de facto tool in examples; in production you may wrap **ffmpeg** CLI or use **libav** APIs, cloud encoders, or GPU farms.

**DAG-based pipeline** example nodes:

| Stage | Output | Depends on |
|-------|--------|------------|
| **Probe** | codecs, duration, audio tracks | raw object |
| **Audio normalize** | AAC 128k stereo | probe |
| **Video ladder** | H.264/H.265/VP9 per rung | probe |
| **Package HLS** | segments + playlist | ladder |
| **Thumbnail strip** | sprite or keyframes | ladder or raw |

```mermaid
flowchart TB
    A[Raw object in storage] --> B[Probe node]
    B --> C[Audio transcode]
    B --> D[Video 1080p]
    B --> E[Video 720p]
    B --> F[Video 480p]
    D --> G[HLS packager 1080p]
    E --> H[HLS packager 720p]
    F --> I[HLS packager 480p]
    G --> J[Master playlist]
    H --> J
    I --> J
    J --> K[Write manifests + segments to storage]
```

{: .note }
> **Adaptive bitrate (ABR)** players switch between renditions based on bandwidth and buffer. **Per-title encoding** optimizes ladders per asset; **chunked CMAF** helps align segment boundaries across renditions.

### 4.3 CDN and Video Delivery (HLS/DASH)

- **Manifest** (master `.m3u8`) lists variant streams; each variant playlist lists **segment URLs**.
- **CDN** caches by URL; use **cache-friendly paths** with version or content hash in path to bust stale entries after re-transcode.
- **TLS** everywhere; **signed URLs** for private or premium content.
- **Origin shield** (optional) collapses requests from many POPs to fewer origin hits.

**Playback URL shape (illustrative):**

```
https://cdn.example.com/v/{video_id}/v3/hls/master.m3u8
```

Low-latency variants (LL-HLS) use partial segments; mostly relevant for live, but worth mentioning if interviewer asks.

### 4.4 Video Metadata and Search

- **Relational store** for channel, video row, status (`PROCESSING`, `READY`, `FAILED`), timestamps.
- **Search index** built from title, description, tags; **async** updates from change data capture or message bus.
- **Ranking** for search blends text relevance, engagement, freshness.

Denormalize **channel name** and **view counts** for display with care (eventual consistency).

### 4.5 Thumbnail Generation

- Extract **keyframes** at intervals or use **scene detection**.
- Store multiple sizes for grid vs hero image.
- **Sprite sheets** for hover preview on seek bar (mapping time range to sprite coordinates).

### 4.6 View Counting and Analytics

At scale, **exact** per-request increments on a single DB row **do not scale** for viral videos.

| Approach | Pros | Cons |
|--------|------|------|
| **Shard counters** | Higher write throughput | Reconciliation, complexity |
| **Buffered increments** (Redis flush) | Fast | Loss window on crash unless careful |
| **Stream processing** (Flink/Kafka) | Scalable aggregation | Delayed visibility |
| **HyperLogLog / sampling** | Cheap | Approximate |

Mitigate **fraud** with bot detection, rate limits per IP/account, and anomaly detection on spikes.

### 4.7 Recommendation Feed

Typically:

- **Candidate generation**: subs, related channels, embeddings nearest neighbors.
- **Ranking**: shallow models online; heavy training offline.
- **Exploration**: multi-armed bandit for new creators.

Log **impressions and clicks** for training; separate **data warehouse** from serving path.

### Deduplication and Content-Addressed Storage

For **ingest deduplication**:

| Step | Mechanism |
|------|-----------|
| Client or server computes **SHA-256** of each uploaded part | Detect identical byte streams |
| After complete, **finalize** hash of full object | Lookup in **content-addressed** table |
| If match exists | Point new `video_id` at existing `storage_key` for transcoded outputs (policy allowing) |

Benefits: lower **storage** and **CPU** for duplicate viral re-uploads. Risks: **legal** nuance (same file, different uploader); product may still require **separate metadata rows** and **per-uploader takedowns**.

### Copyright Detection (High Level)

Production systems combine **multiple signals**:

| Signal | Role |
|--------|------|
| **Perceptual audio/video fingerprints** | Match against rights-holder database (Content ID–class systems) |
| **Allowlist/blocklist hashes** | Known infringing or disallowed files |
| **ML classifiers** | Policy violations (non-copyright) |
| **Dispute workflow** | Counter-notification and appeals outside core infra |

Mention in interviews: detection runs on **processing pipeline** or **separate workers**; **blocking publish** vs **monetization rules** vs **geo-blocking** are product decisions. Never claim you can solve copyright completely in code alone.

---

## Step 5: Scaling & Production

| Area | Technique |
|------|-----------|
| **Upload** | Direct-to-S3 multipart; client-side retry; resume state in local storage |
| **Transcode** | Autoscale worker pools; priority queues for premium creators; spot for batch |
| **Storage** | Tiering; delete aborted multipart uploads; **dedupe** canonical blobs |
| **CDN** | High TTL for immutable segments; short TTL for manifests if needed |
| **DB** | Shard by `channel_id` or `video_id`; read replicas; cache hot metadata |
| **Observability** | Per-stage pipeline latency, queue depth, CDN hit ratio, SLO burn rates |

**Trade-offs (summary):**

| Decision | Option A | Option B |
|----------|----------|----------|
| Transcode location | On-prem GPU | Managed cloud encoder |
| Segment format | TS | fMP4 (CMAF) |
| View count | Approximate fast | Stronger consistency, costlier |
| Search | Managed OpenSearch | Self-operated (ops burden) |

{: .warning }
> **Single hot video** can overload recommendation and metadata caches; use **cache partitioning**, **request coalescing**, and **overload protection** (drop non-critical work first).

### Operational Concerns

| Concern | Practice |
|---------|----------|
| **Stuck transcode** | Visibility timeout on queue consumers; **DLQ** after N attempts; alert on job age p99 |
| **Poison message** | Schema validation at enqueue; **dead-letter** with payload inspection |
| **Codec rollout** | Canary new encoder version on subset of workers; compare **VMAF/PSNR** and bitrate |
| **Cost controls** | Cap concurrent encodes per tenant; defer non-urgent **4K** to off-peak |
| **Regional compliance** | Geo-restrict manifest URLs; metadata residency in specific regions |

### Failure Modes (Short)

1. **Origin overload**: CDN shield, rate limit origin, increase segment immutability so TTL can be long.
2. **Bad upload**: Client retries with same `upload_id`; server rejects incomplete multipart after TTL **AbortMultipartUpload** lifecycle.
3. **Partial transcode**: DAG marks failed stage; **partial outputs** may be deleted or kept for debug; user sees `FAILED` with retry action.

---

## Interview Tips

### Interview Checklist

| Phase | Checklist item |
|-------|----------------|
| **Clarify** | VOD vs live; user scale; regions; monetization (ads) in scope? |
| **Draw** | Upload path, queue, workers, storage, CDN, playback |
| **Numbers** | Rough ingest PB, transcode CPU, CDN egress magnitude |
| **Deep dive** | Pick **one**: transcoding ladder, CDN caching, or view counting |
| **Trade-offs** | Consistency vs cost; processing latency vs upload UX |
| **Wrap** | Monitoring, failure modes (stuck jobs, poison messages), compliance mention |

### Sample Interview Dialogue

**Interviewer:** “Design a video streaming service like YouTube.”

**Candidate:** “I’ll assume **VOD**: upload, process, then playback. Out of scope unless you want it: **live streaming**. Key flows: **resumable upload** to object storage, **async transcoding** to an ABR ladder, **HLS/DASH** delivery via **CDN**, plus **metadata, search, thumbnails**, and **approximate view counts**. I’ll walk through APIs, a rough size estimate, then a diagram.”

**Interviewer:** “How do you handle huge files?”

**Candidate:** “**Multipart upload** with fixed chunk size, checksum per part, and **resume** using uploaded part list. Server returns **pre-signed URLs** so data goes **directly to object storage**. On complete, we emit an event to a **queue** and a **DAG orchestrator** schedules probe, encode, package, and thumbnail jobs. Failed stages **retry** with idempotent job ids.”

**Interviewer:** “What about duplicate videos?”

**Candidate:** “We can hash the **source file** or a normalized intermediate. If it matches an existing **canonical asset**, we **skip transcode** and attach new metadata rows pointing at the same storage prefix, subject to **policy** (copyright still applies per upload).”

**Interviewer:** “How do view counts work at scale?”

**Candidate:** “A per-row counter in one DB row **doesn’t scale** for viral spikes. I’d use **sharded counters**, **Redis buffers** with periodic flush, or a **stream processor** aggregating view events—often **eventually consistent** with fraud controls. **Exact** totals can be reconciled offline.”

{: .tip }
> End with **two failures**: stuck transcoding (dead-letter queue, alert on age), and **CDN miss storm** (origin protection, request coalescing).

### Code Sketches (Multi-Language)

**Java — upload session handler (Spring-style)**

```java
@PostMapping("/v1/uploads")
public ResponseEntity<UploadSessionResponse> startUpload(
    @RequestBody StartUploadRequest req,
    @AuthenticationPrincipal UserPrincipal user) {
  String uploadId = idGenerator.nextId();
  String keyPrefix = "raw/" + user.getId() + "/" + uploadId + "/";
  List<PresignedPart> parts = storageService.presignMultipartUpload(
      BUCKET, keyPrefix, req.getContentType(), req.getTotalBytes());
  UploadSessionResponse body = new UploadSessionResponse(
      uploadId, PART_SIZE_BYTES, parts);
  return ResponseEntity.status(HttpStatus.CREATED).body(body);
}
```

**Python — transcoding worker job (FFmpeg invoke)**

```python
import subprocess
import os

def transcode_rendition(src_path: str, dst_dir: str, height: int, bitrate_k: int) -> str:
    os.makedirs(dst_dir, exist_ok=True)
    out = os.path.join(dst_dir, f"video_{height}p.mp4")
    cmd = [
        "ffmpeg", "-y", "-i", src_path,
        "-vf", f"scale=-2:{height}",
        "-c:v", "libx264", "-b:v", f"{bitrate_k}k",
        "-c:a", "aac", "-b:a", "128k",
        out,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return out

def package_hls(renditions: list[str], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # Simplified: real pipeline uses separate package step or ffmpeg HLS flags
    concat = "|".join(renditions)
    subprocess.run(
        ["ffmpeg", "-y", "-i", f"concat:{concat}", "-c", "copy",
         "-f", "hls", "-hls_time", "6", os.path.join(out_dir, "index.m3u8")],
        check=True,
    )
```

{: .note }
> Production code adds **logging**, **timeouts**, **resource limits**, and often **hardware encoders**; the above illustrates the **FFmpeg boundary**.

**Go — streaming HTTP handler (redirect to CDN manifest)**

```go
package main

import (
	"net/http"
	"path"
)

func (s *Server) ServeHLS(w http.ResponseWriter, r *http.Request) {
	videoID := path.Base(path.Dir(r.URL.Path))
	meta, err := s.Meta.Get(r.Context(), videoID)
	if err != nil {
		http.Error(w, "not found", http.StatusNotFound)
		return
	}
	if meta.Status != "READY" {
		http.Error(w, "processing", http.StatusAccepted)
		return
	}
	// Immutable CDN URL; long cache TTL on segments
	cdnURL := s.CDN.BaseURL + "/v/" + videoID + "/hls/master.m3u8"
	http.Redirect(w, r, cdnURL, http.StatusFound)
}
```

---

## Summary

| Topic | Takeaway |
|-------|----------|
| **Upload** | Multipart, resumable, pre-signed direct-to-storage; complete triggers pipeline |
| **Processing** | **DAG** of probe, encode, package; **FFmpeg**-class workers; idempotent stages |
| **Playback** | **HLS/DASH** ABR; **CDN** caches segments and manifests |
| **Metadata / search** | OLTP + async search index |
| **Thumbnails** | Keyframes or sprites; multiple resolutions |
| **Views** | Sharded or streaming aggregates; fraud controls |
| **Recommendations** | Usually separate; log-rich training data |
| **Compliance** | **Copyright** fingerprinting and policy; safety classifiers |

{: .note }
> You will not build all subsystems in 45 minutes. **Narrow** with the interviewer, draw **one** solid data flow, and **deep dive** on transcoding, CDN, or analytics—whichever they care about most.
