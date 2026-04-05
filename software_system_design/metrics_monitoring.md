---
layout: default
title: Metrics & Monitoring System
parent: System Design Examples
nav_order: 22
---

# Design a Metrics and Monitoring System (Datadog / Prometheus–style)
{: .no_toc }

<details open markdown="block">
  <summary>Table of contents</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

---

## What We're Building

We are designing a **metrics collection, storage, querying, and alerting** platform comparable in scope to **Datadog**, **Prometheus**, **Grafana Mimir**, **Thanos**, or **Google Monarch**: agents ship time-series samples, the backend stores them efficiently at scale, users query and visualize them, and **alert rules** fire when conditions breach SLOs.

| Capability | Why it matters |
|------------|----------------|
| **High-cardinality ingestion** | Modern microservices emit huge label combinations; the system must not collapse under cardinality explosions |
| **Durable, compressed storage** | Time-series data is append-heavy; compression (e.g., Gorilla-style) dominates cost |
| **Fast analytical queries** | Dashboards and incident response need sub-second queries over large windows |
| **Reliable alerting** | Missed pages cost revenue; false positives burn out on-call engineers |
| **Horizontal scale** | Ingestion and query load grow with services, regions, and tenants |

### Real-world scale (illustrative)

| System | Scale signal (public / typical talk) |
|--------|--------------------------------------|
| **Prometheus** | Single-server focus; federation and remote write for scale-out |
| **Datadog** | Trillions of points/day; multi-tenant SaaS |
| **Google Monarch** | Planet-scale; hierarchical aggregation |
| **Cortex / Mimir / Thanos** | Long-term Prometheus-compatible storage at cluster scale |

### Reference systems (interview vocabulary)

| System | Positioning | Typical talking point |
|--------|-------------|------------------------|
| **Prometheus** | Pull-first; local TSDB; single binary simplicity | Federation + remote write for scale; **TSDB block** format |
| **Datadog / Dynatrace / New Relic** | SaaS agents; unified APM + metrics | **Multi-tenant** isolation; **custom** billing per host or per span |
| **Google Monarch** | Global hierarchy; **Borg**-aware | **Aggregation tree**; **SLO** objects as first-class |
| **AWS CloudWatch** | Regional control plane; tight AWS integration | **Metric streams** to Kinesis; **cross-account** dashboards |
| **Grafana Mimir / Cortex** | Horizontally scaled **Prometheus-compatible** backend | **Shuffle sharding**, **limits**, object store backend |

### Why monitoring matters

- **SLOs and error budgets** tie product reliability to engineering velocity; metrics are the feedback loop.
- **Incident detection** depends on **latency histograms**, saturation (CPU, queue depth), and **golden signals** (latency, traffic, errors, saturation).
- **Capacity planning** uses historical trends; without retention and rollups, forecasting is guesswork.
- **Debugging** pairs metrics with traces and logs—but metrics are the cheapest signal at high volume.

{: .note }
> In interviews, **scope cardinality, retention, and multi-tenancy early**. Many designs fail on “we’ll just index everything” or “one global Prometheus.”

---

## Step 1: Requirements Clarification

### Questions to ask the interviewer

| Question | Why it matters |
|----------|----------------|
| Push vs pull for ingestion? | Agent architecture, security, and scrape intervals |
| Single-tenant vs SaaS? | Isolation, noisy-neighbor controls, billing |
| Max **cardinality** per metric / globally? | Index and memory limits; label allowlists |
| Query language: **PromQL**, SQL, or custom? | Parser, planner, and compatibility with Grafana |
| **Long-term retention** vs **raw resolution**? | Downsampling, object storage tiering |
| **Multi-region** active-active or primary + DR? | Replication, clock skew, query consistency |
| Compliance (encryption at rest, audit)? | KMS, per-tenant keys, log redaction |
| Integration with **PagerDuty**, Slack, email? | Notification routing and escalation |

### Functional requirements

| Area | Requirements |
|------|----------------|
| **Metric ingestion** | Accept samples: metric name, labels (dimensions), timestamp, numeric value; support counters, gauges, histograms/summaries |
| **Time-series storage** | Persist ordered (timestamp, value) pairs per series; efficient compression and retention policies |
| **Querying / aggregation** | Range queries, instant queries, `sum by`, `rate`, histogram quantiles, top-k |
| **Dashboards** | Pre-aggregated panels; variable refresh; templated labels |
| **Alerting** | Rule definitions, evaluation window, **for** duration, recovery detection |
| **Anomaly detection (optional)** | Baseline vs threshold; seasonality—often **later phase** or delegated to ML service |

### Non-functional requirements (targets for this walkthrough)

| NFR | Target | Rationale |
|-----|--------|-----------|
| **Ingestion throughput** | **100K+ samples/sec** per shard (cluster scales horizontally) | Large microservice estates |
| **Query latency** | **p99 &lt; 1 s** for dashboard-sized queries | Human-in-the-loop; tighter for critical APIs if needed |
| **Retention** | **1 year** raw or rolled-up; older data at coarser resolution | Cost vs precision |
| **Availability** | **99.99%** for ingestion + query API (excluding client misconfig) | SLO-driven; meta-monitoring separate |
| **Durability** | No silent data loss; WAL + replication | Trust in alerts and capacity graphs |

{: .warning }
> **99.99% for the entire stack** including third-party notification delivery is unrealistic—split SLAs: ingestion, query, **notification delivery** (best-effort with retries).

### Whiteboard checklist (expand each if asked)

| Bucket | Detail |
|--------|--------|
| **Ingestion** | Parse **text (Prometheus exposition)** or **OTLP**; validate label names; reject invalid timestamps |
| **Storage** | **Immutable** blocks after close; **compaction** rewrites for space; **delete** = retention policy (not random row delete) |
| **Query** | **Instant** + **range** API; **subqueries**; optional **SQL** façade for BI |
| **Dashboards** | **Recording rules** precompute expensive expressions; **cache** panel results |
| **Alerting** | **Silences**, **inhibition** (warning suppresses if critical fires), **routes** |
| **Anomaly** | Often **phase 2**: seasonal baselines need **long windows** and **clean** data |

---

## Step 2: Estimation

### Assumptions (adjust in the interview)

```
- 100,000 samples/sec ingested (cluster aggregate)
- Average series: 20 labels × ~24 B each + metric name ~40 B → ~520 B metadata (order of magnitude)
- 8 bytes timestamp + 8 bytes float per sample (before compression)
- Compression ratio ~10× on disk (Gorilla-style; workload-dependent)
- Replication factor 3 for durability
```

### Metrics volume

| Quantity | Calculation | Result |
|----------|----------------|--------|
| Samples per day | `100e3 × 86400` | **8.64×10⁹** samples/day |
| Uncompressed sample payload | 16 B (ts + value) | — |
| Logical sample bytes/day | `8.64e9 × 16` | **~138 GB/day** (values only) |
| With metadata churn (indexes) | +50–200% | Depends on cardinality churn |

### Network and fan-out (order of magnitude)

| Path | Rough math | Comment |
|------|----------------|--------|
| Ingest to gateway | `100K × ~100 B/sample` ≈ **10 MB/s** | Payload + framing; **gRPC** keeps overhead low |
| Replication | × replication factor inside cluster | **Erasure coding** optional for cold tiers |
| Query responses | Highly variable | **JSON** for ad-hoc; **Arrow** / protobuf for bulk |

### Yearly storage envelope (illustrative)

Using **~14 GB/day** logical compressed values (from above) × **365** ≈ **5.1 TB/year** per replica group before index overhead. Real clusters add **posting lists**, **symbol tables**, and **compaction** temporary space—**plan 2×** for operations headroom in interviews.

### Storage (time-series compression)

Compression in production often combines:

- **Delta-of-delta** timestamps (consecutive points ~regular intervals → tiny integers).
- **XOR / leading-zero** encoding for floats (Gorilla paper).
- **Block-level** packing + general-purpose **ZSTD** on cold blocks.

Rough **on-disk** after 10× compression on values: **~14 GB/day** per logical copy before replication. With **3× replication**: **~42 GB/day** physical (ignoring compaction overhead).

### Query patterns

| Pattern | Cost driver |
|---------|-------------|
| **Instant query** | Single evaluation at `t`; may scan recent blocks |
| **Range query** | `start…end` with step; fan-out to many series |
| **Aggregations** | `sum by (service)` shuffles series by label hash |
| **Cardinality queries** | `count by (pod)`—dangerous if `pod` explodes |

{: .tip }
> State clearly: **hot data in SSD**, **cold in object storage** (S3/GCS) with **downsampled** blocks for cheap long-range scans.

### Query load assumptions (for read-path sizing)

| Assumption | Example | Notes |
|------------|---------|--------|
| Concurrent dashboard users | 50–200 | Each panel = 1–5 queries |
| Refresh interval | 30 s | **Caches** dedupe identical queries |
| Expensive queries | `topk`, high-cardinality `count` | **Rate-limit** per tenant |

If **500 QPS** query mix with **50 ms** mean server work, you need enough **querier** replicas and **queue** depth—interviews often ask **only** ingestion math; offering **query** sizing shows maturity.

---

## Step 3: High-Level Design

```mermaid
flowchart LR
  subgraph clients["Data sources"]
    A1[Agents / exporters]
    A2[OpenTelemetry SDK]
    A3[Push gateway]
  end

  subgraph edge["Ingestion"]
    GW[Ingestion Gateway / Distributor]
  end

  subgraph write["Write path"]
    ING[Ingester]
    WAL[(WAL)]
    TS[(Time-series DB / Blocks)]
  end

  subgraph read["Read path"]
    QRY[Query frontend]
    ENG[Query engine / Evaluator]
    CACHE[Cache / Results]
  end

  subgraph async["Async & UX"]
    ALT[Alerting engine]
    RUL[Rule store]
    NOT[Notification router]
    DASH[Dashboard API]
  end

  A1 --> GW
  A2 --> GW
  A3 --> GW
  GW --> ING
  ING --> WAL
  ING --> TS
  QRY --> ENG
  ENG --> TS
  ENG --> CACHE
  ALT --> ENG
  RUL --> ALT
  ALT --> NOT
  DASH --> QRY
```

**Flow:**

1. **Agents** scrape or receive application metrics; they **batch** and send to **Ingestion Gateway** (auth, rate limits, routing).
2. **Ingester** appends to **WAL**, buffers in memory, **flushes** immutable blocks to the **time-series store** (columnar / LSM-like).
3. **Query frontend** parses **PromQL**-style queries, **plans** execution (pushdown aggregations, prune shards), merges results.
4. **Alerting engine** periodically evaluates rules (often same query engine), drives **state machine**, sends to **PagerDuty/Slack** via **notification router** (dedupe, silences).
5. **Dashboard** UI calls the query API (often **Grafana** in front).

{: .note }
> **Read path** and **write path** are often **separated** (different services) so spikes in queries do not starve ingestion.

### Sequence: scrape / push → durable block

```mermaid
sequenceDiagram
  participant Agent
  participant GW as Gateway
  participant ING as Ingester
  participant WAL as WAL
  participant OBJ as Object store

  Agent->>GW: Push batch (or GW pulls /metrics)
  GW->>ING: Route by tenant/hash
  ING->>WAL: Append records (fsync policy)
  ING->>ING: Memory head + chunk encoders
  Note over ING: Flush immutable block (e.g. every 2h)
  ING->>OBJ: Upload block + index
```

### Component responsibilities (talk track)

| Component | Responsibility |
|-----------|------------------|
| **Gateway** | AuthN/Z, **rate limits**, **admission control**, routing key |
| **Ingester** | WAL, **replication**, **cut** blocks, **ship** to storage |
| **Compactor** | Merge small blocks; **downsample**; **retention** deletes |
| **Store gateway** | Serve queries from **object storage** + cache |
| **Querier** | **Parse** PromQL, **fan-out**, **merge**, **dedupe** |
| **Ruler / Alertmanager** | **Rule eval**, **routing**, **silences** |

---

## Step 4: Deep Dive

### 4.1 Time-series data model and compression

**Series identity:** `(metric_name, sorted label map)` → unique **series ID** (internal uint64). Each series is an ordered sequence of `(timestamp_ms, float64)` (or int for counters).

| Concept | Description |
|---------|-------------|
| **Metric name** | e.g., `http_request_duration_seconds` |
| **Labels** | Dimensions: `method`, `path`, `pod`, `region` |
| **Samples** | `(t, v)` pairs; monotonic counters need `rate()` in queries |
| **Histogram** | Multiple series suffixes `_bucket`, `_sum`, `_count` or native histogram type |

**Delta-of-delta timestamps** (Gorilla-style intuition): if deltas between timestamps are near-constant, second-order delta encodes to **few bits**.

**XOR float compression:** store first value raw; for next, XOR with previous; leading zeros often dominate → encode run of zeros compactly.

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class GorillaStyleEncoder:
    """Illustrative tick encoder: delta-of-delta timestamps + XOR floats."""

    timestamps_ms: List[int]
    values: List[float]

    def encode_timestamp_deltas(self) -> List[int]:
        if not self.timestamps_ms:
            return []
        deltas: List[int] = []
        prev_ts = self.timestamps_ms[0]
        prev_delta = 0
        for ts in self.timestamps_ms[1:]:
            delta = ts - prev_ts
            dod = delta - prev_delta  # delta-of-delta
            deltas.append(dod)
            prev_ts, prev_delta = ts, delta
        return deltas

    @staticmethod
    def xor_float_bits(a: float, b: float) -> int:
        import struct

        ab = struct.pack(">d", a)
        bb = struct.pack(">d", b)
        ai = int.from_bytes(ab, "big")
        bi = int.from_bytes(bb, "big")
        return ai ^ bi


# Example: near-regular 15s scrape → small delta-of-delta
enc = GorillaStyleEncoder(
    timestamps_ms=[1_000_000 + i * 15_000 for i in range(5)],
    values=[0.12, 0.13, 0.11, 0.10, 0.09],
)
print(enc.encode_timestamp_deltas())  # mostly zeros after first tick
print(bin(GorillaStyleEncoder.xor_float_bits(enc.values[0], enc.values[1])))
```

**Label hashing → series ID:** production systems **intern** label strings in a **symbol table** and store **sorted** label pairs so `(job="a", env="b")` equals `(env="b", job="a")`.

```python
def canonical_series_key(metric: str, labels: dict[str, str]) -> str:
    """Human-readable key; production would hash to uint64."""
    parts = [metric] + [f'{k}="{labels[k]}"' for k in sorted(labels)]
    return "{" + ",".join(parts) + "}"


print(canonical_series_key("up", {"job": "api", "instance": "10.0.0.1:9100"}))
```

**Delta-of-delta intuition (bits):** if scrape interval is **exactly** 15 s, first-order delta is **15000 ms** every time; second-order delta is **0**. Encoders emit a **short control code** for “same as last interval” vs “patch.”

```python
def leading_zero_bytes(x: int) -> int:
    """How many leading zero bytes in 64-bit XOR (illustrative metric)."""
    if x == 0:
        return 8
    b = x.bit_length()
    return max(0, 8 - (b + 7) // 8)
```

{: .tip }
> Mention the **Gorilla** (Facebook) paper explicitly—it is interview currency. Note **ZSTD** on **frozen blocks** for cold tiers.

---

### 4.2 Write path and ingestion

**Push vs pull:**

| Mode | Pros | Cons |
|------|------|------|
| **Pull (scrape)** | Simple mental model; targets expose `/metrics`; Prometheus-native | Requires reachable endpoints; NAT/firewall pain |
| **Push** | Works behind NAT; batching to gateway | Need dedupe, back-pressure, auth |

Production systems often support **both**: OTel **push** to collector → same pipeline as scrape.

**Batching:** agents buffer **N** samples or **T** ms, **snappy-compress** payloads, use **HTTP/2** or gRPC.

**WAL + memory:** ingester writes **append-only WAL** before acknowledging; on crash, replay WAL. Memory holds **active head block**; when full or time window closes, **flush** immutable block.

**LSM-tree analogy:** recent writes go to **memtable**; flush produces **SSTable-like** files; **compaction** merges overlapping windows and drops expired data per retention.

```python
import json
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List


class IngesterBuffer:
    """Simplified in-memory shard buffer → flush to JSONL 'blocks' (illustrative)."""

    def __init__(self, wal_path: Path, block_dir: Path) -> None:
        self.wal_path = wal_path
        self.block_dir = block_dir
        self.buf: DefaultDict[str, List[Dict]] = defaultdict(list)

    def append_wal(self, record: Dict) -> None:
        with self.wal_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def ingest(self, series_key: str, ts: int, value: float) -> None:
        rec = {"series": series_key, "ts": ts, "v": value}
        self.append_wal(rec)
        self.buf[series_key].append(rec)

    def flush_block(self, block_id: str) -> None:
        self.block_dir.mkdir(parents=True, exist_ok=True)
        out = self.block_dir / f"block-{block_id}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for series, rows in self.buf.items():
                for r in sorted(rows, key=lambda x: x["ts"]):
                    f.write(json.dumps(r) + "\n")
        self.buf.clear()
```

**Back-pressure:** gateway returns **429** or **drops** lowest-priority tenants when **ingestion lag** &gt; SLO; **adaptive** batching reduces scrape frequency for **noisy** jobs.

```python
from dataclasses import dataclass
from typing import List


@dataclass
class Batcher:
    max_points: int
    max_wait_ms: int
    buf: List[dict]

    def should_flush(self, elapsed_ms: int) -> bool:
        return len(self.buf) >= self.max_points or elapsed_ms >= self.max_wait_ms
```

{: .warning }
> **High-cardinality** labels (e.g., `user_id`) will **OOM** the ingester and index—enforce **limits**, **drop rules**, or **aggregations** at the edge.

---

### 4.3 Storage engine

**Columnar / time-oriented layout:** store **columns** (timestamps, values) per series or per **block** of time; good for **scanning one metric across many series** (aggregation).

**Block-based:** immutable **2-hour** (example) blocks with **index** (label → posting lists) + **chunks** for values.

**Retention:** drop blocks older than policy; **object storage** for archive.

**Downsampling / rollups:** for data older than **30 days**, keep **5-minute** aggregates; older → **1-hour**—trades precision for cost.

| Tier | Age | Resolution | Use case |
|------|-----|------------|----------|
| **Hot** | 0–7 d | Native scrape (e.g. 15 s) | Incident drill-down |
| **Warm** | 7–90 d | 1 min (recording rules) | Weekly trends |
| **Cold** | 90 d–1 y | 5–60 min | Capacity, finance |
| **Archive** | &gt; 1 y | Daily / weekly | Compliance (optional) |

```mermaid
flowchart TB
  subgraph hot["Hot SSD"]
    H[Open head block + recent blocks]
  end
  subgraph warm["Warm / object store"]
    W[Compacted blocks]
  end
  subgraph cold["Cold / glacier"]
    C[Downsampled blocks]
  end
  hot --> warm --> cold
```

```python
from dataclasses import dataclass
from typing import List


@dataclass
class Rollup:
    window_sec: int

    def downsample(
        self, timestamps: List[int], values: List[float]
    ) -> tuple[List[int], List[float]]:
        bucket: dict[int, List[float]] = {}
        w_ms = self.window_sec * 1000
        for ts, v in zip(timestamps, values):
            b = (ts // w_ms) * w_ms
            bucket.setdefault(b, []).append(v)
        out_ts = sorted(bucket)
        out_v = [sum(bucket[t]) / len(bucket[t]) for t in out_ts]
        return out_ts, out_v
```

**Posting lists:** inverted index from **label=value** → list of **series IDs** containing it. Query `job="api"` **AND** `region="us"` = **intersect** posting lists (skip if one side is tiny).

---

### 4.4 Query engine

**PromQL-style** (conceptual): selectors `metric{label="v"}`, **range vectors** `rate(http_requests_total[5m])`, aggregations `sum by (job) (...)`.

**Aggregation operators:** vector matching **one-to-one**, **many-to-one** with `group_left`.

**Fan-out:** **querier** asks **all ingesters / store-gateways** holding data for time range; **merge** deduplicated samples (replication-aware).

**Query planning:** prune **label matchers** early; push down **`sum`** where possible; use **indexes** to reduce posting lists.

```python
from typing import Dict, List


def plan_simple_instant_query(
    matchers: Dict[str, str], shard_fn=lambda series: hash(series) % 4
) -> Dict[int, Dict[str, str]]:
    """Assign scan to shards by hashed series id (illustrative)."""
    planned: Dict[int, Dict[str, str]] = {}
    for shard in range(4):
        planned[shard] = dict(matchers)
    return planned


def merge_fanout_partial_vectors(partials: List[List[float]]) -> List[float]:
    """Dumb sum merge if time-aligned (real systems align on timestamps)."""
    if not partials:
        return []
    length = len(partials[0])
    return [sum(p[i] for p in partials) for i in range(length)]
```

**`rate()` for counters** (conceptual): take the last two samples in the window, divide delta by time—handle **counter reset** (delta &lt; 0) and **missing** points.

```python
def rate_per_second(timestamps: List[float], values: List[float]) -> float:
    """Instant rate over window using first/last (Prometheus does more)."""
    if len(timestamps) < 2:
        return 0.0
    dt = timestamps[-1] - timestamps[0]
    dv = values[-1] - values[0]
    if dv < 0:
        dv = values[-1]  # counter reset heuristic
    return dv / dt if dt > 0 else 0.0
```

**Shard pruning:** if matchers include `region="us-east"`, only **touch** ingesters owning that **tenant/region** slice—critical for latency.

{: .note }
> **Parallelism** is keyed by **time range** and **shard**; protect with **query concurrency** limits per tenant.

---

### 4.5 Alerting pipeline

**Alert rule:** `expr` + **`for`** duration + labels + **annotations**.

**Evaluation:** scheduler runs **every `eval_interval`**; executes query; compares to threshold.

**State machine:**

```mermaid
stateDiagram-v2
    [*] --> Inactive
    Inactive --> Pending: condition true
    Pending --> Firing: condition true for "for" duration
    Pending --> Inactive: recovered
    Firing --> Resolved: condition false
    Resolved --> Inactive
```

**Notification routing:** route by **severity**, **team** label, **on-call** schedule (PagerDuty **service** / **escalation policy**).

**Deduplication:** same alert key `(rule, labels)` within **group_wait** → one notification; **group_interval** for updates.

```python
from enum import Enum, auto
from typing import Dict, Optional


class AlertState(Enum):
    INACTIVE = auto()
    PENDING = auto()
    FIRING = auto()


class Alert:
    def __init__(self, name: str, labels: Dict[str, str], for_sec: int) -> None:
        self.name = name
        self.labels = labels
        self.for_sec = for_sec
        self.state = AlertState.INACTIVE
        self.true_since: Optional[float] = None

    def tick(self, now: float, condition_met: bool) -> None:
        if not condition_met:
            self.state = AlertState.INACTIVE
            self.true_since = None
            return
        if self.state == AlertState.INACTIVE:
            self.state = AlertState.PENDING
            self.true_since = now
            return
        if self.state == AlertState.PENDING and self.true_since is not None:
            if now - self.true_since >= self.for_sec:
                self.state = AlertState.FIRING
```

**Notification deduplication (sketch):**

```python
from time import time
from typing import Dict, Tuple

AlertKey = Tuple[str, Tuple[Tuple[str, str], ...]]


class NotificationDeduper:
    """Drop duplicate notifications for the same alert key within window_sec."""

    def __init__(self, window_sec: int) -> None:
        self.window_sec = window_sec
        self.seen: Dict[AlertKey, float] = {}

    def should_send(self, key: AlertKey) -> bool:
        now = time()
        last = self.seen.get(key)
        if last is not None and now - last < self.window_sec:
            return False
        self.seen[key] = now
        return True
```

**On-call integration:** **Alertmanager** sends JSON webhook to **PagerDuty** Events API v2; include **runbook** URL and **dashboard** deep link in **annotations**.

---

### 4.6 Scalability — sharding and federation

**Metric-based sharding:** **hash(series_id) → shard** or **range** on `tenant_id` + metric prefix. Avoid hotspots by **shuffle sharding** tenants across nodes.

**Hierarchical federation:** **regional** Prometheus **federates** up to **global** read-only queries for **SLI dashboards**; avoids shipping all raw data globally.

**Cross-datacenter:** **dual write** or **async replication** with **eventual consistency**; queries may use **local** DC for freshness or **global** view with **staleness** bounds.

```python
def shard_for_series(series_id: int, num_shards: int) -> int:
    return series_id % num_shards


def federate_selectors(global_query: str) -> list[str]:
    """Illustrative: split one global metric into per-region selectors."""
    regions = ["us-east", "eu-west", "ap-south"]
    return [f'{global_query}{{region="{r}"}}' for r in regions]
```

```mermaid
flowchart TB
  subgraph r1["Region A"]
    P1[Prometheus / Agent pool]
  end
  subgraph r2["Region B"]
    P2[Prometheus / Agent pool]
  end
  subgraph global["Global read path"]
    FED[Federation / Thanos querier]
    GRA[Grafana]
  end
  P1 --> FED
  P2 --> FED
  FED --> GRA
```

**Cross-datacenter aggregation:** **write locally**, **query globally** with **staleness** annotations; avoid **synchronous** cross-region writes on the hot path.

| Pattern | When to use |
|---------|-------------|
| **Dedicated global metrics** | SLIs that must be **comparable** across regions |
| **Region-local dashboards** | **Latency** inside one DC—global merge is misleading |
| **Replication** | **DR** and **query availability**—not a substitute for **correct** aggregation semantics |

{: .tip }
> Name-check **Thanos / Cortex / Mimir** for **long-term Prometheus**; **Monarch** for **hierarchical global** monitoring at Google scale.

---

## Step 5: Scaling & Production

### Failure handling

| Failure | Mitigation |
|---------|------------|
| **Ingester crash** | WAL replay; another replica serves **duplicate** data until compacted |
| **Zone outage** | Replicate shards across AZs; **read** from healthy replicas |
| **Query OOM** | **Limits** on max series, max range, max points; **query queue** with timeout |
| **Notification provider down** | Retry with backoff; **dead-letter** queue; multi-channel fallback |

### Meta-monitoring

Monitor the monitors: **ingestion lag**, **WAL age**, **compaction backlog**, **query latency**, **alert evaluation lag**, **notification success rate**. Use **separate** infrastructure or **vendor** for **paging the paging system**.

| Signal | What breaks if it degrades |
|--------|----------------------------|
| **Ingestion lag** | Dashboards look “fine” while reality is on fire |
| **Compaction backlog** | Disk fills; queries slow; **read amplification** spikes |
| **Rule eval lag** | **Late** alerts; **thundering herd** after catch-up |
| **Memberlist / ring health** | **Wrong** shard routing → **partial** data loss visibility |

### Runbooks and human factors

- **SLO-based alerts** beat static thresholds; pair metrics with **error budgets**.
- **Cascading failures:** protect the **control plane** with **bulkheads** (separate etcd/consul for monitoring).
- **Game days** for “metrics backend down”—operators should know **fallback** (synthetic checks only, etc.).

### Trade-offs

| Choice | Upside | Downside |
|--------|--------|----------|
| **Strong vs eventual** replication | Linearizable reads | Latency + coordination cost |
| **Long raw retention** | Fine-grained forensics | Storage cost |
| **High cardinality** | Powerful drill-down | Memory and index explosion |
| **Pull-only** | Idempotent scrape | Operational complexity in k8s |

{: .warning }
> **Cardinality** and **query cost** are the #1 production failure modes—interviewers reward **guardrails** (limits, recording rules, aggressive downsampling).

---

## Interview Tips — Google follow-up questions

| Topic | What they probe |
|-------|-----------------|
| **Cardinality explosion** | How do you cap labels? **Recording rules**? **Adaptive sampling**? |
| **Exactly-once ingestion** | At-least-once typical; **idempotent** keys for push; **dedupe** windows |
| **Clock skew** | **NTP**; reject too-future timestamps; **out-of-order** handling in TSDB |
| **Histogram quantiles** | **t-digest**, **HLL** vs exact; error bounds |
| **Multi-tenancy fairness** | **Quotas**, **shuffle sharding**, **noisy neighbor** isolation |
| **Cost** | Cold storage, **query pushdown**, **aggregations at write** (recording rules) |
| **vs Logs/traces** | Metrics for **aggregates**; **Exemplars** link to traces |

### Follow-up drill (short answers)

| Question | Strong answer shape |
|----------|---------------------|
| “How is this different from **InfluxDB**?” | **Column/time** layout vs **full** database features; **PromQL** ecosystem vs **InfluxQL** |
| “**Global** percentiles?” | **Mergeable** sketches (t-digest) per shard + **merge**; exact global p99 needs **heavy** data movement |
| “**Billing** in SaaS?” | **Ingested samples**, **query cost units**, **cardinality** peaks—meter at **gateway** |
| “**ML anomaly**?” | **Offline** train on rollups; **online** scorer on **stream**—keep **separate** from **hot path** |

{: .note }
> Close with **what you’d build first**: ingestion + **WAL** + **query on recent blocks**, then **long-term store**, then **alerting**—mirrors **MVP → scale** path.

### MVP vs scale (explicit)

| Phase | Build | Defer |
|-------|-------|-------|
| **MVP** | Scrape/push → **WAL** → **query** last 14 d | Multi-year archive |
| **Scale** | **Horizontal** ingesters, **object store** blocks | Fancy ML |
| **Mature** | **Global** federation, **tiered** storage | — |

---

## Summary

| Layer | Core idea |
|-------|-----------|
| **Model** | Metric + labels = series; compress **time** and **value** streams |
| **Write** | WAL, batched flush, **block** files, retention + **rollups** |
| **Read** | Label index → posting lists; **fan-out** + merge; **PromQL** planning |
| **Alerts** | **Pending → firing** with **`for`**; dedupe + routes |
| **Scale** | **Shard** by series/tenant; **federation** for hierarchy |

This walkthrough is a **structured narrative** for system design interviews—not a production architecture for any one product. Adapt numbers, **SLAs**, and **compliance** to the prompt and your experience.
