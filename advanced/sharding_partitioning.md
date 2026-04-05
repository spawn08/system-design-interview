---
layout: default
title: Sharding & Partitioning
parent: Advanced Topics
nav_order: 12
---

# Database Sharding & Partitioning Strategies
{: .no_toc }

<details open markdown="block">
  <summary>Table of contents</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

---

## Why Sharding Matters for Staff Engineers

Sharding is the most common technique for horizontal scaling of databases. At L6, interviewers expect you to **choose the right partition key, anticipate hot spots, and plan for resharding**—not just say "we'll shard the data."

{: .note }
> Every system design question eventually leads to "how do you scale the data layer?" Your sharding answer determines whether your design works at 10x the initial traffic.

---

## Partitioning vs. Sharding

| Term | Meaning |
|------|---------|
| **Partitioning** | Splitting data within a single database engine (e.g., PostgreSQL table partitioning) |
| **Sharding** | Distributing data across multiple independent database instances |

In interviews, the terms are often used interchangeably. Clarify with the interviewer if needed.

---

## Partitioning Strategies

### Key-Based (Hash) Partitioning

Compute `shard = hash(key) % num_shards`. Data distributes evenly if the hash function is uniform.

| Pros | Cons |
|------|------|
| Even distribution | Adding/removing shards reshuffles most keys (modulo change) |
| Simple routing | Range queries across shards are expensive |
| No hotspots (if hash is good) | Cannot exploit data locality |

### Range Partitioning

Assign contiguous key ranges to shards (e.g., A-M on shard 1, N-Z on shard 2).

| Pros | Cons |
|------|------|
| Efficient range queries | Uneven distribution (some ranges are hotter) |
| Natural ordering | Requires periodic rebalancing |
| Easy to understand | Sequential keys create write hotspots |

### Consistent Hashing

Map both keys and nodes to a ring. Each key is assigned to the first node clockwise from its hash position. Adding a node only moves keys between neighbors.

| Pros | Cons |
|------|------|
| Minimal key movement on node add/remove | Uneven load with few nodes (solved with virtual nodes) |
| No central routing table needed | More complex than simple modulo |
| Used by Cassandra, DynamoDB, Riak | Ring metadata must be distributed |

### Directory-Based Partitioning

A lookup service maps keys to shards. Maximum flexibility but adds a dependency.

| Pros | Cons |
|------|------|
| Complete flexibility in placement | Lookup service is a single point of failure |
| Can move individual keys | Additional latency for lookup |
| Supports heterogeneous shards | Must be highly available and fast |

---

## Choosing a Partition Key

The partition key determines data distribution, query patterns, and hotspot risk.

| Consideration | Guidance |
|---------------|----------|
| **Cardinality** | High cardinality (e.g., user_id) distributes well; low cardinality (e.g., country) creates hot shards |
| **Query patterns** | If most queries filter by `tenant_id`, shard by `tenant_id` to avoid scatter-gather |
| **Write distribution** | If writes are timestamped and sequential, hashing the timestamp avoids a single hot shard |
| **Join locality** | Co-locate data that is frequently joined (e.g., user + orders on the same shard) |

### Common Partition Key Choices

| System | Partition Key | Rationale |
|--------|---------------|-----------|
| **Chat system** | `conversation_id` | All messages in a conversation are on one shard; efficient reads |
| **E-commerce orders** | `user_id` | Most queries are "my orders"; avoids cross-shard queries |
| **Time-series metrics** | `hash(metric_name)` | Avoids sequential write hotspots; range queries use time index within shard |
| **Multi-tenant SaaS** | `tenant_id` | Data isolation; compliance; per-tenant backup/restore |

{: .warning }
> **Anti-pattern:** Sharding by auto-increment ID puts all recent writes on the last shard. Use a hash of the ID or a compound key.

---

## Hot Spot Mitigation

| Technique | Description |
|-----------|-------------|
| **Salting** | Append a random suffix to the key before hashing (e.g., `user_123_salt_7`); distributes writes but complicates reads (must query all salt variants) |
| **Key splitting** | Split a hot partition into sub-partitions; route based on secondary key |
| **Caching** | Cache hot reads in front of the shard; reduces read load on the hot shard |
| **Rate limiting per shard** | Protect hot shards from overwhelming writes; backpressure to clients |
| **Virtual nodes** | In consistent hashing, more virtual nodes per physical node smooths distribution |

---

## Cross-Shard Operations

| Operation | Challenge | Solutions |
|-----------|-----------|----------|
| **Cross-shard queries** | Must fan out to all shards; aggregate results | Application-level scatter-gather; or denormalize data to avoid cross-shard reads |
| **Cross-shard joins** | No native SQL join across shards | Denormalize; or use a materialized view updated via CDC |
| **Cross-shard transactions** | No single-DB ACID guarantee | 2PC (if same DB engine); Saga pattern; or design to avoid cross-shard transactions |
| **Global secondary indexes** | Index entries span multiple shards | Local indexes (per-shard, scatter on query) or global index (updated async, eventually consistent) |

{: .tip }
> **Staff-level answer:** *"I'd design the schema so that 95% of queries hit a single shard. For the rare cross-shard query (e.g., admin dashboard), I'd build a read-optimized materialized view in a separate analytical store, updated via CDC from the sharded OLTP database."*

---

## Resharding Strategies

As data grows, the initial shard count becomes insufficient. Resharding is one of the hardest operational challenges.

| Strategy | Description | Downtime |
|----------|-------------|----------|
| **Logical sharding** | Use many more logical shards than physical nodes (e.g., 1024 logical shards on 16 nodes); move logical shards between nodes without re-hashing | Zero (with careful migration) |
| **Dual-write migration** | Write to both old and new shard layout; backfill historical data; cut over reads; stop old writes | Zero (but complex) |
| **Shadow traffic** | Route a copy of traffic to the new shard layout; compare results; switch when consistent | Zero |
| **Stop-the-world** | Take the system offline; redistribute data; bring back up | Minutes to hours |

{: .warning }
> **Staff-level insight:** Always over-provision logical shards at design time. Starting with 1024 logical shards on 8 physical nodes is far easier to scale (just move logical shards) than starting with 8 logical shards and needing to re-hash everything.

---

## Replication Within Shards

Each shard should be replicated for durability and read scaling:

| Topology | Description |
|----------|-------------|
| **Leader-follower** | One leader handles writes; followers serve reads; failover promotes a follower |
| **Leader-leader** | Both nodes accept writes; conflict resolution needed (LWW, vector clocks) |
| **Leaderless (quorum)** | Any node accepts writes; quorum ensures overlap (Dynamo-style) |

---

## Sharding in Real Systems

| System | Sharding Approach |
|--------|-------------------|
| **MySQL (Vitess)** | Application-level sharding; Vitess adds routing, resharding, and schema management |
| **PostgreSQL (Citus)** | Extension that distributes tables across worker nodes; co-located joins |
| **MongoDB** | Built-in sharding with configurable shard key; automatic balancing |
| **Cassandra** | Consistent hashing with virtual nodes; automatic rebalancing |
| **Google Spanner** | Range-based splits with automatic split/merge; Paxos per split |
| **DynamoDB** | Hash partitioning; automatic splitting of hot partitions |

---

## Interview Checklist

| Topic | What to Cover |
|-------|---------------|
| **Why shard?** | "A single node cannot handle X writes/sec or Y TB of data" |
| **Partition key** | Choose based on query patterns, cardinality, and write distribution |
| **Hot spots** | Identify and mitigate with salting, splitting, or caching |
| **Cross-shard** | Design to minimize; use denormalization or materialized views for the rest |
| **Resharding** | Logical shards > physical nodes; dual-write migration for zero downtime |
| **Replication** | Each shard has leader-follower replication for durability |
| **Secondary indexes** | Local (per-shard) vs. global (async) with clear trade-offs |

---

_Last updated: 2026-04-05_
