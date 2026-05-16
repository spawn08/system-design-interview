# Notation & Abbreviation Reference

Every abbreviation used across this guide is listed here. On the rendered site, hovering over any underlined term in a document shows a tooltip with its full form and meaning — this page is the master reference you can bookmark or search.

---

## How to Use

- **Hover any dotted-underlined term** on any page to see its full form in a tooltip.
- **Search this page** (Ctrl/Cmd + F) to look up a specific abbreviation.
- Abbreviations are grouped by domain. Each entry shows: **Abbreviation → Full Form** and a short explanation of how it is used in system design context.

---

## Distributed Systems & Consistency

| Abbreviation | Full Form | Context |
|---|---|---|
| **ACID** | Atomicity, Consistency, Isolation, Durability | Properties that guarantee reliable database transactions |
| **BASE** | Basically Available, Soft state, Eventually consistent | Trade-off model for highly available distributed systems (contrast with ACID) |
| **CAP** | Consistency, Availability, Partition tolerance | Theorem: a distributed system can guarantee at most two of these three properties simultaneously |
| **CP** | Consistency and Partition tolerance | A CAP trade-off: system chooses correctness over availability during partitions |
| **AP** | Availability and Partition tolerance | A CAP trade-off: system stays available but may return stale data during partitions |
| **CRDT** | Conflict-free Replicated Data Type | Data structure that can be merged across replicas without conflicts (e.g., G-Counter, LWW-Register) |
| **LWW** | Last-Write-Wins | Conflict resolution strategy that keeps the most recent write (by timestamp) |
| **HLC** | Hybrid Logical Clock | Clock combining physical time and logical counters for distributed event ordering |
| **2PC** | Two-Phase Commit | Protocol ensuring all-or-nothing writes across multiple nodes (prepare → commit/abort) |
| **CQRS** | Command Query Responsibility Segregation | Pattern separating write (command) and read (query) models |
| **DAG** | Directed Acyclic Graph | Graph with directed edges and no cycles; used to model task dependencies (e.g. workflow DAGs) |
| **ZAB** | Zookeeper Atomic Broadcast | Consensus and leader-election protocol used by Apache ZooKeeper |
| **OT** | Operational Transformation | Algorithm for real-time collaborative editing that merges concurrent edits without conflicts |

---

## Storage & Databases

| Abbreviation | Full Form | Context |
|---|---|---|
| **SQL** | Structured Query Language | Language for relational databases; also used as a shorthand for "relational database" |
| **OLTP** | Online Transaction Processing | Workload optimised for short reads/writes (e.g. web apps, payment systems) |
| **OLAP** | Online Analytical Processing | Workload optimised for large analytical scans (e.g. data warehouses, BI) |
| **LSM** | Log-Structured Merge-tree | Write-optimised storage engine (used by Cassandra, RocksDB) — appends to memtable then flushes to SSTables |
| **SSTable** | Sorted String Table | Immutable on-disk file produced by an LSM flush; keys are sorted for efficient range scans |
| **B+tree** | Balanced Plus Tree | Disk-optimised search tree used by most RDBMS (PostgreSQL, MySQL) for index pages |
| **WAL** | Write-Ahead Log | Durability mechanism: changes written to log before being applied to main storage |
| **CDC** | Change Data Capture | Technique for streaming row-level changes out of a database (e.g. Debezium over Postgres WAL) |
| **STCS** | Size-Tiered Compaction Strategy | Cassandra compaction: merges similarly-sized SSTables; good for write-heavy workloads |
| **LCS** | Leveled Compaction Strategy | Cassandra compaction: organises SSTables into fixed-size levels; better for read-heavy workloads |
| **TWCS** | Time-Window Compaction Strategy | Cassandra compaction for time-series data; compacts within rolling time windows |
| **KV** | Key-Value | Simplest data model: store and retrieve opaque values by key (e.g. Redis, DynamoDB) |
| **FK** | Foreign Key | Relational constraint linking a column to a primary key in another table |
| **RDD** | Resilient Distributed Dataset | Spark's core abstraction: an immutable, fault-tolerant distributed collection |

---

## Networking & Protocols

| Abbreviation | Full Form | Context |
|---|---|---|
| **HTTP** | Hypertext Transfer Protocol | Application-layer protocol for web communication |
| **HTTPS** | Hypertext Transfer Protocol Secure | HTTP over TLS — encrypted web communication |
| **TCP** | Transmission Control Protocol | Reliable, ordered, connection-oriented transport protocol |
| **UDP** | User Datagram Protocol | Connectionless, best-effort transport; lower latency than TCP (used in video, DNS) |
| **TLS** | Transport Layer Security | Cryptographic protocol for secure communication; successor to SSL |
| **SSL** | Secure Sockets Layer | Predecessor to TLS; deprecated but term still used colloquially |
| **mTLS** | Mutual TLS | Both client and server authenticate each other with certificates; used in service mesh (e.g. Istio) |
| **DNS** | Domain Name System | Resolves human-readable hostnames to IP addresses |
| **IP** | Internet Protocol | Network-layer protocol for addressing and routing packets |
| **gRPC** | Google Remote Procedure Call | High-performance RPC framework using HTTP/2 and Protocol Buffers |
| **RPC** | Remote Procedure Call | Pattern for calling functions on a remote server as if local |
| **REST** | Representational State Transfer | Architectural style for HTTP APIs using resource-based URLs and standard methods |
| **WebRTC** | Web Real-Time Communication | Browser API for peer-to-peer audio, video, and data channels |
| **WSS** | WebSocket Secure | WebSocket protocol over TLS |
| **SSE** | Server-Sent Events | HTTP-based one-way streaming from server to browser |
| **RTT** | Round-Trip Time | Time for a message to travel from sender to receiver and back |
| **FQDN** | Fully Qualified Domain Name | Complete domain name including all labels (e.g. `api.example.com`) |
| **AMQP** | Advanced Message Queuing Protocol | Wire-level messaging protocol; used by RabbitMQ |
| **NTP** | Network Time Protocol | Protocol for synchronising clocks across networked computers |

---

## Performance & Reliability

| Abbreviation | Full Form | Context |
|---|---|---|
| **SLA** | Service Level Agreement | Contractual commitment between provider and customer (e.g. 99.9% uptime) |
| **SLO** | Service Level Objective | Internal target derived from an SLA (e.g. P99 latency < 200 ms) |
| **SLI** | Service Level Indicator | Metric used to measure whether an SLO is being met (e.g. request success rate) |
| **P50** | 50th Percentile | Median — 50% of requests are faster than this |
| **P95** | 95th Percentile | 95% of requests are faster than this; filters out top-5% outliers |
| **P99** | 99th Percentile | 99% of requests are faster than this; the standard "tail latency" benchmark |
| **QPS** | Queries Per Second | Request throughput — often used for read-heavy systems |
| **RPS** | Requests Per Second | Request throughput — general alternative to QPS |
| **RPM** | Revenue Per Mille / Requests Per Minute | In ads: revenue per 1,000 impressions. In infra: requests per minute |
| **IOPS** | Input/Output Operations Per Second | Disk or storage throughput metric |
| **MTBF** | Mean Time Between Failures | Average time between successive failures of a system |
| **MTTR** | Mean Time To Recovery | Average time to restore a system after a failure |
| **RPO** | Recovery Point Objective | Maximum acceptable data loss measured in time (e.g. 5 minutes of data) |
| **RTO** | Recovery Time Objective | Maximum acceptable downtime before service must be restored |
| **SPOF** | Single Point of Failure | A component whose failure brings down the entire system |
| **NFR** | Non-Functional Requirement | System quality attribute: latency, availability, scalability, security, etc. |
| **QoS** | Quality of Service | Mechanisms for prioritising traffic or guaranteeing performance levels |

---

## Caching & Storage Hardware

| Abbreviation | Full Form | Context |
|---|---|---|
| **TTL** | Time To Live | Duration after which a cached entry or DNS record expires |
| **SSD** | Solid State Drive | Flash-based storage; much faster random I/O than HDD |
| **HDD** | Hard Disk Drive | Magnetic spinning disk; high capacity, low random IOPS |
| **RAM** | Random Access Memory | Fast volatile memory; used for in-memory caches (Redis, Memcached) |
| **GB** | Gigabyte | 10⁹ bytes (or 2³⁰ bytes in binary) |
| **MB** | Megabyte | 10⁶ bytes |
| **KB** | Kilobyte | 10³ bytes |
| **TB** | Terabyte | 10¹² bytes |
| **PB** | Petabyte | 10¹⁵ bytes |
| **I/O** | Input/Output | Read/write operations to storage or peripherals |
| **OOM** | Out of Memory | Condition where a process exceeds available RAM; often triggers eviction or crash |
| **GC** | Garbage Collection | Automatic memory reclamation; GC pauses can cause latency spikes |

---

## Infrastructure & Cloud

| Abbreviation | Full Form | Context |
|---|---|---|
| **API** | Application Programming Interface | Contract defining how two software components communicate |
| **SDK** | Software Development Kit | Library + tools for interacting with a platform or service |
| **CI/CD** | Continuous Integration / Continuous Deployment | Automated pipeline: build → test → deploy on every commit |
| **CDN** | Content Delivery Network | Geographically distributed cache for static assets and media |
| **DNS** | Domain Name System | See Networking section |
| **VPC** | Virtual Private Cloud | Isolated private network inside a public cloud provider |
| **AZ** | Availability Zone | Isolated data-centre within a cloud region; failure domains |
| **CRR** | Cross-Region Replication | Automatically replicating data to a second geographic region |
| **S3** | Simple Storage Service | AWS object storage service; often used generically for object/blob storage |
| **GCS** | Google Cloud Storage | Google Cloud's object storage service |
| **LB** | Load Balancer | Distributes incoming traffic across multiple backend instances |
| **HPA** | Horizontal Pod Autoscaler | Kubernetes controller that scales pod replicas based on CPU/memory metrics |
| **VPA** | Vertical Pod Autoscaler | Kubernetes controller that adjusts pod CPU/memory requests automatically |
| **SRE** | Site Reliability Engineering | Discipline applying software engineering to operations (Google-origin) |
| **DevOps** | Development and Operations | Culture and practice of shared ownership across dev and ops teams |
| **MLOps** | Machine Learning Operations | DevOps practices applied to ML: versioning, monitoring, retraining pipelines |
| **SaaS** | Software as a Service | Cloud delivery model: vendor manages everything; customer uses via browser |
| **DLQ** | Dead Letter Queue | Queue that receives messages that could not be processed after retries |
| **MPSC** | Multi-Producer Single-Consumer | Queue pattern common in high-throughput logging pipelines |
| **P2P** | Peer-to-Peer | Network architecture where nodes communicate directly (no central server) |
| **UUID** | Universally Unique Identifier | 128-bit identifier generated to be unique without central coordination |
| **DDoS** | Distributed Denial of Service | Attack using many sources to overwhelm a system with traffic |

---

## Security & Privacy

| Abbreviation | Full Form | Context |
|---|---|---|
| **TLS** | Transport Layer Security | See Networking section |
| **JWT** | JSON Web Token | Compact, signed token for stateless authentication and claims |
| **RBAC** | Role-Based Access Control | Permissions granted to roles; users inherit role permissions |
| **ACL** | Access Control List | Per-resource permission list mapping principals to allowed actions |
| **GDPR** | General Data Protection Regulation | EU regulation governing personal data collection, storage, and consent |
| **PII** | Personally Identifiable Information | Data that can identify an individual (name, email, IP address, etc.) |
| **XSS** | Cross-Site Scripting | Web vulnerability injecting malicious scripts into pages viewed by other users |
| **PCI-DSS** | Payment Card Industry Data Security Standard | Compliance framework for systems handling card payment data |
| **PCI** | Payment Card Industry | Industry body that sets PCI-DSS standards |
| **PAN** | Primary Account Number | The 16-digit card number; must be tokenised or encrypted, never stored in plain text |
| **SEO** | Search Engine Optimization | Practices to improve a page's ranking in organic search results |

---

## Payments & Finance

| Abbreviation | Full Form | Context |
|---|---|---|
| **PSP** | Payment Service Provider | Third-party that processes card/bank payments (e.g. Stripe, Adyen, PayPal) |
| **ACH** | Automated Clearing House | US electronic bank-to-bank transfer network (direct debit / payroll) |
| **SEPA** | Single Euro Payments Area | EU equivalent of ACH for euro-denominated bank transfers |

---

## ML & AI Fundamentals

| Abbreviation | Full Form | Context |
|---|---|---|
| **LLM** | Large Language Model | Transformer-based model trained on text at scale (GPT-4, Gemini, Claude) |
| **RAG** | Retrieval-Augmented Generation | Architecture combining a retriever with an LLM to ground answers in documents |
| **MLP** | Multi-Layer Perceptron | Fully-connected feedforward neural network |
| **CNN** | Convolutional Neural Network | Network using convolutional filters; standard for image and audio tasks |
| **RNN** | Recurrent Neural Network | Sequential model with hidden state; largely superseded by Transformers |
| **LSTM** | Long Short-Term Memory | RNN variant with gating that mitigates the vanishing gradient problem |
| **BERT** | Bidirectional Encoder Representations from Transformers | Pre-trained encoder model; fine-tuned for NLU tasks |
| **ViT** | Vision Transformer | Transformer applied to image patches; state-of-the-art for image understanding |
| **GNN** | Graph Neural Network | Neural network that operates on graph-structured data |
| **MoE** | Mixture of Experts | Architecture with many "expert" sub-networks; only a few are active per token (e.g. Mixtral) |
| **CLIP** | Contrastive Language-Image Pre-Training | OpenAI model that jointly embeds images and text; foundational for multimodal retrieval |
| **ANN** | Approximate Nearest Neighbor | Fast similarity search that trades exact results for speed (HNSW, FAISS, ScaNN) |
| **KNN** | K-Nearest Neighbors | Exact nearest-neighbor search; practical only for small datasets |
| **HNSW** | Hierarchical Navigable Small World | Graph-based ANN index algorithm; best recall/speed trade-off for dense vectors |
| **FAISS** | Facebook AI Similarity Search | Meta's library of ANN indexes; GPU-accelerated options available |
| **BM25** | Best Matching 25 | Probabilistic ranking function for keyword (sparse) retrieval |
| **TF-IDF** | Term Frequency-Inverse Document Frequency | Classical keyword weighting: high weight for rare terms in a document |
| **IDF** | Inverse Document Frequency | The log-scaled penalty for common terms; the "IDF" component of TF-IDF |
| **NLP** | Natural Language Processing | Broad field of ML applied to text understanding and generation |
| **NLU** | Natural Language Understanding | Subset of NLP focused on intent, entities, and semantic meaning |
| **NER** | Named Entity Recognition | Task of labelling entities (person, location, organisation) in text |
| **NLI** | Natural Language Inference | Task of determining if a premise entails, contradicts, or is neutral to a hypothesis |
| **OCR** | Optical Character Recognition | Converting images of text into machine-readable characters |

---

## ML Training

| Abbreviation | Full Form | Context |
|---|---|---|
| **RLHF** | Reinforcement Learning from Human Feedback | Training paradigm: LLM fine-tuned with a reward model trained from human preferences |
| **SFT** | Supervised Fine-Tuning | First stage of RLHF: fine-tune a base LLM on curated instruction-response pairs |
| **DPO** | Direct Preference Optimization | RLHF alternative: optimises directly on preference pairs without a separate reward model |
| **PPO** | Proximal Policy Optimization | RL algorithm used in RLHF reward phase to constrain policy updates |
| **LoRA** | Low-Rank Adaptation | PEFT technique: inject small trainable rank-decomposition matrices into frozen weights |
| **QLoRA** | Quantized Low-Rank Adaptation | LoRA + 4-bit quantized base weights; enables fine-tuning 65B+ models on a single GPU |
| **PEFT** | Parameter-Efficient Fine-Tuning | Family of techniques that update only a small fraction of weights (LoRA, adapters, prefix tuning) |
| **FSDP** | Fully Sharded Data Parallel | PyTorch training strategy: shards model parameters, gradients, and optimizer state across GPUs |
| **DDP** | Distributed Data Parallel | PyTorch training strategy: each GPU holds a full model copy; gradients are all-reduced |
| **ZeRO** | Zero Redundancy Optimizer | DeepSpeed memory optimization: eliminates redundant parameter/gradient/optimizer state across GPUs |
| **FP16** | 16-bit Floating Point | Half-precision format; reduces memory by 2× vs FP32; may require loss scaling |
| **BF16** | BFloat16 | 16-bit format with the same exponent range as FP32; more numerically stable than FP16 for training |
| **FP32** | 32-bit Floating Point | Single-precision; standard for master weights and optimizer state |
| **INT8** | 8-bit Integer | Quantization format for inference; reduces memory and latency at the cost of precision |

---

## ML Serving & Infrastructure

| Abbreviation | Full Form | Context |
|---|---|---|
| **GPU** | Graphics Processing Unit | Massively parallel processor; dominant compute for neural network training and inference |
| **TPU** | Tensor Processing Unit | Google's custom ASIC for matrix multiplications; used in Google Cloud |
| **CPU** | Central Processing Unit | General-purpose processor; used for preprocessing, light inference, orchestration |
| **A100** | NVIDIA A100 GPU | 80 GB HBM2e; dominant for LLM training and large-batch inference |
| **H100** | NVIDIA H100 GPU | Successor to A100; Transformer Engine, NVLink 4; 3× the LLM training throughput of A100 |
| **TPM** | Tokens Per Minute | Rate limit unit for LLM APIs |
| **DAG** | Directed Acyclic Graph | See Distributed Systems section; also used for ML pipeline and workflow graphs |
| **ETL** | Extract, Transform, Load | Classic data pipeline pattern: pull from source, transform, load into warehouse |
| **ELT** | Extract, Load, Transform | Modern variant: load raw data first, transform inside the data warehouse |

---

## ML Metrics & Evaluation

| Abbreviation | Full Form | Context |
|---|---|---|
| **AUC** | Area Under the ROC Curve | Model discrimination metric: probability that a random positive ranks above a random negative |
| **ECE** | Expected Calibration Error | Measures calibration: weighted average gap between predicted confidence and actual accuracy |
| **BLEU** | Bilingual Evaluation Understudy | N-gram precision metric for machine translation quality |
| **ROUGE** | Recall-Oriented Understudy for Gisting Evaluation | Recall-based metric for summarisation; ROUGE-L measures longest common subsequence |
| **WER** | Word Error Rate | Speech recognition metric: (substitutions + deletions + insertions) / total reference words |
| **CTC** | Connectionist Temporal Classification | Loss function for sequence alignment without frame-level labels; used in ASR and OCR |
| **ASR** | Automatic Speech Recognition | Task of converting spoken audio to text (e.g. Whisper, DeepSpeech) |
| **TTS** | Text-to-Speech | Task of synthesising spoken audio from text |

---

## Ads & Monetisation

| Abbreviation | Full Form | Context |
|---|---|---|
| **CTR** | Click-Through Rate | Fraction of impressions that result in a click; core signal for ad quality |
| **pCTR** | Predicted Click-Through Rate | Model's estimate of P(click \| impression, context); used in auction ranking |
| **CVR** | Conversion Rate | Fraction of clicks that result in a conversion (purchase, sign-up, install) |
| **pCVR** | Predicted Conversion Rate | Model's estimate of P(conversion \| click, context) |
| **CPC** | Cost Per Click | Billing model: advertiser pays per click on their ad |
| **CPM** | Cost Per Mille | Billing model: advertiser pays per 1,000 impressions |
| **CPA** | Cost Per Action | Billing model: advertiser pays per conversion/acquisition |
| **eCPM** | Effective Cost Per Mille | Expected revenue per 1,000 impressions; used to normalise bids across billing models |
| **RPM** | Revenue Per Mille | Publisher-side revenue per 1,000 impressions |
| **ROAS** | Return on Ad Spend | Advertiser metric: conversion value earned per dollar spent on ads |
| **tCPA** | Target Cost Per Acquisition | Automated bidding objective: platform adjusts bids to achieve a specified CPA target |
| **tROAS** | Target Return on Ad Spend | Automated bidding objective: platform adjusts bids to achieve a specified ROAS target |
| **GSP** | Generalized Second-Price | Multi-slot auction mechanism used in sponsored search; winners pay the next competitor's externality |
| **VCG** | Vickrey-Clarke-Groves | Theoretically efficient and incentive-compatible auction mechanism; charges each winner their externality on others |
| **IVT** | Invalid Traffic | Non-human or fraudulent ad traffic (bots, click farms); excluded from billing and training labels |
| **LTV** | Lifetime Value | Expected total revenue from a user (or customer) over their relationship with the product |
| **IPS** | Inverse Propensity Scoring | Debiasing technique: reweights logged examples by the inverse probability of their being observed |

---

## Algorithms & Consensus

| Abbreviation | Full Form | Context |
|---|---|---|
| **FIFO** | First In, First Out | Queue ordering: oldest item is processed first |
| **LWW** | Last-Write-Wins | See Distributed Systems section |
| **CRDT** | Conflict-free Replicated Data Type | See Distributed Systems section |
| **ZAB** | Zookeeper Atomic Broadcast | See Distributed Systems section |
| **PID** | Proportional-Integral-Derivative | Classic control loop algorithm; used in auto-bidding and pacing controllers |
| **BM25** | Best Matching 25 | See ML Fundamentals section |

---

!!! tip "Abbreviation not listed here?"
    Open an issue or PR on the [GitHub repository](https://github.com/spawn08/system-design-interview). All additions are welcome.
