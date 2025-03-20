# System Design Interview Preparation Guide

This guide provides a comprehensive overview of topics and example questions for system design interviews, particularly for roles in GenAI/ML and Senior Software Engineering.

## I. Essential System Design Topics

These topics are fundamental to system design.  A strong understanding of these concepts is crucial, regardless of your specific role.

### 1. Load Balancing

*   **Types:** Round Robin, Least Connections, IP Hash, Weighted Round Robin, etc.
*   **Hardware vs. Software Load Balancers**
*   **Session Management:** Sticky Sessions
*   **Health Checks**
*   **Pros and Cons** of different algorithms

### 2. Caching

*   **Cache Types:** In-memory (Redis, Memcached), CDN, Browser Cache, Database Cache
*   **Cache Eviction Policies:** LRU, LFU, FIFO, TTL
*   **Cache Invalidation Strategies**
*   **Write Policies:** Write-through, Write-back, Write-around
*   **Cache Coherency**

### 3. Databases

*   **Relational Databases (SQL):**
    *   ACID properties
    *   Normalization
    *   Indexing
    *   Transactions
    *   Sharding
    *   Replication
*   **NoSQL Databases:**
    *   Key-Value, Document, Column-family, Graph databases
    *   CAP Theorem, BASE properties
    *   Use cases for each type
*   **Database Scaling:**
    *   Vertical vs. Horizontal Scaling
    *   Read Replicas
    *   Master-Slave, Master-Master
*   **Data Modeling**

### 4. Networking

*   **TCP/IP, UDP**
*   **HTTP/HTTPS, REST, gRPC**
*   **DNS**
*   **Proxies:** Forward and Reverse
*   **WebSockets**
*   **Key Metrics:** Latency, Bandwidth, Throughput

### 5. Concurrency

*   **Threads, Processes**
*   **Locks, Mutexes, Semaphores**
*   **Deadlocks, Race Conditions**
*   **Concurrency Patterns:** e.g., Producer-Consumer

### 6. Distributed Systems Concepts

*   **Consistency and Availability:** CAP Theorem
*   **Distributed Consensus:** Paxos, Raft
*   **Eventual Consistency**
*   **Message Queues:** Kafka, RabbitMQ, SQS
*   **Distributed Hash Tables (DHTs)**
*   **Leader Election**

### 7. API Design

*   **RESTful APIs**
*   **GraphQL**
*   **API Versioning**
*   **Rate Limiting**
*   **Authentication and Authorization:** OAuth, JWT

### 8. Security

*   **Common Vulnerabilities:** SQL Injection, XSS, CSRF
*   **Encryption:** Symmetric, Asymmetric
*   **Hashing**
*   **TLS/SSL**

### 9. Scalability, Availability, and Reliability

*   **Horizontal vs. Vertical Scaling**
*   **Redundancy and Failover**
*   **Monitoring and Alerting**
*   **Disaster Recovery**

### 10. Estimation and Capacity Planning

*   Ability to estimate storage, bandwidth, and compute needs based on user numbers, request rates, and data sizes.
*   Back-of-the-envelope calculations.

## II. Advanced Topics

These topics are generally more relevant for Senior/Staff roles and specialized areas.

### 1. Message Queues and Stream Processing

*   **Kafka, RabbitMQ, SQS, Pulsar**
*   **Stream Processing Frameworks:** Apache Flink, Apache Spark Streaming

### 2. Search Systems

*   **Inverted Indexes**
*   **Elasticsearch, Solr**

### 3. Data Warehousing and Data Lakes

*   **Data Warehousing Concepts:** ETL, Star Schema, Snowflake Schema
*   **Data Lake Concepts:** Hadoop, Spark

### 4. Microservices Architecture

*   **Service Discovery**
*   **API Gateways**
*   **Circuit Breakers**
*   **Containerization:** Docker, Kubernetes

### 5. Consistency Patterns

*   **Strong Consistency**
*   **Eventual Consistency**
*   **Causal Consistency**

## III. GenAI/ML Specific Topics

These topics are particularly important for system design interviews focused on Generative AI and Machine Learning.

### 1. Model Serving

*   **REST APIs for model inference**
*   **Batch vs. Online Prediction**
*   **Model Versioning**
*   **A/B Testing of Models**
*   **Model Monitoring:** drift detection, performance metrics
*   **Serving Frameworks:** TensorFlow Serving, TorchServe, Triton Inference Server

### 2. Feature Stores

*   **Centralized management of features** for training and inference
*   **Consistency** between training and serving data
*   **Feature versioning**

### 3. Data Pipelines for ML

*   **Data Ingestion, Transformation, and Validation**
*   **Workflow Orchestration:** Airflow, Kubeflow

### 4. Large Language Models (LLMs)

*   **Prompt Engineering**
*   **Fine-tuning**
*   **Retrieval-Augmented Generation (RAG)**
*   **Vector Databases:** for similarity search
*   **Model Deployment and Scaling** for LLMs

### 5. Distributed Training

*   **Data Parallelism**
*   **Model Parallelism**
*   **Parameter Servers**

## IV. Top 25 System Design Interview Questions

These questions are categorized and cover a range of difficulty levels.  Remember that the *process* of how you approach the problem is often more important than finding a "perfect" solution.

### General System Design (Applicable to all roles)

1.  **Design a URL Shortener (TinyURL):**  Hashing, databases, scaling.
2.  **Design a Rate Limiter:** Algorithms (token bucket, leaky bucket), distributed systems.
3.  **Design a Web Crawler:** Concurrency, distributed processing, politeness policies.
4.  **Design a Notification System:** Message queues, push vs. pull, scalability.
5.  **Design a Distributed Cache:** Caching strategies, consistency, eviction policies.
6.  **Design a Key-Value Store:** Data structures, consistency, distributed systems.
7.  **Design a Proximity Service (e.g., find nearby restaurants):** Geospatial indexing, data structures (quadtrees, geohashes).
8.  **Design a System for Processing a High Volume of Events:** Message queues, stream processing, data pipelines.
9.  **Design a Social Media Feed (e.g., Twitter, Facebook):** Data modeling, read-heavy vs. write-heavy, caching.
10. **Design a Distributed Message Queue:** Message delivery guarantees, fault tolerance, scalability.
11. **Design a system to handle large file uploads:** Chunking, resumable uploads, storage.
12. **Design a system for collaborative text editing (like Google Docs):** Operational transforms, conflict resolution, real-time updates.

### GenAI/ML Specific System Design

13. **Design a Recommendation System (e.g., for Netflix, Amazon):** Collaborative filtering, content-based filtering, hybrid approaches, cold start.
14. **Design a System for Real-time Fraud Detection:** Feature engineering, model serving, low-latency.
15. **Design a System for Image Search:** Feature extraction, similarity search, indexing, vector databases.
16. **Design a System for Training Large Language Models:** Distributed training, data pipelines, model parallelism.
17. **Design a System for Serving LLM Predictions:** Model deployment, scaling, caching, prompt engineering.
18. **Design a Feature Store:** Feature management, consistency, versioning, serving.
19. **Design a system for A/B testing different ML models:** Experiment tracking, metrics, traffic splitting.
20. **Design a system for detecting and mitigating model drift:** Monitoring, retraining, data validation.
21. **Design a system for personalized search:** User profiling, query understanding, ranking models.
22. **Design a system for generating captions for images:** Image understanding, text generation, model evaluation.

### Senior Software Engineer System Design (Focus on Architecture & Trade-offs)

23. **Design a system to handle a sudden surge in traffic (e.g., a viral event).** Load balancing, auto-scaling, caching, circuit breakers.
24. **You are tasked with migrating a monolithic application to a microservices architecture.  Describe your approach.** Service decomposition, API design, data consistency, deployment.
25. **Design a system that needs to be highly available and fault-tolerant across multiple data centers.** Replication, consistency, disaster recovery, network considerations.

## V. Key Tips for System Design Interviews

*   **Clarify Requirements:** Ask clarifying questions! Don't make assumptions. Understand the scale, constraints, and non-functional requirements (availability, consistency, latency, etc.).
*   **Start Simple:** Begin with a high-level design and gradually add details.
*   **Think Out Loud:** Explain your thought process, trade-offs, and design choices.
*   **Use Diagrams:** Draw diagrams to illustrate your design.
*   **Consider Trade-offs:** There's rarely a single "right" answer. Discuss pros and cons.
*   **Scale Incrementally:** Start with a design for a smaller scale, then discuss scaling.
*   **Handle Failure:** Discuss how your system would handle failures.
*   **Data Modeling:** Pay attention to data storage and access. Choose appropriate databases.
*   **Bottlenecks:** Identify potential bottlenecks and discuss solutions.
*   **Practice:** The more you practice, the better you'll become.

Good luck with your interviews!
