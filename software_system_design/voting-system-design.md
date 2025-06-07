Designing a voting system is a fascinating challenge because it touches upon security, scalability, accuracy, and auditability in very critical ways.

## 1. REQUIREMENTS GATHERING

**Clarifying Questions:**

*   **What is the context of the voting?** (e.g., National elections, corporate board elections, simple online polls, talent show voting, feature prioritization). *This dramatically changes security and anonymity requirements.*
*   **Who are the voters?** (e.g., Registered citizens, employees, authenticated users, anonymous public).
*   **What is the scale?** (e.g., Tens, hundreds, millions of voters? Number of concurrent votes?)
*   **What is the desired level of anonymity for voters?** (Fully anonymous, pseudo-anonymous, fully identifiable).
*   **What is the desired level of transparency for results?** (Real-time, after polls close, only to administrators).
*   **How critical is it to prevent duplicate votes?** (Strictly one vote per person, or more relaxed for informal polls).
*   **Are there specific time windows for voting?** (Start and end dates/times).
*   **Who creates and manages the polls/elections?** (Administrators, general users).
*   **What kind of options are being voted on?** (Single choice, multiple choice, ranked choice).
*   **Are there any legal or regulatory compliance requirements?** (e.g., GDPR, election laws).

Let's assume for this design we're building a **general-purpose online platform for creating and participating in polls/elections, targeting authenticated users, with a strong emphasis on preventing duplicate votes and ensuring result integrity.**

**Functional Requirements:**

1.  **User Management:**
    *   User registration and authentication (e.g., email/password, OAuth).
    *   User roles (e.g., Admin, Poll Creator, Voter).
2.  **Poll/Election Management (by Admins/Poll Creators):**
    *   Create new polls/elections with a title, description, options, start time, and end time.
    *   Define eligibility (e.g., all authenticated users, specific groups).
    *   Edit/delete polls (before voting starts).
    *   Close polls manually or automatically at the end time.
3.  **Voting:**
    *   Eligible users can view active polls.
    *   Eligible users can cast one vote per poll for one option (simplifying for now).
    *   Votes should be recorded accurately.
    *   Users should not be able to change their vote after casting (can be debated, let's assume no change for now).
4.  **Results Viewing:**
    *   View aggregated results for a poll (typically after it closes).
    *   Results should show the count/percentage for each option.
    *   (Optional) Real-time results viewing (can be configured per poll).
5.  **Auditability:**
    *   Track key actions: poll creation, vote casting (anonymized if needed), result generation.

**Non-Functional Requirements:**

1.  **Security:**
    *   **Integrity:** Votes cannot be tampered with.
    *   **Authenticity:** Only eligible, authenticated users can vote.
    *   **Non-repudiation (for voter):** A voter cannot deny they cast a vote (if not anonymous).
    *   **Confidentiality/Anonymity (configurable):** Voter choices should be kept private if specified. For our general system, let's aim for voter choices being private from other users, but admins might be able to audit *that* a user voted, not *how* they voted, to ensure one-vote-per-person.
    *   Prevent duplicate voting strictly.
    *   Protection against DoS attacks.
2.  **Scalability:**
    *   Handle a large number of concurrent voters, especially during peak voting periods (e.g., start/end of popular polls). Aim for 1000 votes/sec, scalable to 10,000+ votes/sec.
    *   Handle a large number of polls.
3.  **Availability:** High availability, especially during voting periods (target 99.99%).
4.  **Accuracy:** Votes must be counted correctly without loss or misattribution.
5.  **Performance:**
    *   Low latency for casting a vote (< 500ms).
    *   Fast loading of poll lists and results.
6.  **Auditability:** All significant events must be logged for auditing purposes.

**Constraints:**

*   Cloud-based deployment.
*   Standard web technologies.

**Key Metrics:**

*   Votes per second (QPS for vote casting).
*   Latency for vote casting.
*   Latency for result viewing.
*   System Uptime.
*   Error rate for vote casting.

## 2. SYSTEM ARCHITECTURE

**High-Level Architecture:**

We'll use a microservices-based architecture to separate concerns and allow independent scaling.

1.  **API Gateway:** Entry point for all client requests, handles routing, authentication, rate limiting.
2.  **User Service:** Manages user registration, authentication, profiles, and roles.
3.  **Poll Service:** Manages creation, configuration, and lifecycle of polls/elections.
4.  **Voting Service:** Handles the critical path of casting and recording votes.
5.  **Results Service:** Aggregates votes and provides results.
6.  **Notification Service:** (Optional) Sends notifications (e.g., poll opening/closing).
7.  **Message Queue:** Decouples vote submission from persistent storage, enhancing scalability and resilience.
8.  **Databases:** Separate databases for different services or data types.
9.  **Cache:** For poll details, hot poll results.

**Mermaid.js Diagram:**

```mermaid
graph LR
    subgraph Client Layer
        WebApp[Web Application]
        MobileApp[Mobile Application]
    end

    subgraph API Gateway Layer
        APIGateway[API Gateway]
    end

    subgraph Core Services
        UserService[User Service]
        PollService[Poll Service]
        VotingService[Voting Service]
        ResultsService[Results Service]
    end

    subgraph Asynchronous Processing
        MQ[Message Queue (Kafka/RabbitMQ)]
        VoteProcessor[Vote Processor Worker]
    end

    subgraph Data Stores
        UserDB[(User DB - PostgreSQL)]
        PollDB[(Poll DB - PostgreSQL)]
        VoteDB[(Vote DB - PostgreSQL/NoSQL)]
        ResultsCache[(Results Cache - Redis)]
        AuditLogStore[(Audit Log Store - Elasticsearch)]
    end

    subgraph Monitoring & Logging
        Monitoring[Monitoring (Prometheus/Grafana)]
        Logging[Logging (ELK Stack)]
    end

    WebApp -- HTTPS --> APIGateway
    MobileApp -- HTTPS --> APIGateway

    APIGateway -- Authenticates/Routes --> UserService
    APIGateway -- CRUD Polls --> PollService
    APIGateway -- Cast Vote --> VotingService
    APIGateway -- Get Results --> ResultsService

    UserService -- Reads/Writes --> UserDB
    PollService -- Reads/Writes --> PollDB
    PollService -- Reads --> ResultsCache  // For poll details
    PollService -- Caches --> ResultsCache // Poll details

    VotingService -- Publishes Vote --> MQ
    VotingService -- Checks Eligibility --> PollDB
    VotingService -- Checks if Voted --> VoteDB  // Or a bloom filter + DB

    VoteProcessor -- Consumes from --> MQ
    VoteProcessor -- Writes Validated Vote --> VoteDB
    VoteProcessor -- Updates --> ResultsCache // Increment counters
    VoteProcessor -- Writes --> AuditLogStore

    ResultsService -- Reads Aggregated Data --> ResultsCache
    ResultsService -- Reads from (fallback/detailed) --> VoteDB
    ResultsService -- Reads Poll Info --> PollDB

    UserService -- Writes --> AuditLogStore
    PollService -- Writes --> AuditLogStore

    CoreServices --> Monitoring
    CoreServices --> Logging
    MQ --> Monitoring
    VoteProcessor --> Monitoring
    DataStores --> Monitoring
```

**Data Flow (Casting a Vote):**

1.  User submits a vote via WebApp/MobileApp.
2.  Request hits API Gateway, which authenticates the user (via User Service or JWT validation) and routes to Voting Service.
3.  **Voting Service:**
    *   Validates the request (poll exists, poll is active, user is eligible).
    *   **Crucially, checks if the user has already voted for this poll.** This can be done by querying `VoteDB` or a more optimized structure like a Bloom filter backed by `VoteDB`.
    *   If all checks pass, it constructs a vote message (user_id, poll_id, option_id, timestamp).
    *   Publishes the vote message to the Message Queue (e.g., Kafka).
    *   Returns an "Accepted" (HTTP 202) response to the client immediately.
4.  **Vote Processor Worker:**
    *   Consumes vote messages from the queue.
    *   Performs final validation (e.g., idempotent check to prevent processing duplicates from the queue if a worker restarted).
    *   Atomically records the vote in `VoteDB`. This write *must* enforce uniqueness for `(user_id, poll_id)`.
    *   Updates aggregated counts in `ResultsCache` (e.g., Redis counters).
    *   Writes an entry to the `AuditLogStore`.
5.  **Results Service:**
    *   When results are requested, it primarily reads from `ResultsCache`.
    *   For final, definitive counts or if cache is cold, it might query `VoteDB`.

## 3. COMPONENT SELECTION & JUSTIFICATION

*   **API Gateway (AWS API Gateway / Kong / Apigee):**
    *   **Why:** Standard for microservices. Handles auth, rate limiting, routing. Managed services reduce operational overhead.
    *   **Alternatives:** Nginx, Envoy.
    *   **Trade-offs:** Managed is easier, self-hosted gives more control.

*   **User Service, Poll Service, Voting Service, Results Service (Node.js/Python/Go/Java with Spring Boot):**
    *   **Why:** Language choice depends on team expertise and performance needs. Go for CPU-bound tasks like vote processing, Node.js/Python for I/O-bound API services. Microservices allow for different tech per service.
    *   **Trade-offs:** Polyglot can add complexity. Monolith simpler initially but scales poorly.

*   **Message Queue (Apache Kafka):**
    *   **Why:** High throughput, durability, fault tolerance, stream processing capabilities. Ideal for absorbing vote bursts.
    *   **Alternatives:** RabbitMQ (good for complex routing, simpler), AWS SQS (fully managed, simpler).
    *   **Trade-offs:** Kafka has a higher operational complexity but offers superior scalability for very high loads. SQS is easiest to manage. Let's pick Kafka due to the potential for high vote bursts.

*   **Databases:**
    *   **UserDB, PollDB (PostgreSQL):**
        *   **Why:** Relational data, ACID compliance for user accounts and poll configurations. Strong consistency is important.
        *   **Alternatives:** MySQL.
    *   **VoteDB (PostgreSQL or a NoSQL like Cassandra/DynamoDB):**
        *   **PostgreSQL:** Can handle this with a `UNIQUE` constraint on `(user_id, poll_id)`. Good for consistency. Read replicas for scaling reads. Sharding by `poll_id` might be needed at extreme scale.
        *   **Cassandra/DynamoDB:** If write scalability for votes is the absolute paramount concern beyond what sharded PostgreSQL can offer. Partition key `(poll_id, user_id)` for uniqueness and efficient lookups.
        *   **Decision:** Start with PostgreSQL for `VoteDB` due to its strong consistency and familiar ACID properties, especially the unique constraint. Optimize and consider NoSQL if it becomes a bottleneck.
    *   **ResultsCache (Redis):**
        *   **Why:** In-memory, fast key-value store. Ideal for counters (`INCRBY`) and caching poll details or pre-computed results.
        *   **Alternatives:** Memcached.
        *   **Trade-offs:** Redis offers more data structures and persistence options.

*   **AuditLogStore (Elasticsearch / OpenSearch):**
    *   **Why:** Powerful search and aggregation capabilities for logs.
    *   **Alternatives:** Storing logs in files and shipping to a log management system.
    *   **Trade-offs:** Elasticsearch provides better query capabilities than simple file storage.

## 4. DATABASE DESIGN

**UserDB (PostgreSQL):**

```sql
CREATE TABLE users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password TEXT NOT NULL,
    role VARCHAR(50) NOT NULL DEFAULT 'voter', -- 'admin', 'poll_creator', 'voter'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

**PollDB (PostgreSQL):**

```sql
CREATE TABLE polls (
    poll_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    creator_id UUID REFERENCES users(user_id),
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE NOT NULL,
    eligibility_criteria JSONB, -- e.g., {"group_id": "some_group"} or NULL for all users
    allow_realtime_results BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'pending' -- 'pending', 'active', 'closed', 'archived'
);

CREATE TABLE poll_options (
    option_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    poll_id UUID REFERENCES polls(poll_id) ON DELETE CASCADE,
    option_text VARCHAR(255) NOT NULL,
    display_order INT
);

-- Index for querying active polls
CREATE INDEX idx_polls_active ON polls (status, start_time, end_time);
```

**VoteDB (PostgreSQL):**

```sql
CREATE TABLE votes (
    vote_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    poll_id UUID NOT NULL, -- No FK to PollDB to allow decoupling, but app logic enforces existence
    option_id UUID NOT NULL, -- No FK for decoupling
    user_id UUID NOT NULL,   -- No FK for decoupling
    voted_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_user_poll_vote UNIQUE (user_id, poll_id) -- CRITICAL for preventing duplicate votes
);

-- Index for querying votes for a poll (for aggregation if cache fails)
CREATE INDEX idx_votes_poll_option ON votes (poll_id, option_id);
-- Index for quick check if user voted on a poll (supports the unique constraint)
CREATE INDEX idx_votes_user_poll ON votes (user_id, poll_id);
```
*(Note: Not using foreign keys on `VoteDB` to `PollDB` or `UserDB` can improve decoupling and performance in a microservices setup where databases might be physically separate. Application logic must ensure integrity.)*

**ResultsCache (Redis):**

*   Key for poll details: `poll:<poll_id>:details` (Hash storing title, options, etc.)
*   Key for vote counts: `poll:<poll_id>:results` (Hash where field is `option_id` and value is count)
    *   Example: `HINCRBY poll:some-poll-id:results option-abc-123 1`

**AuditLogStore (Elasticsearch Document Structure):**

```json
{
    "timestamp": "2023-10-27T10:00:00Z",
    "event_type": "VOTE_CAST", // POLL_CREATED, USER_LOGIN, etc.
    "user_id": "user-uuid-123", // Can be null for system events
    "poll_id": "poll-uuid-456", // If applicable
    "details": {
        "option_id": "option-uuid-789",
        "ip_address": "192.168.1.100",
        "user_agent": "Chrome/..."
    },
    "status": "SUCCESS" // or FAILURE
}
```

## 5. API DESIGN

*   **Authentication:** JWT Bearer Token in `Authorization` header.

*   **Users:**
    *   `POST /auth/register` (username, email, password) -> `201 Created`
    *   `POST /auth/login` (email, password) -> `200 OK` { token }
*   **Polls:**
    *   `POST /polls` (Admin/PollCreator only)
        *   Request: `{ "title": "...", "description": "...", "options": ["Opt1", "Opt2"], "start_time": "...", "end_time": "..." }`
        *   Response: `201 Created` { poll_id, ... }
    *   `GET /polls` (List active/upcoming polls, pagination) -> `200 OK`
    *   `GET /polls/{poll_id}` -> `200 OK` { poll details }
    *   `PUT /polls/{poll_id}` (Admin/PollCreator, before voting starts) -> `200 OK`
    *   `DELETE /polls/{poll_id}` (Admin/PollCreator, before voting starts) -> `204 No Content`
*   **Voting:**
    *   `POST /polls/{poll_id}/vote`
        *   Request: `{ "option_id": "..." }`
        *   Response: `202 Accepted` (Vote enqueued) or `400 Bad Request` (invalid input), `403 Forbidden` (not eligible/already voted), `404 Not Found` (poll/option not found)
*   **Results:**
    *   `GET /polls/{poll_id}/results`
        *   Response: `200 OK` `{ "poll_id": "...", "results": { "option_id_1": count1, "option_id_2": count2 }, "status": "CLOSED"|"ACTIVE_REALTIME" }`
        *   Response: `403 Forbidden` (if results not yet public)

## 6. LOW-LEVEL IMPLEMENTATION (Conceptual)

**VotingService - Handling a Vote Request (Python/Flask like):**

```python
# Simplified VotingService endpoint
from flask import Flask, request, jsonify
import kafka_producer # Assume a Kafka producer utility
import redis_client # Assume a Redis client utility
import db_client # Assume a DB client utility

app = Flask(__name__)

# Pre-check for duplicate voting (Bloom Filter + Cache)
# This is an optimization. The DB unique constraint is the source of truth.
VOTED_CACHE_PREFIX = "voted:" # voted:<poll_id>:<user_id> -> 1 (if voted)
BLOOM_FILTER_PER_POLL = {} # In-memory, needs persistence or recreation

def has_user_voted_fast_check(user_id, poll_id):
    # 1. Check local Bloom Filter (if poll is very active)
    # if poll_id in BLOOM_FILTER_PER_POLL and BLOOM_FILTER_PER_POLL[poll_id].check(user_id):
    #     # Likely voted, proceed to cache/DB check for confirmation
    #     pass # For brevity, skipping bloom filter details

    # 2. Check distributed cache (e.g., Redis)
    cache_key = f"{VOTED_CACHE_PREFIX}{poll_id}:{user_id}"
    if redis_client.exists(cache_key):
        return True
    return False

@app.route('/polls/<poll_id>/vote', methods=['POST'])
def cast_vote(poll_id):
    auth_token = request.headers.get('Authorization')
    user_id = get_user_from_token(auth_token) # Assume this function exists
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    option_id = data.get('option_id')

    # 1. Validate poll and option (query PollDB or cache)
    poll = db_client.get_poll_details(poll_id)
    if not poll or poll.status != 'active' or not is_option_valid(poll, option_id):
        return jsonify({"error": "Invalid poll or option, or poll not active"}), 400

    # 2. Check eligibility (based on poll.eligibility_criteria and user_id)
    if not is_user_eligible(user_id, poll.eligibility_criteria):
        return jsonify({"error": "User not eligible to vote"}), 403

    # 3. Check if user already voted (fast check first, DB is ultimate source of truth)
    if has_user_voted_fast_check(user_id, poll_id):
         # If fast check says yes, it's a strong indicator.
         # For 100% certainty, the VoteProcessor will hit the DB unique constraint.
         # For this API, we might rely on the fast check or do a quick DB check here if latency allows.
         # Let's assume the VoteProcessor will handle the final DB check.
         # To be safer and give immediate feedback:
        if db_client.has_user_voted_in_db(user_id, poll_id): # This queries VoteDB
             return jsonify({"error": "Already voted"}), 403


    # 4. Construct vote message
    vote_message = {
        "user_id": user_id,
        "poll_id": poll_id,
        "option_id": option_id,
        "timestamp": datetime.utcnow().isoformat()
    }

    # 5. Publish to Kafka
    try:
        kafka_producer.send_message('votes_topic', vote_message)
    except Exception as e:
        # Log error
        return jsonify({"error": "Failed to enqueue vote, please try again"}), 500

    return jsonify({"message": "Vote accepted"}), 202

# Vote Processor Worker (Conceptual)
def process_vote_message(message):
    user_id = message['user_id']
    poll_id = message['poll_id']
    option_id = message['option_id']

    try:
        # Atomically insert into VoteDB.
        # This will fail if (user_id, poll_id) unique constraint is violated.
        db_client.insert_vote(user_id, poll_id, option_id, message['timestamp'])

        # If insert successful, update Redis counters
        redis_client.hincrby(f"poll:{poll_id}:results", option_id, 1)
        # Add to voted cache for fast checks by VotingService
        redis_client.set(f"{VOTED_CACHE_PREFIX}{poll_id}:{user_id}", "1", ex=POLL_DURATION)


        # Log to audit store
        log_audit_event("VOTE_CAST_SUCCESS", user_id, poll_id, {"option_id": option_id})

    except db_client.DuplicateVoteError: # Custom exception for unique constraint violation
        log_audit_event("VOTE_CAST_DUPLICATE_ATTEMPT", user_id, poll_id, {"option_id": option_id})
        # Message is acknowledged, as it's a valid duplicate attempt, not a system error
    except Exception as e:
        # Log error, potentially requeue message if it's a transient failure
        log_audit_event("VOTE_PROCESSING_ERROR", user_id, poll_id, {"error": str(e)})
        raise # Re-raise to trigger requeue based on Kafka consumer config
```

## 7. QUEUE SOLUTION (Kafka)

*   **Topic:** `votes_topic`
*   **Partitions:** Partition by `poll_id`. This ensures all votes for the same poll go to the same partition, which can be useful for localized processing or stateful operations (like using a Bloom filter per partition/poll in the consumer). However, if one poll is extremely hot, it can overload one consumer. A more general approach is to partition by `user_id` or round-robin for better load distribution across consumers, and handle poll-specific logic stateless-ly. For preventing duplicate votes, `user_id` as partition key helps ensure subsequent votes from the same user go to the same consumer, potentially aiding faster duplicate detection if consumers maintain some local state, but the DB unique constraint is the ultimate guarantee. **Let's partition by `user_id` to distribute load from high-activity users.**
*   **Consumer Groups:** A single consumer group for `VoteProcessor` workers. Kafka distributes partitions among consumers in the group.
*   **Delivery Guarantees:** "At-least-once." The Vote Processor must be idempotent (handle message replays without double-counting, e.g., by relying on the DB unique constraint).

## 8. AI MODEL INTEGRATION

*   Not directly applicable for the core voting mechanism.
*   Could be used for:
    *   **Fraud Detection:** Analyzing voting patterns to detect suspicious activity (e.g., bots).
    *   **Sentiment Analysis:** On poll descriptions or comments (if comments are allowed).
    *   **Recommendation:** Suggesting polls to users based on their voting history.

## 9. SCALING STRATEGY

*   **Stateless Services (User, Poll, Voting, Results):** Horizontally scale by adding more instances behind the API Gateway/Load Balancer.
*   **Vote Processor Workers:** Scale by adding more consumer instances to the Kafka consumer group. Kafka will rebalance partitions.
*   **Kafka:** Scale Kafka brokers and partitions as needed.
*   **Databases:**
    *   **PostgreSQL:**
        *   Read Replicas for UserDB, PollDB, and VoteDB (for result aggregation reads).
        *   Vertical scaling (more powerful instances).
        *   Sharding `VoteDB` by `poll_id` or a composite key if it becomes a major bottleneck for writes, though this adds complexity.
    *   **Redis:** Use Redis Cluster for sharding and HA.
*   **Caching:** Aggressively cache poll details, user sessions, and (if allowed) real-time results. Use a CDN for static assets.

## 10. FAILURE HANDLING

*   **Service Failure:** Load balancer/API Gateway reroutes traffic to healthy instances. Kubernetes/ECS can auto-restart failed containers.
*   **Database Failure:**
    *   Use managed DB services with automated failover to replicas (e.g., AWS RDS Multi-AZ).
    *   Regular backups.
*   **Kafka Failure:** Kafka is designed for fault tolerance with data replication across brokers.
*   **Vote Processor Failure:** Kafka retains messages. If a worker dies, another picks up. "At-least-once" delivery with idempotent consumers ensures correctness.
*   **Dead Letter Queue (DLQ):** Messages that consistently fail processing in `VoteProcessor` (after retries) are sent to a DLQ for manual inspection. This prevents poison pills from blocking the queue.
*   **Circuit Breakers:** Implement circuit breakers in services to prevent cascading failures when downstream dependencies are unavailable.

## 11. BOTTLENECKS & MITIGATIONS

1.  **Vote Ingestion (Writing to `VoteDB`):**
    *   **Bottleneck:** High concurrent writes during peak voting.
    *   **Mitigation:**
        *   **Message Queue (Kafka):** Decouples submission from DB write, absorbing bursts.
        *   **Batching by Vote Processor:** Workers can batch writes to DB (if DB supports efficient batch inserts and uniqueness check across batch).
        *   **Database Optimization:** Proper indexing (`(user_id, poll_id)`), connection pooling.
        *   **Sharding `VoteDB`:** If single instance write capacity is exceeded.
        *   **Optimistic Concurrency/Fast Duplicate Check:** Use Redis/Bloom Filter to quickly reject most duplicate votes before hitting the DB.

2.  **Duplicate Vote Check:**
    *   **Bottleneck:** Querying `VoteDB` for every vote can be slow.
    *   **Mitigation:**
        *   **Caching voted status:** Use Redis (`SETEX voted:<poll_id>:<user_id> 1 <duration>`) after a successful vote. The Voting Service checks this cache before publishing to Kafka.
        *   **Bloom Filters:** For very hot polls, a Bloom filter per poll (in memory of Voting Service instances or shared via Redis) can quickly tell if a user *might* have voted. If yes, then check cache/DB. Reduces DB load for non-duplicate votes.
        *   **DB Unique Constraint:** This is the ultimate, non-negotiable guard.

3.  **Result Aggregation:**
    *   **Bottleneck:** Querying and aggregating from `VoteDB` for popular polls.
    *   **Mitigation:**
        *   **Real-time counters in Redis:** `VoteProcessor` increments counters in Redis. `ResultsService` reads from Redis.
        *   **Periodic Materialized Views/Snapshotting:** For non-real-time results, a background job can aggregate from `VoteDB` and store in a summary table or update Redis.

4.  **Poll Listing (Many Active Polls):**
    *   **Bottleneck:** Querying `PollDB` for all active polls.
    *   **Mitigation:**
        *   Pagination.
        *   Caching of frequently accessed poll lists.
        *   Good indexing on `PollDB` (status, start/end times).

**Security Considerations - Deep Dive:**

*   **Authentication:** Strong authentication (MFA if highly sensitive).
*   **Authorization:** Strict role-based access control.
*   **Input Validation:** Sanitize all inputs to prevent XSS, SQLi.
*   **Rate Limiting:** On API Gateway and critical endpoints (vote casting, login).
*   **HTTPS Everywhere:** Encrypt all communication.
*   **Vote Tampering:**
    *   Votes written to Kafka are immutable.
    *   DB transactions for vote recording.
    *   Regular audits and checksums if extreme security is needed.
*   **Voter Anonymity vs. Prevent Duplicates:**
    *   The current design links `user_id` to a vote in `VoteDB` to enforce uniqueness. This means votes are not truly anonymous from the system's perspective.
    *   For stronger anonymity (while still preventing duplicates):
        *   **Cryptographic Schemes:** Blind signatures, zero-knowledge proofs. These are complex.
        *   **Token-based Anonymous Voting:** User requests a unique, one-time voting token. The token is marked as used when a vote is cast, but the vote itself isn't directly linked to the user_id, only to the token. This requires a secure token generation and management system.
