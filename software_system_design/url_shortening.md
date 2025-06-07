## Design a URL Shortener (TinyURL or Bitly)

## 1. REQUIREMENTS GATHERING

**Clarifying Questions:**

*   **What is the expected scale?** (Number of URLs to be shortened per day? Number of redirect requests per second?)
*   **Should short URLs be unique globally, or per user?** (Global uniqueness is standard).
*   **Do we need custom short URLs (vanity URLs)?** (e.g., `tiny.co/mybrand`)
*   **Do we need analytics for clicks?** (e.g., how many times a short URL was clicked).
*   **Should short URLs expire?**
*   **Are there any constraints on the length or characters of the short URL?**
*   **Do we need user accounts to manage their URLs?**
*   **What is the desired availability and latency?**

Let's assume the following based on common expectations for such a service:

**Functional Requirements:**

1.  **URL Shortening:** Users can submit a long URL and receive a unique, much shorter URL.
2.  **URL Redirection:** When a user accesses a short URL, they should be redirected to the original long URL.
3.  **Uniqueness:** Every generated short URL must be unique.
4.  **Custom Short URLs (Optional but common):** Users should be able to suggest a custom short code (if available).
5.  **Analytics (Optional but common):** Track the number of clicks for each short URL.
6.  **URL Expiration (Optional):** Allow users to set an expiration date for short URLs.
7.  **API Access:** Provide an API for third-party services to shorten URLs.

**Non-Functional Requirements:**

1.  **High Availability:** The service (especially redirection) must be highly available (e.g., 99.99%).
2.  **Low Latency:**
    *   Redirection should be extremely fast (e.g., < 50ms, excluding network latency to the client).
    *   Shortening can have slightly higher latency (e.g., < 200ms).
3.  **Scalability:**
    *   Handle a large number of redirection requests (read-heavy). Let's aim for 10,000 redirects/sec.
    *   Handle a moderate number of shortening requests (write-heavy, but less frequent than reads). Let's aim for 100 shortenings/sec.
4.  **Durability:** Shortened URL mappings should not be lost.
5.  **Predictable Short URL Length:** Generated short URLs should have a consistent length (e.g., 6-8 characters for the code part).
6.  **Security:** Prevent abuse (e.g., shortening malicious URLs, DoS attacks).

**Constraints:**

*   The short URL will have a fixed domain (e.g., `tiny.co/`).
*   The short code part will consist of alphanumeric characters (e.g., `a-z`, `A-Z`, `0-9`). This gives 26+26+10 = 62 possible characters.
    *   With 6 characters: 62^6 ≈ 56.8 billion combinations.
    *   With 7 characters: 62^7 ≈ 3.5 trillion combinations.
    *   With 8 characters: 62^8 ≈ 218 trillion combinations.
    *   Let's aim for 7 characters to provide ample capacity for a long time.

**Key Metrics:**

*   **Redirect QPS:** 10,000/sec.
*   **Shortening QPS:** 100/sec.
*   **Redirect Latency (p99):** < 50ms.
*   **Shortening Latency (p99):** < 200ms.
*   **Number of active short URLs:** Billions.

## 2. SYSTEM ARCHITECTURE

**High-Level Architecture:**

The system can be broadly divided into two main flows:

1.  **Write Path (Shortening URL):**
    *   Client sends a long URL.
    *   Service generates a unique short code.
    *   Service stores the mapping (`short_code -> long_URL`).
    *   Service returns the short URL.
2.  **Read Path (Redirecting URL):**
    *   Client requests a short URL.
    *   Service looks up the short code.
    *   Service responds with an HTTP redirect (301 or 302) to the long URL.

**Mermaid.js Diagram:**

```
graph LR
    subgraph Client
        User[User/Application]
    end

    subgraph API Layer
        APIGateway[API Gateway]
    end

    subgraph Write Path
        ShorteningService[URL Shortening Service]
        IDGenerator[Distributed ID Generator]
        DB_Write[(Database - Write Replica/Primary)]
    end

    subgraph Read Path
        RedirectService[URL Redirect Service]
        Cache[(Distributed Cache - Redis/Memcached)]
        DB_Read[(Database - Read Replicas)]
    end

    subgraph Analytics (Async)
        AnalyticsQueue[Message Queue (Kafka/SQS)]
        AnalyticsProcessor[Analytics Processor Worker]
        AnalyticsDB[(Analytics DB - Time Series/NoSQL)]
    end

    subgraph Data Stores
        MainDB[Primary Database (e.g., PostgreSQL, MySQL)]
    end
    
    subgraph Monitoring & Logging
        Monitoring[Monitoring (Prometheus/Grafana)]
        Logging[Logging (ELK Stack)]
    end

    User -- Shorten Request (Long URL) --> APIGateway
    APIGateway -- Route to --> ShorteningService
    ShorteningService -- Request Unique ID --> IDGenerator
    IDGenerator -- Returns Unique ID --> ShorteningService
    ShorteningService -- Encode ID to Short Code --> ShorteningService
    ShorteningService -- Store (Short Code, Long URL) --> MainDB
    ShorteningService -- Returns Short URL --> APIGateway
    APIGateway -- Short URL --> User

    User -- Access Short URL --> APIGateway
    APIGateway -- Route to --> RedirectService
    RedirectService -- Lookup Short Code --> Cache
    Cache -- Cache Miss --> RedirectService
    Cache -- Cache Hit (Long URL) --> RedirectService
    RedirectService -- DB Lookup (if Cache Miss) --> MainDB
    MainDB -- Long URL --> RedirectService
    RedirectService -- Update Cache --> Cache
    RedirectService -- Publish Click Event --> AnalyticsQueue
    RedirectService -- HTTP 301/302 Redirect --> APIGateway
    APIGateway -- Redirect --> User

    AnalyticsQueue -- Click Event --> AnalyticsProcessor
    AnalyticsProcessor -- Update Click Count --> AnalyticsDB
    AnalyticsProcessor -- (Optional) Update MainDB --> MainDB

    ShorteningService --> Monitoring
    RedirectService --> Monitoring
    IDGenerator --> Monitoring
    AnalyticsProcessor --> Monitoring
    MainDB --> Monitoring
    Cache --> Monitoring
    APIGateway --> Logging
    ShorteningService --> Logging
    RedirectService --> Logging
```

**Data Flow (Shortening):**

1.  Client sends `POST /shorten` request with `long_url` to API Gateway.
2.  API Gateway routes to URL Shortening Service.
3.  Shortening Service first checks if this `long_url` (or its hash) already exists to avoid duplicates and return the existing short URL.
4.  If new, it requests a unique ID from the Distributed ID Generator.
5.  The ID is base-62 encoded to generate the `short_code`.
6.  The mapping (`short_code`, `long_url`, `id`, `user_id`, `expires_at`) is stored in the Main Database.
7.  The full short URL (e.g., `https://tiny.co/{short_code}`) is returned.

**Data Flow (Redirection):**

1.  Client browser requests `GET /{short_code}`.
2.  API Gateway routes to URL Redirect Service.
3.  Redirect Service extracts `short_code` from the path.
4.  It first checks the Distributed Cache for `short_code`.
    *   **Cache Hit:** Retrieves `long_url` from cache.
    *   **Cache Miss:** Queries Main Database (Read Replica) for `short_code`.
        *   If found, stores `long_url` in cache for future requests.
        *   If not found, returns a 404 error.
5.  If `long_url` is found, Redirect Service publishes a "click event" (containing `short_code`, timestamp, IP, user-agent) to the Analytics Queue.
6.  Redirect Service issues an HTTP 301 (Permanent Redirect) or 302 (Found/Temporary Redirect) to the `long_url`. 301 is better for SEO if the short URL is meant to be canonical; 302 allows for easier changes to the target URL later or if analytics are critical for every hit (as 301s can be heavily cached by browsers/proxies). For a generic shortener, 302 is often preferred to ensure analytics are captured.

## 3. COMPONENT SELECTION & JUSTIFICATION

*   **API Gateway (AWS API Gateway / Kong / Apigee):**
    *   **Why:** Manages incoming requests, routing, rate limiting, authentication, SSL termination.
    *   **Alternatives:** Nginx, Traefik.
    *   **Trade-offs:** Managed services are easier but might have vendor lock-in.

*   **URL Shortening Service & URL Redirect Service (Go / Node.js / Java):**
    *   **Why:** These can be stateless services. Go is excellent for high-concurrency, low-latency network services (ideal for Redirect Service). Node.js is good for I/O-bound tasks.
    *   **Alternatives:** Python (Flask/FastAPI).
    *   **Trade-offs:** Choice depends on team expertise and specific performance needs.

*   **Distributed ID Generator:**
    *   **Why:** Essential for generating unique short codes without collision and without a single DB sequence becoming a bottleneck.
    *   **Approaches:**
        1.  **Snowflake (Twitter's):** Generates 64-bit unique IDs composed of timestamp, worker ID, and sequence number. Ensures IDs are roughly time-sortable.
        2.  **Zookeeper/Etcd for sequence blocks:** Service instances request a block of IDs (e.g., 1000 IDs) from Zookeeper.
        3.  **Database sequence with multiple DBs/shards:** Each shard has its own sequence, combine with shard ID.
    *   **Choice:** Snowflake-like approach is robust and widely used.

*   **Base-62 Encoding Logic:** Simple algorithm to convert the numeric ID to an alphanumeric string.

*   **Main Database (PostgreSQL / MySQL / Cassandra / DynamoDB):**
    *   **Why:** Stores the `short_code -> long_URL` mapping and other metadata.
    *   **SQL (PostgreSQL/MySQL):**
        *   **Pros:** Strong consistency, ACID, mature. `short_code` can be a unique key.
        *   **Cons:** Can be a bottleneck for extreme write loads if not sharded. Read replicas help with read scaling.
    *   **NoSQL (Cassandra/DynamoDB):**
        *   **Pros:** Excellent horizontal scalability for key-value lookups (`short_code` as key).
        *   **Cons:** Eventual consistency (for some), unique constraint enforcement might be application-level for some NoSQL types.
    *   **Decision:** Start with **PostgreSQL** due to its strong consistency, unique constraints, and good balance. Use read replicas for redirect path. If write throughput or total data size becomes an issue, consider sharding PostgreSQL or moving the core mapping to a K-V NoSQL store like DynamoDB.

*   **Distributed Cache (Redis / Memcached):**
    *   **Why:** Crucial for low-latency redirects. Caches `short_code -> long_URL` mapping.
    *   **Redis:** Offers persistence options and more data structures.
    *   **Memcached:** Simpler, pure in-memory cache.
    *   **Decision:** **Redis** is generally preferred for its versatility.

*   **Analytics Queue (Apache Kafka / AWS SQS / RabbitMQ):**
    *   **Why:** Decouples click tracking from the critical redirect path. Absorbs bursts of click events.
    *   **Kafka:** High throughput, durable.
    *   **SQS:** Fully managed, simple.
    *   **Decision:** **Kafka** if extreme throughput is needed, otherwise SQS/RabbitMQ for simplicity.

*   **Analytics Processor & Analytics DB:**
    *   **Processor (Flink / Spark Streaming / Custom Worker):** Consumes from queue, aggregates data.
    *   **DB (ClickHouse / Druid / TimescaleDB / Elasticsearch):** Optimized for time-series data and analytical queries. For simple click counts, even the Main DB or Redis could suffice initially.
    *   **Decision:** For click counts, a worker incrementing counters in **Redis** or a separate table in the Main DB is often sufficient to start. More advanced analytics would need a specialized time-series DB.

## 4. DATABASE DESIGN

**Main Database (PostgreSQL):**

```sql
CREATE TABLE urls (
    id BIGINT PRIMARY KEY,                      -- Unique ID from Distributed ID Generator
    short_code VARCHAR(10) UNIQUE NOT NULL,     -- Base-62 encoded version of ID (e.g., 7-8 chars)
    long_url TEXT NOT NULL,
    long_url_hash VARCHAR(64),                  -- SHA-256 hash of long_url for quick duplicate checks
    user_id UUID,                               -- Optional: If user accounts exist
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    -- click_count INT DEFAULT 0,               -- Option 1: Store click count here (updated by analytics processor)
    --                                          -- Option 2: Store in a separate analytics DB or Redis
    custom_alias VARCHAR(50) UNIQUE             -- For custom short URLs
);

-- Index for redirect lookups (critical)
CREATE INDEX idx_urls_short_code ON urls (short_code);

-- Index for checking if a long URL has already been shortened
CREATE INDEX idx_urls_long_url_hash ON urls (long_url_hash);

-- Index for user-specific URL listing
CREATE INDEX idx_urls_user_id ON urls (user_id) WHERE user_id IS NOT NULL;

-- Index for custom alias lookup
CREATE INDEX idx_urls_custom_alias ON urls (custom_alias) WHERE custom_alias IS NOT NULL;

-- Optional: Table for detailed click analytics if not using a separate Analytics DB
CREATE TABLE clicks (
    click_id BIGSERIAL PRIMARY KEY,
    short_code_ref VARCHAR(10) NOT NULL, -- Can FK to urls.short_code if needed, or just store the code
    -- url_id BIGINT REFERENCES urls(id), -- Alternative FK
    clicked_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    ip_address INET,
    user_agent TEXT,
    referrer TEXT,
    country VARCHAR(2)
);
CREATE INDEX idx_clicks_short_code_ref_timestamp ON clicks (short_code_ref, clicked_at DESC);
```

**Analytics Cache (Redis):**

*   Key: `clicks:{short_code}` -> Value: (integer count)
    *   Use `INCR clicks:{short_code}` for atomic increment.

## 5. API DESIGN

*   **Authentication:** API Key for API clients, JWT for logged-in users (if user accounts are implemented).

1.  **Shorten URL:**
    *   `POST /api/v1/shorten`
    *   Request Body:
        ```json
        {
            "long_url": "https://very.long/url/to/be/shortened?param1=value1",
            "custom_alias": "mycustomlink", // Optional
            "expires_in_days": 30 // Optional
        }
        ```
    *   Success Response (201 Created):
        ```json
        {
            "short_url": "https://tiny.co/aB3xYz1",
            "long_url": "https://very.long/url/to/be/shortened?param1=value1",
            "expires_at": "2024-01-15T10:00:00Z" // If applicable
        }
        ```
    *   Error Responses: `400 Bad Request` (invalid URL, alias taken), `401 Unauthorized`, `500 Internal Server Error`.

2.  **Redirect Short URL:**
    *   `GET /{short_code}` (e.g., `https://tiny.co/aB3xYz1`)
    *   Success Response: `HTTP 302 Found` (or `301 Moved Permanently`)
        *   `Location: {long_url}`
    *   Error Response: `404 Not Found`.

3.  **Get URL Analytics (Optional):**
    *   `GET /api/v1/stats/{short_code}`
    *   Success Response (200 OK):
        ```json
        {
            "short_code": "aB3xYz1",
            "long_url": "https://very.long/url/...",
            "created_at": "2023-12-15T10:00:00Z",
            "click_count": 1205,
            "daily_clicks": [{"date": "2023-12-15", "count": 500}, ...] // More advanced
        }
        ```

## 6. LOW-LEVEL IMPLEMENTATION (Conceptual)

**Base-62 Encoding (Python example):**

```python
CHARSET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
BASE = len(CHARSET)

def id_to_short_code(num_id):
    if num_id == 0:
        return CHARSET[0]
    
    short_code_chars = []
    while num_id > 0:
        remainder = num_id % BASE
        short_code_chars.append(CHARSET[remainder])
        num_id //= BASE
    return "".join(reversed(short_code_chars))

def short_code_to_id(short_code):
    num_id = 0
    for char in short_code:
        num_id = num_id * BASE + CHARSET.index(char)
    return num_id

# Example:
# unique_id = 1234567890 # From ID Generator
# short_code = id_to_short_code(unique_id) # e.g., "4C9GzE" (example, actual output depends on ID)
# print(f"ID {unique_id} -> Short Code {short_code}")
# print(f"Short Code {short_code} -> ID {short_code_to_id(short_code)}")
```

**Distributed ID Generator (Conceptual - using a range pre-fetcher):**

Each instance of the Shortening Service could fetch a range of IDs (e.g., 1000) from a central coordinator (like Zookeeper or a dedicated ID service backed by a DB sequence) and use them locally. This reduces contention on the central ID source.

**Analytics Processor (Conceptual Python Worker for SQS/Kafka):**

```python
# Assuming messages from queue are like: {"short_code": "aB3xYz1", "timestamp": "...", ...}
import redis_client # Assume a Redis client
import db_client    # Assume a DB client for detailed analytics

def process_click_event(message_data):
    short_code = message_data.get("short_code")
    if not short_code:
        # Log error, skip message
        return

    # Option 1: Increment Redis counter
    redis_client.incr(f"clicks:{short_code}")

    # Option 2: Write to detailed analytics DB (if needed)
    # db_client.log_click_details(
    #     short_code=short_code,
    #     timestamp=message_data.get("timestamp"),
    #     ip_address=message_data.get("ip_address"),
    #     user_agent=message_data.get("user_agent")
    # )
    print(f"Processed click for {short_code}")

# Main loop for worker
# while True:
#   message = analytics_queue.receive_message()
#   if message:
#     process_click_event(message.body)
#     analytics_queue.delete_message(message.handle)
```

## 7. QUEUE SOLUTION (for Analytics)

*   **Technology:** AWS SQS or Apache Kafka.
    *   **SQS:** Simpler, fully managed, good for decoupling.
    *   **Kafka:** Higher throughput, more features, but more complex to manage.
    *   **Decision:** SQS is likely sufficient for click analytics unless volume is astronomically high.
*   **Partitioning (Kafka):** Can partition by `short_code` if per-URL ordered processing is needed, or round-robin for even distribution.
*   **Consumer Groups (Kafka):** Analytics Processors will form a consumer group.
*   **Delivery Guarantees:** At-least-once. Analytics Processor should be idempotent (e.g., if updating a DB, ensure no double counting, though for Redis INCR this is fine).

## 8. AI MODEL INTEGRATION

*   Not directly needed for core functionality.
*   **Possible Uses:**
    *   **Malicious URL Detection:** An ML model could be trained to identify potentially malicious URLs (phishing, malware) during the shortening process. This could integrate with services like Google Safe Browsing.
    *   **Trend Analysis:** Analyzing click patterns to identify trending URLs or topics.

## 9. SCALING STRATEGY

*   **Stateless Services (Shortening, Redirect):** Horizontally scale by adding more instances behind a load balancer.
*   **Database:**
    *   **Read Scaling:** Use read replicas for the Main Database, especially for the Redirect Service.
    *   **Write Scaling/Data Volume:**
        *   Vertical scaling (more powerful DB instance).
        *   **Sharding:** If the `urls` table becomes too large or write contention is high. Shard by `id` (from ID generator) or a hash of `short_code`. The `id` being somewhat time-ordered (from Snowflake-like generator) can lead to hot shards if not careful; hash sharding distributes better.
*   **Distributed Cache (Redis):** Use Redis Cluster for distributed caching, allowing horizontal scaling and higher availability.
*   **ID Generator:** Ensure it's distributed and doesn't become a bottleneck.
*   **Analytics System:** Scale Kafka/SQS and Analytics Processor workers independently. Scale Analytics DB as needed.
*   **Rate Limiting:** Implement at API Gateway or service level to prevent abuse.

## 10. FAILURE HANDLING

*   **Service Instance Failure:** Load balancer reroutes traffic. Auto-scaling groups replace failed instances.
*   **Database Failure:**
    *   **Read Replicas:** Redirect Service can still function using read replicas.
    *   **Primary Failure:** Shortening service might be temporarily impaired or return errors. Managed DB services (e.g., AWS RDS Multi-AZ) provide automatic failover.
*   **Cache Failure:** Redirect Service falls back to querying the database. Performance degrades, but system remains functional. Implement cache warming strategies.
*   **ID Generator Failure:** Shortening service will fail. Requires a highly available ID generation system.
*   **Analytics Queue/Processor Failure:** Click data might be delayed or lost (if queue is not durable). Use durable queues and retry mechanisms. A Dead Letter Queue (DLQ) for messages that consistently fail processing.

## 11. BOTTLENECKS & MITIGATIONS

1.  **Redirect Database Reads:**
    *   **Bottleneck:** High QPS hitting the database.
    *   **Mitigation:** Aggressive caching with Redis. Ensure high cache hit ratio. Use read replicas.

2.  **Shortening Database Writes / ID Generation:**
    *   **Bottleneck:** Contention on generating unique IDs or writing to the `urls` table.
    *   **Mitigation:** Distributed ID generator. Database connection pooling. Consider sharding if write QPS is extremely high. Asynchronous writes (client gets an ack, actual DB write happens slightly later - adds complexity and potential for eventual failure).

3.  **Short Code Collision (for custom aliases or poorly designed random generator):**
    *   **Bottleneck:** Retries if a chosen custom alias or random code is already taken.
    *   **Mitigation:** For custom aliases, check availability directly. For generated codes, the ID-to-base62 approach avoids collisions by design. If using random generation (not recommended for primary approach), have a retry mechanism with a limit.

4.  **Analytics Data Ingestion/Processing:**
    *   **Bottleneck:** High volume of click events overwhelming the analytics pipeline.
    *   **Mitigation:** Scalable message queue (Kafka). Horizontally scale analytics processors. Batch writes to analytics DB.

5.  **"Thundering Herd" Problem:** If a very popular URL's cache entry expires, many requests might hit the DB simultaneously.
    *   **Mitigation:** Probabilistic early cache refresh (a small percentage of requests refresh the cache just before it expires). Slightly longer TTLs. Use a locking mechanism (e.g., distributed lock in Redis) so only one request fetches from DB and updates cache on a miss.
