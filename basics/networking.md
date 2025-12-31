---
layout: default
title: Networking
parent: Fundamentals
nav_order: 4
---

# Networking
{: .no_toc }

<details open markdown="block">
  <summary>Table of Contents</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

---

## Why Networking Matters in System Design

Every distributed system is fundamentally a network of computers communicating with each other. Understanding networking is crucial because:

1. **Network is often the bottleneck:** CPU and memory are fast; networks are slow and unreliable
2. **Latency kills user experience:** A 100ms delay feels instant, 1 second feels slow, 10 seconds and users leave
3. **Networks fail in weird ways:** Packets get lost, connections hang, routers crash
4. **Protocol choice affects everything:** REST vs gRPC, HTTP/1.1 vs HTTP/2, WebSocket vs polling

When you design a system, you're really designing how machines talk to each other over a network.

---

## Network Fundamentals

### The Speed of Light Problem

Data travels at the speed of light through fiber optic cables—but even light isn't fast enough for global systems.

**Round-trip times (RTT) for light in fiber:**

| Route | Distance | RTT (minimum) |
|-------|----------|---------------|
| Same data center | < 1 km | < 0.01 ms |
| Same region (East Coast) | 500 km | 3.3 ms |
| Cross-country (NY to LA) | 4,000 km | 27 ms |
| Transatlantic (NY to London) | 5,500 km | 37 ms |
| Around the world | 40,000 km | 267 ms |

These are **theoretical minimums**. Real latencies are 2-5x higher due to routing, processing, and non-straight paths.

**What this means for system design:**
- If your server is in Virginia and your user is in Tokyo, minimum latency is ~150ms per round trip
- A typical HTTP request involves 3+ round trips (DNS, TCP handshake, TLS handshake, actual request)
- This is why CDNs, edge computing, and regional deployments matter

### Bandwidth vs Latency

**Bandwidth:** How much data you can transfer per unit time (e.g., 1 Gbps)
**Latency:** How long it takes for a single piece of data to get there (e.g., 50ms)

Think of it like a highway:
- **Bandwidth** = Number of lanes (how many cars per hour)
- **Latency** = Distance to destination (how long one car takes)

A 10 Gbps connection with 100ms latency might be worse for interactive applications than a 100 Mbps connection with 10ms latency.

**For most web applications, latency matters more than bandwidth.** Users care about how fast the page starts loading, not how fast a 10GB file transfers.

### TCP vs UDP

These are the two main transport protocols. Understanding their differences is essential.

#### TCP (Transmission Control Protocol)

TCP provides reliable, ordered delivery of data. If you send bytes 1, 2, 3, the receiver gets 1, 2, 3—in that order, guaranteed.

**How TCP ensures reliability:**

1. **Three-way handshake (connection establishment):**
```
Client: SYN (Hey, want to talk?)
Server: SYN-ACK (Sure, I'm ready)
Client: ACK (Great, let's go)
```
This adds 1 round-trip before any data can flow.

2. **Acknowledgments:** Receiver confirms each packet. If no ACK, sender retransmits.

3. **Ordering:** Each packet has a sequence number. Out-of-order packets are reordered.

4. **Flow control:** Receiver tells sender how much data it can handle (prevents overwhelming slow receivers).

5. **Congestion control:** Sender slows down if network is congested (prevents network collapse).

**TCP's problem: Head-of-line blocking**

If packet 1 is lost, the receiver must wait for retransmission before processing packets 2, 3, 4... even if they've already arrived.

```
Sent: [1] [2] [3] [4] [5]
Received: [ ] [2] [3] [4] [5]
Application sees: waiting... waiting... [1] [2] [3] [4] [5]
```

**When to use TCP:**
- Web traffic (HTTP/HTTPS)
- File transfers
- Email
- Any time you need reliability

#### UDP (User Datagram Protocol)

UDP is "fire and forget." Send a packet and hope it arrives. No guarantees.

**UDP provides:**
- No connection setup (faster first message)
- No ordering guarantees
- No delivery guarantees
- No congestion control

**Why would anyone use UDP?**

1. **Speed:** No handshake, no waiting for ACKs
2. **Real-time applications:** For video calls, a late packet is useless. Better to skip it than wait.
3. **Application-level reliability:** Sometimes you want custom reliability logic

**When to use UDP:**
- Video/audio streaming
- Online gaming
- DNS queries
- VPN tunnels
- IoT sensors (high volume, loss tolerance)

**Modern protocols built on UDP:**
- QUIC (used by HTTP/3): Adds reliability and encryption on top of UDP
- WebRTC: Real-time video/audio

---

## HTTP and HTTPS

HTTP (Hypertext Transfer Protocol) is the foundation of web communication. Every time you load a website or make an API call, you're using HTTP.

### HTTP Request/Response Cycle

**Request:**
```http
GET /api/users/123 HTTP/1.1
Host: api.example.com
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
Accept: application/json
```

**Response:**
```http
HTTP/1.1 200 OK
Content-Type: application/json
Cache-Control: max-age=3600

{"id": 123, "name": "Alice", "email": "alice@example.com"}
```

### HTTP Methods

| Method | Purpose | Idempotent? | Safe? |
|--------|---------|-------------|-------|
| GET | Retrieve data | Yes | Yes |
| POST | Create new resource | No | No |
| PUT | Replace entire resource | Yes | No |
| PATCH | Partial update | No | No |
| DELETE | Remove resource | Yes | No |
| HEAD | GET without body (check existence) | Yes | Yes |
| OPTIONS | Get allowed methods | Yes | Yes |

**Idempotent:** Calling multiple times has same effect as once. GET, PUT, DELETE are idempotent. POST is not.

**Safe:** Doesn't modify server state. GET and HEAD are safe.

**Why idempotency matters:**

If a network request times out, should you retry? With idempotent methods, yes—worst case you do the same thing twice. With POST, you might create duplicate records.

```python
# Safe to retry (idempotent)
PUT /orders/123  # Same order created/updated

# Dangerous to retry (not idempotent)
POST /orders  # Might create duplicate orders!
```

### HTTP Status Codes

| Range | Category | Examples |
|-------|----------|----------|
| 2xx | Success | 200 OK, 201 Created, 204 No Content |
| 3xx | Redirection | 301 Moved Permanently, 304 Not Modified |
| 4xx | Client Error | 400 Bad Request, 401 Unauthorized, 404 Not Found |
| 5xx | Server Error | 500 Internal Error, 502 Bad Gateway, 503 Service Unavailable |

**Common codes and when to use them:**

| Code | Meaning | When to Use |
|------|---------|-------------|
| 200 | OK | Successful GET, PUT, PATCH |
| 201 | Created | Successful POST that created a resource |
| 204 | No Content | Successful DELETE |
| 400 | Bad Request | Invalid request syntax or parameters |
| 401 | Unauthorized | Authentication required or failed |
| 403 | Forbidden | Authenticated but not allowed |
| 404 | Not Found | Resource doesn't exist |
| 409 | Conflict | Conflict with current state (e.g., duplicate) |
| 429 | Too Many Requests | Rate limited |
| 500 | Internal Server Error | Unexpected server error (catch-all) |
| 502 | Bad Gateway | Upstream service failed |
| 503 | Service Unavailable | Server temporarily overloaded |
| 504 | Gateway Timeout | Upstream service timed out |

### HTTP/1.1 vs HTTP/2 vs HTTP/3

**HTTP/1.1 (1997):**
- One request per connection at a time
- To parallelize, browsers open 6+ connections per domain
- Plain text headers (verbose, repeated)
- No compression for headers

**HTTP/2 (2015):**
- Multiple requests over single connection (multiplexing)
- Binary protocol (more efficient parsing)
- Header compression (HPACK)
- Server push (server can send resources before client asks)
- Still uses TCP

**HTTP/3 (2022):**
- Uses QUIC (over UDP) instead of TCP
- Eliminates head-of-line blocking at transport layer
- Faster connection establishment (0-RTT possible)
- Better mobile performance (handles network switching)

**Performance comparison:**

```
Loading 50 small resources:

HTTP/1.1: 6 connections × 8 requests/connection = 48 round trips
HTTP/2: 1 connection, all requests multiplexed = 1 round trip (mostly)
HTTP/3: Same as HTTP/2 but no HOL blocking
```

**When to use which:**
- HTTP/2 is the default choice today—widely supported, significant benefits
- HTTP/3 for mobile-heavy or latency-critical applications
- HTTP/1.1 only for legacy compatibility

### HTTPS and TLS

HTTPS = HTTP over TLS (Transport Layer Security). All traffic is encrypted.

**TLS Handshake (simplified):**

```
1. Client: "Hello, I support these cipher suites"
2. Server: "Let's use TLS 1.3 with AES-256-GCM. Here's my certificate."
3. Client: "Certificate is valid. Here's my key share."
4. Server: "Here's my key share. Let's encrypt!"
5. (Encrypted communication begins)
```

**TLS 1.3 improvements:**
- 1-RTT handshake (instead of 2)
- 0-RTT resumption for returning clients
- Removed legacy insecure cipher suites
- Encrypted more of the handshake

**Certificates and Trust:**

Your browser trusts a set of Certificate Authorities (CAs). When a server presents a certificate:

1. Is it signed by a trusted CA?
2. Is the domain name in the certificate correct?
3. Is the certificate not expired?
4. Is it not revoked?

If all checks pass, connection is trusted. This is how you know you're talking to the real google.com.

---

## DNS (Domain Name System)

DNS translates human-readable domain names (google.com) to IP addresses (142.250.80.46).

### How DNS Resolution Works

When you type `www.example.com` in your browser:

```mermaid
flowchart LR
    Browser[Browser] --> Resolver[DNS Resolver]
    Resolver --> Root[Root DNS Server]
    Root --> TLD[.com TLD Server]
    TLD --> Auth[example.com Authoritative Server]
    Auth --> Resolver
    Resolver --> Browser
```

**Step by step:**

1. **Browser cache:** Check if we recently resolved this domain
2. **OS cache:** Check operating system's DNS cache
3. **Recursive resolver:** Query your ISP's (or configured) DNS resolver
4. **Root servers:** Resolver asks root server "Who handles .com?"
5. **TLD servers:** Root points to .com TLD servers
6. **Authoritative servers:** TLD points to example.com's nameservers
7. **Answer:** Authoritative server returns the IP address
8. **Caching:** Result is cached at each level

**DNS Record Types:**

| Type | Purpose | Example |
|------|---------|---------|
| A | IPv4 address | `example.com → 93.184.216.34` |
| AAAA | IPv6 address | `example.com → 2606:2800:220:1:248:1893:25c8:1946` |
| CNAME | Alias to another domain | `www.example.com → example.com` |
| MX | Mail server | `example.com → mail.example.com` |
| TXT | Arbitrary text | Used for verification, SPF records |
| NS | Nameserver | `example.com → ns1.example.com` |

### DNS and Latency

DNS resolution adds latency to the first request to any domain. Typically 20-100ms, but can be 200ms+ if caches are cold.

**Optimization strategies:**

1. **DNS prefetching:** Tell the browser to resolve domains early
```html
<link rel="dns-prefetch" href="//api.example.com">
```

2. **Minimize third-party domains:** Each new domain = another DNS lookup

3. **Use a fast DNS provider:** Cloudflare (1.1.1.1), Google (8.8.8.8)

4. **Low TTLs have a cost:** Lower TTL = more DNS queries

### DNS for Load Balancing

DNS can distribute traffic across multiple servers:

```
example.com → [93.184.216.34, 93.184.216.35, 93.184.216.36]
```

**Round-robin DNS:** Return IPs in rotating order.

**GeoDNS:** Return different IPs based on client location.

**Limitations:**
- No health checking (DNS doesn't know if a server is down)
- TTL delays (takes time for clients to see changes)
- Client caching (clients may hold onto old IPs)
- Uneven distribution (one IP might get more traffic)

For production load balancing, use a proper load balancer, not DNS alone.

---

## REST API Design

REST (Representational State Transfer) is an architectural style for designing networked applications. Most web APIs today are RESTful.

### REST Principles

1. **Stateless:** Each request contains all information needed. Server doesn't store client state.

2. **Resource-based:** Everything is a resource identified by a URL.

3. **HTTP methods as verbs:** GET, POST, PUT, DELETE map to CRUD operations.

4. **Representations:** Resources can have multiple representations (JSON, XML).

### Designing Good REST APIs

**Resources, not actions:**

```
Bad:  GET /getUser?id=123
Bad:  POST /createOrder
Bad:  GET /getAllProducts

Good: GET /users/123
Good: POST /orders
Good: GET /products
```

**Use nouns for resources:**

```
Bad:  GET /getUserOrders/123
Good: GET /users/123/orders
```

**Pluralize resource names:**

```
GET /users        # List all users
GET /users/123    # Get user 123
POST /users       # Create new user
PUT /users/123    # Update user 123
DELETE /users/123 # Delete user 123
```

**Nest related resources:**

```
GET /users/123/orders       # Orders for user 123
GET /orders/456/items       # Items in order 456
POST /users/123/orders      # Create order for user 123
```

**Use query parameters for filtering, sorting, pagination:**

```
GET /products?category=electronics&sort=price&order=asc
GET /orders?status=pending&page=2&limit=20
GET /users?search=john&fields=id,name,email
```

### API Versioning

APIs evolve. How do you handle breaking changes?

**Option 1: URL path versioning (most common)**
```
GET /v1/users/123
GET /v2/users/123
```

**Option 2: Header versioning**
```
GET /users/123
Accept: application/vnd.api+json; version=2
```

**Option 3: Query parameter**
```
GET /users/123?version=2
```

**Best practice:** Use URL versioning. It's explicit and easy to understand.

### Error Handling

Return meaningful errors with:
- Appropriate status code
- Human-readable message
- Machine-parseable error code
- Optional: details for debugging

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": [
      {"field": "email", "message": "Invalid email format"},
      {"field": "age", "message": "Must be a positive integer"}
    ],
    "request_id": "abc123"
  }
}
```

---

## gRPC and Protocol Buffers

gRPC is a high-performance RPC (Remote Procedure Call) framework from Google.

### Why gRPC?

| Aspect | REST/JSON | gRPC |
|--------|-----------|------|
| Protocol | HTTP/1.1 or HTTP/2 | HTTP/2 only |
| Format | JSON (text) | Protocol Buffers (binary) |
| Schema | Optional (OpenAPI) | Required (.proto files) |
| Streaming | Complex | Native support |
| Browser support | Native | Requires proxy |
| Typical size | Larger | 3-10x smaller |
| Typical speed | Slower | 3-10x faster |

### Protocol Buffers

Protocol Buffers (protobuf) is a binary serialization format.

**Define your schema (.proto file):**

```protobuf
syntax = "proto3";

service UserService {
  rpc GetUser (GetUserRequest) returns (User);
  rpc ListUsers (ListUsersRequest) returns (stream User);
  rpc CreateUser (CreateUserRequest) returns (User);
}

message GetUserRequest {
  int64 id = 1;
}

message User {
  int64 id = 1;
  string name = 2;
  string email = 3;
  repeated Order orders = 4;
}

message Order {
  int64 id = 1;
  double amount = 2;
  string status = 3;
}
```

**Generate code from schema:**

```bash
protoc --python_out=. --grpc_python_out=. user.proto
```

This generates client and server stubs in your language.

### gRPC Streaming

gRPC supports four communication patterns:

1. **Unary:** Single request, single response (like REST)
```protobuf
rpc GetUser(GetUserRequest) returns (User);
```

2. **Server streaming:** Single request, stream of responses
```protobuf
rpc ListUsers(ListUsersRequest) returns (stream User);
```

3. **Client streaming:** Stream of requests, single response
```protobuf
rpc UploadFile(stream Chunk) returns (UploadResponse);
```

4. **Bidirectional streaming:** Both sides stream
```protobuf
rpc Chat(stream Message) returns (stream Message);
```

### When to Use gRPC

**Use gRPC when:**
- Internal microservice communication
- High throughput, low latency requirements
- You control both client and server
- You need streaming
- Payload size matters (mobile, IoT)

**Use REST when:**
- Public APIs (browser compatibility)
- Simple CRUD operations
- Team familiarity with REST
- Debugging ease (human-readable)

---

## WebSockets

WebSockets provide full-duplex, persistent connections between client and server.

### HTTP vs WebSocket

**HTTP:** Client initiates request, server responds, connection closes.
```
Client: "Give me data"
Server: "Here's data"
(Connection closed)
```

**WebSocket:** Persistent connection, either side can send at any time.
```
Client: "Let's stay connected"
Server: "OK"
(Connection stays open)
Server: "Here's an update"
Server: "Another update"
Client: "Sending some data"
Server: "Got it, here's more"
```

### WebSocket Handshake

WebSocket starts as an HTTP request, then upgrades:

```http
GET /chat HTTP/1.1
Host: example.com
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
Sec-WebSocket-Version: 13
```

```http
HTTP/1.1 101 Switching Protocols
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=
```

After this, the connection uses the WebSocket protocol.

### When to Use WebSockets

**Good for:**
- Real-time notifications
- Chat applications
- Live sports scores / stock tickers
- Collaborative editing
- Online gaming
- Live dashboards

**Not good for:**
- Simple request/response
- Infrequent updates (polling is simpler)
- Stateless servers (WebSockets are stateful)

### WebSocket Scaling Challenges

**Problem 1: Sticky sessions**

WebSocket connections are stateful. If you have multiple servers, you need to route the same client to the same server.

**Solution:** Use a load balancer with sticky sessions, or use a pub/sub system (Redis, Kafka) so any server can send to any client.

```mermaid
flowchart TD
    Client1[Client 1] --> LB[Load Balancer]
    Client2[Client 2] --> LB
    Client3[Client 3] --> LB
    
    LB --> Server1[Server 1]
    LB --> Server2[Server 2]
    
    Server1 --> Redis[(Redis Pub/Sub)]
    Server2 --> Redis
```

**Problem 2: Connection limits**

Each WebSocket connection uses a file descriptor. Servers have limits on open file descriptors.

**Solution:** Tune OS limits, use connection pooling, or use multiple server instances.

**Problem 3: Reconnection handling**

Connections drop. Clients need to reconnect and resync state.

**Solution:** Use libraries like Socket.IO that handle reconnection, or implement exponential backoff.

---

## Polling, Long Polling, and Server-Sent Events

When you need real-time updates but can't use WebSockets.

### Short Polling

Client repeatedly asks "Any updates?"

```javascript
setInterval(async () => {
  const response = await fetch('/api/notifications');
  if (response.data.length > 0) {
    displayNotifications(response.data);
  }
}, 5000);  // Every 5 seconds
```

**Pros:** Simple, works everywhere
**Cons:** Wasteful (mostly empty responses), higher latency (up to polling interval)

### Long Polling

Server holds the request until there's data to send.

```javascript
async function poll() {
  try {
    const response = await fetch('/api/notifications?wait=true');
    displayNotifications(response.data);
  } finally {
    poll();  // Immediately poll again
  }
}
poll();
```

**Server side:**
```python
@app.route('/api/notifications')
async def notifications():
    # Wait up to 30 seconds for new data
    data = await wait_for_notifications(timeout=30)
    return jsonify(data)
```

**Pros:** Real-time feel, no wasted requests
**Cons:** Holds server resources, connection reestablishment overhead

### Server-Sent Events (SSE)

Server pushes updates over a persistent HTTP connection. One-way only (server to client).

```javascript
const eventSource = new EventSource('/api/stream');

eventSource.onmessage = (event) => {
  console.log('Received:', event.data);
};

eventSource.onerror = () => {
  console.log('Connection lost, reconnecting...');
};
```

**Server side:**
```python
@app.route('/api/stream')
def stream():
    def generate():
        while True:
            data = get_next_update()
            yield f"data: {json.dumps(data)}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')
```

**Pros:** Native browser support, auto-reconnect, simpler than WebSocket
**Cons:** One-way only, limited browser connections per domain

### Comparison

| Method | Real-time | Bidirectional | Complexity | Server Load |
|--------|-----------|---------------|------------|-------------|
| Short Polling | No (delay) | Yes | Low | High |
| Long Polling | Yes | Yes | Medium | Medium |
| SSE | Yes | No | Low | Low |
| WebSocket | Yes | Yes | High | Low |

---

## Proxies and Load Balancers

### Forward Proxy

A forward proxy sits between clients and the internet. Clients talk to the proxy, proxy talks to servers.

```mermaid
flowchart LR
    Client1[Client] --> Proxy[Forward Proxy]
    Client2[Client] --> Proxy
    Proxy --> Internet[Internet]
    Internet --> Server1[Server]
    Internet --> Server2[Server]
```

**Use cases:**
- **Privacy:** Hide client IP from servers
- **Filtering:** Block certain sites (corporate/school)
- **Caching:** Cache responses for multiple clients
- **Bypass restrictions:** Access geo-blocked content

### Reverse Proxy

A reverse proxy sits between clients and your servers. Clients talk to the proxy, proxy routes to servers.

```mermaid
flowchart LR
    Internet[Internet] --> Proxy[Reverse Proxy]
    Proxy --> Server1[Server 1]
    Proxy --> Server2[Server 2]
    Proxy --> Server3[Server 3]
```

**Use cases:**
- **Load balancing:** Distribute requests across servers
- **SSL termination:** Handle HTTPS, send HTTP to backend
- **Caching:** Cache responses
- **Security:** Hide server details, block attacks
- **Compression:** Compress responses

**Popular reverse proxies:** Nginx, HAProxy, Traefik, Envoy

### Load Balancer

A load balancer distributes traffic across multiple servers to improve reliability and performance.

**Load balancing algorithms:**

| Algorithm | How it Works | Best For |
|-----------|--------------|----------|
| Round Robin | Rotate through servers | Equal capacity servers |
| Weighted Round Robin | Rotate with weights | Different capacity servers |
| Least Connections | Send to server with fewest connections | Long-running requests |
| IP Hash | Hash client IP to server | Session affinity |
| Random | Random server selection | Simple, even distribution |

**Health checks:**

Load balancers regularly check if servers are healthy:

```
Every 5 seconds, send GET /health to each server.
If server fails 3 checks in a row, remove from pool.
When server passes 2 checks, add back to pool.
```

**Layer 4 vs Layer 7:**

| Layer | OSI Layer | Operates On | Examples |
|-------|-----------|-------------|----------|
| L4 | Transport | TCP/UDP | HAProxy (TCP mode), AWS NLB |
| L7 | Application | HTTP/HTTPS | Nginx, HAProxy (HTTP mode), AWS ALB |

L7 can make routing decisions based on URL, headers, cookies. L4 is faster but less flexible.

---

## Network Security Basics

### Common Attacks

**DDoS (Distributed Denial of Service):**
Overwhelm servers with traffic from many sources.

*Mitigations:*
- CDN (absorbs traffic)
- Rate limiting
- DDoS protection services (Cloudflare, AWS Shield)
- Auto-scaling

**Man-in-the-Middle (MITM):**
Attacker intercepts communication between client and server.

*Mitigations:*
- Always use HTTPS
- Certificate pinning (mobile apps)
- HSTS (force HTTPS)

**SQL Injection / XSS:**
Injecting malicious code through user input.

*Mitigations:*
- Parameterized queries
- Input validation
- Content Security Policy (CSP)
- Output encoding

### CORS (Cross-Origin Resource Sharing)

Browsers block requests from one origin (domain) to another by default. CORS allows controlled cross-origin access.

**How it works:**

```http
# Request from example.com to api.other.com
Origin: https://example.com

# Response from api.other.com
Access-Control-Allow-Origin: https://example.com
Access-Control-Allow-Methods: GET, POST, PUT
Access-Control-Allow-Headers: Content-Type, Authorization
```

**Preflight requests:**

For "complex" requests (non-GET with custom headers), browser first sends OPTIONS:

```http
OPTIONS /api/data HTTP/1.1
Origin: https://example.com
Access-Control-Request-Method: POST
Access-Control-Request-Headers: Content-Type
```

Server responds with what's allowed. If allowed, browser sends actual request.

---

## Interview Tips

### Common Interview Questions

**Q: "How would you design a real-time notification system?"**

*Good answer:*
> "I'd use WebSockets for clients that support them, with SSE as fallback. For scaling, I'd have notification servers subscribe to a Redis Pub/Sub channel. When a notification is created, it's published to Redis, and all servers forward it to connected clients. For offline users, notifications are stored in a database and delivered when they reconnect."

**Q: "Why is HTTP/2 faster than HTTP/1.1?"**

*Good answer:*
> "HTTP/2 introduces multiplexing—multiple requests share a single TCP connection. HTTP/1.1 requires one request at a time per connection, so browsers open 6+ connections in parallel. HTTP/2 also compresses headers (HPACK) since headers are often repetitive, and uses binary framing which is more efficient to parse than text."

**Q: "Explain what happens when you type a URL in your browser."**

*Good answer (abbreviated):*
> "First, DNS resolution to get the IP address. Then TCP three-way handshake, followed by TLS handshake for HTTPS. The browser sends an HTTP GET request. The server processes it and returns a response. The browser parses HTML, discovers additional resources (CSS, JS, images), and makes additional requests (often in parallel via HTTP/2). The DOM is constructed, CSS is applied, JavaScript executes, and the page renders."

---

## Summary

| Concept | Key Takeaway |
|---------|--------------|
| **TCP vs UDP** | TCP: reliable, ordered. UDP: fast, unreliable. Use TCP for most apps. |
| **HTTP/2** | Multiplexing, header compression. Default choice today. |
| **HTTPS/TLS** | Always use it. TLS 1.3 for speed. |
| **DNS** | Adds latency. Use fast resolvers, prefetching, low TTLs carefully. |
| **REST** | Resource-based URLs, HTTP methods as verbs, proper status codes. |
| **gRPC** | Binary, fast, typed. Great for microservices. |
| **WebSockets** | Full-duplex, persistent. For real-time bidirectional communication. |
| **SSE** | Simple one-way server push. Good for notifications. |
| **Proxies** | Forward (client-side), Reverse (server-side). |
| **Load Balancers** | Distribute traffic, health checks, L4 vs L7. |
