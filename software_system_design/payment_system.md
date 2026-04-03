---
layout: default
title: Payment System
parent: System Design Examples
nav_order: 18
---

# Payment System
{: .no_toc }

<details open markdown="block">
  <summary>Table of contents</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

---

## What We're Building

A **payment system** lets customers pay merchants for goods or services using cards, bank transfers, or digital wallets. At a high level, it must:

- Accept payment instructions from clients (web, mobile, partner APIs).
- Move money safely between buyers, your platform, and sellers.
- Stay consistent with external networks (card schemes, banks, PSPs such as Stripe or PayPal).
- Support refunds, disputes (chargebacks), reconciliation, and fraud controls.
- Meet regulatory and industry rules, especially **PCI-DSS** for card data.

Real-world examples include Stripe Connect, Adyen for Platforms, PayPal Commerce, and in-house systems at marketplaces and SaaS billing engines.

{: .note }
> In interviews, scope matters: clarify one-time checkout vs subscriptions, domestic vs cross-border, and whether you are the merchant of record or a marketplace splitting funds.

### Why This Problem Shows Up in Interviews

Payment systems combine **distributed systems**, **correctness under retries**, **external integrations**, and **security/compliance**. Interviewers often probe: idempotency, exactly-once-ish semantics, ledger design, webhook handling, and operational reconciliation.

---

## Step 1: Requirements

Clarify assumptions before drawing boxes. Payments are correctness-critical; wrong assumptions about settlement timing or dispute ownership are expensive.

### Functional Requirements

| Requirement | Priority | Notes |
|-------------|----------|-------|
| Create and confirm a payment for an order | Must have | Card or wallet; user-visible status |
| Authorize funds (hold) before capture | Must have | Two-phase card flow |
| Capture settled amount after fulfillment | Must have | May be partial capture |
| Idempotent API and webhook processing | Must have | Retries are guaranteed in production |
| Refund (full or partial) | Must have | Tied to original payment |
| Handle PSP asynchronous events (success, failure, dispute) | Must have | Webhooks from Stripe/PayPal-style APIs |
| Merchant reporting and customer receipts | Should have | Often async |
| Subscription billing with retries | Nice to have | Adds dunning, proration |

### Non-Functional Requirements

| Requirement | Target | Rationale |
|-------------|--------|-----------|
| **Consistency** | Strong for money movement | No double charges; ledger must balance |
| **Availability** | 99.95%+ for API; PSP may have own SLA | Degrade gracefully; queue webhooks |
| **Latency (API)** | p99 under a few hundred ms for create | User waits at checkout |
| **Durability** | No lost payments or events | Audit trail for disputes |
| **Auditability** | Immutable append-only financial records | Regulators and accountants |
| **Security** | PCI scope minimization | Tokenization, no PAN storage if possible |

### API Design

Represent payments as first-class resources with explicit states and idempotency.

**Create payment (client or server):**

```http
POST /api/v1/payments
Idempotency-Key: 7b2c9e1a-4f3d-4c2b-9e8a-1d2c3b4a5e6f
Content-Type: application/json
Authorization: Bearer <token>

{
  "order_id": "ord_9x7k2m",
  "amount": { "value": 4999, "currency": "USD" },
  "payment_method": "pm_card_visa_****4242",
  "capture_strategy": "manual",
  "metadata": { "cart_id": "cart_abc" }
}
```

**Response (202 or 200 depending on sync/async PSP):**

```json
{
  "payment_id": "pay_1a2b3c4d",
  "status": "pending",
  "psp_reference": "pi_3QxYz...",
  "client_secret": "pi_..._secret_...",
  "created_at": "2026-04-03T12:00:00Z"
}
```

**Capture:**

```http
POST /api/v1/payments/pay_1a2b3c4d/capture
Idempotency-Key: cap-ord_9x7k2m-001
Content-Type: application/json

{ "amount": { "value": 4999, "currency": "USD" } }
```

**Refund:**

```http
POST /api/v1/payments/pay_1a2b3c4d/refunds
Idempotency-Key: ref-req-88aa
Content-Type: application/json

{ "amount": { "value": 1999, "currency": "USD" }, "reason": "customer_request" }
```

**PSP webhook endpoint (server-to-server):**

```http
POST /internal/webhooks/psp/stripe
Stripe-Signature: t=...,v1=...
Content-Type: application/json

{ "id": "evt_...", "type": "charge.captured", "data": { "object": { ... } } }
```

{: .tip }
> Put `Idempotency-Key` on every mutating call. Store keys with the resulting payment id and status so retries replay the same outcome without duplicate side effects.

---

## Step 2: Back-of-the-Envelope Estimation

Assume a mid-size e-commerce platform (tune numbers in the interview).

### Traffic

```
Orders: 2M/day
Payment attempts: ~2.2M/day (10% retries / abandoned re-attempts)
Peak factor: 5x average

Average QPS: 2.2M / 86,400 ≈ 25.5
Peak QPS: ~130 payment API calls/sec (your edge)

Webhook deliveries from PSP: similar order of magnitude, burstier (batch settlements)
```

### Storage (operational DB)

```
Per payment row: ~500 bytes (ids, amounts, state, idempotency key refs)
2.2M/day × 400 days ≈ 880M rows/year → ~440 GB/year raw
With indexes and ledger entries (3–10x): plan for low single-digit TB/year in OLTP

Immutable ledger append: higher volume; often separate store or partitioned table
```

### External dependencies

```
Card authorization latency: 200ms–2s (PSP + network)
Webhook processing: must be fast (ack) but heavy work async (queue)
```

{: .warning }
> Estimation proves you think about **data growth** and **PSP rate limits**. Mention webhook signing verification and backoff if the interviewer cares about operations.

---

## Step 3: High-Level Design

### Architecture Overview

```mermaid
flowchart TB
  subgraph clients["Clients"]
    WEB[Web / Mobile]
    PARTNER[Partner API]
  end

  subgraph edge["Edge"]
    GW[API Gateway]
    AUTH[Auth / OAuth]
  end

  subgraph core["Payment Core"]
    PS[Payment Service]
    LED[Ledger Service]
    IDEM[Idempotency Store]
    OUTBOX[Outbox / Events]
  end

  subgraph async["Async Processing"]
    Q[(Message Queue)]
    WH[Webhook Worker]
    REC[Reconciliation Worker]
    FRAUD[Fraud Service]
  end

  subgraph external["External"]
    PSP[PSP - Stripe / PayPal]
    BANK[Issuing / Acquiring Networks]
  end

  subgraph data["Data Stores"]
    PAYDB[(Payment DB)]
    LEDDB[(Ledger DB)]
    REDIS[(Redis - locks / cache)]
  end

  WEB --> GW
  PARTNER --> GW
  GW --> AUTH
  AUTH --> PS
  PS --> IDEM
  PS --> PAYDB
  PS --> LED
  LED --> LEDDB
  PS --> OUTBOX
  OUTBOX --> Q
  Q --> WH
  Q --> REC
  PS --> FRAUD
  WH --> PSP
  PS --> PSP
  PSP --> WH
  PS --> REDIS
```

**Responsibilities:**

| Component | Role |
|-----------|------|
| **Payment Service** | State machine, orchestration, maps internal payment to PSP objects |
| **Ledger Service** | Double-entry balances; source of truth for “who owes whom” |
| **Idempotency Store** | Dedupes API requests and webhook deliveries |
| **Outbox** | Reliable domain events to downstream systems (inventory, shipping) |
| **Webhook Worker** | Verifies signatures, updates state, posts ledger entries idempotently |
| **Reconciliation Worker** | Matches PSP settlements to internal ledger |
| **Fraud Service** | Rules + ML; may decline before PSP call |

---

## Step 4: Deep Dive

### 4.1 Payment Flow and State Machine

Typical card flow separates **authorization** (hold) from **capture** (take money), then **settlement** (funds movement across networks), which is often asynchronous.

**States (simplified production model):**

| State | Meaning |
|-------|---------|
| `pending` | Created locally; may await client confirmation or 3DS |
| `authorized` | Funds held; not yet captured |
| `captured` | Capture succeeded at PSP; merchant-side fulfillment can proceed |
| `settled` | PSP reporting matches; internal ledger aligned with cash movement |
| `failed` | Terminal failure (decline, expired auth, invalid request) |
| `refunding` / `refunded` | Refund in flight or complete |
| `chargeback_open` | Dispute filed; funds may be reversed |

```mermaid
stateDiagram-v2
  [*] --> pending
  pending --> authorized: Auth OK
  pending --> failed: Decline / timeout
  authorized --> captured: Capture OK
  authorized --> failed: Void / expiry
  captured --> settled: Reconciliation OK
  captured --> refunding: Refund started
  settled --> refunding: Refund after settle
  refunding --> refunded: Refund confirmed
  captured --> chargeback_open: Dispute
  settled --> chargeback_open: Dispute
  chargeback_open --> settled: Won
  chargeback_open --> refunded: Lost
  failed --> [*]
  refunded --> [*]
```

**Sequence: authorize and capture via PSP**

```mermaid
sequenceDiagram
  participant C as Client
  participant PS as Payment Service
  participant PSP as PSP API
  participant WH as Webhook Worker

  C->>PS: POST /payments (Idempotency-Key)
  PS->>PS: Reserve idempotency slot
  PS->>PSP: Create PaymentIntent / Order
  PSP-->>PS: requires_action / succeeded
  PS-->>C: pending + client_secret
  Note over C,PSP: Customer completes 3DS if required
  C->>PS: POST /confirm
  PS->>PSP: Confirm / authorize
  PSP-->>PS: authorized
  PS-->>C: status authorized
  PSP->>WH: charge.succeeded (async)
  WH->>PS: apply event (idempotent)
  PS->>PS: transition if needed
  PS->>PS: POST /capture (batch or manual)
  PS->>PSP: capture
  PSP-->>PS: captured
  PSP->>WH: charge.captured
  WH->>PS: mark captured / enqueue settlement
```

{: .important }
> **Settlement** timing is PSP-specific. Your `captured` state usually means “we got a successful capture from PSP,” while **bank settlement** may lag; use `settled` when your reconciliation matches PSP reports and bank deposits.

### 4.2 Idempotency and Exactly-Once Processing

Networks retry; users double-click; workers crash mid-flight. **Exactly-once side effects** do not exist across heterogeneous systems—you aim for **idempotent effects**:

1. **Client idempotency keys** on `POST` mutations (create payment, capture, refund).
2. **Server-side deduplication** table: `(idempotency_key, scope) -> response snapshot or resource id`.
3. **Webhook event ids** from PSP: store processed event ids forever (or long retention) to skip duplicates.
4. **Natural keys** on ledger postings: `(payment_id, entry_type, correlation_id)` unique.

**Java (Spring-style): idempotent payment creation**

```java
@Service
public class PaymentCommandService {

    private final IdempotencyRepository idempotencyRepo;
    private final PaymentRepository paymentRepo;
    private final PspClient pspClient;

    @Transactional
    public PaymentResponse createPayment(String idempotencyKey, CreatePaymentRequest req) {
        Optional<IdempotencyRecord> existing =
            idempotencyRepo.findByKeyAndScope(idempotencyKey, "create_payment");
        if (existing.isPresent()) {
            return existing.get().getCachedResponse();
        }

        Payment p = paymentRepo.save(Payment.pending(req.getOrderId(), req.getAmount()));
        PspPaymentIntent intent = pspClient.createIntent(p.getId(), req.getAmount());

        idempotencyRepo.save(IdempotencyRecord.lock(
            idempotencyKey, "create_payment", p.getId(), toSnapshot(p, intent)));

        return PaymentResponse.from(p, intent);
    }
}
```

**Retry rule of thumb:** safe retries use the **same** idempotency key; after a timeout, either query payment status by `order_id` or use a **new** key only if the operation is provably not created (careful: prefer lookup-first).

### 4.3 Double-Entry Ledger

A **ledger** records economic reality with **balanced journals**: every movement debits one account and credits another by the same amount.

Example accounts: `customer_funds_in_transit`, `merchant_payable`, `platform_fee_revenue`, `psp_clearing`, `cash`.

**Invariant:** sum of all posted amounts per currency is zero across paired lines.

**Python: append balanced ledger entries**

```python
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum


class Account(str, Enum):
    CUSTOMER_IN_TRANSIT = "customer_in_transit"
    MERCHANT_PAYABLE = "merchant_payable"
    PLATFORM_FEE = "platform_fee"
    PSP_CLEARING = "psp_clearing"


@dataclass(frozen=True)
class LedgerLine:
    account: Account
    amount_cents: int  # signed: debit positive for assets convention — pick one standard


def post_capture(
    payment_id: str,
    gross_cents: int,
    fee_cents: int,
    merchant_cents: int,
) -> list[LedgerLine]:
    if gross_cents != fee_cents + merchant_cents:
        raise ValueError("lines must balance to gross")
    # Example: move from in-transit to merchant + fee; net to PSP clearing
    return [
        LedgerLine(Account.CUSTOMER_IN_TRANSIT, -gross_cents),
        LedgerLine(Account.MERCHANT_PAYABLE, merchant_cents),
        LedgerLine(Account.PLATFORM_FEE, fee_cents),
        LedgerLine(Account.PSP_CLEARING, gross_cents - merchant_cents - fee_cents),
    ]
```

{: .note }
> Pick **one** debit/credit convention and stick to it. Many systems store signed amounts with account types (asset/liability/revenue) to enforce balancing rules in code or DB constraints.

### 4.4 Payment Service Provider (PSP) Integration

PSPs (Stripe, PayPal, Adyen, etc.) expose:

- **APIs** to create payments, capture, refund.
- **Webhooks** for asynchronous lifecycle (succeeded, failed, dispute, transfer paid).

**Go: verify Stripe webhook and hand off to idempotent processor**

```go
package webhook

import (
	"io"
	"net/http"

	"github.com/stripe/stripe-go/v76/webhook"
)

type EventHandler struct {
	Secret      []byte
	Processor   PaymentEventProcessor
	EventStore  ProcessedEventStore
}

func (h *EventHandler) ServeStripe(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "read body", http.StatusBadRequest)
		return
	}
	event, err := webhook.ConstructEvent(body, r.Header.Get("Stripe-Signature"), string(h.Secret))
	if err != nil {
		http.Error(w, "invalid signature", http.StatusBadRequest)
		return
	}

	if applied, _ := h.EventStore.AlreadyApplied(event.ID); applied {
		w.WriteHeader(http.StatusOK)
		return
	}

	if err := h.Processor.ApplyStripeEvent(event); err != nil {
		http.Error(w, "processing failed", http.StatusInternalServerError)
		return
	}
	_ = h.EventStore.MarkApplied(event.ID)

	w.WriteHeader(http.StatusOK)
}
```

**PayPal** patterns are similar: verify signatures or certificates, treat `event id` as dedupe key, respond quickly and offload work to a queue if processing is heavy.

{: .warning }
> Always verify webhook authenticity **before** parsing business payloads. Return non-2xx only when you want the PSP to retry; avoid tight coupling between verification and long DB transactions.

### 4.5 Reconciliation

**Reconciliation** aligns three views:

1. **Internal ledger** (what you think happened).
2. **PSP reports** (charges, fees, refunds, disputes).
3. **Bank deposits** (cash in your account).

```mermaid
flowchart LR
  subgraph daily["Daily reconciliation job"]
    A[Download PSP report CSV / API]
    B[Normalize to canonical rows]
    C[Match by PSP reference + amount + date]
    D{Matched?}
    E[Mark settled / flag OK]
    F[Exception queue]
  end

  A --> B --> C --> D
  D -->|yes| E
  D -->|no| F
```

| Exception type | Typical action |
|----------------|----------------|
| Amount mismatch | Investigate partial capture, FX, or rounding |
| Missing PSP row | Check webhook backlog; query PSP API by id |
| Duplicate PSP row | Idempotent apply; confirm event store |
| Timing skew | Retry next day; use tolerance windows |

### 4.6 Refund and Chargeback Handling

**Refunds** must be idempotent, tied to the original `payment_id`, and reflected in the ledger as reversing entries (or separate contra accounts).

**Chargebacks** arrive asynchronously:

1. PSP notifies via webhook (`charge.dispute.created`).
2. Freeze or debit merchant balance per policy.
3. Evidence window for contesting; terminal outcomes update `chargeback_open` to won/lost.

{: .tip }
> Separate **customer refund** (merchant-initiated) from **chargeback** (issuer-initiated reversal). They follow different timelines and accounting treatment.

### 4.7 Fraud Prevention

| Layer | Examples |
|-------|----------|
| **Rules** | Velocity limits, blocklists, high-risk BINs |
| **Device / behavior** | IP reputation, impossible travel, session fingerprint |
| **PSP tools** | Stripe Radar, PayPal risk scores, 3-D Secure |
| **Manual review** | Queue for edge cases |

Fraud checks ideally run **before** expensive operations and integrate with **step-up authentication** (3DS) rather than only post-hoc blocking.

### 4.8 PCI-DSS Compliance

You **reduce scope** by never storing raw card numbers (PAN) or CVV.

| Approach | PCI impact |
|----------|------------|
| **Hosted fields / PSP tokenization** | Card data touches PSP directly; you store tokens only |
| **Vault + tokenization** | Small footprint if you must store references |
| **Full PAN storage** | Heavy PCI controls—avoid in most designs |

Practices:

- TLS everywhere; HSTS at edge.
- No card data in logs, URLs, or error messages.
- Access control and audit for anyone touching payment configuration.
- Regular vulnerability scanning and key rotation.

{: .important }
> In interviews, saying **“we use Stripe.js / Elements and only handle tokens”** is often the expected answer for minimizing PCI scope.

---

## Step 5: Scaling & Production

| Concern | Approach |
|---------|----------|
| **Hot partitions** | Shard by `merchant_id` or `tenant_id`; avoid single hot merchant monopolizing one shard |
| **Webhook bursts** | Queue + autoscaling consumers; rate-limit per PSP signing secret rotation |
| **Ledger contention** | Serialize per account if needed; partition journals by account |
| **Read scaling** | CQRS-style read models for dashboards; OLAP for finance |
| **Outbox pattern** | Reliable cross-service notifications without dual-write bugs |
| **Disaster recovery** | Backups, replay from PSP; event log for financial reconstruction |

**Retry with idempotency (operations checklist):**

1. Client retries `POST` with same `Idempotency-Key`.
2. Worker retries PSP calls with same client request id where supported.
3. Webhook handler retries on `5xx` from your endpoint; your handler must tolerate duplicate deliveries.

---

## Interview Tips

### Interview Checklist

- [ ] Clarify marketplace vs merchant-of-record, currencies, and refund policy.
- [ ] State machine covers `pending` → `authorized` → `captured` → `settled`.
- [ ] Explain idempotency keys for APIs and event ids for webhooks.
- [ ] Sketch double-entry ledger and why single-table “balance updates” are risky.
- [ ] Discuss PSP integration: API + signed webhooks + reconciliation.
- [ ] Mention PCI scope reduction via tokenization; no PAN in logs.
- [ ] Cover refunds vs chargebacks and async dispute lifecycle.
- [ ] Call out reconciliation between internal state, PSP reports, and bank deposits.

### Sample Interview Dialogue

**Interviewer:** Walk me through what happens when a user checks out.

**You:** I’d start by confirming payment method and amount with our `Payment` aggregate in `pending`. We call the PSP to create and confirm a payment intent; on success we transition to `authorized` if we’re doing auth/capture split, otherwise we might capture immediately. We persist the PSP reference and return the client anything needed for 3DS. As asynchronous events arrive—webhooks—we update state idempotently using PSP event ids. After fulfillment, capture moves us to `captured`, and our reconciliation job later marks `settled` when PSP payouts match the ledger.

**Interviewer:** How do you prevent double charges?

**You:** All mutating endpoints require an idempotency key scoped to the operation. The server stores the first successful result and replays it on retries. Webhooks are deduped by event id. The ledger uses unique constraints on natural keys for postings so we can’t apply the same financial movement twice.

**Interviewer:** Where does PCI fit?

**You:** We avoid handling raw PANs by using PSP-hosted card entry and storing only tokens. Our servers stay out of PCI scope as much as possible, and we keep card data out of logs and traces.

---

## Summary

Designing a **payment system** is about **correct orchestration** with external PSPs, a clear **state machine** from authorization through settlement, **idempotent** APIs and webhooks, and a **double-entry ledger** that mirrors economic reality. **Reconciliation** closes the loop between internal records and PSP/bank truth; **refunds and chargebacks** extend that lifecycle with dispute handling. **PCI-DSS** is managed primarily by **not** storing sensitive card data and by strong operational hygiene. In interviews, emphasize **retries with idempotency**, **webhook verification**, and **financial auditability**—that signals you can ship payments without losing money or trust.
