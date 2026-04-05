---
layout: default
title: AI Agent System
parent: GenAI System Design
nav_order: 7
---

# Design an AI Agent System
{: .no_toc }

<details open markdown="block">
  <summary>Table of Contents</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

---

## What We're Building

We are designing an **autonomous AI agent system** that can **plan**, **invoke tools**, **maintain memory**, and **execute multi-step tasks** toward a user goal — not just emit the next token in a chat thread. Think **Google’s agentic assistants**, **Anthropic’s computer use**, **OpenAI’s deep research**, or internal “copilots” that browse, code, and call APIs until the task is done.

**This is not a single LLM endpoint.** It is a **control loop** (observe → think → act), a **tool surface** with strict contracts, **durable memory**, and **governance** (sandboxing, budgets, human checkpoints).

### Representative Scale (Hypothetical Production Service)

| Dimension | Order-of-magnitude |
|-----------|-------------------|
| **Concurrent agent runs** | 10K–100K (burst) |
| **Avg tool calls per completed task** | 8–40 (domain-dependent) |
| **LLM reasoning steps per task** | 8–15 “macro” steps (each may hide micro-turns) |
| **Working memory (scratchpad)** | 4K–32K tokens per run (compressed over time) |
| **Long-term memory vectors** | 10M–1B+ embeddings (namespace/partitioned) |
| **Sandboxed code executions / day** | 1M–50M (CPU-bound; heavily quotaed) |
| **Regions** | Multi-region; data residency per tenant |

{: .note }
> Interview tip: give **ranges** and say what drives the upper bound (web research vs. local file Q&A). Numbers are illustrative; **reasoning** matters more than precision.

### Agents vs. Chatbots — Why the Design Changes

| Aspect | **Chatbot (single-turn / multi-turn chat)** | **Agent system** |
|--------|---------------------------------------------|------------------|
| **Objective** | Helpful response per message | **Task completion** with measurable outcome |
| **Control flow** | Mostly user-driven turns | **Model-driven loop** + planner |
| **External world** | Optional tools | **Tools are first-class** (search, code, APIs, browser) |
| **State** | Conversation buffer | **Scratchpad + episodic + semantic memory** |
| **Failure mode** | Wrong answer | Wrong answer **plus** runaway loops, tool abuse, cost spikes |
| **Evaluation** | Helpfulness, safety | **Task success**, tool correctness, **latency to outcome**, cost |
| **Ops** | Model serving | Serving **plus** sandboxes, **secrets**, **queues**, **human review** |

---

## Key Concepts Primer

### ReAct (Reason + Act)

**ReAct** interleaves natural-language **reasoning** (“thought”) with **actions** (tool calls) and **observations** (tool results). It reduces blind tool spamming by forcing explicit intermediate steps.

```
Thought: I need current revenue; search SEC filings.
Action: search(query="ACME 10-K revenue 2023 site:sec.gov")
Observation: [snippet with FY2023 revenue $X]

Thought: Summarize and cite.
Action: finish(answer="FY2023 revenue was $X (10-K)...")
```

### Tool Calling / Function Calling

The model emits **structured** tool invocations (often JSON) against a **schema** (name, description, parameters). The runtime **validates**, **authorizes**, **executes**, and returns **observations**. This is the **contract** between “brain” and “hands.”

### Planning

| Technique | Idea | When it shows up |
|-----------|------|------------------|
| **Chain-of-thought (CoT)** | Step-by-step deliberation in one pass | Lightweight planning inside one LLM call |
| **Tree-of-thought (ToT)** | Explore multiple plans; score/prune | Hard tasks, higher cost |
| **Dependency graph / DAG** | Subtasks with edges | Multi-step workflows, parallelization |
| **Dynamic replanning** | Revise plan after failure or new observation | Production agents (essential) |

### Memory

| Layer | Role | Typical implementation |
|-------|------|------------------------|
| **Short-term / working** | Current goal, recent observations, scratchpad | Prompt window + rolling summary |
| **Episodic** | Past runs, decisions, failures | Event log, structured traces |
| **Semantic** | Facts/skills reusable across sessions | **Vector store** + metadata filters |

### Agent Orchestration

| Pattern | Description |
|---------|-------------|
| **Single agent** | One loop with many tools — simplest |
| **Multi-agent** | Specialists (researcher, coder, critic) + coordination |
| **Hierarchical** | Manager delegates to workers; consolidates results |

### Guardrails and Sandboxing

**Guardrails** = policy checks on inputs/outputs/plans (PII, malware, jailbreaks, disallowed tools). **Sandboxing** = **isolate execution** (containers, no host FS, restricted egress) so the model never gets raw superpowers.

---

## Step 1: Requirements

### Questions to Ask the Interviewer

| Question | Why it matters |
|----------|----------------|
| What **task domains**? (research, coding, ops, customer support) | Tooling + safety + latency |
| **Untrusted code** from the model? | Full sandbox vs. read-only APIs |
| **Human approval** for which actions? (payments, deletes, external posts) | HITL gates + audit |
| **Multi-tenant**? Data isolation? | Memory partitioning, KMS, network policy |
| **SLO**: time-to-complete vs. cost? | Depth of search, model size, parallelism |
| **Evaluation**: who labels success? | Offline suites, human eval, user thumbs |

### Functional Requirements

| Requirement | Detail |
|-------------|--------|
| **Multi-step task execution** | Decompose goal → steps → execute until stop condition |
| **Tool use** | Web search, **code execution**, **HTTP/API** calls, optional **browser/computer use** |
| **Persistent memory** | Session + user/org knowledge across runs (with consent) |
| **Self-correction / retry** | Detect tool errors, parse failures, **replan** |
| **Human-in-the-loop** | Checkpoints for sensitive tools, escalation to reviewer |

### Non-Functional Requirements

| NFR | Target (example for interview) |
|-----|--------------------------------|
| **End-to-end completion** | **< 5 minutes** for “simple” tasks (P95) — define “simple” explicitly |
| **Tool call latency** | **< 2s** (P95) for typical tools excluding long-running jobs |
| **Safety** | **Sandboxed** execution; default-deny network; secrets injected by platform |
| **Cost** | **< $1 per task** average at moderate scale — combine small model for routing + large for reasoning |

{: .warning }
> Always tie NFRs to **measurement**: what is a “task,” what is included in “tool latency,” and are we counting queueing?

---

## Step 2: Estimation

### LLM Calls per Task

Assume **8–15 “reasoning steps”** per average task:

- Each step may be **1–3 LLM calls** (plan, optionally critique, format).
- **ReAct** often costs **1 call per tool round** + occasional summarization.
- **Self-correction** adds **20–40%** extra calls.

**Back-of-envelope for one task:**

```
LLM calls ≈ 10 steps × 1.5 calls/step ≈ 15 calls (median)
P90 might be 25–40 calls if searches fail or code doesn't run
```

### Tool Execution Costs

| Tool type | Dominant cost | Notes |
|-----------|---------------|-------|
| **Search** | Provider $ + your index | Cache aggressively; dedupe queries |
| **Code sandbox** | CPU seconds + image startup | Warm pools; cap runtime; binary allowlists |
| **Browser automation** | VMs + bandwidth | Slowest; queue separately |
| **Private APIs** | Latency + partner rate limits | Client-side retries with jitter |

### Memory Storage

```
Working memory: ephemeral (Redis / in-process) — MBs per active run
Episodic logs: TB-scale at large orgs — partitioned by tenant, TTL, legal hold
Vector store: embedding dim × vectors × replicas
  Example: 768-dim × 100M vectors × 4 bytes ≈ 300 GB raw (before compression)
```

{: .tip }
> Mention **compaction**: summarize old scratchpad into bullet memories → fewer tokens → lower cost and fewer errors.

---

## Step 3: High-Level Design

### System Diagram

```mermaid
flowchart LR
    U[User / Client] --> O[Orchestrator]
    O --> P[Planner LLM]
    P --> TR[Tool Router]
    TR --> T1[Search]
    TR --> T2[Code Sandbox]
    TR --> T3[HTTP / APIs]
    TR --> T4[Browser / Computer Use]
    TR --> MS[(Memory Store)]
    MS --> P
    T1 --> EV[Evaluator]
    T2 --> EV
    T3 --> EV
    T4 --> EV
    EV --> O
    O --> R[Response / Artifacts]
```

**Flow (one iteration):**

1. **Orchestrator** accepts task, loads **policy** + **memory**, initializes trace.
2. **Planner** proposes next action(s) with explicit rationale (ReAct-style).
3. **Tool Router** picks implementation, validates args, enforces auth/budget.
4. **Tools** run in **sandboxes** or through controlled gateways.
5. **Memory Store** records observations; retrieves relevant long-term facts.
6. **Evaluator** checks progress, safety, stopping criteria; may **loop** or **escalate**.

### Sequence: One ReAct Iteration (Happy Path)

```mermaid
sequenceDiagram
    participant U as User
    participant O as Orchestrator
    participant P as Planner LLM
    participant TR as Tool Router
    participant T as Tool Worker
    participant M as Memory Store
    participant E as Evaluator

    U->>O: Task + session_id
    O->>M: Load scratchpad + retrieve (optional)
    M-->>O: Context bundle
    O->>P: Prompt with tools + policy
    P-->>O: tool_call JSON
    O->>TR: validate + authorize
    TR->>T: execute (sandbox)
    T-->>TR: observation
    TR-->>O: normalized result
    O->>M: append observation / episodic event
    O->>E: progress + safety check
    alt continue
        E-->>O: continue
        O->>P: next iteration
    else finish
        E-->>O: stop
        O-->>U: final answer + citations
    end
```

{: .note }
> **Async path:** long-running tools (deep crawl, batch tests) return a **job id**; orchestrator **suspends** the run and **resumes** on webhook or poll — state machine, not a single blocking thread.

---

## Step 4: Deep Dive

### 4.1 Agent Loop and ReAct Pattern

The **Observe → Thought → Action** cycle continues until a terminal action (`finish`, `ask_human`, `abort`) or a global budget is exceeded.

```python
from dataclasses import dataclass
from typing import Any, Callable
import json
import re

Action = dict[str, Any]  # {"tool": str, "args": dict}
Observation = dict[str, Any]

@dataclass
class Step:
    thought: str
    action: Action
    observation: Observation | None = None

class ReactAgent:
    def __init__(
        self,
        llm: Callable[[str], str],
        tools: dict[str, Callable[..., Observation]],
        max_steps: int = 32,
    ):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps

    def _parse(self, text: str) -> tuple[str, Action]:
        """Very small illustrative parser — production uses JSON schema + constrained decoding."""
        m_thought = re.search(r"Thought:\s*(.+?)(?=Action:)", text, re.S | re.I)
        m_action = re.search(r"Action:\s*(\{.*\})", text, re.S | re.I)
        if not m_thought or not m_action:
            raise ValueError("Unparseable LLM output")
        thought = m_thought.group(1).strip()
        action = json.loads(m_action.group(1))
        return thought, action

    def run(self, task: str) -> list[Step]:
        history: list[Step] = []
        ctx = f"Task:\n{task}\n"
        for i in range(self.max_steps):
            prompt = (
                ctx
                + "Reply with Thought: ... and Action: {\"tool\": ..., \"args\": ...}.\n"
                + "If done, use tool \"finish\".\n"
            )
            raw = self.llm(prompt)
            thought, action = self._parse(raw)

            if action["tool"] == "finish":
                history.append(Step(thought, action, {"status": "done"}))
                return history

            tool_fn = self.tools.get(action["tool"])
            if tool_fn is None:
                obs: Observation = {"error": "unknown_tool", "tool": action["tool"]}
            else:
                try:
                    obs = tool_fn(**action.get("args", {}))
                except Exception as e:  # noqa: BLE001 — illustrative
                    obs = {"error": "tool_failure", "detail": str(e)}

            history.append(Step(thought, action, obs))
            ctx += f"\nObservation: {json.dumps(obs)}\n"

        history.append(
            Step("abort: step budget exceeded", {"tool": "abort", "args": {}}, {})
        )
        return history
```

**Structured output & retries:** Prefer **JSON schema**, **tool-calling API**, or **constrained decoding** over regex. Retry with **shorter prompt**, **error feedback**, or **fallback model** when parsing fails.

---

### 4.2 Planning and Task Decomposition

Represent the plan as a **DAG** of subtasks with **dependencies**; execute ready nodes in parallel when safe.

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

@dataclass
class Subtask:
    id: str
    title: str
    deps: set[str] = field(default_factory=set)
    status: Literal["pending", "running", "done", "failed"] = "pending"
    result: object | None = None

class PlanGraph:
    def __init__(self, nodes: Iterable[Subtask]):
        self.nodes = {n.id: n for n in nodes}

    def ready(self) -> list[Subtask]:
        out: list[Subtask] = []
        for n in self.nodes.values():
            if n.status != "pending":
                continue
            if all(self.nodes[d].status == "done" for d in n.deps):
                out.append(n)
        return out

    def replan_on_failure(self, failed_id: str, new_nodes: list[Subtask]) -> None:
        """Dynamic replanning: replace or extend subgraph after a failure."""
        self.nodes[failed_id].status = "failed"
        for nn in new_nodes:
            self.nodes[nn.id] = nn
```

**Dynamic replanning:** On **tool error** or **evaluator rejection**, patch the graph (add diagnostic subtask, switch strategy, or ask for missing credentials — via HITL).

---

### 4.3 Tool Registry and Execution

Central **registry** holds schemas, handlers, risk tiers, and timeouts.

```python
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any, Literal, TypedDict

class ToolSpec(TypedDict):
    name: str
    description: str
    json_schema: dict[str, Any]
    risk_tier: Literal["low", "medium", "high"]

class ToolRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._impl: dict[str, Any] = {}

    def register(self, spec: ToolSpec, fn: Any) -> None:
        self._specs[spec["name"]] = spec
        self._impl[spec["name"]] = fn

    def validate(self, name: str, args: dict[str, Any]) -> None:
        # Production: use jsonschema, protobuf, or pydantic
        if name not in self._specs:
            raise KeyError(name)
        # ... validate args against spec["json_schema"]

    def execute(self, name: str, args: dict[str, Any], timeout_s: float = 30.0) -> Any:
        self.validate(name, args)
        fn = self._impl[name]
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(fn, **args)
            try:
                return fut.result(timeout=timeout_s)
            except FuturesTimeout:
                return {"error": "timeout", "tool": name}
```

**Sandboxed execution:** High-risk tools run in **workers** with **seccomp/AppArmor**, **no host mounts**, **egress proxy** with allowlists, and **per-tenant** credentials injected as **environment** (never passed through model text).

---

### 4.4 Memory Architecture

| Layer | Purpose | Implementation sketch |
|-------|---------|-------------------------|
| **Working / scratchpad** | Current plan + last N observations | Ring buffer in prompt; periodic summarization |
| **Episodic** | “What happened” in past runs | Append-only event store (Kafka/BigQuery) |
| **Semantic** | Reusable facts | Embeddings + vector DB; **metadata** filters (user, project) |

```python
from collections import deque
from typing import Any, Callable, Deque

class WorkingMemory:
    def __init__(self, max_chars: int = 24000):
        self.max_chars = max_chars
        self._chunks: Deque[str] = deque()

    def append(self, text: str) -> None:
        self._chunks.append(text)
        self._compress_if_needed()

    def _compress_if_needed(self) -> None:
        body = "\n".join(self._chunks)
        while len(body) > self.max_chars and len(self._chunks) > 1:
            merged = "[SUMMARY] " + self._chunks.popleft()[:2000]
            self._chunks.appendleft(merged)
            body = "\n".join(self._chunks)

class SemanticMemory:
    def __init__(self, embed: Callable[[str], list[float]], store: Any):
        self.embed = embed
        self.store = store  # e.g., vector DB client

    def remember(self, text: str, meta: dict) -> str:
        vec = self.embed(text)
        return self.store.upsert(text=text, vector=vec, metadata=meta)

    def recall(self, query: str, k: int = 8, filters: dict | None = None) -> list[dict]:
        qv = self.embed(query)
        return self.store.search(vector=qv, k=k, filters=filters or {})
```

**Retrieval hygiene:** Deduplicate, **time-decay**, and **ground** answers with citations from tool outputs — not from unchecked memory.

---

### 4.5 Multi-Agent Orchestration

**Supervisor pattern:** a **manager** model delegates to **specialists** with narrow tools; aggregates results.

```python
import queue
import threading
from typing import Any, Callable, Literal, TypedDict

class Message(TypedDict):
    from_agent: str
    to_agent: str
    kind: Literal["task", "result", "question"]
    payload: dict[str, Any]

class Bus:
    """Minimal message bus — production uses durable queues + idempotency keys."""

    def __init__(self) -> None:
        self._q: dict[str, queue.Queue[Message]] = {}

    def register(self, agent_id: str) -> None:
        self._q.setdefault(agent_id, queue.Queue())

    def send(self, m: Message) -> None:
        self._q[m["to_agent"]].put(m)

    def recv(self, agent_id: str, timeout: float | None = None) -> Message:
        return self._q[agent_id].get(timeout=timeout)

def supervisor_run(
    user_task: str,
    workers: dict[str, Callable[[str], dict]],
    resolve_conflict: Callable[[list[dict]], dict],
) -> dict:
    partials: list[dict] = []
    threads = []

    def run_worker(wid: str, fn: Callable[[str], dict]) -> None:
        partials.append(fn(user_task))

    for wid, fn in workers.items():
        t = threading.Thread(target=run_worker, args=(wid, fn))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    return resolve_conflict(partials)
```

**Parallelism:** Only where **dependencies allow**; otherwise race conditions duplicate work. **Conflict resolution** = structured merge (critic model, voting, or deterministic rules).

**Agent communication protocol:** versioned **message schema**, **correlation IDs**, **capability advertisement** (`tools_i_support`), and **deny-by-default** for cross-agent actions.

---

### 4.6 Safety and Sandboxing

| Control | Mechanism |
|---------|-----------|
| **Code execution** | **gVisor/Firecracker** VMs or hardened containers; non-root; read-only FS except `/tmp` |
| **Filesystem** | **No host paths**; ephemeral volumes; size quotas |
| **Network** | **Egress proxy**; domain allowlists per tool; block metadata endpoints |
| **Cost limits** | Per-task **token budget**, **tool count**, **wall-clock** deadline |
| **Human gates** | Workflow for **high-risk** tool tier (e.g., send email, spend money) |

```python
from dataclasses import dataclass
from enum import Enum

class RiskTier(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3

@dataclass
class Policy:
    allow_tiers: set[RiskTier]
    require_human_for: set[str]

def gate(tool_name: str, tier: RiskTier, policy: Policy, human_approved: bool) -> bool:
    if tool_name in policy.require_human_for and not human_approved:
        return False
    return tier in policy.allow_tiers
```

{: .warning }
> **Never** let the model receive raw OAuth tokens. Use **platform-mediated** auth with **scoped**, **short-lived** credentials.

---

## Step 5: Scaling & Production

### Failure Handling

| Failure | Mitigation |
|---------|------------|
| **Tool timeout** | Retry with backoff; switch strategy; partial results |
| **Bad JSON / schema** | Repair prompt; constrained decoding; smaller “formatter” model |
| **Sandbox crash** | Fresh VM; deterministic replay from last safe checkpoint |
| **Upstream LLM outage** | Fallback model; degrade to retrieve-only mode |
| **User changes goal mid-flight** | **Interrupt** signal; snapshot memory; clarify |

### Monitoring

- **Trace IDs** per agent run (spans: plan, tool, memory, eval).
- Metrics: **steps/run**, **tool error rate**, **tokens/task**, **$ / task**, **human review rate**.
- **Safety** alerts: jailbreak patterns, exfiltration attempts, unusual egress.

### Trade-offs

| Choice | Upside | Downside |
|--------|--------|----------|
| **Single agent** | Simpler, fewer coordination bugs | Weaker specialization |
| **Multi-agent** | Parallelism, modularity | Higher cost, deadlock/conflict risk |
| **Big model everywhere** | Higher reasoning quality | Cost + latency |
| **Small router + big worker** | Cheaper routing | Router errors |
| **Long scratchpad** | More context | Drift, distraction, $$ |
| **Heavy sandbox** | Strong security | Cold start, throughput limits |

### Capacity, Queues, and Backpressure

Agent workloads are **spiky**: one task may enqueue **dozens** of tool calls while another stays “think-only.” Treat **orchestrator**, **tool pools**, and **LLM inference** as **separate** scaled services.

| Component | Scaling knob | Backpressure signal |
|-----------|--------------|---------------------|
| **Planner LLM** | GPU replicas, max concurrent generations | Queue depth, TTFT SLO breach |
| **Tool workers** | Horizontal pods + **warm pools** for sandboxes | P95 queue wait > threshold |
| **Search / browser** | Rate limits per tenant + **token bucket** | 429s, shed low-priority tasks |
| **Memory / vector DB** | Read replicas, partition by tenant | Retrieval latency SLO |

{: .warning }
> Without **per-tenant concurrency caps**, one customer can **starve** others — classic noisy-neighbor, worse than REST APIs because agent runs are **long** and **stateful**.

### Idempotency and Side Effects

Tools that **mutate** the world (tickets, payments, PR merges) must use **idempotency keys** generated by the **platform**, not the model:

```
Client request → Orchestrator assigns run_id
Each side-effecting tool call carries:
  Idempotency-Key: sha256(run_id + step_index + tool_name + canonical_args)
```

**Retries** after timeouts must not **double-charge** or **double-post**. Read-only tools can retry freely.

### State Storage Model (Sketch)

| Entity | Stored where | Retention |
|--------|--------------|-----------|
| **AgentRun** | OLTP DB (Postgres/Cockroach) | 30–90 days hot |
| **Step / ToolEvent** | Append log + object store for payloads | Compliance-driven |
| **Scratchpad blob** | Redis / in-memory with spill | TTL = session |
| **Semantic vectors** | Vector index per region | User-controlled delete |

### Evaluation in Production

| Layer | What you measure | How |
|-------|------------------|-----|
| **Task success** | Did the user get the outcome? | Rubric + binary checks on artifacts |
| **Tool accuracy** | Correct tool + args | Offline traces with labels |
| **Safety** | Policy violations | Red-team suites, canary prompts |
| **Efficiency** | Steps, tokens, $ | Automatic from telemetry |
| **Human burden** | % escalations | Queue depth + reviewer time |

**Regression gates:** ship planner/router changes only if **offline** suite and **shadow** traffic show **no** degradation on **success rate** and **safety** metrics.

### Deployment Topology (Multi-Region Sketch)

```mermaid
flowchart TB
    subgraph region_eu[Region EU]
        OE[Orchestrator]
        PE[Planner pool]
        TE[Tool workers]
        ME[(Memory EU)]
    end
    subgraph region_us[Region US]
        OU[Orchestrator]
        PU[Planner pool]
        TU[Tool workers]
        MU[(Memory US)]
    end
    G[Global DNS / LB] --> region_eu
    G --> region_us
```

**Sticky routing** per tenant keeps **memory** and **compliance** coherent; **async** cross-region replication only where policy allows.

---

## Interview Tips

{: .tip }
> **Strong answers** separate **planning**, **tool execution**, **memory**, and **governance** — and show how they fail independently without taking down the whole platform.

**Do:**

- Draw the **loop** and where **state** lives.
- Name **concrete tools** and **sandbox** boundaries.
- Discuss **evaluation**: task success, tool accuracy, **human audit** sampling.
- Address **cost** (model tiers, caching search, summarization).

**Don’t:**

- Hide everything behind “the LLM will figure it out.”
- Ignore **timeouts**, **retries**, and **malicious prompts**.
- Conflate **RAG** with **agent memory** without retrieval strategy.

---

## Hypothetical Interview Transcript

**Setting:** 45-minute system design. **Candidate** = you. **Interviewer** = Staff Engineer, **DeepMind-adjacent agents** team (fictionalized). Focus: **architecture, planning, tools, memory, safety, evaluation, multi-agent**.

---

**[00:00] Interviewer:**  
Thanks for joining. The question is: **design a production AI agent system** that helps users complete multi-step tasks — research, code changes, API calls — not just chat. You have the whiteboard; start with requirements.

**[00:45] Candidate:**  
I’ll clarify scope first. **Domains** — general web research + code execution, or also **enterprise SSO** tools? **Tenancy** — single-user vs. B2B with isolation? **Risk posture** — can the agent **send email** or open PRs without a human? And **SLOs** — target time-to-complete and cost per task?

**[01:20] Interviewer:**  
Good. Assume **B2B**, strong isolation, **default-deny** for anything externally visible unless approved. Target **< 5 minutes** for simple tasks, **tool latency** in the **low seconds** excluding long jobs. **Cost** should stay roughly **under a dollar** for typical tasks at moderate scale.

**[02:00] Candidate:**  
Functional requirements: **multi-step execution**, **tools** for search, **sandboxed code**, **HTTP APIs** with platform-managed auth, **persistent memory** per user/org with consent, **self-correction** when tools fail, and **human-in-the-loop** for high-risk actions. Non-functional: **latency**, **safety**, **cost**, **availability** of orchestration — the LLM can degrade if we still return partial results.

**[02:40] Interviewer:**  
How is this different from a chatbot?

**[03:00] Candidate:**  
A chatbot optimizes **next response quality**. An agent optimizes **task outcome** — so we need an **explicit control loop**, **structured actions**, **environment feedback**, and **termination**. Failure modes include **infinite loops** and **runaway spend**, which we don’t treat as second-class.

**[03:30] Interviewer:**  
Walk me through the high-level architecture.

**[04:10] Candidate:**  
**Client** hits an **Orchestrator** that owns state machine + budgets. **Planner LLM** proposes the next step in a **ReAct** style — thought + action. **Tool Router** validates and dispatches to **Search**, **Code Sandbox**, **API Gateway**, maybe **Browser** workers. Observations go to **Memory Store** — scratchpad + episodic logs + semantic retrieval. An **Evaluator** checks progress and safety; loop until finish or escalation.

**[05:10] Interviewer:**  
Where would you put **caching**?

**[05:30] Candidate:**  
**Search** results and **page fetches** — cache normalized by URL with TTL. **Embeddings** for memory lookups — cache query vectors per session. **Not** caching raw LLM outputs for tool actions unless idempotent and policy-safe.

**[06:00] Interviewer:**  
Describe the **agent loop** more concretely.

**[06:40] Candidate:**  
Each iteration: append prior observations to context — sometimes **summarized** to save tokens. Planner emits a **tool call** with schema-validated JSON. Router enforces **auth**, **risk tier**, and **timeout**. Observation returns; **evaluator** decides continue vs. **replan** vs. **HITL**. We cap **steps** and **tokens** globally.

**[07:30] Interviewer:**  
How do you handle **malformed tool calls**?

**[08:00] Candidate:**  
Prefer **native tool-calling** with JSON schema. On failure: **one repair attempt** with the error, then **fallback** to a smaller model that only reformats. If still bad, **skip** and log; don’t silently execute.

**[08:40] Interviewer:**  
**Planning** — do you use a DAG, or free-form ReAct?

**[09:20] Candidate:**  
Both layers. **DAG** for tasks with clear decomposition — e.g., parallelizable research sub-questions. **ReAct** inside each node for flexibility. On failure, **replan** — maybe add a diagnostic subtask or switch from browsing to API if available.

**[10:10] Interviewer:**  
What if the planner **hallucinates** a dependency that doesn’t exist?

**[10:40] Candidate:**  
**Evaluator** checks against **tool registry** — unknown tools are rejected. For **data dependencies**, we only mark tasks **ready** when **deps** succeeded; failed nodes trigger **replan**. We can add a **critic** pass before execution for high-risk tiers.

**[11:30] Interviewer:**  
Deep dive on **sandboxing** for code execution.

**[12:20] Candidate:**  
Run in **minimal container** or **microVM**, **non-root**, **read-only** image, **writable /tmp** only with size cap. **Network** via **egress proxy** with per-tenant allowlists. **No secrets** in the prompt — **sidecar** injects short-lived tokens. **CPU/time** quotas; **kill** on violation.

**[13:20] Interviewer:**  
Model suggests `curl http://169.254.169.254` — what happens?

**[13:50] Candidate:**  
**Blocked** at proxy — metadata endpoints on the **deny** list. Alert + **trace flag**; possibly **user/org** policy review if repeated.

**[14:30] Interviewer:**  
How do you do **browser automation** safely?

**[15:10] Candidate:**  
Isolated **browser pods**, **no** clipboard to internal networks, **download** scanning, **URL allowlists** for sensitive flows. Prefer **APIs** over raw browsing when possible — faster and more stable.

**[16:00] Interviewer:**  
**Memory** — what do you store in vectors vs. logs?

**[16:40] Candidate:**  
**Episodic** — structured events for audit and debugging. **Semantic** — embeddings of **facts** the user/org opted into, with metadata for ACLs. **Scratchpad** — short-term; **compressed** summaries when long. **Ground** answers with fresh tool output when correctness matters.

**[17:30] Interviewer:**  
How do you avoid **stale** memory poisoning answers?

**[18:00] Candidate:**  
**TTL**, **source tags**, **confidence**, and **mandatory** tool refresh for **time-sensitive** queries. Show **citations** to user-facing artifacts.

**[18:40] Interviewer:**  
**Multi-agent** — when would you split agents?

**[19:20] Candidate:**  
When **skills** and **tooling** differ — researcher vs. coder vs. **verifier**. Use a **supervisor** with a **message bus**, **schema-versioned** messages, **idempotent** tasks, and **conflict resolution** — often a **critic** or **deterministic merge** for code.

**[20:10] Interviewer:**  
Deadlocks?

**[20:40] Candidate:**  
**Timeouts** on waits, **single-owner** per subtask, **escalate** to supervisor with partial results. Avoid **cyclic** asks between agents by design.

**[21:20] Interviewer:**  
**Evaluation** — how do you know the system works?

**[22:00] Candidate:**  
**Offline** task suites with golden checks; **tool** correctness metrics; **human** review on sampled traces; **online** user success signals. Separate **safety** evals for prompt injections and exfiltration.

**[22:50] Interviewer:**  
Say more about **injection** via tools.

**[23:30] Candidate:**  
Treat tool outputs as **untrusted**. **Sanitize** HTML, **block** script execution in preview panes, **separate** secrets from LLM context. **Monitor** for **data exfil** patterns in outbound calls.

**[24:10] Interviewer:**  
**Cost** control in one sentence?

**[24:40] Candidate:**  
**Cap** steps/tokens, **cache** search, **route** easy tasks to smaller models, and **summarize** aggressively — dollars are part of the SLA.

**[25:20] Interviewer:**  
How would you **roll out** safely?

**[26:00] Candidate:**  
**Shadow mode** — plan without executing. **Canary** tenants, **feature flags** per tool, **kill switch** for high-risk tools. **Progressive** memory retention defaults.

**[26:50] Interviewer:**  
**Observability** — what’s on your dashboard?

**[27:30] Candidate:**  
**p95 latency** per tool, **error taxonomy**, **tokens/task**, **$ / task**, **HITL rate**, **sandbox crash rate**, **policy denials**.

**[28:10] Interviewer:**  
**Failure** scenario: search API is slow — how does UX look?

**[28:50] Candidate:**  
**Streaming** status to user, **async** partial results, **deadline** with fallback strategies — narrower query, different provider, or ask user to **refine**. Never spin silently.

**[29:30] Interviewer:**  
**Fairness** question: cheaper tenants could starve if noisy neighbors burn GPUs — mitigation?

**[30:10] Candidate:**  
**Fair queuing**, **per-tenant** concurrency caps, **preemption** of long low-priority jobs, **autoscale** orchestrator workers separately from **GPU** inference pools.

**[30:50] Interviewer:**  
If you had **one** extra component budget, what would you add?

**[31:30] Candidate:**  
A **learned critic / reward model** for **stopping** and **choosing tools** — reduces wasted steps. Still keep **rules** for safety.

**[32:10] Interviewer:**  
Push: reward models drift — concern?

**[32:50] Candidate:**  
**Continuous** eval on **frozen** suites, **shadow** comparisons, **rollback** on regression — same as any ML service.

**[33:30] Interviewer:**  
How does **data residency** affect memory?

**[34:10] Candidate:**  
**Region-scoped** vector indices and **KMS** keys; **no** cross-region replication for regulated data unless allowed. Router pins **storage** to tenant region.

**[34:50] Interviewer:**  
**Open question:** would you expose **raw chain-of-thought** to end users?

**[35:30] Candidate:**  
**No** by default — **summaries** for UX; **full traces** for **enterprise audit** roles with scrubbing. Raw chains can leak **secrets** or **policy**.

**[36:10] Interviewer:**  
**Multi-modal** later — what changes?

**[36:50] Candidate:**  
**Tool** surface adds OCR/screenshot understanding; **bigger** contexts; **different** sandboxing for **media** processing — still **quota** and **virus** scanning.

**[37:30] Interviewer:**  
Wrap up — **three** key risks you’d report to leadership.

**[38:10] Candidate:**  
**Safety** of tool egress, **cost** runaway at scale, **eval** gaps on long-horizon tasks — propose **tiered** autonomy and **human** oversight where ROI is unclear.

**[38:50] Interviewer:**  
Solid. We have a few minutes — **your questions**.

**[39:15] Candidate:**  
How do you **trade off** research freedom vs **production** safety when adding a new tool surface — what is the **release gate**?

**[39:45] Interviewer:**  
**Offline** red-team + **shadow** execution, **tiered** rollout, and **hard** kill switches. **Irreversible** side effects default to **HITL** until we have **proven** low error rates on **frozen** eval suites.

**[40:15] Candidate:**  
Do you **centralize** scratchpad **compression** and **token accounting**, or is it every team for themselves?

**[40:45] Interviewer:**  
**Platform** owns **budgets**, **telemetry**, and **hooks** for summarization; teams own **prompt** details. Central enforcement prevents **silent** cost blowups — we’ve seen that movie.

**[41:15] Candidate:**  
Last one: **biggest** lesson from **real** incidents?

**[41:40] Interviewer:**  
**Untrusted** tool outputs are as dangerous as untrusted user inputs — **treat observations like MIME from the internet**. **Second:** **idempotency** everywhere; agents **retry** aggressively.

**[42:10] Interviewer:**  
We’re at time. Thanks — this was a strong **systems** discussion; next round will go deeper on **one** area you touched — often **sandboxing** or **eval**.

---

{: .note }
> This transcript is a **study aid**, not a verbatim Google interview. Use it to practice **structured** answers and **trade-offs**, not to memorize lines.

---

## Quick Reference Card

| You must mention | One-liner |
|------------------|-----------|
| **ReAct / loop** | Thought + action + observation until stop |
| **Tool schema** | Validate, authorize, timeout, parse |
| **Memory** | Scratchpad + episodic + semantic (with ACL + TTL) |
| **Sandbox** | VMs/containers + egress control + no secrets in prompts |
| **Multi-agent** | Supervisor + bus + conflict resolution |
| **Eval** | Offline tasks + live metrics + human audit |

---

## Further Reading (Conceptual)

- **ReAct** pattern: interleave reasoning traces with grounded tool use (paper-style framing for interviews).  
- **Function / tool calling** in modern LLM APIs: JSON schema, parallel calls, refusal handling.  
- **MicroVM** isolation (**Firecracker**, **gVisor**) for running **untrusted** code from model-generated programs.  
- **Distributed systems** primitives — **timeouts**, **idempotency**, **backpressure**, **bulkheads** — agent fleets stress each of these harder than stateless APIs.  
- **Multi-agent** coordination: message **schemas**, **supervisor** patterns, and **conflict resolution** (avoid “chatroom of models” without contracts).  
- **Evaluation**: task rubrics, **tool-use** accuracy, **long-horizon** success — expect interview follow-ups on **what you measure** and **how you regress**.

### Mental Model (One Diagram)

```mermaid
flowchart TB
    subgraph control[Control Plane]
        POL[Policy + Budgets]
        REG[Tool Registry]
        EV[Evaluator]
    end
    subgraph data[Data Plane]
        PL[Planner LLM]
        TW[Tool Workers]
        MEM[Memory]
    end
    POL --> PL
    REG --> TW
    PL <--> TW
    TW --> MEM
    MEM --> PL
    EV --> PL
```

{: .tip }
> In interviews, **point** to **control plane vs data plane** — it signals you know how to **operate** the system, not only draw boxes.
