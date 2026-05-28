# Chapter 6: Agents and Tool Use

The word "agent" has been stretched to mean everything from a chatbot with a system prompt to a fully autonomous coding assistant that spins up cloud infrastructure. That ambiguity costs teams real money -- they either over-build when a simple chain would suffice, or under-build when genuine autonomy is required. This chapter cuts through the noise. We will define what an agent actually is, walk through the core patterns for building them, cover the protocols that connect them to the outside world (including MCP), and establish the guardrails that keep them from going off the rails in production.

---

## 6.1 What Is an Agent

An agent is an LLM that decides its own control flow. That is the entire distinction. A chain executes a fixed sequence of steps. An agent observes, reasons about what to do next, takes an action, and loops until it decides the task is done.

The four components:

- **LLM backbone** -- the reasoning engine that interprets context and decides what to do next.
- **Reasoning** -- the model's ability to break a problem into steps, evaluate progress, and adjust course.
- **Tools** -- functions the agent can invoke to interact with the outside world (APIs, databases, file systems, search engines).
- **Loop** -- the execution cycle that feeds tool results back into the model and repeats until a termination condition is met.

Strip away any one of these and you have something else. An LLM with tools but no loop is function calling. An LLM with a loop but no tools is chain-of-thought with retries. An LLM with neither is just a prompt.

---

## 6.2 The ReAct Pattern

ReAct (Reason + Act) is the foundational agent loop. The model alternates between reasoning about the current state, taking an action, and observing the result. Here is a concrete trace for a task: "What is the current stock price of NVIDIA and how does it compare to one year ago?"

```
Thought: I need the current NVIDIA stock price. I will call the stock_price tool.
Action: stock_price(symbol="NVDA")
Observation: {"symbol": "NVDA", "price": 132.65, "currency": "USD", "timestamp": "2025-05-28T14:30:00Z"}

Thought: Now I need the price from one year ago. I will call the historical_price tool.
Action: historical_price(symbol="NVDA", date="2024-05-28")
Observation: {"symbol": "NVDA", "price": 95.20, "currency": "USD", "date": "2024-05-28"}

Thought: I have both prices. The current price is $132.65 and the price one year ago was
$95.20. That is a gain of approximately 39.3%. I can now answer the user.
Action: respond("NVIDIA (NVDA) is currently trading at $132.65. One year ago it was $95.20,
representing an increase of approximately 39.3%.")
```

Each iteration feeds the observation back into the model's context, so the model builds up a running record of what it has tried and learned. The loop terminates when the model decides to respond directly rather than invoke another tool.

The strength of ReAct is its transparency. Every step is legible, which makes debugging straightforward. The weakness is token cost -- the full trace lives in context, and complex tasks can burn through context windows fast.

---

## 6.3 Tool and Function Calling

Tools are how agents interact with the world. The model does not execute code directly; it emits structured requests that your application intercepts and routes to the appropriate function.

### Defining Tools with JSON Schema

Every major model provider uses a variant of JSON Schema to describe available tools. Here is a well-designed tool definition:

```json
{
  "name": "search_orders",
  "description": "Search customer orders by order ID, email, or date range. Returns up to 20 matching orders with status and line items.",
  "parameters": {
    "type": "object",
    "properties": {
      "order_id": {
        "type": "string",
        "description": "Exact order ID (e.g., ORD-20250528-1234)"
      },
      "customer_email": {
        "type": "string",
        "description": "Customer email address for lookup"
      },
      "date_from": {
        "type": "string",
        "description": "Start date in YYYY-MM-DD format"
      },
      "date_to": {
        "type": "string",
        "description": "End date in YYYY-MM-DD format"
      },
      "status": {
        "type": "string",
        "enum": ["pending", "shipped", "delivered", "cancelled"],
        "description": "Filter by order status"
      }
    },
    "required": []
  }
}
```

### Tool Design Principles

1. **Write descriptions for the model, not for humans.** The description is a prompt. Tell the model exactly when to use this tool and what it returns. Vague descriptions like "handles orders" lead to misuse.

2. **One tool, one job.** A `search_orders` tool should not also create orders. Split operations into separate tools so the model can reason about which action to take.

3. **Predictable output shape.** Always return the same structure. If a search returns no results, return an empty array, not null, not an error string, not a different schema.

4. **Constrain inputs with enums and formats.** The JSON Schema is your first line of defense. Use `enum` for categorical values, `pattern` for formatted strings, and `minimum`/`maximum` for numeric bounds.

5. **Fail loudly.** When a tool call fails, return a clear error message the model can reason about: `{"error": "Order ORD-123 not found"}` is far more useful than a 500 stack trace.

### Calling Tools in Practice (Python, OpenAI SDK)

```python
import openai
import json

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_orders",
            "description": "Search customer orders by order ID, email, or date range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "customer_email": {"type": "string"},
                    "status": {"type": "string", "enum": ["pending", "shipped", "delivered", "cancelled"]}
                }
            }
        }
    }
]

def agent_loop(user_message: str, max_iterations: int = 10):
    messages = [{"role": "user", "content": user_message}]

    for _ in range(max_iterations):
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
        )
        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            return msg.content  # Agent decided to respond

        for call in msg.tool_calls:
            result = execute_tool(call.function.name, json.loads(call.function.arguments))
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result),
            })

    return "Agent reached maximum iterations without completing the task."
```

---

## 6.4 Planning and Decomposition

How much planning should an agent do before acting? There are three approaches, and the right one depends on task complexity.

**No planning (direct ReAct).** The agent reasons one step at a time. Best for simple, well-scoped tasks with 1-3 tool calls. Adding a planning step to "look up this customer's last order" just wastes tokens.

**Plan-then-execute.** The agent generates a full plan before taking any action, then executes steps sequentially. Works well when the task is predictable and the steps are mostly independent. Risk: the plan goes stale if early steps return unexpected results.

```
Plan:
1. Search for customer by email
2. Retrieve their last 5 orders
3. Check refund eligibility for the most recent order
4. Summarize findings for the user

Executing step 1...
```

**Iterative planning.** The agent creates an initial plan, executes a few steps, then re-plans based on what it has learned. This is the most robust approach for complex tasks but also the most expensive. Use it when the problem space is genuinely uncertain -- research tasks, multi-system debugging, open-ended analysis.

The choice is a cost-quality trade-off. Start with no planning. Upgrade to plan-then-execute when you see agents floundering on multi-step tasks. Reserve iterative planning for your hardest workflows.

---

## 6.5 Memory Systems

An agent without memory is stateless between invocations. For non-trivial applications, you need to think about three layers.

**Short-term memory (conversation context).** This is the message history within a single session. It is the simplest form of memory and is limited by the model's context window. Manage it with summarization or sliding-window truncation when conversations get long.

**Working memory (scratchpad).** A structured space where the agent tracks intermediate state during a task. This can be as simple as a JSON object the agent updates at each step:

```python
scratchpad = {
    "goal": "Resolve customer billing dispute",
    "findings": [],
    "pending_actions": ["check_payment_history", "review_invoice"],
    "completed_actions": []
}
```

The scratchpad is injected into the system prompt at each iteration. It keeps the agent oriented without relying on the model to parse the full conversation history.

**Long-term memory (persistent store).** Facts, preferences, and outcomes that persist across sessions. Implementation options range from a simple key-value store to a vector database for semantic retrieval:

```python
# Store a memory after resolving a task
memory_store.upsert(
    user_id="cust_123",
    content="Customer prefers email communication. Has a history of disputing shipping charges.",
    metadata={"type": "preference", "source": "support_session_456"}
)

# Retrieve relevant memories at the start of a new session
memories = memory_store.search(
    query="customer billing dispute",
    user_id="cust_123",
    top_k=5
)
```

The practical rule: start with conversation context only. Add a scratchpad when tasks exceed 5-6 tool calls. Add long-term memory only when you have a clear retrieval use case and can measure whether the recalled context actually improves outcomes.

---

## 6.6 Multi-Agent Architectures

When a single agent's tool set or reasoning scope becomes too large, you split it into multiple agents. Four patterns dominate.

**Supervisor.** A central agent receives the task, delegates subtasks to specialist agents, and synthesizes their outputs. Good for customer service (routing to billing, shipping, or technical agents) and workflows where one entity needs to maintain global state.

**Debate / Critique.** Two or more agents review each other's work. One drafts, another critiques, the first revises. Effective for content generation, code review, and any task where a second perspective catches errors. Cost scales linearly with the number of review rounds.

**Pipeline.** Agents execute in sequence, each transforming the output of the previous one. Research agent gathers data, analysis agent interprets it, writing agent produces the report. Simple to reason about and debug, but rigid -- a failure in one stage blocks the whole pipeline.

**Swarm.** Agents operate semi-independently on subtasks, coordinating through shared state (a message queue, a shared document, a database). Best for parallelizable work like processing a batch of documents or monitoring multiple data streams. Hardest to debug because control flow is emergent.

Choose the simplest architecture that handles your task. Most production agent systems are either single agents or supervisor patterns. Swarms are powerful but operationally complex -- treat them as a last resort.

---

## 6.7 When NOT to Use Agents

Agents are the most complex pattern in the LLM toolkit. Complexity has costs: latency, token spend, unpredictable behavior, and debugging difficulty. Use this decision framework:

**Use simple prompting when:** the task has a single, well-defined input and output. Summarization, classification, extraction, reformatting. No external data needed.

**Use RAG when:** the model needs access to your data but the retrieval-then-generate flow is fixed. Question answering over documents, knowledge base search, contextual chat.

**Use agents when:** the task requires multiple steps that depend on intermediate results, the number of steps is not known in advance, and the model needs to make decisions about which tools to call and in what order.

The agent decision is where the build trap hits hardest. Agents are impressive to demo and satisfying to build -- which is exactly why teams reach for them when a three-line prompt would do the job. Before you wire up an agent loop, ask: is the complexity earning its keep, or is it hiding the lack of a clear user need?

Concrete signals that you do NOT need an agent:

- You can write the tool call sequence on a whiteboard before runtime.
- The "loop" always runs exactly the same number of iterations.
- The task does not require conditional branching based on tool outputs.
- Your users are waiting synchronously and need sub-second responses.

If all of these are true, a chain or a simple function-calling pass is the right tool.

---

## 6.8 Frameworks

| Criteria | LangChain / LangGraph | LlamaIndex | AutoGen | CrewAI | Custom |
|---|---|---|---|---|---|
| **Best for** | General-purpose chains and agents | Data-centric RAG and agents | Multi-agent research | Role-based multi-agent teams | Full control, minimal dependencies |
| **Learning curve** | Moderate-high (large API surface) | Moderate | Moderate | Low-moderate | High (you build everything) |
| **Abstraction level** | High | High | High | Very high | None |
| **Flexibility** | Good with LangGraph | Moderate | Good | Limited | Total |
| **Debugging** | Improving (LangSmith) | Moderate | Moderate | Difficult | You own it |
| **Production readiness** | Mature | Mature | Maturing | Early | Depends on your team |
| **Lock-in risk** | Moderate | Moderate | Moderate | High | None |
| **When to avoid** | When you need tight control over every LLM call | When your task is not data retrieval | When you need a simple single agent | When you need custom coordination logic | When time-to-prototype matters more than control |

The honest recommendation: start with raw SDK calls (OpenAI, Anthropic, etc.) and build the minimal loop yourself. You will understand what is happening at every step. Adopt a framework only when you find yourself re-implementing something it provides well -- observability, complex graph execution, or multi-agent coordination. The worst outcome is debugging a framework abstraction you do not understand on top of model behavior you do not understand.

---

## 6.9 Human-in-the-Loop Patterns

Fully autonomous agents are appropriate for low-stakes, reversible tasks. For everything else, you need human checkpoints.

**Approval gates.** The agent pauses before executing high-impact actions (deleting data, sending emails, making purchases) and presents its plan to a human for approval. Implementation is straightforward: check a policy table before executing any tool call.

```python
REQUIRES_APPROVAL = {"send_email", "delete_record", "process_refund", "deploy_service"}

async def execute_with_approval(tool_name: str, args: dict) -> dict:
    if tool_name in REQUIRES_APPROVAL:
        approved = await request_human_approval(
            action=tool_name,
            args=args,
            reason=f"Agent wants to execute {tool_name} with {args}"
        )
        if not approved:
            return {"error": "Action rejected by human reviewer."}
    return execute_tool(tool_name, args)
```

**Confidence thresholds.** The agent self-reports confidence. Below a threshold, it escalates to a human rather than guessing. This works best when you fine-tune or prompt the model to output calibrated confidence scores alongside its decisions.

**Escalation paths.** Define explicit conditions under which the agent hands off to a human entirely: repeated failures, user frustration signals, tasks outside its defined scope. The agent should explain what it tried and what it learned before handing off, so the human does not start from scratch.

---

## 6.10 Safety and Guardrails

Production agents need hard limits. Hope is not a safety strategy.

**Tool call limits.** Cap the number of tool calls per session. An agent stuck in a loop will drain your budget. Typical limits: 10-25 calls for focused tasks, 50-100 for complex research. Terminate with a clear message when the limit is hit.

**Budget caps.** Track token usage per session and per user. Set hard ceilings. When a budget cap is reached, the agent should summarize its progress and stop, not silently fail.

**Output validation.** Validate tool call arguments before execution. Validate tool results before feeding them back to the model. Treat every boundary between the model and external systems as a trust boundary.

```python
def validate_tool_call(name: str, args: dict) -> bool:
    schema = TOOL_SCHEMAS.get(name)
    if not schema:
        return False
    try:
        jsonschema.validate(args, schema)
        return True
    except jsonschema.ValidationError:
        return False
```

**Sandboxing.** If the agent can execute code or modify files, run those operations in a sandboxed environment (containers, VMs, restricted file system permissions). Never give an agent write access to production databases through the same credentials your application uses.

**Audit logging.** Log every tool call, every argument, every result. This is non-negotiable for debugging, compliance, and understanding agent behavior over time.

---

## 6.11 Tool Protocols: Model Context Protocol (MCP)

### What MCP Is

The Model Context Protocol is an open standard created by Anthropic that defines how AI applications connect to external data sources and tools. Think of it as USB-C for AI integrations: a single protocol that replaces the need to build custom connectors for every tool and every model.

Before MCP, connecting an LLM to your database required writing custom integration code. Connecting it to your CRM required different custom code. Every new tool meant another bespoke integration. MCP standardizes this into a client-server architecture with a well-defined contract.

### Core Concepts

MCP defines three primitives:

**Resources** -- read-only data that the AI can access. A file's contents, a database query result, a configuration document. Resources are identified by URIs (`file:///path/to/doc.txt`, `postgres://db/customers`). They are pulled into context, not executed.

**Tools** -- actions that the AI can invoke to produce side effects. Sending an email, creating a record, running a query. Tools have typed input schemas and return structured results. The model decides when to call them.

**Prompts** -- reusable prompt templates that servers can expose. A server might offer a "summarize_table" prompt that takes a table name and returns a well-structured summary prompt. These are optional and less commonly used than resources and tools.

### Architecture

MCP uses JSON-RPC 2.0 as its wire protocol, transported over one of two channels:

- **stdio** -- the MCP server runs as a subprocess. The host application communicates via stdin/stdout. Best for local tools (file system access, local databases, CLI tools).
- **Streamable HTTP** -- the MCP server exposes an HTTP endpoint. The client connects over the network. Required for remote servers and multi-user deployments.

The architecture has three layers:

- **Host** -- the application the user interacts with (Claude Desktop, Cursor, your custom app).
- **Client** -- a protocol client that maintains a 1:1 connection with a server. The host creates one client per server.
- **Server** -- a lightweight process that exposes resources, tools, and prompts for a specific integration.

### Building an MCP Server

**TypeScript (using the official SDK):**

```typescript
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const server = new McpServer({
  name: "inventory-server",
  version: "1.0.0",
});

server.tool(
  "check_inventory",
  "Check current inventory level for a product SKU",
  { sku: z.string().describe("Product SKU (e.g., WIDGET-001)") },
  async ({ sku }) => {
    const stock = await db.query("SELECT quantity FROM inventory WHERE sku = $1", [sku]);
    return {
      content: [{ type: "text", text: JSON.stringify(stock.rows[0] ?? { error: "SKU not found" }) }],
    };
  }
);

server.tool(
  "update_inventory",
  "Adjust inventory quantity for a product SKU",
  {
    sku: z.string(),
    adjustment: z.number().describe("Positive to add stock, negative to remove"),
  },
  async ({ sku, adjustment }) => {
    const result = await db.query(
      "UPDATE inventory SET quantity = quantity + $1 WHERE sku = $2 RETURNING quantity",
      [adjustment, sku]
    );
    return {
      content: [{ type: "text", text: JSON.stringify(result.rows[0]) }],
    };
  }
);

const transport = new StdioServerTransport();
await server.connect(transport);
```

**Python (using the official SDK):**

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("inventory-server")

@mcp.tool()
async def check_inventory(sku: str) -> str:
    """Check current inventory level for a product SKU."""
    row = await db.fetchone("SELECT quantity FROM inventory WHERE sku = $1", sku)
    return json.dumps(row or {"error": "SKU not found"})

@mcp.tool()
async def update_inventory(sku: str, adjustment: int) -> str:
    """Adjust inventory quantity. Positive to add, negative to remove."""
    row = await db.fetchone(
        "UPDATE inventory SET quantity = quantity + $1 WHERE sku = $2 RETURNING quantity",
        adjustment, sku
    )
    return json.dumps(row)

mcp.run()
```

### Connecting to Claude Desktop and Cursor

Configuration is declarative. For Claude Desktop, add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "inventory": {
      "command": "node",
      "args": ["./build/inventory-server.js"],
      "env": { "DATABASE_URL": "postgres://localhost/inventory" }
    }
  }
}
```

For Cursor, add to `.cursor/mcp.json` in your project root:

```json
{
  "mcpServers": {
    "inventory": {
      "command": "node",
      "args": ["./build/inventory-server.js"]
    }
  }
}
```

Once configured, the tools appear in the model's tool list automatically. No code changes to your prompts or agent logic.

### Popular MCP Servers

The ecosystem has grown rapidly. High-value servers include:

- **filesystem** -- read, write, search, and manage local files with configurable access controls.
- **git** -- repository operations: status, diff, log, commit, branch management.
- **postgres / sqlite** -- query databases, inspect schemas, run migrations.
- **slack** -- read channels, post messages, search message history.
- **github** -- manage issues, pull requests, code search, repository operations.
- **fetch** -- make HTTP requests to arbitrary URLs with response parsing.

### Design Patterns

**Database Gateway.** An MCP server sits in front of your database and exposes read-only query tools with parameterized queries. The model never sees raw SQL -- it calls tools like `search_customers(name="Acme")` and gets structured results. This enforces access control at the tool layer.

**API Aggregator.** A single MCP server wraps multiple related APIs behind a unified interface. Instead of the model learning three different API schemas for your CRM, ticketing system, and billing platform, it interacts with one server that exposes `get_customer`, `create_ticket`, and `check_invoice` as coherent tools.

**Context Provider.** An MCP server that primarily exposes resources rather than tools. It provides the model with relevant documentation, configuration files, or reference data on demand. The model requests `resource://docs/api-reference` and gets the current API docs injected into context.

### MCP Security

MCP servers are trust boundaries. Treat them accordingly.

- **Input validation.** Validate every parameter against its schema before executing. Never interpolate user-controlled strings into SQL or shell commands.
- **Least privilege.** Database servers should use read-only credentials unless write access is explicitly required. File system servers should be scoped to specific directories.
- **Rate limiting.** Cap the number of tool calls per session and per time window. An agent in a loop can generate hundreds of calls per minute.
- **Audit logging.** Log every tool invocation with timestamp, caller identity, arguments, and result. This is essential for debugging and compliance.
- **Transport security.** For HTTP-based servers, use TLS. Authenticate clients with API keys or OAuth tokens. Never expose an MCP server to the public internet without authentication.

### MCP vs Alternatives

| Criteria | MCP | OpenAI Plugins (deprecated) | LangChain Tools | Custom REST Integration |
|---|---|---|---|---|
| **Protocol** | JSON-RPC 2.0 | OpenAPI/REST | Python functions | Custom |
| **Transport** | stdio, Streamable HTTP | HTTPS only | In-process | HTTP/gRPC |
| **Model support** | Claude, GPT (via adapters), open models | OpenAI only (discontinued) | Any (via LangChain) | Any |
| **Standardized** | Yes (open spec) | No (proprietary, deprecated) | No (framework-specific) | No |
| **Discovery** | Built-in capability negotiation | Manifest file | Manual registration | Manual |
| **Resources (read-only data)** | First-class | Not supported | Not standardized | Custom |
| **Community ecosystem** | Growing rapidly | Dead | Large but fragmented | N/A |
| **Best for** | Standardized tool integration across hosts | N/A (deprecated) | Python-first prototyping | Full control, specific needs |

MCP's primary advantage is that you build the integration once and it works across any MCP-compatible host. As adoption spreads, the protocol is becoming the default way to connect AI applications to external systems. If you are building a new tool integration today and do not have a strong reason to do otherwise, build it as an MCP server.

---

## Summary

Agents are powerful when the problem genuinely requires autonomous, multi-step reasoning with tools. The core loop is simple: reason, act, observe, repeat. Everything else -- planning strategies, memory systems, multi-agent architectures, human-in-the-loop gates -- is scaffolding around that loop to handle real-world complexity.

MCP is rapidly becoming the standard protocol for the tool-connection layer. Building your integrations as MCP servers gives you portability across hosts and a clean separation between your agent logic and your tool implementations.

The hardest skill in agent engineering is not building agents. It is knowing when not to. Start with the simplest approach that could work. Validate that the complexity of an agent loop is earning its keep before you commit to it. Your users care about outcomes, not architecture.
