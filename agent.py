"""
ShopEase Agentic RAG – Agent Layer
Uses Anthropic tool use with a manual agentic loop (claude-sonnet-4-6).
"""

from __future__ import annotations

import json
import random
import string
from datetime import datetime, timedelta
from typing import Any

import anthropic

from knowledge_base import search_knowledge_base_raw

# ---------------------------------------------------------------------------
# Anthropic client
# ---------------------------------------------------------------------------
_client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are Sage, a friendly and knowledgeable customer support agent for ShopEase — a leading online retailer.

Your goal is to provide accurate, helpful, and empathetic responses to customer questions.

Guidelines:
- Always search the knowledge base before answering policy or product questions.
- If you need order-specific details, use the check_order_status tool.
- If a situation is urgent, complex, or requires action beyond your authority, use the escalate_to_human tool.
- Be concise yet thorough. Use bullet points when listing multiple items.
- Acknowledge customer frustration with empathy before jumping to solutions.
- Never make up policies or invent information not supported by your tools.
- Always greet the customer warmly on the first message."""

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "search_knowledge_base",
        "description": (
            "Search ShopEase's internal knowledge base for policy information, FAQs, "
            "and support articles. Use this tool whenever a customer asks about returns, "
            "shipping, payments, warranties, account management, or any ShopEase policy. "
            "Returns the most relevant documents."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query — describe what information you need.",
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of results to return (1-5). Default 3.",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "check_order_status",
        "description": (
            "Look up the current status of a customer's order by order ID. "
            "Use this when a customer mentions an order number or asks about "
            "delivery, tracking, or order details. Order IDs follow the format ORD-XXXXX."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order ID, e.g. ORD-12345.",
                }
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": (
            "Escalate this support ticket to a human agent. Use this when: "
            "(1) the customer is very upset or the issue is sensitive, "
            "(2) the question requires account modifications you cannot perform, "
            "(3) the issue involves fraud, legal, or safety concerns, "
            "(4) you have been unable to resolve the issue after searching the knowledge base."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why this ticket needs human review.",
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "urgent"],
                    "description": "Priority level for the human agent queue.",
                },
            },
            "required": ["reason", "priority"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _tool_search_knowledge_base(query: str, n_results: int = 3, collection=None) -> str:
    """Semantic search wrapper called by the agentic loop."""
    if collection is None:
        return json.dumps({"error": "Knowledge base not initialised."})

    results = search_knowledge_base_raw(collection, query, n_results)
    if not results:
        return json.dumps({"results": [], "message": "No relevant documents found."})

    formatted = []
    for r in results:
        formatted.append({
            "doc_id": r["doc_id"],
            "category": r["category"],
            "content": r["text"],
            "relevance_score": round(1 - r["distance"], 4),
        })
    return json.dumps({"results": formatted, "count": len(formatted)})


def _tool_check_order_status(order_id: str) -> str:
    """Mock order status lookup – returns plausible fake data."""
    order_id = order_id.strip().upper()

    # Deterministic "random" based on order_id so same ID always returns same data
    seed = sum(ord(c) for c in order_id)
    rng = random.Random(seed)

    statuses = [
        ("Processing", "Your order is being prepared for shipment."),
        ("Shipped", "Your order is on its way!"),
        ("Out for Delivery", "Your order will be delivered today."),
        ("Delivered", "Your order was delivered successfully."),
        ("Return Initiated", "Your return request has been received."),
    ]

    status, status_msg = rng.choice(statuses)

    order_date = datetime.now() - timedelta(days=rng.randint(1, 14))
    estimated_delivery = order_date + timedelta(days=rng.randint(3, 10))

    carriers = ["UPS", "FedEx", "USPS"]
    carrier = rng.choice(carriers)
    tracking = "".join(rng.choices(string.digits + string.ascii_uppercase, k=12))

    items = rng.randint(1, 4)
    total = round(rng.uniform(19.99, 249.99), 2)

    result = {
        "order_id": order_id,
        "status": status,
        "status_message": status_msg,
        "order_date": order_date.strftime("%Y-%m-%d"),
        "estimated_delivery": estimated_delivery.strftime("%Y-%m-%d"),
        "carrier": carrier,
        "tracking_number": tracking,
        "items_count": items,
        "order_total": f"${total:.2f}",
        "shipping_address": "on file",
    }

    # If order_id looks totally invalid, return error
    if not order_id.startswith("ORD-"):
        return json.dumps({
            "error": f"Order '{order_id}' not found. Please verify the order ID format (e.g. ORD-12345)."
        })

    return json.dumps(result)


def _tool_escalate_to_human(reason: str, priority: str) -> str:
    """Mark this ticket for human escalation."""
    ticket_id = "TKT-" + "".join(random.choices(string.digits, k=6))
    return json.dumps({
        "escalated": True,
        "ticket_id": ticket_id,
        "priority": priority,
        "reason": reason,
        "message": (
            f"Ticket {ticket_id} has been created with {priority.upper()} priority. "
            "A human agent will contact the customer within the SLA window: "
            "Urgent=30 min, High=2 hrs, Medium=4 hrs, Low=1 business day."
        ),
        "estimated_response": {
            "urgent": "30 minutes",
            "high": "2 hours",
            "medium": "4 hours",
            "low": "1 business day",
        }.get(priority, "4 hours"),
    })


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

def _dispatch_tool(tool_name: str, tool_input: dict[str, Any], collection=None) -> str:
    """Route tool calls to the correct implementation."""
    if tool_name == "search_knowledge_base":
        return _tool_search_knowledge_base(
            query=tool_input["query"],
            n_results=tool_input.get("n_results", 3),
            collection=collection,
        )
    elif tool_name == "check_order_status":
        return _tool_check_order_status(tool_input["order_id"])
    elif tool_name == "escalate_to_human":
        return _tool_escalate_to_human(
            reason=tool_input["reason"],
            priority=tool_input["priority"],
        )
    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})


# ---------------------------------------------------------------------------
# Agentic response structure
# ---------------------------------------------------------------------------

class AgentResponse:
    """Structured output from a single agent turn."""

    def __init__(self):
        self.answer: str = ""
        self.sources_used: list[dict] = []
        self.escalated: bool = False
        self.escalation_ticket: str | None = None
        self.tool_calls_made: list[dict] = []

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources_used": self.sources_used,
            "escalated": self.escalated,
            "escalation_ticket": self.escalation_ticket,
            "tool_calls_made": self.tool_calls_made,
        }


# ---------------------------------------------------------------------------
# Main agentic loop
# ---------------------------------------------------------------------------

def run_agent(
    user_message: str,
    conversation_history: list[dict],
    collection=None,
) -> AgentResponse:
    """
    Run the agentic loop for a single customer turn.

    Args:
        user_message: The customer's latest message.
        conversation_history: List of prior messages in the format
            [{"role": "user"|"assistant", "content": ...}, ...].
        collection: ChromaDB collection instance.

    Returns:
        AgentResponse with the final answer and metadata.
    """
    response_obj = AgentResponse()

    # Build the message list for this turn
    messages = list(conversation_history)
    messages.append({"role": "user", "content": user_message})

    # ── Agentic loop ──────────────────────────────────────────────────────
    max_iterations = 10  # safety cap
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        api_response = _client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Append assistant response to conversation
        messages.append({"role": "assistant", "content": api_response.content})

        # ── Terminal condition ─────────────────────────────────────────────
        if api_response.stop_reason == "end_turn":
            # Extract the final text answer
            for block in api_response.content:
                if hasattr(block, "text"):
                    response_obj.answer = block.text
                    break
            break

        # ── Tool use ──────────────────────────────────────────────────────
        if api_response.stop_reason == "tool_use":
            tool_results = []

            for block in api_response.content:
                if block.type != "tool_use":
                    continue

                tool_name = block.name
                tool_input = block.input
                tool_use_id = block.id

                # Execute the tool
                result_str = _dispatch_tool(tool_name, tool_input, collection)
                result_data = json.loads(result_str)

                # Record tool call metadata
                call_record = {
                    "tool": tool_name,
                    "input": tool_input,
                    "result_summary": _summarise_tool_result(tool_name, result_data),
                }
                response_obj.tool_calls_made.append(call_record)

                # Handle side effects
                if tool_name == "search_knowledge_base":
                    _collect_sources(result_data, response_obj)
                elif tool_name == "escalate_to_human":
                    response_obj.escalated = True
                    response_obj.escalation_ticket = result_data.get("ticket_id")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result_str,
                })

            # Feed results back to Claude
            messages.append({"role": "user", "content": tool_results})

        else:
            # Unexpected stop_reason – extract whatever text is available and stop
            for block in api_response.content:
                if hasattr(block, "text"):
                    response_obj.answer = block.text
                    break
            break

    # Fallback if no answer was captured
    if not response_obj.answer:
        response_obj.answer = (
            "I'm sorry, I wasn't able to generate a complete response. "
            "Please try again or contact support directly."
        )

    return response_obj


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _collect_sources(result_data: dict, response_obj: AgentResponse) -> None:
    """Extract source doc references from a KB search result."""
    for item in result_data.get("results", []):
        source = {
            "doc_id": item.get("doc_id"),
            "category": item.get("category"),
            "relevance": item.get("relevance_score"),
            "snippet": item.get("content", "")[:200] + "…",
        }
        # Deduplicate by doc_id
        existing_ids = {s["doc_id"] for s in response_obj.sources_used}
        if source["doc_id"] not in existing_ids:
            response_obj.sources_used.append(source)


def _summarise_tool_result(tool_name: str, result_data: dict) -> str:
    """Return a short human-readable summary of a tool result for logging."""
    if tool_name == "search_knowledge_base":
        count = result_data.get("count", 0)
        return f"Found {count} document(s)"
    elif tool_name == "check_order_status":
        if "error" in result_data:
            return f"Error: {result_data['error']}"
        return f"Order {result_data.get('order_id')} – {result_data.get('status')}"
    elif tool_name == "escalate_to_human":
        return f"Ticket {result_data.get('ticket_id')} created ({result_data.get('priority')} priority)"
    return str(result_data)[:100]
