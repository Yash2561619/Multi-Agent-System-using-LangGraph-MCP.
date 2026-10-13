import asyncio
import json
import operator
import os
import uuid
from typing import Annotated, Any, TypedDict

import certifi
from dotenv import load_dotenv

load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

import psycopg
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from psycopg.rows import dict_row

from mcp_client import (
    aviation_mcp_call,
    extract_destination,
    forecast_mcp_search,
    tavily_mcp_search,
    weather_mcp_search,
)


def get_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. Please add your PostgreSQL URL to .env"
        )

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file.")

# =========================
# LLM
# =========================
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
)


# =========================
# Safe Async Helper for Sync Nodes
# =========================
def _run_async(coro):
    """Safely run an async MCP call from inside a synchronous node."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        import nest_asyncio

        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    return loop.run_until_complete(coro)


# =========================
# State Schema
# =========================
class TravelState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str

    guardrail_allowed: bool
    guardrail_reason: str
    selected_agents: list[str]
    trip_constraints: dict[str, Any]
    supervisor_reasoning: str

    flight_results: str
    hotel_results: str
    weather_results: str
    itinerary: str

    budget_results: str
    approval_request: str
    approved: bool
    human_feedback: str
    final_response: str

    llm_calls: int


KNOWN_AGENTS = {
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
}

AGENT_ORDER = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
]


def _llm_text(system_prompt: str, user_prompt: str) -> str:
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return str(response.content)


def _json_from_llm(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("The model did not return a valid JSON object.")
    return json.loads(text[start : end + 1])


def _empty_constraints() -> dict[str, Any]:
    return {
        "destination": "",
        "origin": "",
        "duration": "",
        "budget": "",
        "travel_style": "",
        "special_preferences": [],
    }


# =========================
# Supervisor & Guardrail
# =========================
def supervisor_agent(state: TravelState):
    query = state["user_query"]
    llm_calls = state.get("llm_calls", 0)

    guardrail_prompt = f"""Determine whether the following request belongs to travel planning or travel information.
Valid requests include destinations, flights, hotels, weather, budgets, visas, food, packing, or itineraries.
Block clearly unrelated, harmful, or illegal requests.

Return strict JSON only:
{{
  "allowed": true,
  "reason": ""
}}

User request: {query}"""

    try:
        guardrail_raw = _llm_text(
            "You are the input guardrail for a travel app. Return strict JSON only.",
            guardrail_prompt,
        )
        guardrail_result = _json_from_llm(guardrail_raw)
        allowed = bool(guardrail_result.get("allowed", True))
        guardrail_reason = str(guardrail_result.get("reason", "")).strip()
        llm_calls += 1
    except Exception as exc:
        print(f"Guardrail fallback used: {exc}")
        allowed = True
        guardrail_reason = "Guardrail validation fallback allowed the request."

    if not allowed:
        reason = (
            guardrail_reason
            or "TripMate AI can only help with travel-planning requests."
        )
        return {
            "guardrail_allowed": False,
            "guardrail_reason": reason,
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [AIMessage(content=f"Guardrail blocked request: {reason}")],
            "llm_calls": llm_calls,
        }

    supervisor_prompt = f"""You are the supervisor of a multi-agent travel planner.
Choose only the specialist agents needed for the request:
- flight_agent: flights, airlines, routes, or airfare
- hotel_agent: hotels, accommodation, or neighborhoods
- weather_agent: weather, climate, forecast, or packing
- budget_agent: cost, affordability, or pricing
- itinerary_agent: creates the travel plan (always include)

Return strict JSON:
{{
  "selected_agents": ["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"],
  "trip_constraints": {{
    "destination": "",
    "origin": "",
    "duration": "",
    "budget": "",
    "travel_style": "",
    "special_preferences": []
  }},
  "reasoning": ""
}}

User request: {query}"""

    try:
        supervisor_raw = _llm_text(
            "Route work to travel specialists. Return strict JSON only.",
            supervisor_prompt,
        )
        parsed = _json_from_llm(supervisor_raw)
        requested_agents = parsed.get("selected_agents", [])
        selected_agents = [
            name
            for name in AGENT_ORDER
            if name in requested_agents and name in KNOWN_AGENTS
        ]
        if "itinerary_agent" not in selected_agents:
            selected_agents.append("itinerary_agent")

        constraints = _empty_constraints()
        parsed_constraints = parsed.get("trip_constraints", {})
        if isinstance(parsed_constraints, dict):
            constraints.update(parsed_constraints)

        reasoning = str(parsed.get("reasoning", "")).strip()
        llm_calls += 1
    except Exception as exc:
        print(f"Supervisor fallback used: {exc}")
        selected_agents = AGENT_ORDER.copy()
        constraints = _empty_constraints()
        reasoning = "Supervisor fallback selected full travel workflow."

    return {
        "guardrail_allowed": True,
        "guardrail_reason": guardrail_reason,
        "selected_agents": selected_agents,
        "trip_constraints": constraints,
        "supervisor_reasoning": reasoning,
        "messages": [AIMessage(content="Supervisor created execution plan.")],
        "llm_calls": llm_calls,
    }


def guardrail_blocked_agent(state: TravelState):
    reason = (
        state.get("final_response")
        or state.get("guardrail_reason")
        or "Request blocked by travel guardrail."
    )
    return {
        "final_response": reason,
        "messages": [AIMessage(content=reason)],
    }


# =========================
# Specialist Agents
# =========================
FLIGHT_AGENT_PROMPT = """You are a travel flight expert.
User Query: {query}
Airport Information: {airport_data}
Airline Information: {airline_data}

Generate:
1. Likely departure airport
2. Likely arrival airport
3. Airlines serving this route
4. Typical flight duration
5. Estimated airfare range
6. Peak season pricing warning
7. Booking advice"""


def flight_agent(state: TravelState):
    query = state["user_query"]
    try:
        airports = _run_async(aviation_mcp_call("list_airports"))
        airlines = _run_async(aviation_mcp_call("list_airlines"))

        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            airport_data=str(airports)[:2000],
            airline_data=str(airlines)[:2000],
        )

        response = llm.invoke(
            [
                SystemMessage(content="You are an expert flight planner."),
                HumanMessage(content=prompt),
            ]
        )
        flight_data = response.content
    except Exception as exc:
        flight_data = f"Flight information unavailable: {exc}"

    return {
        "flight_results": flight_data,
        "messages": [AIMessage(content="Flight recommendations generated.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def hotel_agent(state: TravelState):
    query = f"Best hotels for {state['user_query']}"
    try:
        hotel_results = _run_async(tavily_mcp_search(query))
    except Exception as exc:
        print(f"Hotel MCP error: {exc}")
        hotel_results = (
            "Live hotel search unavailable. Providing general neighborhood options."
        )

    return {
        "hotel_results": str(hotel_results),
        "messages": [AIMessage(content="Hotel recommendations generated.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def weather_agent(state: TravelState):
    city = extract_destination(state["user_query"])
    try:
        weather_data = _run_async(weather_mcp_search(city))
        forecast_data = _run_async(forecast_mcp_search(city))
        weather_results = f"Current Weather:\n{weather_data}\n\nForecast:\n{forecast_data}"
    except Exception as exc:
        print(f"Weather MCP error: {exc}")
        weather_results = f"Live weather for {city} unavailable. Seasonal guidance recommended."

    return {
        "weather_results": weather_results,
        "messages": [AIMessage(content="Weather data gathered.")],
    }


def budget_agent(state: TravelState):
    prompt = f"""Analyze the trip budget feasibility:
User Query: {state['user_query']}
Constraints: {state.get('trip_constraints', {})}
Flight Data: {state.get('flight_results', '')}
Hotel Data: {state.get('hotel_results', '')}

Provide:
1. Estimated cost breakdown
2. Budget risk areas
3. Money-saving tips
4. Overall feasibility verdict"""

    response = llm.invoke(
        [
            SystemMessage(content="You are a travel budget analyst."),
            HumanMessage(content=prompt),
        ]
    )

    return {
        "budget_results": response.content,
        "messages": [AIMessage(content="Budget feasibility assessed.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def itinerary_agent(state: TravelState):
    prompt = f"""Create a comprehensive draft travel itinerary:
Query: {state['user_query']}
Constraints: {state.get('trip_constraints', {})}
Flights: {state.get('flight_results', '')}
Hotels: {state.get('hotel_results', '')}
Weather: {state.get('weather_results', '')}
Budget: {state.get('budget_results', '')}

Make the draft practical, day-by-day, and ready for human review."""

    response = llm.invoke(
        [
            SystemMessage(content="You are an expert travel planner."),
            HumanMessage(content=prompt),
        ]
    )

    approval_request = (
        "Please review the generated draft itinerary. Approve it to create the final plan, "
        "or request revisions."
    )

    return {
        "itinerary": response.content,
        "approval_request": approval_request,
        "messages": [AIMessage(content="Draft itinerary prepared for review.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Human-in-the-Loop Interruption
# =========================
def human_approval_agent(state: TravelState):
    review = interrupt(
        {
            "question": "Do you approve this itinerary?",
            "draft_itinerary": state.get("itinerary", ""),
            "approval_request": state.get("approval_request", ""),
            "selected_agents": state.get("selected_agents", []),
            "supervisor_reasoning": state.get("supervisor_reasoning", ""),
        }
    )

    approved = bool(review.get("approved", False))
    human_feedback = str(review.get("feedback", "")).strip()

    return {
        "approved": approved,
        "human_feedback": human_feedback,
        "messages": [AIMessage(content="Human review processed.")],
    }


def final_agent(state: TravelState):
    if state.get("approved", False):
        review_instruction = "The user approved the draft itinerary. Polish and present clearly."
    else:
        review_instruction = (
            f"The user requested changes: {state.get('human_feedback', '') or 'Revise draft.'}"
        )

    final_prompt = f"""Generate the final comprehensive travel plan.
Review status: {review_instruction}
User Query: {state['user_query']}
Flights: {state.get('flight_results', '')}
Hotels: {state.get('hotel_results', '')}
Weather: {state.get('weather_results', '')}
Budget: {state.get('budget_results', '')}
Draft: {state.get('itinerary', '')}

Format cleanly:
1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Weather & Packing Guidance
5. Day-by-Day Itinerary
6. Estimated Budget
7. Final Recommendations"""

    response = llm.invoke(
        [
            SystemMessage(content="You are a professional travel assistant."),
            HumanMessage(content=final_prompt),
        ]
    )

    return {
        "final_response": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Routing Logic
# =========================
ROUTE_MAP = {
    "guardrail_blocked": "guardrail_blocked",
    "flight_agent": "flight_agent",
    "hotel_agent": "hotel_agent",
    "weather_agent": "weather_agent",
    "budget_agent": "budget_agent",
    "itinerary_agent": "itinerary_agent",
}


def _selected_agents(state: TravelState) -> list[str]:
    selected = state.get("selected_agents", [])
    return [agent for agent in AGENT_ORDER if agent in selected]


def route_from_supervisor(state: TravelState) -> str:
    if not state.get("guardrail_allowed", True):
        return "guardrail_blocked"
    selected = _selected_agents(state)
    return selected[0] if selected else "itinerary_agent"


def route_after_agent(current_agent: str):
    def route(state: TravelState) -> str:
        selected = _selected_agents(state)
        current_index = AGENT_ORDER.index(current_agent)
        for next_agent in AGENT_ORDER[current_index + 1 :]:
            if next_agent in selected:
                return next_agent
        return "itinerary_agent"

    return route


# =========================
# Graph Construction & PostgreSQL Checkpointer
# =========================
graph = StateGraph(TravelState)

graph.add_node("supervisor", supervisor_agent)
graph.add_node("guardrail_blocked", guardrail_blocked_agent)
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("budget_agent", budget_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("human_approval", human_approval_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", route_from_supervisor, ROUTE_MAP)
graph.add_conditional_edges("flight_agent", route_after_agent("flight_agent"), ROUTE_MAP)
graph.add_conditional_edges("hotel_agent", route_after_agent("hotel_agent"), ROUTE_MAP)
graph.add_conditional_edges("weather_agent", route_after_agent("weather_agent"), ROUTE_MAP)
graph.add_conditional_edges("budget_agent", route_after_agent("budget_agent"), ROUTE_MAP)

graph.add_edge("itinerary_agent", "human_approval")
graph.add_edge("human_approval", "final_agent")
graph.add_edge("final_agent", END)
graph.add_edge("guardrail_blocked", END)

DATABASE_URL = get_database_url()

# Setup PostgreSQL Checkpointer with reconnect resilience
_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row,
)
checkpointer = PostgresSaver(_conn)
checkpointer.setup()

travel_graph = graph.compile(checkpointer=checkpointer)


# =========================
# Execution Handlers
# =========================
def _extract_interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("__interrupt__", [])
    if not interrupts:
        return None
    first = interrupts[0]
    payload = getattr(first, "value", first)
    return payload if isinstance(payload, dict) else {"value": payload}


def _serialize_state(
    state_values: dict[str, Any],
    thread_id: str,
    interrupt_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    messages = state_values.get("messages", [])
    last_message = messages[-1].content if messages else ""
    answer = state_values.get("final_response") or last_message

    if interrupt_payload:
        answer = interrupt_payload.get("draft_itinerary") or state_values.get(
            "itinerary", ""
        )

    return {
        "thread_id": thread_id,
        "answer": answer,
        "requires_approval": interrupt_payload is not None,
        "approval_request": (
            interrupt_payload.get("approval_request", "")
            if interrupt_payload
            else state_values.get("approval_request", "")
        ),
        "flight_results": state_values.get("flight_results", ""),
        "hotel_results": state_values.get("hotel_results", ""),
        "weather_results": state_values.get("weather_results", ""),
        "budget_results": state_values.get("budget_results", ""),
        "itinerary": state_values.get("itinerary", ""),
        "selected_agents": state_values.get("selected_agents", []),
        "trip_constraints": state_values.get("trip_constraints", {}),
        "supervisor_reasoning": state_values.get("supervisor_reasoning", ""),
        "guardrail_allowed": state_values.get("guardrail_allowed", True),
        "guardrail_reason": state_values.get("guardrail_reason", ""),
        "approved": state_values.get("approved"),
        "human_feedback": state_values.get("human_feedback", ""),
        "llm_calls": state_values.get("llm_calls", 0),
    }


def run_travel_agent(user_input: str, thread_id: str | None = None):
    """Initiates execution, stopping at human_approval interrupt."""
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {"configurable": {"thread_id": thread_id}}

    result = travel_graph.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "guardrail_allowed": True,
            "guardrail_reason": "",
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": "",
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "budget_results": "",
            "itinerary": "",
            "approval_request": "",
            "approved": False,
            "human_feedback": "",
            "final_response": "",
            "llm_calls": 0,
        },
        config=config,
    )

    interrupt_payload = _extract_interrupt_payload(result)
    snapshot = travel_graph.get_state(config)
    return _serialize_state(snapshot.values, thread_id, interrupt_payload)


def resume_travel_agent(thread_id: str, approved: bool, feedback: str = ""):
    """Resumes after human approval and completes the final response node."""
    if not thread_id:
        raise ValueError("thread_id is required to resume an agent run.")

    config = {"configurable": {"thread_id": thread_id}}

    result = travel_graph.invoke(
        Command(
            resume={
                "approved": approved,
                "feedback": feedback.strip(),
            }
        ),
        config=config,
    )

    interrupt_payload = _extract_interrupt_payload(result)
    snapshot = travel_graph.get_state(config)
    return _serialize_state(snapshot.values, thread_id, interrupt_payload)