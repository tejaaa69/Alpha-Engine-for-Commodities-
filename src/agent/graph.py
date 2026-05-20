"""
src/agent/graph.py

Compiles the unified Alchemist LangGraph agent.

Upgrades:
  1. Global Memory Persistence: `MemorySaver` is now a singleton to ensure 
     conversational memory persists across multiple Streamlit chat inputs.
  2. Message State Injection: Properly wires the user's question into 
     LangGraph's native `messages` state.
  3. Contextual Symbol Binding: Injects the active dashboard symbol directly 
     into the agent's context.
"""

import uuid
from typing import Optional

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from loguru import logger

from src.agent.nodes import execute_tools_node, router_node, synthesizer_node
from src.agent.state import ResearchState

# ── GLOBAL MEMORY SAVER (CRITICAL FOR STREAMLIT CHAT) ──
# This must exist outside the function so memory isn't wiped on every run.
_global_memory = MemorySaver()

def build_graph(use_memory: bool = True):
    """Build and compile the LangGraph agent."""
    workflow = StateGraph(ResearchState)

    workflow.add_node("router",        router_node)
    workflow.add_node("execute_tools", execute_tools_node)
    workflow.add_node("synthesizer",   synthesizer_node)

    workflow.add_edge(START,           "router")
    workflow.add_edge("router",        "execute_tools")
    workflow.add_edge("execute_tools", "synthesizer")
    workflow.add_edge("synthesizer",   END)

    checkpointer = _global_memory if use_memory else None
    app = workflow.compile(checkpointer=checkpointer)
    
    logger.info("✅ Alchemist LangGraph Agent compiled successfully.")
    return app

# Compile the app globally once to save initialization time
agent_app = build_graph()

def run_query(question: str, symbol: str = "GLD", thread_id: Optional[str] = None) -> dict:
    """
    Single entry point for all agent queries.
    Safely integrates with Streamlit session state.
    """
    thread_id = thread_id or f"alchemist_{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    # Contextualize the question with the active dashboard symbol
    contextual_question = f"[Active Dashboard Asset: {symbol}] {question}"

    # Initialize state, explicitly mapping to the LangGraph `messages` array
    initial_state = {
        "question":      contextual_question,
        "messages":      [HumanMessage(content=contextual_question)],
        "plan":          [],
        "tool_outputs":  [],
        "ml_prediction": None,
        "final_report":  "",
    }

    logger.info(f"\n{'═'*60}")
    logger.info(f"🧠 ALCHEMIST AGENT QUERY | Thread: {thread_id}")
    logger.info(f"Question: {question}")
    logger.info(f"{'═'*60}\n")

    # Invoke the graph
    final_state = agent_app.invoke(initial_state, config=config)

    # Safely extract the quantitative payload for the UI
    ml = final_state.get("ml_prediction") or {}
    
    return {
        "thread_id":    thread_id,
        "question":     question,
        "final_report": final_state.get("final_report", "Error: No report generated."),
        "probability":  ml.get("probability", 0.0),
        "signal":       ml.get("signal", "N/A"),
        "regime":       ml.get("regime",  "UNKNOWN"),
        "bt_sharpe":    ml.get("bt_sharpe", 0.0),
        "bt_win_rate":  ml.get("bt_win_rate", 0.0),
        "shap":         ml.get("shap_narrative", ""),
    }