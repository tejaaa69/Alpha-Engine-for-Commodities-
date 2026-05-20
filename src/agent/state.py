"""
src/agent/state.py

Unified state for the merged Alchemist + CIE agent.
"""

import operator
from typing import Annotated, Optional, TypedDict
from langgraph.graph.message import add_messages

class MLPrediction(TypedDict):
    symbol:          str
    prediction_date: str
    probability:     float          # calibrated P(profit target hit)
    signal:          str            # BUY / NO_SIGNAL
    regime:          str            # BULL / BEAR / SIDEWAYS
    shap_narrative:  str            # Top drivers as text
    bt_sharpe:       str            # Model's historical Sharpe
    bt_win_rate:     str            # Model's historical Win Rate

class ResearchState(TypedDict):
    # Core Conversational Memory (Mandatory for LangGraph)
    messages:        Annotated[list, add_messages]
    
    # Task specific tracking
    question:        str
    plan:            list[dict]

    # Tool outputs — append-only
    tool_outputs:    Annotated[list, operator.add]

    # ML prediction — the ground truth verdict
    ml_prediction:   Optional[MLPrediction]

    # Final output
    final_report:    str