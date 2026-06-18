"""
src/agent/state.py

Unified state for the merged Alchemist + CIE agent.
"""

import operator
from typing import Annotated, Optional, TypedDict
from langgraph.graph.message import add_messages
from sympy import Float

class MLPrediction(TypedDict):
    symbol:          str
    prediction_date: str
    probability:     float          # calibrated P(profit target hit)
    signal:          str            # BUY / NO_SIGNAL
    regime:          str            # BULL / BEAR / SIDEWAYS
    shap_narrative:  str            # Top drivers as text
    bt_sharpe:       Float            # Model's historical Sharpe
    bt_win_rate:     Float            # Model's historical Win Rate
    regime_prob:     float          # P(regime) from regime classifier
    confidence_pct:  int           # Confidence in the prediction (0-100)

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