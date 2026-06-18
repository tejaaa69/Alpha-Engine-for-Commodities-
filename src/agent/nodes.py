"""
src/agent/nodes.py

Three nodes. Clean, no redundancy.
Perfectly aligned with the Alchemist ML outputs and LangGraph memory.
"""

import re
from datetime import datetime
from typing import List

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from loguru import logger
from pydantic import BaseModel, Field

from src.agent.state import ResearchState, MLPrediction
from src.agent.tools import TOOL_MAP

# LLM — shared across nodes

def _get_llm():
    import os
    return ChatGroq(
        model_name  = "llama-3.3-70b-versatile",
        temperature = 0, # Strict, deterministic reasoning
        api_key     = os.environ.get("GROQ_API_KEY", ""),
    )

# ROUTER NODE

class ToolCall(BaseModel):
    tool_name:  str = Field(
        description="One of: alchemist_prediction, rag_wgc, tavily_search, fred_live"
    )
    tool_query: str = Field(
        description=(
            "For alchemist_prediction: the symbol (GLD or SLV). "
            "For rag_wgc: specific topic to search in documents. "
            "For tavily_search: concise news query. "
            "For fred_live: FRED series ID (e.g. FEDFUNDS, DFII10, CPIAUCSL)."
        )
    )

class RoutePlan(BaseModel):
    plan: List[ToolCall] = Field(description="Ordered list of tool calls.")


ROUTER_PROMPT = """You are a routing agent for a quantitative gold/silver research system.
Your job: decide which tools to call based on the user's question.

AVAILABLE TOOLS:
1. alchemist_prediction  — ML model signal. Call with 'GLD' for gold, 'SLV' for silver.
                           ALWAYS call this first for any gold/silver price query.
2. rag_wgc               — Searches WGC/institutional research documents.
                           Use for fundamental supply/demand questions.
3. tavily_search         — Live web news. Use for recent events and context.
4. fred_live             — Fetches one FRED series for human-readable context.
                           Only call for: FEDFUNDS, DFII10, CPIAUCSL.

RULES:
- For gold queries: ALWAYS call alchemist_prediction('GLD') first.
- For silver queries: ALWAYS call alchemist_prediction('SLV') first.
- Keep the plan to 4-6 tool calls maximum.
- Do NOT call fred_live for more than 2 series — the ML model handles scoring internally.
"""

def router_node(state: ResearchState) -> dict:
    logger.info(f"[ROUTER] Planning for: '{state['question']}'")

    llm = _get_llm()
    structured_llm = llm.with_structured_output(RoutePlan)
    prompt = ChatPromptTemplate.from_messages([
        ("system", ROUTER_PROMPT),
        ("user", "{question}"),
    ])
    
    result = (prompt | structured_llm).invoke({"question": state["question"]})
    plan = [{"tool_name": tc.tool_name, "tool_query": tc.tool_query} for tc in result.plan]

    logger.info(f"[ROUTER] Plan: {[(t['tool_name'], t['tool_query']) for t in plan]}")
    return {"plan": plan}


# EXECUTOR NODE

def execute_tools_node(state: ResearchState) -> dict:
    logger.info("[EXECUTOR] Running tool plan...")
    outputs = []
    ml_prediction = None

    for task in state["plan"]:
        tool_name  = task["tool_name"]
        tool_query = task["tool_query"]

        if tool_name not in TOOL_MAP:
            logger.warning(f"[EXECUTOR] Unknown tool: {tool_name}")
            continue

        tool_fn = TOOL_MAP[tool_name]
        result  = tool_fn.invoke(tool_query)
        formatted = f"--- {tool_name} (query: {tool_query}) ---\n{result}\n"
        outputs.append(formatted)

        # Extract ML prediction using the corrected Regex mappings
        if tool_name == "alchemist_prediction":
            ml_prediction = _parse_ml_output(result, tool_query)

    update = {"tool_outputs": outputs}
    if ml_prediction:
        update["ml_prediction"] = ml_prediction

    return update


def _parse_ml_output(raw_output: str, symbol: str) -> MLPrediction:
    """
    Parse the structured text output of AlchemistPredictionTool
    into a typed MLPrediction dict. Meticulously mapped to tools.py output.
    """
    def extract(pattern, text, default):
        m = re.search(pattern, text, re.DOTALL)
        return m.group(1).strip() if m else default

    # Perfected regex targeting the elite tools.py output
    prob_str  = extract(r"Probability.*?:\s*([\d.]+)%", raw_output, "0")
    signal    = extract(r"Actionable Signal:\s*(.*?)\n", raw_output, "NO_SIGNAL")
    regime    = extract(r"Current Market Regime:\s*(\w+)", raw_output, "UNKNOWN")
    shap      = extract(r"Mathematical Drivers \(SHAP\):\n(.*?)\n\n", raw_output, "N/A")
    sharpe    = extract(r"Sharpe Ratio:\s*([A-Za-z0-9/.-]+)", raw_output, "0")
    win_rate  = extract(r"Win Rate:\s*([A-Za-z0-9/.-]+)%?", raw_output, "0")

    try:
        prob = float(prob_str) / 100
    except ValueError:
        prob = 0.0

    # Safely handle N/A from un-backtested models
    bt_sharpe = 0.0
    if sharpe and sharpe != "N/A":
        try: bt_sharpe = float(sharpe)
        except ValueError: pass

    bt_win_rate = 0.0
    if win_rate and win_rate != "N/A":
        try: bt_win_rate = float(win_rate) / 100
        except ValueError: pass

    return MLPrediction(
        symbol          = symbol,
        prediction_date = datetime.today().strftime("%Y-%m-%d"),
        probability     = round(prob, 4),
        signal          = signal.strip(),
        regime          = regime,
        regime_prob     = 0.0,
        shap_narrative  = " | ".join(shap.split("\n")) if shap != "N/A" else "N/A",
        bt_sharpe       = bt_sharpe,
        bt_win_rate     = bt_win_rate,
        confidence_pct  = 100,
    )


# SYNTHESIZER NODE

SYNTHESIZER_PROMPT = """You are a senior quantitative analyst writing an investment brief.

The ML model has already made its decision — your job is to explain WHY it makes sense
given the supporting evidence, and what it means for an investor.

RULES:
1. The ML probability is ground truth. Do not contradict it.
2. Cite FRED numbers, RAG findings, and news as supporting evidence.
3. NEVER invent prices or statistics. Only use numbers from the tool outputs.
4. 4-6 sentences maximum for the rationale. Be direct. No hedging.
5. End with one concrete actionable statement.
"""

class InvestmentBrief(BaseModel):
    verdict:     str = Field(description="Must exactly reflect the ML model's Actionable Signal")
    probability: str = Field(description="ML probability as percentage string e.g. '73%'")
    regime:      str = Field(description="Current market regime: LOW_VOL / MID_VOL / HIGH_VOL")
    rationale:   str = Field(description="4-6 sentences. Cite specific numbers. No fluff.")
    key_risk:    str = Field(description="Single most important risk to this signal from the news/RAG.")
    action:      str = Field(description="One concrete actionable statement.")


def synthesizer_node(state: ResearchState) -> dict:
    logger.info("[SYNTHESIZER] Writing investment brief...")

    ml   = state.get("ml_prediction")
    raw  = "\n".join(state.get("tool_outputs", []))
    llm  = _get_llm()

    if ml is None:
        error_msg = "No quantitative ML prediction available. Please run the training pipeline first."
        return {"final_report": error_msg, "messages": [AIMessage(content=error_msg)]}

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYNTHESIZER_PROMPT),
        ("user", (
            "Question: {question}\n\n"
            "ML MODEL OUTPUT (GROUND TRUTH):\n"
            "  Symbol:       {symbol}\n"
            "  Probability:  {prob:.1%}\n"
            "  Signal:       {signal}\n"
            "  Regime:       {regime}\n"
            "  SHAP drivers: {shap}\n"
            "  Backtest Sharpe: {sharpe}\n\n"
            "SUPPORTING EVIDENCE (tool outputs):\n{raw}"
        )),
    ])

    structured_llm = llm.with_structured_output(InvestmentBrief)
    brief = (prompt | structured_llm).invoke({
        "question": state["question"],
        "symbol":   ml["symbol"],
        "prob":     ml["probability"],
        "signal":   ml["signal"],
        "regime":   ml["regime"],
        "shap":     ml["shap_narrative"],
        "sharpe":   ml["bt_sharpe"],
        "raw":      raw[:5000],   # Safely truncated to avoid context overflow
    })

    prob_pct   = int(ml["probability"] * 100)
    verdict_icon = "🟢" if "BUY" in ml["signal"].upper() else "🟡"
    
    # Updated to our Volatility-based Regime Labels
    regime_icon  = {"LOW_VOL": "📈 (Stable Bull)", "MID_VOL": "➡️ (Transition)", "HIGH_VOL": "📉 (Crisis/Bear)"}.get(ml["regime"], "❓")

    final_report = f"""
{'═'*60}
ALCHEMIST INVESTMENT BRIEF  |  {ml['symbol']}  |  {ml['prediction_date']}
{'═'*60}

{verdict_icon} SIGNAL:      {ml['signal']}
📊 PROBABILITY:  {prob_pct}% (Calibrated algorithmic probability)
{regime_icon} REGIME:        {ml['regime']}
📈 SHAP DRIVERS: {ml['shap_narrative']}

{'─'*60}
RATIONALE:
{brief.rationale}

KEY RISK:
{brief.key_risk}

ACTION:
{brief.action}

{'─'*60}
BACKTEST CONTEXT:
  Sharpe Ratio (OOS): {ml['bt_sharpe']}
  Win Rate:           {ml['bt_win_rate']:.1%}
{'═'*60}
"""
    # Append to messages for LangGraph conversational memory
    return {
        "final_report": final_report,
        "messages": [AIMessage(content=final_report)]
    }