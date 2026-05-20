"""
agent.py

Command-line entry point for rapid testing of the Alchemist Agent.
Bypasses the Streamlit UI for fast debugging.

Usage:
  python agent.py "What is the current gold outlook?"
  python agent.py "Compare the fundamentals to the technicals." --symbol SLV
"""

import argparse
from dotenv import load_dotenv
from loguru import logger

# Suppress debug logs from HTTP requests to keep the terminal clean
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Alchemist Agent CLI")
    parser.add_argument("question", type=str, help="Your question for the agent")
    parser.add_argument("--symbol", type=str, default="GLD", help="Asset symbol context (GLD or SLV)")
    args = parser.parse_args()

    from src.agent.graph import run_query
    
    try:
        result = run_query(args.question, symbol=args.symbol)
        print("\n" + result["final_report"])
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")

if __name__ == "__main__":
    main()