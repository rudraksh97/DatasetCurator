import argparse
import asyncio
import sys
from pathlib import Path

# Add backend to path
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BACKEND_DIR))

from evals.eval_intent import run_evals as run_intent_evals
from evals.eval_planner import run_evals as run_planner_evals

async def main():
    parser = argparse.ArgumentParser(description="Run Dataset Curator Evals")
    parser.add_argument("--type", choices=["intent", "planner", "all"], default="all", help="Type of eval to run")
    args = parser.parse_args()
    
    if args.type in ["intent", "all"]:
        print("\n" + "="*50)
        print("RUNNING INTENT EVALS")
        print("="*50 + "\n")
        await run_intent_evals()
        
    if args.type in ["planner", "all"]:
        print("\n" + "="*50)
        print("RUNNING PLANNER EVALS")
        print("="*50 + "\n")
        await run_planner_evals()

if __name__ == "__main__":
    asyncio.run(main())
