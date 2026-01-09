import asyncio
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any

# Add backend directory to path so we can import services
# Assuming this script is at backend/evals/eval_intent.py
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BACKEND_DIR))

try:
    from services.llm.intent import IntentClassifierService
except ImportError as e:
    print(f"Error importing services: {e}")
    print(f"sys.path: {sys.path}")
    sys.exit(1)

WHITE = "\033[0m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
BOLD = "\033[1m"

async def run_evals():
    if not os.environ.get("OPENROUTER_API_KEY"):
        print(f"{RED}Error: OPENROUTER_API_KEY environment variable not set.{WHITE}")
        return

    classifier = IntentClassifierService()
    
    dataset_path = Path(__file__).parent / "datasets" / "intent_cases.json"
    if not dataset_path.exists():
        print(f"{RED}Error: Dataset not found at {dataset_path}{WHITE}")
        return
        
    with open(dataset_path, "r") as f:
        cases = json.load(f)
        
    print(f"{BOLD}Running {len(cases)} intent evaluation cases...{WHITE}\n")
    
    results = []
    correct_count = 0
    by_category = defaultdict(lambda: {"total": 0, "correct": 0})
    
    for i, case in enumerate(cases):
        message = case["message"]
        expected = case["expected_intent"]
        category = case.get("category", "unknown")
        
        print(f"[{i+1}/{len(cases)}] '{message}'...", end="", flush=True)
        
        try:
            result = await classifier.classify(
                message=message,
                has_data=case.get("has_data", False),
                columns=case.get("columns", [])
            )
            
            actual = result.get("intent")
            is_correct = (actual == expected)
            
            if is_correct:
                print(f" {GREEN}✓{WHITE}")
                correct_count += 1
                by_category[category]["correct"] += 1
            else:
                print(f" {RED}✗ (Expected: {expected}, Got: {actual}){WHITE}")
                
            by_category[category]["total"] += 1
            
            results.append({
                "case": case,
                "actual": actual,
                "correct": is_correct,
                "explanation": result.get("explanation")
            })
            
        except Exception as e:
            print(f" {RED}Error: {e}{WHITE}")
            results.append({
                "case": case,
                "error": str(e),
                "correct": False
            })
            by_category[category]["total"] += 1

        # Add small delay to avoid rate limits
        await asyncio.sleep(0.5)

    # Summary
    print(f"\n{BOLD}=== Summary ==={WHITE}")
    accuracy = (correct_count / len(cases)) * 100
    color = GREEN if accuracy == 100 else (YELLOW if accuracy >= 80 else RED)
    print(f"Total Accuracy: {color}{accuracy:.1f}% ({correct_count}/{len(cases)}){WHITE}")
    
    print(f"\n{BOLD}By Category:{WHITE}")
    for cat, stats in sorted(by_category.items()):
        cat_acc = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        cat_color = GREEN if cat_acc == 100 else (YELLOW if cat_acc >= 80 else RED)
        print(f"  {cat:<15}: {cat_color}{cat_acc:.1f}% ({stats['correct']}/{stats['total']}){WHITE}")
        
    # Write detailed results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "intent_results.json"
    
    with open(results_file, "w") as f:
        json.dump({
            "summary": {
                "total": len(cases),
                "correct": correct_count,
                "accuracy": accuracy,
                "by_category": dict(by_category)
            },
            "detailed_results": results
        }, f, indent=2)
        
    print(f"\nDetailed results saved to {results_file}")

if __name__ == "__main__":
    asyncio.run(run_evals())
