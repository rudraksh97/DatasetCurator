import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add backend directory to path
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BACKEND_DIR))

try:
    from services.llm.planner import ExecutionPlannerService
except ImportError as e:
    print(f"Error importing services: {e}")
    sys.exit(1)

WHITE = "\033[0m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
BOLD = "\033[1m"

def check_assertion(plan: Dict[str, Any], assertion: Dict[str, Any]) -> bool:
    steps = plan.get("steps", [])
    atype = assertion["type"]
    
    if atype == "contains_operation":
        target_op = assertion["operation"]
        target_params = assertion.get("params_subset", {})
        
        for step in steps:
            if step.get("operation") == target_op:
                # Check params if specified
                if target_params:
                    step_params = step.get("params", {})
                    # Check if all target params are in step params with strict equality
                    if all(step_params.get(k) == v for k, v in target_params.items()):
                        return True
                else:
                    return True
        return False
        
    elif atype == "step_count_equals":
        return len(steps) == assertion["count"]
    
    elif atype == "ordered_operations":
        expected_sequence = assertion["operations"]
        plan_ops = [(s.get("operation"), s.get("params", {})) for s in steps]
        
        current_idx = 0
        for expected_op in expected_sequence:
            target_op = expected_op["operation"]
            target_params = expected_op.get("params_subset", {})
            
            found = False
            # Search for the next expected op starting from current_idx
            for i in range(current_idx, len(plan_ops)):
                actual_op, actual_params = plan_ops[i]
                if actual_op == target_op:
                    # Check params if specified
                    if target_params:
                        if all(actual_params.get(k) == v for k, v in target_params.items()):
                            current_idx = i + 1
                            found = True
                            break
                    else:
                        current_idx = i + 1
                        found = True
                        break
            
            if not found:
                return False
                
        return True
        
    return False

async def run_evals():
    if not os.environ.get("OPENROUTER_API_KEY"):
        print(f"{RED}Error: OPENROUTER_API_KEY environment variable not set.{WHITE}")
        return

    planner = ExecutionPlannerService()
    
    dataset_path = Path(__file__).parent / "datasets" / "planner_cases.json"
    if not dataset_path.exists():
        print(f"{RED}Error: Dataset not found at {dataset_path}{WHITE}")
        return
        
    with open(dataset_path, "r") as f:
        cases = json.load(f)
        
    print(f"{BOLD}Running {len(cases)} planner evaluation cases...{WHITE}\n")
    
    results = []
    correct_count = 0
    
    for i, case in enumerate(cases):
        message = case["message"]
        assertions = case.get("assertions", [])
        columns = case.get("columns", [])
        
        print(f"[{i+1}/{len(cases)}] '{message}'...", end="", flush=True)
        
        try:
            plan = await planner.create_plan(
                user_message=message,
                columns=columns
            )
            
            failed_assertions = []
            for assertion in assertions:
                if not check_assertion(plan, assertion):
                    failed_assertions.append(assertion)
            
            is_correct = len(failed_assertions) == 0
            
            if is_correct:
                print(f" {GREEN}✓{WHITE}")
                correct_count += 1
            else:
                print(f" {RED}✗{WHITE}")
                for fail in failed_assertions:
                    print(f"   Failed assertion: {fail}")
                print(f"   Actual plan steps: {[s.get('operation') for s in plan.get('steps', [])]}")
            
            results.append({
                "case": case,
                "plan": plan,
                "correct": is_correct,
                "failed_assertions": failed_assertions
            })
            
        except Exception as e:
            print(f" {RED}Error: {e}{WHITE}")
            results.append({
                "case": case,
                "error": str(e),
                "correct": False
            })
            
        # Add delay to avoid rate limits
        await asyncio.sleep(2)

    # Summary
    print(f"\n{BOLD}=== Summary ==={WHITE}")
    accuracy = (correct_count / len(cases)) * 100
    color = GREEN if accuracy == 100 else (YELLOW if accuracy >= 80 else RED)
    print(f"Total Accuracy: {color}{accuracy:.1f}% ({correct_count}/{len(cases)}){WHITE}")
        
    # Write results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "planner_results.json"
    
    with open(results_file, "w") as f:
        json.dump({
            "summary": {
                "total": len(cases),
                "correct": correct_count,
                "accuracy": accuracy
            },
            "detailed_results": results
        }, f, indent=2)
        
    print(f"\nDetailed results saved to {results_file}")

if __name__ == "__main__":
    asyncio.run(run_evals())
