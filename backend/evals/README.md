# Dataset Curator Evals

This directory contains evaluation frameworks for the Dataset Curator backend.

## Structure

- `datasets/`: Contains JSON files with test cases and expected results.
    - `intent_cases.json`: Test cases for Intent Classification using `IntentClassifierService`.
    - `planner_cases.json`: Test cases for Execution Planning using `ExecutionPlannerService`.
- `results/`: Output directory for detailed evaluation results (JSON).
- `eval_intent.py`: Runner for intent evals.
- `eval_planner.py`: Runner for planner evals.
- `run.py`: Unified runner.

## Running Evals

You must have the `OPENROUTER_API_KEY` environment variable set.

Run from the project root:

```bash
# Run all evals
python backend/evals/run.py

# Run only intent evals
python backend/evals/run.py --type intent

# Run only planner evals
python backend/evals/run.py --type planner
```

## Adding New Cases

### Intent Cases
Add to `datasets/intent_cases.json`:
```json
{
  "message": "remove the email column",
  "has_data": true,
  "columns": ["id", "email"],
  "expected_intent": "transform_data",
  "category": "modification"
}
```

### Planner Cases
Add to `datasets/planner_cases.json`:
```json
{
  "message": "remove email",
  "columns": ["email", "id"],
  "assertions": [
    {
      "type": "contains_operation",
      "operation": "drop_column", 
      "params_subset": {"column": "email"}
    }
  ]
}
```
Assertion types:
- `contains_operation`: Checks if an operation exists. Optional `params_subset`.
- `step_count_equals`: Checks exact number of steps.
