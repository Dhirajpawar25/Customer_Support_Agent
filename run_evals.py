import json
from pathlib import Path
from typing import List, Dict

def load_cases() -> List[Dict]:
    cases_path = Path(__file__).parent / "evaluation" / "visible-cases.json"
    with cases_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("cases", [])

def main():
    cases = load_cases()
    print(f"Loaded {len(cases)} visible evaluation cases.")
    # Placeholder: actual agent invocation and assertions would go here.
    for case in cases:
        case_id = case.get("id", "<no-id>")
        print(f"[TODO] Evaluate case {case_id}")

if __name__ == "__main__":
    main()
