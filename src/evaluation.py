"""Evaluation suite for the Aster & Row AI Support Agent."""
import json
import uuid
from typing import List, Dict, Any, Callable
from dataclasses import dataclass, field
from src.agent import agent
from src.config import config
from src.models import AgentResponse


@dataclass
class EvaluationCase:
    id: str
    category: str
    messages: List[Dict[str, str]]
    expect: Dict[str, Any]
    
    def __post_init__(self):
        if isinstance(self.messages, list) and self.messages and isinstance(self.messages[0], dict):
            pass
        else:
            self.messages = [{"role": "user", "content": str(self.messages)}]


@dataclass
class EvaluationResult:
    case_id: str
    category: str
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)
    response: AgentResponse = None
    error: str = ""


class Evaluator:
    def __init__(self):
        self.results: List[EvaluationResult] = []
    
    def load_cases(self, filepath: str) -> List[EvaluationCase]:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        cases = []
        for case_data in data.get("cases", []):
            case = EvaluationCase(
                id=case_data["id"],
                category=case_data.get("category", "unknown"),
                messages=case_data["messages"],
                expect=case_data["expect"]
            )
            cases.append(case)
        
        return cases

    def evaluate_case(self, case: EvaluationCase) -> EvaluationResult:
        session_id = f"eval-{case.id}-{uuid.uuid4().hex[:8]}"
        
        try:
            final_response = None
            for msg in case.messages:
                final_response = agent.process_message(session_id, msg["content"])
            
            passed, details = self._run_assertions(case, final_response)
            
            return EvaluationResult(
                case_id=case.id,
                category=case.category,
                passed=passed,
                details=details,
                response=final_response
            )
        except Exception as e:
            return EvaluationResult(
                case_id=case.id,
                category=case.category,
                passed=False,
                details={"error": str(e)},
                error=str(e)
            )

    def _run_assertions(self, case: EvaluationCase, response: AgentResponse) -> tuple:
        details = {}
        all_passed = True
        
        expect = case.expect
        
        if "must_include" in expect:
            for phrase in expect["must_include"]:
                found = phrase.lower() in response.answer.lower()
                details[f"must_include:{phrase}"] = found
                if not found:
                    all_passed = False
        
        if "must_not_include" in expect:
            for phrase in expect["must_not_include"]:
                found = phrase.lower() in response.answer.lower()
                details[f"must_not_include:{phrase}"] = not found
                if found:
                    all_passed = False
        
        if "must_include_concepts" in expect:
            for concept in expect["must_include_concepts"]:
                concept_keywords = concept.lower().split()
                found = any(kw in response.answer.lower() for kw in concept_keywords)
                details[f"must_include_concepts:{concept}"] = found
                if not found:
                    all_passed = False
        
        if "required_sources" in expect:
            for src in expect["required_sources"]:
                found = any(src in s for s in response.sources)
                details[f"required_sources:{src}"] = found
                if not found:
                    all_passed = False
        
        if "forbidden_sources_as_authority" in expect:
            for src in expect["forbidden_sources_as_authority"]:
                found = any(src in s for s in response.sources)
                details[f"forbidden_sources:{src}"] = not found
                if found:
                    all_passed = False
        
        if "tool" in expect:
            expected_tool = expect["tool"]
            if expected_tool == "not_called":
                tool_called = len(response.tool_calls) > 0
                details["tool_not_called"] = not tool_called
                if tool_called:
                    all_passed = False
            elif expected_tool == "optional_sanitized_lookup":
                lookup_called = any(tc.name == "order_lookup" for tc in response.tool_calls)
                details["tool_lookup_called"] = lookup_called
                if lookup_called:
                    for tc in response.tool_calls:
                        if tc.name == "order_lookup" and tc.result:
                            result = tc.result
                            if result.get("success") and "order" in result:
                                order = result["order"]
                                forbidden_fields = ["customer_email", "shipping_address", "internal_risk_score", "internal_warehouse_note", "internal_support_tags"]
                                for field in forbidden_fields:
                                    if field in order:
                                        details[f"privacy_leak:{field}"] = True
                                        all_passed = False
        
        if "handoff" in expect:
            expected_handoff = expect["handoff"]
            details["handoff_expected"] = expected_handoff
            details["handoff_actual"] = response.handoff_recommended
            if expected_handoff != response.handoff_recommended:
                all_passed = False
        
        if "must_refuse_to_disclose" in expect:
            for field in expect["must_refuse_to_disclose"]:
                leaked = field.lower() in response.answer.lower()
                details[f"refuse_disclose:{field}"] = not leaked
                if leaked:
                    all_passed = False
        
        return all_passed, details

    def run_all(self, cases: List[EvaluationCase]) -> List[EvaluationResult]:
        self.results = []
        for case in cases:
            print(f"Evaluating: {case.id}...")
            result = self.evaluate_case(case)
            self.results.append(result)
            status = "PASS" if result.passed else "FAIL"
            print(f"  {status}")
            if not result.passed:
                for k, v in result.details.items():
                    if v is False:
                        print(f"    ✗ {k}")
        return self.results

    def print_summary(self):
        if not self.results:
            print("No results to summarize.")
            return
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        
        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total: {total} | Passed: {passed} | Failed: {failed} | Pass Rate: {passed/total*100:.1f}%")
        
        categories = {}
        for r in self.results:
            if r.category not in categories:
                categories[r.category] = {"total": 0, "passed": 0}
            categories[r.category]["total"] += 1
            if r.passed:
                categories[r.category]["passed"] += 1
        
        print(f"\nBy Category:")
        for cat, stats in sorted(categories.items()):
            rate = stats["passed"] / stats["total"] * 100
            print(f"  {cat}: {stats['passed']}/{stats['total']} ({rate:.1f}%)")
        
        failed_cases = [r for r in self.results if not r.passed]
        if failed_cases:
            print(f"\nFailed Cases:")
            for r in failed_cases:
                print(f"  {r.case_id} ({r.category})")
                for k, v in r.details.items():
                    if v is False:
                        print(f"    ✗ {k}")
                if r.error:
                    print(f"    Error: {r.error}")


def run_evaluation():
    evaluator = Evaluator()
    cases = evaluator.load_cases(config.evaluation_path)
    print(f"Loaded {len(cases)} evaluation cases")
    
    results = evaluator.run_all(cases)
    evaluator.print_summary()
    
    return results


if __name__ == "__main__":
    run_evaluation()