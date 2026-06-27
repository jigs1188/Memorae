"""Final engineering review - regression + unseen query suite."""
import json, logging, time
logging.basicConfig(level=logging.WARNING)

from evaluation.evaluation import _load_store, NOW, RegressionEvaluator, display_report, EvalReport
from core.project_builder import ProjectBuilder
from core.query_engine import QueryEngine

store = _load_store()
projects = ProjectBuilder(NOW).build_projects(store.memories)
engine = QueryEngine(store, NOW, projects=projects)

report = EvalReport()

# Regression tests (the 5 original preset-query golden tests)
print("\n=== REGRESSION TESTS ===")
reg = RegressionEvaluator(engine)

# Run tests one at a time with a delay to respect 5 RPM free-tier limit
reg_tests = [
    reg.test_focus_today_mentions_uie_review,
    reg.test_risk_mentions_overdue_rubric,
    reg.test_procrastination_includes_reimbursement,
    reg.test_uie_summary_includes_key_facts,
    reg.test_uie_summary_no_stale_deadline,
]
for i, test_fn in enumerate(reg_tests):
    r = test_fn()
    report.add(r)
    status = "PASS" if r.passed else "FAIL"
    print(f"[{status}] {r.name} (score={r.score:.2f}): {r.details[:100]}")
    if i < len(reg_tests) - 1:
        time.sleep(13)  # stay under 5 RPM on free tier

# Generalization tests (3 unseen-query structural tests)
print("\n=== GENERALIZATION TESTS ===")
gen_tests = [reg.test_who_is_waiting_on_me, reg.test_what_changed_recently, reg.test_which_projects_are_blocked]
for i, test_fn in enumerate(gen_tests):
    r = test_fn()
    report.add(r)
    status = "PASS" if r.passed else "FAIL"
    print(f"[{status}] {r.name} (score={r.score:.2f}): {r.details[:100]}")
    if i < len(gen_tests) - 1:
        time.sleep(13)

# Extended unseen-query suite (LLM answers)
print("\n=== EXTENDED UNSEEN QUERIES ===")
unseen = [
    "Who am I waiting on?",
    "Which deadlines moved?",
    "Summarize Southridge.",
    "Summarize Hiring.",
    "What does Nina need from me?",
    "What did Ravi update?",
    "Which meetings are approaching?",
    "What became unblocked?",
    "What dependencies remain?",
    "What personal/family tasks need my attention?",
]

extended_results = []
for i, q in enumerate(unseen):
    time.sleep(13)  # respect 5 RPM limit between every LLM call
    try:
        result = engine.run(q)
        extended_results.append({"query": q, "answer": result.answer, "model": result.model_used})
        print(f"\nQ: {q}")
        print(f"A: {result.answer[:300]}")
    except Exception as e:
        extended_results.append({"query": q, "error": str(e)})
        print(f"\nQ: {q}")
        print(f"ERROR: {e}")


# Save all results
out = {
    "regression_pass_rate": sum(1 for r in report.results if r.passed) / len(report.results),
    "regression_results": [
        {"name": r.name, "category": r.category, "passed": bool(r.passed), "score": float(r.score), "details": r.details}
        for r in report.results
    ],
    "extended_unseen_results": extended_results,
}
with open("final_review_results.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("\nSaved to final_review_results.json")
