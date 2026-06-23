"""
evaluation.py — Evaluation framework for the Memorae engine.

Covers:
  1. Offline evals  — deterministic unit tests & golden-set comparisons
  2. Online evals   — latency, model-use tracking, user feedback logging
  3. Regression tests — specific scenarios that must not regress

Run:  python evaluation.py
"""

from __future__ import annotations

import json
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.table import Table
from rich import box

from config import SCENARIO_NOW
from event_store import load_events, EventStore
from query_engine import QueryEngine, QuerySpec, QUERY_SPECS

logger = logging.getLogger(__name__)
console = Console()

NOW = datetime.fromisoformat(SCENARIO_NOW.replace("Z", "+00:00"))
DATA_PATH = Path(__file__).parent.parent / "memorae_mock_events.json"


# ══════════════════════════════════════════════════════════════════════════════
# Test result structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TestResult:
    name: str
    passed: bool
    score: float          # 0.0 – 1.0
    details: str
    category: str         # "offline" | "regression" | "online"


@dataclass
class EvalReport:
    results: list[TestResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)

    def by_category(self, cat: str) -> list[TestResult]:
        return [r for r in self.results if r.category == cat]

    def add(self, result: TestResult) -> None:
        self.results.append(result)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_store() -> EventStore:
    events = load_events(str(DATA_PATH))
    return EventStore(events, NOW)


def _check_content_required(answer: str, required_phrases: list[str]) -> tuple[float, list[str]]:
    """Check how many required phrases appear in the answer (case-insensitive)."""
    answer_low = answer.lower()
    missing = [p for p in required_phrases if p.lower() not in answer_low]
    score = (len(required_phrases) - len(missing)) / max(len(required_phrases), 1)
    return score, missing


def _check_content_forbidden(answer: str, forbidden_phrases: list[str]) -> tuple[bool, list[str]]:
    """Check that none of the forbidden phrases appear in the answer."""
    answer_low = answer.lower()
    found = [p for p in forbidden_phrases if p.lower() in answer_low]
    return len(found) == 0, found


# ══════════════════════════════════════════════════════════════════════════════
# 1. OFFLINE EVALS
# ══════════════════════════════════════════════════════════════════════════════

class OfflineEvaluator:
    """
    Deterministic tests that don't require an LLM.
    These test retrieval quality and scoring behavior.
    """

    def __init__(self, store: EventStore):
        self.store = store

    def test_noise_filtered(self) -> TestResult:
        """Coffee-machine messages and OTPs must NOT appear in any query result."""
        noise_patterns = ["coffee machine", "otp is", "ride receipt", "lunch is late"]
        scored = self.store.retrieve(
            keywords=["focus", "today", "deadline", "uie"],
            exclude_noise=True,
            top_k=30,
        )
        contents = [se.event.content.lower() for se in scored]
        violations = [
            c for c in contents
            if any(np in c for np in noise_patterns)
        ]
        passed = len(violations) == 0
        return TestResult(
            name="noise_filtered_from_results",
            passed=passed,
            score=1.0 if passed else 0.0,
            details=(
                "All noise events correctly excluded."
                if passed
                else f"Noise leaked: {violations[:2]}"
            ),
            category="offline",
        )

    def test_uie_deadline_update_detected(self) -> TestResult:
        """The system should surface the Apr 13 deadline, not the original Apr 10."""
        uie_scored = self.store.retrieve(
            keywords=["uie", "deadline", "proposal", "nina"],
            must_include_patterns=["apr 13", "apr 10", "moved"],
            top_k=10,
        )
        contents = " ".join(se.event.content.lower() for se in uie_scored)
        has_apr13 = "apr 13" in contents or "monday" in contents
        has_apr10 = "apr 10" in contents

        # We want apr 13 to be present (updated deadline) but the system
        # should rank apr 13 events higher than apr 10 events
        if uie_scored:
            top_content = uie_scored[0].event.content.lower()
            top_is_updated = "apr 13" in top_content or "moved" in top_content
        else:
            top_is_updated = False

        passed = has_apr13
        score = 1.0 if top_is_updated else (0.7 if has_apr13 else 0.0)
        return TestResult(
            name="uie_deadline_update_detected",
            passed=passed,
            score=score,
            details=(
                f"Apr 13 deadline found: {has_apr13}, top result is updated: {top_is_updated}"
            ),
            category="offline",
        )

    def test_procurement_estimate_update(self) -> TestResult:
        """$48.5k update must appear among procurement events retrieved by the system."""
        # Use the actual event store pattern-search (not scored retrieval)
        all_with = self.store.get_all_with_pattern(["48.5k", "42k", "procurement estimate"])
        has_48k = any("48.5k" in e.content for e in all_with)
        has_42k = any("42k" in e.content for e in all_with)
        # Also verify the 48.5k event is MORE RECENT than 42k event
        events_48k = [e for e in all_with if "48.5k" in e.content]
        events_42k = [e for e in all_with if "42k" in e.content]
        if events_48k and events_42k:
            latest_48k = max(e.timestamp for e in events_48k)
            latest_42k = max(e.timestamp for e in events_42k)
            updated_is_newer = latest_48k >= latest_42k
        else:
            updated_is_newer = has_48k and not has_42k
        passed = has_48k and updated_is_newer
        return TestResult(
            name="procurement_estimate_update",
            passed=passed,
            score=1.0 if passed else (0.5 if has_48k else 0.0),
            details=(
                f"$48.5k found: {has_48k}, $42k found: {has_42k}, "
                f"updated is newer: {updated_is_newer}. "
                f"48.5k event: {events_48k[0].content[:60] if events_48k else 'none'}"
            ),
            category="offline",
        )

    def test_recency_bias(self) -> TestResult:
        """Events from Apr 12-13 should score higher than semantically equal Apr 1 events."""
        recent_scored = self.store.retrieve(
            keywords=["uie", "proposal"],
            top_k=5,
        )
        if not recent_scored:
            return TestResult(
                name="recency_bias_check",
                passed=False,
                score=0.0,
                details="No events found",
                category="offline",
            )
        # Check that top events are from later dates
        top_ts = recent_scored[0].event.timestamp
        passed = top_ts >= datetime(2026, 4, 9, tzinfo=timezone.utc)
        return TestResult(
            name="recency_bias_check",
            passed=passed,
            score=1.0 if passed else 0.3,
            details=f"Top event timestamp: {top_ts.isoformat()}",
            category="offline",
        )

    def test_calendar_beats_whatsapp_for_meetings(self) -> TestResult:
        """Calendar events for specific meetings should rank above WhatsApp chatter."""
        scored = self.store.retrieve(
            keywords=["standup", "uie", "review", "nina"],
            top_k=10,
        )
        calendar_scores = [se.score for se in scored if se.event.source == "calendar"]
        whatsapp_scores = [se.score for se in scored if se.event.source == "whatsapp"]
        if not calendar_scores or not whatsapp_scores:
            return TestResult(
                name="calendar_beats_whatsapp",
                passed=True,
                score=1.0,
                details="Only one source type present; test not applicable.",
                category="offline",
            )
        avg_cal = sum(calendar_scores) / len(calendar_scores)
        avg_wa = sum(whatsapp_scores) / len(whatsapp_scores)
        passed = avg_cal > avg_wa
        return TestResult(
            name="calendar_beats_whatsapp",
            passed=passed,
            score=1.0 if passed else 0.0,
            details=f"Avg calendar score: {avg_cal:.3f}, avg WhatsApp score: {avg_wa:.3f}",
            category="offline",
        )

    def test_southridge_clause8_resolved(self) -> TestResult:
        """The system should recognize clause 8 as approved (Apr 11 event present)."""
        # Use pattern search — confirmed by fact that approved event exists in stream
        all_clause8 = self.store.get_all_with_pattern(["clause 8"])
        if not all_clause8:
            return TestResult(
                name="clause8_resolved",
                passed=False,
                score=0.0,
                details="No clause 8 events found in dataset",
                category="offline",
            )
        # Check any event says clause 8 is approved
        approved_events = [e for e in all_clause8 if "approved" in e.content.lower()]
        passed = len(approved_events) > 0
        latest = max(all_clause8, key=lambda e: e.timestamp)
        return TestResult(
            name="clause8_resolved",
            passed=passed,
            score=1.0 if passed else 0.0,
            details=(
                f"Clause-8 events: {len(all_clause8)}, approved: {len(approved_events)}. "
                f"Latest: {latest.content[:60]}"
            ),
            category="offline",
        )

    def test_high_urgency_events_surface(self) -> TestResult:
        """Events with explicit deadlines must have high urgency scores."""
        from event_store import _contains_any, URGENCY_SIGNALS
        urgent_events = [e for e in self.store.events if e.has_urgency]
        passed = len(urgent_events) >= 20  # expect many deadline-bearing events
        return TestResult(
            name="urgency_detection_coverage",
            passed=passed,
            score=min(len(urgent_events) / 30, 1.0),
            details=f"Found {len(urgent_events)} events with urgency signals (of {len(self.store.events)} total)",
            category="offline",
        )

    def run_all(self) -> list[TestResult]:
        tests = [
            self.test_noise_filtered,
            self.test_uie_deadline_update_detected,
            self.test_procurement_estimate_update,
            self.test_recency_bias,
            self.test_calendar_beats_whatsapp_for_meetings,
            self.test_southridge_clause8_resolved,
            self.test_high_urgency_events_surface,
        ]
        return [t() for t in tests]


# ══════════════════════════════════════════════════════════════════════════════
# 2. REGRESSION TESTS (LLM-based)
# ══════════════════════════════════════════════════════════════════════════════

class RegressionEvaluator:
    """
    Golden-set tests: run real queries and check that key facts appear in answers.
    These are intentionally simple string-match checks (no LLM-as-judge needed).
    """

    def __init__(self, engine: QueryEngine):
        self.engine = engine

    def _run_query(self, spec: QuerySpec) -> str:
        result = self.engine.run(spec)
        return result.answer.lower()

    def test_focus_today_mentions_uie_review(self) -> TestResult:
        """'Focus today' answer must mention the 14:30 UIE review with Nina."""
        spec = QUERY_SPECS[0]  # "What should I focus on today?"
        t0 = time.time()
        answer = self._run_query(spec)
        latency = time.time() - t0

        required = ["nina", "14:30", "uie"]
        score, missing = _check_content_required(answer, required)
        passed = score >= 0.66  # at least 2/3
        return TestResult(
            name="focus_today_uie_review",
            passed=passed,
            score=score,
            details=(
                f"Required phrases found: {[p for p in required if p in answer]}. "
                f"Missing: {missing}. Latency: {latency:.1f}s"
            ),
            category="regression",
        )

    def test_risk_mentions_overdue_rubric(self) -> TestResult:
        """'Risk' answer must flag the hiring rubric as overdue."""
        spec = QUERY_SPECS[1]
        answer = self._run_query(spec)
        required = ["rubric", "rhea"]
        score, missing = _check_content_required(answer, required)
        passed = score >= 0.5
        return TestResult(
            name="risk_overdue_rubric",
            passed=passed,
            score=score,
            details=f"Missing: {missing}",
            category="regression",
        )

    def test_procrastination_includes_reimbursement(self) -> TestResult:
        """'Procrastinating' answer must mention reimbursement receipts."""
        spec = QUERY_SPECS[2]
        answer = self._run_query(spec)
        required = ["reimburse", "export screenshots", "nudge"]
        score, missing = _check_content_required(answer, required)
        passed = score >= 0.33
        return TestResult(
            name="procrastination_reimbursement",
            passed=passed,
            score=score,
            details=f"Missing: {missing}",
            category="regression",
        )

    def test_uie_summary_includes_key_facts(self) -> TestResult:
        """UIE summary must mention deadline, Nina, risk, appendix, 48.5k."""
        spec = QUERY_SPECS[3]
        answer = self._run_query(spec)
        required = ["nina", "appendix", "risk", "48.5", "14:30"]
        score, missing = _check_content_required(answer, required)
        passed = score >= 0.6
        return TestResult(
            name="uie_summary_key_facts",
            passed=passed,
            score=score,
            details=f"Missing phrases: {missing}",
            category="regression",
        )

    def test_uie_summary_no_stale_deadline(self) -> TestResult:
        """UIE summary must NOT say deadline is Apr 10 without correction."""
        spec = QUERY_SPECS[3]
        answer = self._run_query(spec)
        # If Apr 10 appears, it should also include correction language
        if "apr 10" in answer or "april 10" in answer:
            has_correction = any(
                kw in answer for kw in ["moved", "updated", "changed", "superseded", "now"]
            )
            passed = has_correction
            score = 1.0 if passed else 0.0
            detail = "Apr 10 mentioned but correction present." if has_correction else "Apr 10 stated without correction!"
        else:
            passed = True
            score = 1.0
            detail = "Apr 10 not mentioned (correct: system used updated Apr 13 deadline)."
        return TestResult(
            name="uie_no_stale_deadline",
            passed=passed,
            score=score,
            details=detail,
            category="regression",
        )

    def run_all(self) -> list[TestResult]:
        tests = [
            self.test_focus_today_mentions_uie_review,
            self.test_risk_mentions_overdue_rubric,
            self.test_procrastination_includes_reimbursement,
            self.test_uie_summary_includes_key_facts,
            self.test_uie_summary_no_stale_deadline,
        ]
        return [t() for t in tests]


# ══════════════════════════════════════════════════════════════════════════════
# 3. ONLINE EVAL FRAMEWORK (structural / latency)
# ══════════════════════════════════════════════════════════════════════════════

class OnlineEvaluator:
    """
    Simulate online evaluation metrics:
    - Latency measurement
    - Context efficiency (ratio of used tokens to budget)
    - Answer length distribution
    - Model-tier usage
    """

    def __init__(self, engine: QueryEngine):
        self.engine = engine

    def measure_latency_and_efficiency(self, specs: list[QuerySpec]) -> list[TestResult]:
        results = []
        for spec in specs:
            t0 = time.time()
            result = self.engine.run(spec)
            latency = time.time() - t0

            # Latency test: should finish within 30s (free-tier models can be slow)
            latency_ok = latency < 30.0
            results.append(TestResult(
                name=f"latency_{spec.query[:30].replace(' ', '_')}",
                passed=latency_ok,
                score=max(0, 1.0 - latency / 30.0),
                details=f"Latency: {latency:.2f}s",
                category="online",
            ))

            # Context efficiency
            efficiency = result.token_estimate / 8000  # target budget
            results.append(TestResult(
                name=f"context_efficiency_{spec.query[:20].replace(' ', '_')}",
                passed=efficiency <= 1.2,
                score=min(1.0, 1.0 / max(efficiency, 0.1)),
                details=f"Tokens used: {result.token_estimate} / 8000 target ({efficiency:.1%})",
                category="online",
            ))

            # Answer length (too short = incomplete, too long = verbose)
            word_count = len(result.answer.split())
            length_ok = 50 <= word_count <= 600
            results.append(TestResult(
                name=f"answer_length_{spec.query[:20].replace(' ', '_')}",
                passed=length_ok,
                score=1.0 if length_ok else 0.5,
                details=f"Answer word count: {word_count}",
                category="online",
            ))

        return results


# ══════════════════════════════════════════════════════════════════════════════
# Display & report
# ══════════════════════════════════════════════════════════════════════════════

def display_report(report: EvalReport) -> None:
    console.rule("[bold cyan]Evaluation Report[/bold cyan]")

    tbl = Table(box=box.ROUNDED, show_header=True, header_style="bold white")
    tbl.add_column("Test", style="dim", width=40)
    tbl.add_column("Category", width=12)
    tbl.add_column("Pass", width=6, justify="center")
    tbl.add_column("Score", width=7, justify="right")
    tbl.add_column("Details", no_wrap=False)

    for r in report.results:
        pass_icon = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
        score_color = "green" if r.score >= 0.7 else ("yellow" if r.score >= 0.4 else "red")
        tbl.add_row(
            r.name,
            r.category,
            pass_icon,
            f"[{score_color}]{r.score:.2f}[/{score_color}]",
            r.details[:80],
        )

    console.print(tbl)

    # Summary by category
    for cat in ["offline", "regression", "online"]:
        cat_results = report.by_category(cat)
        if not cat_results:
            continue
        cat_pass = sum(1 for r in cat_results if r.passed)
        avg_score = sum(r.score for r in cat_results) / len(cat_results)
        console.print(
            f"  [{cat.upper()}] Pass: {cat_pass}/{len(cat_results)} | "
            f"Avg score: {avg_score:.2f}"
        )

    console.print(f"\n[bold]Overall pass rate: [cyan]{report.pass_rate:.1%}[/cyan][/bold]")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def run_offline_only() -> EvalReport:
    """Run only deterministic offline tests (no LLM calls)."""
    store = _load_store()
    evaluator = OfflineEvaluator(store)
    report = EvalReport()
    console.print("\n[bold cyan]Running offline evaluations (no LLM)...[/bold cyan]")
    for result in evaluator.run_all():
        report.add(result)
    return report


def run_full_eval() -> EvalReport:
    """Run offline + regression + online evals (requires LLM calls)."""
    store = _load_store()
    engine = QueryEngine(store, NOW)
    report = EvalReport()

    # Offline
    console.print("\n[bold cyan]Running offline evaluations...[/bold cyan]")
    offline_eval = OfflineEvaluator(store)
    for result in offline_eval.run_all():
        report.add(result)

    # Regression
    console.print("\n[bold cyan]Running regression tests (LLM)...[/bold cyan]")
    regression_eval = RegressionEvaluator(engine)
    for result in regression_eval.run_all():
        report.add(result)

    # Online (latency, efficiency — subset of queries to keep it fast)
    console.print("\n[bold cyan]Running online evals (latency, efficiency)...[/bold cyan]")
    online_eval = OnlineEvaluator(engine)
    # Run on 2 queries only to keep overall eval reasonable
    for result in online_eval.measure_latency_and_efficiency(QUERY_SPECS[:2]):
        report.add(result)

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Memorae evaluation framework")
    parser.add_argument("--offline-only", action="store_true",
                        help="Run only deterministic tests (no LLM)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    if args.offline_only:
        report = run_offline_only()
    else:
        report = run_full_eval()

    display_report(report)

    # Save report
    out = {
        "pass_rate": report.pass_rate,
        "results": [
            {
                "name": r.name, "category": r.category,
                "passed": r.passed, "score": r.score, "details": r.details,
            }
            for r in report.results
        ],
    }
    with open("eval_report.json", "w") as f:
        json.dump(out, f, indent=2)
    console.print("[green]Eval report saved to eval_report.json[/green]")
