import re
import spacy
from typing import Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from core.event_store import Event

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None

@dataclass
class Memory:
    id: str
    memory_type: str
    content: str
    timestamp: datetime
    source: str
    importance: float
    project: Optional[str] = None
    people: list[str] = field(default_factory=list)
    relationships: dict[str, str] = field(default_factory=dict)
    state_transitions: dict[str, str] = field(default_factory=dict)
    raw_event: Optional[Event] = None


class MemoryExtractor:
    """
    Extracts Memory objects from raw Events using Regex, Keyword patterns, and spaCy.

    The extractor is intentionally dataset-agnostic at the algorithm level:
      - PROJECT_KEYWORDS / KNOWN_PEOPLE are *seed defaults* for the mock dataset.
        Any caller can inject different values via __init__ parameters — no
        subclassing required — making the extractor reusable across datasets.
      - State-transition detection (_extract_state_transitions) uses generic
        regex patterns; it does not hard-code any entity names.
      - spaCy NER (ORG / PERSON) provides a zero-shot fallback for unseen entities.
    """

    # Seed defaults — override via __init__ for different datasets.
    # Key = lowercase substring that appears in text; value = canonical project name.
    DEFAULT_PROJECT_KEYWORDS: dict[str, str] = {
        "uie": "UIE",
        "unified intelligence engine": "UIE",
        "southridge": "Southridge",
        "hiring rubric": "Hiring",
        "rubric": "Hiring",
        "interview": "Hiring",
        "candidate": "Hiring",
    }

    # Seed person list — augmented at runtime by spaCy PERSON entity recognition.
    DEFAULT_KNOWN_PEOPLE: list[str] = [
        "Nina", "Ravi", "Rhea", "Cedric", "Pari", "Mom", "Karan", "Aarav"
    ]

    IMPORTANCE_KEYWORDS = [
        "promised", "must", "due", "urgent", "critical", "review",
        "deadline", "risk", "blocked", "missing",
    ]

    def __init__(
        self,
        project_keywords: dict[str, str] | None = None,
        known_people: list[str] | None = None,
    ):
        """Initialise the extractor with optional dataset-specific overrides.

        Args:
            project_keywords: Mapping of lowercase keyword → canonical project
                label.  If None, DEFAULT_PROJECT_KEYWORDS is used.
            known_people: List of person names to use as a fallback seed when
                spaCy is unavailable.  If None, DEFAULT_KNOWN_PEOPLE is used.
        """
        self.PROJECT_KEYWORDS: dict[str, str] = (
            project_keywords if project_keywords is not None
            else dict(self.DEFAULT_PROJECT_KEYWORDS)
        )
        self.KNOWN_PEOPLE: list[str] = (
            known_people if known_people is not None
            else list(self.DEFAULT_KNOWN_PEOPLE)
        )

    def _extract_projects(self, text: str) -> Optional[str]:
        """Extract project label from text.

        Priority order:
          1. Exact keyword match from PROJECT_KEYWORDS (dataset-seeded labels).
          2. spaCy ORG entity (dynamic discovery for unseen project names).
        """
        text_lower = text.lower()
        for kw, proj in self.PROJECT_KEYWORDS.items():
            if kw in text_lower:
                return proj
        # Dynamic fallback: use spaCy ORG detection for unknown project names
        if nlp:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    name = ent.text.strip()
                    if name and len(name) > 2 and name not in self.KNOWN_PEOPLE:
                        return name
        return None

    def _calculate_importance(self, text: str) -> float:
        text_lower = text.lower()
        score = 0.0
        for kw in self.IMPORTANCE_KEYWORDS:
            if kw in text_lower:
                score += 0.2
        return min(score, 1.0)

    def _extract_people(self, text: str) -> list[str]:
        # Fallback list if spaCy is missing
        people = []
        
        # spaCy extraction
        if nlp:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    # basic cleanup
                    name = ent.text.strip()
                    if name and name not in people:
                        people.append(name)
        
        # Regex/Keyword fallback
        for person in self.KNOWN_PEOPLE:
            if person in text and person not in people:
                people.append(person)
                
        return people

    def _extract_relationships_and_type(self, text: str, people: list[str]) -> tuple[str, dict[str, list[str]]]:
        text_lower = text.lower()
        relationships: dict[str, list[str]] = {}
        memory_type = "note"

        # Relationships: waiting_on vs waiting_for (direction matters)
        waiting_kws = ["waiting on", "waiting for"]
        blocked_kws = ["blocked by", "blocks", "blocking"]
        
        for kw in waiting_kws:
            if kw in text_lower:
                memory_type = "dependency"
                kw_idx = text_lower.find(kw)
                for p in people:
                    p_idx = text_lower.find(p.lower())
                    if p_idx != -1:
                        if p_idx < kw_idx:
                            # e.g. "Rhea is waiting for..." -> Rhea is waiting on me
                            relationships.setdefault("person_waiting_on_me", []).append(p)
                        else:
                            # e.g. "I am waiting on Ravi..." -> I am waiting on Ravi
                            relationships.setdefault("i_am_waiting_on_person", []).append(p)
                break
                
        for kw in blocked_kws:
            if kw in text_lower:
                memory_type = "dependency"
                for p in people:
                    if p.lower() in text_lower:
                        # For simplicity, if they block or are blocked by, they are a blocker to us
                        relationships.setdefault("blocked_by", []).append(p)
                break

        # Other types
        if "due" in text_lower or "deadline" in text_lower:
            memory_type = "deadline"
        elif "promised" in text_lower or "will send" in text_lower or "owe" in text_lower:
            memory_type = "commitment"
        elif "risk" in text_lower or "failure" in text_lower:
            memory_type = "risk"
        elif "standup" in text_lower or "sync" in text_lower or "review" in text_lower or "meeting" in text_lower:
            memory_type = "meeting"

        return memory_type, relationships

    # Generic status vocabulary for state-transition detection.
    # These are domain-agnostic resolution/completion signals.
    _STATUS_RESOLVED = {"approved", "resolved", "confirmed", "accepted", "done",
                        "completed", "signed", "merged", "shipped", "closed", "unblocked"}
    _STATUS_BLOCKED  = {"blocked", "rejected", "declined", "cancelled", "failed",
                        "pending", "waiting", "stalled", "overdue", "stuck"}

    # Regex: "X is now <status>" / "X has been <status>" / "<status>: X"
    _TRANSITION_RE = re.compile(
        r"(?P<entity>[A-Z][A-Za-z0-9\s\-_]{2,40})"   # leading-cap entity name (≥3 chars)
        r"\s+(?:is now|has been|was|got|marked as|set to)\s+"
        r"(?P<state>[a-z]+)",
        re.IGNORECASE,
    )

    def _extract_state_transitions(self, text: str) -> dict[str, str]:
        """Generic state-transition extractor — works on any entity name.

        Strategy:
          1. Regex scan for "<Entity> is now/has been/was <state>" patterns.
          2. Validate <state> against known resolved/blocked vocabularies.
          3. Return {entity: state} for any match.
        """
        text_lower = text.lower()
        transitions: dict[str, str] = {}

        # Pattern-based extraction
        for m in self._TRANSITION_RE.finditer(text):
            entity = m.group("entity").strip()
            state  = m.group("state").lower()
            if state in self._STATUS_RESOLVED or state in self._STATUS_BLOCKED:
                transitions[entity] = state

        # Fallback heuristic: single-sentence texts like
        #   "Clause 8 approved." / "Invoice blocked pending review."
        #   where the entity precedes the status word with no auxiliary verb.
        all_status = self._STATUS_RESOLVED | self._STATUS_BLOCKED
        if not transitions:
            words = text.split()
            for i, word in enumerate(words):
                if word.lower().rstrip(".,!?") in all_status and i > 0:
                    # Take up to 3 preceding words as the entity
                    entity_words = words[max(0, i - 3):i]
                    entity = " ".join(entity_words).strip(".,!?;:").strip()
                    if entity and len(entity) >= 3:
                        transitions[entity] = word.lower().rstrip(".,!?")
                        break  # one transition per sentence is enough

        return transitions

    def extract_memories(self, events: list[Event]) -> list[Memory]:
        memories = []
        for i, ev in enumerate(events):
            if ev.is_noise:
                continue
                
            text = ev.content
            project = self._extract_projects(text)
            importance = self._calculate_importance(text)
            people = self._extract_people(text)
            memory_type, relationships = self._extract_relationships_and_type(text, people)
            state_transitions = self._extract_state_transitions(text)
            
            # If importance is very high and it's a note, upgrade it
            if importance >= 0.4 and memory_type == "note":
                memory_type = "commitment"
                
            memories.append(Memory(
                id=f"mem_{i}",
                memory_type=memory_type,
                content=text,
                timestamp=ev.timestamp,
                source=ev.source,
                importance=importance,
                project=project,
                people=people,
                relationships=relationships,
                state_transitions=state_transitions,
                raw_event=ev
            ))
        return memories
