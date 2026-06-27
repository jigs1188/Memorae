from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime

from core.memory_extractor import Memory

@dataclass
class ProjectState:
    name: str
    status: str = "active"
    health: str = "green"  # green, yellow, red
    open_commitments: int = 0
    overdue_commitments: int = 0
    blocked_dependencies: int = 0
    key_people: set[str] = field(default_factory=set)
    related_memories: list[Memory] = field(default_factory=list)


class ProjectBuilder:
    """
    Aggregates Memory objects into higher-level Project states.
    """

    def __init__(self, now: datetime):
        self.now = now

    def build_projects(self, memories: list[Memory]) -> dict[str, ProjectState]:
        projects: dict[str, ProjectState] = {}

        for mem in memories:
            if not mem.project:
                continue
                
            proj_name = mem.project
            if proj_name not in projects:
                projects[proj_name] = ProjectState(name=proj_name)
                
            p = projects[proj_name]
            p.related_memories.append(mem)
            p.key_people.update(mem.people)

            # Analyze memory type for health/status
            if mem.memory_type == "commitment" or mem.memory_type == "deadline":
                # Very simple heuristic: if it mentions 'due', 'promised'
                p.open_commitments += 1
                
                # Assume memory timestamp might have passed
                # In a real system, we'd extract the actual date entity.
                # For this mock, if urgency keywords are present, assume it's at risk.
                content_lower = mem.content.lower()
                if "risk" in content_lower or "overdue" in content_lower or "was due" in content_lower or "past due" in content_lower:
                    p.overdue_commitments += 1
                    
            if mem.relationships.get("blocked_by") or mem.relationships.get("person_waiting_on_me") or mem.relationships.get("i_am_waiting_on_person"):
                p.blocked_dependencies += 1

        # Calculate health
        for p in projects.values():
            if p.overdue_commitments > 0 or p.blocked_dependencies > 0:
                p.health = "red"
            elif p.open_commitments > 2:
                p.health = "yellow"
            else:
                p.health = "green"

        return projects
