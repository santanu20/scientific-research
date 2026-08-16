"""Shared dataclasses (leaf module — imports nothing local). Phase 1 cycle-break."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TopicTemplate:
    """Template defining how to search, filter, rank, and organize a topic."""

    name: str
    display_name: str
    synonyms: list[str] = field(default_factory=list)
    landmarks: list[str] = field(default_factory=list)
    must_have: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    report_sections: list[str] = field(default_factory=list)
    boost_terms: dict[str, float] = field(default_factory=dict)
    target_property: str = ""
    unit: str = ""
    valid_range: tuple[float, float] = (0.0, 0.0)
