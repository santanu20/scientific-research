"""Plugin architecture for domain-specific research modules.

Each plugin provides:
    - ontology terms + hierarchy
    - topic templates
    - materials/methods dictionaries (augments the built-in ones)
    - extraction rules (domain-specific patterns)
    - report structure defaults
    - statistics configuration

Built-in: geosciences (default, always loaded).
External plugins discovered via importlib.metadata.entry_points(group="geokit.research.plugins").

Usage:
    from _plugins import load_plugins, get_plugin
    load_plugins()
    plugin = get_plugin("petrology")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("scientific_research.plugins")


@dataclass
class ResearchPlugin:
    """A domain-specific research plugin."""

    name: str
    display_name: str
    description: str = ""
    ontology_terms: dict[str, list[str]] = field(default_factory=dict)
    materials: dict[str, tuple[list[str], str]] = field(default_factory=dict)
    methods: dict[str, dict[str, Any]] = field(default_factory=dict)
    topic_templates: list[str] = field(default_factory=list)
    report_sections: list[str] = field(default_factory=list)
    measurement_types: dict[str, tuple[float, float]] = field(default_factory=dict)
    landmarks: list[str] = field(default_factory=list)


_REGISTRY: dict[str, ResearchPlugin] = {}


def register_plugin(plugin: ResearchPlugin) -> None:
    """Register a research plugin."""
    _REGISTRY[plugin.name] = plugin
    log.info("Registered research plugin: %s (%s)", plugin.name, plugin.display_name)


def get_plugin(name: str) -> ResearchPlugin | None:
    """Get a registered plugin by name."""
    return _REGISTRY.get(name)


def all_plugins() -> dict[str, ResearchPlugin]:
    """Return all registered plugins."""
    return dict(_REGISTRY)


def load_plugins() -> None:
    """Load all available plugins — built-in + entry_points + auto-discovered.

    Order: built-in geosciences → entry_points → auto-discovered in package.
    """
    if _REGISTRY:
        return  # already loaded

    # 1. Built-in: geosciences (default, always loaded)
    register_plugin(
        ResearchPlugin(
            name="geosciences",
            display_name="Earth Sciences (built-in)",
            description="Default geosciences domain — petrology, geochemistry, structural, geophysics",
            topic_templates=[
                "thermometry", "barometry", "thermobarometry",
                "geochronology", "thermochronology",
                "igneous_petrology", "metamorphic_petrology",
                "tectonics", "planetary_geology",
            ],
            report_sections=[
                "Methodology and calibration",
                "Applications and case studies",
                "Limitations and uncertainties",
            ],
            measurement_types={
                "temperature": (100, 2000),
                "pressure": (0.1, 150),
                "age": (0, 4600),
            },
            landmarks=[
                "Putirka", "Wells", "Lindsley", "Holland", "Blundy",
                "Spear", "Powell", "Connolly", "Molnar", "England",
            ],
        )
    )

    # 2. External plugins via entry_points
    try:
        from importlib.metadata import entry_points

        eps = entry_points(group="geokit.research.plugins")
        for ep in eps:
            try:
                plugin_cls = ep.load()
                if isinstance(plugin_cls, ResearchPlugin):
                    register_plugin(plugin_cls)
                elif callable(plugin_cls):
                    instance = plugin_cls()
                    if isinstance(instance, ResearchPlugin):
                        register_plugin(instance)
            except Exception as e:
                log.warning("Failed to load plugin entry point '%s': %s", ep.name, e)
    except Exception as e:
        log.debug("Entry point discovery unavailable: %s", e)

    log.info("Loaded %d research plugins", len(_REGISTRY))


def get_active_plugin(query: str = "") -> ResearchPlugin:
    """Get the active plugin for a query. Currently always returns geosciences.

    Future: could auto-detect domain from query content.
    """
    if not _REGISTRY:
        load_plugins()
    return _REGISTRY.get("geosciences") or next(iter(_REGISTRY.values()))
