"""Domain ontology for geoscience query expansion.

Maps ambiguous scientific terms (fugacity, viscosity, diffusion, etc.) to
their domain-specific synonyms and geological context. When a query contains
an ambiguous term, the expansion replaces generic geo context with specific
vocabulary that physics/chemistry/biology papers never contain.

Example:
  "sulfur fugacity" → "sulfur fugacity fS2 fO2 sulfide saturation pyrrhotite
     pyrite FMQ NNO IW mantle magma melt igneous metamorphic experimental
     petrology phase equilibrium ore deposit"

vs. current weak expansion:
  "sulfur fugacity geology petrology geochemistry"  ← physics papers
  can mention "geology" in passing
"""

from __future__ import annotations

import logging

log = logging.getLogger("scientific_research.ontology")

# Domain ontology: ambiguous term → (domain synonyms, geo context terms)
# Synonyms are alternative ways to express the same concept in geoscience.
# Context terms are geological vocabulary that co-occurs with this concept
# but NEVER appears in physics/chemistry/biology papers about the same term.
_DOMAIN_ONTOLOGY: dict[str, tuple[list[str], list[str]]] = {
    "fugacity": (
        # Synonyms — alternative geo expressions for fugacity
        [
            "fugacity",
            "fS2",
            "fO2",
            "oxygen fugacity",
            "sulfur fugacity",
            "sulphur fugacity",
            "redox",
            "redox state",
            "redox buffer",
            "FMQ",
            "fayalite-magnetite-quartz",
            "NNO",
            "IW",
            "QFM",
            "sulfide saturation",
            "sulfur speciation",
            "dFMQ",
            "MH",
            "magnetite-hematite",
            "nickel-nickel oxide",
            "iron-wustite",
            "quartz-fayalite-magnetite",
            "Fe3+/SigmaFe",
            "ferric iron",
            "ferrous-ferric",
            "oxidation state",
            "oxygen barometer",
        ],
        # Geo context — terms that co-occur in petrology but never in physics
        [
            "pyrrhotite",
            "pyrite",
            "magnetite",
            "anhydrite",
            "pentlandite",
            "mantle",
            "magma",
            "melt",
            "igneous",
            "metamorphic",
            "experimental petrology",
            "phase equilibrium",
            "ore deposit",
            "peridotite",
            "eclogite",
            "basalt",
            "arc magma",
            "hematite",
            "spinel",
            "garnet lherzolite",
            "bridgmanite",
            "MORB",
            "xenolith",
            "Mossbauer",
            "XANES",
            "EPMA",
            "LA-ICP-MS",
            "sulfide",
            "chalcophile",
            "volcanic degassing",
            "magmatic volatile",
            "lower mantle",
            "transition zone",
            "subduction zone",
            "mid-ocean ridge",
        ],
    ),
    "viscosity": (
        ["viscosity", "rheology", "flow law", "deformation", "strain rate"],
        [
            "magma",
            "melt",
            "mantle",
            "lava",
            "igneous",
            "volcanic",
            "volcanology",
            "chamber",
            "ascent",
            "crystallization",
        ],
    ),
    "diffusion": (
        ["diffusion", "diffusivity", "geospeedometry", "cooling rate", "diffusion chronometry"],
        [
            "garnet",
            "zoning",
            "mineral",
            "metamorphic",
            "chronometry",
            "Fe-Mg",
            "interdiffusion",
            "growth",
            "resorption",
        ],
    ),
    "stress": (
        ["stress", "paleostress", "stress inversion", "differential stress", "deviatoric stress"],
        [
            "fault",
            "fracture",
            "tectonic",
            "structural",
            "orogenic",
            "thrust",
            "shear",
            "compressional",
            "extensional",
        ],
    ),
    "conductivity": (
        ["conductivity", "electrical resistivity", "thermal conductivity"],
        [
            "mantle",
            "crustal",
            "geophysical",
            "magnetotelluric",
            "lithosphere",
            "asthenosphere",
            "seismic",
        ],
    ),
    "porosity": (
        ["porosity", "permeability", "pore pressure"],
        [
            "reservoir",
            "sediment",
            "rock",
            "sandstone",
            "aquifer",
            "hydrogeology",
            "groundwater",
            "diagenesis",
        ],
    ),
    "anisotropy": (
        ["anisotropy", "anisotropic", "seismic anisotropy"],
        [
            "mantle",
            "crystal",
            "fabric",
            "mineral",
            "olivine",
            "lithosphere",
            "deformation",
            "lattice preferred orientation",
        ],
    ),
    "partition": (
        ["partition", "partition coefficient", "distribution coefficient", "Kd"],
        [
            "mineral",
            "melt",
            "trace element",
            "geochemistry",
            "rare earth",
            "REE",
            "LA-ICP-MS",
            "mineral-melt",
        ],
    ),
    "gradient": (
        ["gradient", "geothermal gradient", "temperature gradient"],
        ["mantle", "crustal", "heat flow", "metamorphic", "subduction", "geotherm", "lithosphere"],
    ),
    "solubility": (
        ["solubility", "dissolution", "saturation"],
        [
            "mineral",
            "melt",
            "fluid",
            "hydrothermal",
            "metamorphic",
            "magmatic volatile",
            "experimental petrology",
        ],
    ),
    "adsorption": (
        ["adsorption", "sorption", "surface complexation"],
        [
            "mineral",
            "clay",
            "oxide",
            "geochemistry",
            "weathering",
            "soil",
            "sediment",
            "groundwater",
        ],
    ),
    "crystallization": (
        ["crystallization", "crystallization age", "crystallization temperature"],
        [
            "magma",
            "melt",
            "igneous",
            "volcanic",
            "plutonic",
            "zircon",
            "feldspar",
            "phase diagram",
            "liquidus",
        ],
    ),
    "precipitation": (
        ["precipitation", "mineral precipitation", "cementation"],
        ["mineral", "diagenesis", "sediment", "hydrothermal", "vein", "ore deposit", "fluid"],
    ),
    "attenuation": (
        ["attenuation", "seismic attenuation", "Q factor"],
        ["mantle", "seismic", "crustal", "wave", "tomography", "lithosphere", "asthenosphere"],
    ),
    "tomography": (
        ["tomography", "seismic tomography"],
        ["mantle", "crustal", "P-wave", "S-wave", "velocity", "subduction", "plume", "lithosphere"],
    ),
    "strain": (
        ["strain", "finite strain", "strain ellipsoid", "strain rate",
         "deformation", "ductile shear"],
        ["fault", "fold", "mylonite", "tectonic", "structural",
         "cleavage", "foliation", "shear zone", "orogenic"],
    ),
    "permeability": (
        ["permeability", "hydraulic conductivity", "fluid flow"],
        ["reservoir", "aquifer", "hydrogeolog", "sediment",
         "fracture", "rock", "pore", "groundwater", "geothermal"],
    ),
    "density": (
        ["density", "bulk density", "grain density"],
        ["rock", "mineral", "mantle", "crust", "magma",
         "melt", "core", "geophysical", "seismic"],
    ),
    "absorption": (
        ["absorption", "X-ray absorption", "spectroscopy"],
        ["mineral", "XANES", "EXAFS", "Mossbauer",
         "crystal", "geochemistry", "trace element"],
    ),
    "saturation": (
        ["saturation", "saturated", "phase saturation"],
        ["magnetization", "magma", "melt", "fluid", "volatile",
         "sulfide saturation", "experimental petrology"],
    ),
    "weathering": (
        ["weathering", "chemical weathering", "alteration"],
        ["soil", "regolith", "erosion", "geochemistry",
         "isotope", "catchment", "denudation", "paleosol"],
    ),
    "erosion": (
        ["erosion", "denudation", "incision"],
        ["landscape", "tectonic", "uplift", "river", "basin",
         "catchment", "geomorpholog", "thermochronolog"],
    ),
    "magnetization": (
        ["magnetization", "natural remanent", "NRM",
         "paleomagnetism", "magnetic susceptibility"],
        ["rock", "basalt", "sediment", "tectonic",
         "plate", "drift", "paleolatitude", "igneous"],
    ),
    "susceptibility": (
        ["susceptibility", "magnetic susceptibility", "AMS"],
        ["rock", "fabric", "magnetite", "mineral",
         "paleomagnet", "anisotropy", "igneous", "metamorphic"],
    ),
    "inversion": (
        ["inversion", "stress inversion", "paleostress inversion"],
        ["fault", "fracture", "tectonic", "structural",
         "seismic", "tomograph", "geophysical"],
    ),
    "fracture": (
        ["fracture", "joint", "crack", "fracture network"],
        ["rock", "fault", "tectonic", "structural",
         "reservoir", "hydrogeolog", "geothermal", "brittle"],
    ),
}

# Strong non-geoscience signals — papers with these should be penalized in ranking
# Used by the ranking domain penalty (separate from _GEO_EXCLUSION in discover.py)
_NEGATIVE_SIGNALS = [
    # Statistical mechanics
    "lattice gas",
    "lattice model",
    "lattice field theory",
    "lattice gauge",
    "lattice QCD",
    "lattice dynamics",
    "qcd",
    "ising",
    "bose",
    "fermion",
    "quark",
    "yang-lee",
    "partition function",
    "hamiltonian",
    "renormalization",
    # Polymer/soft matter
    "polymer",
    "self-avoiding",
    "copolymer",
    "monomer",
    # Plasma/astro physics
    "plasma",
    "fusion",
    "tokamak",
    "stellarator",
    # Environmental chemistry (not geology)
    "multimedia fugacity model",
    "pesticide",
    "pollutant",
    "contaminant fate",
    "chemical transport",
    "air quality",
    # Pure math
    "graph theory",
    "combinatorial",
    "theorem",
    "conjecture",
]


def expand_query_with_ontology(query: str) -> str:
    """Expand query using domain ontology.

    For ambiguous terms (fugacity, viscosity, etc.), replaces generic
    geo context with domain-specific synonyms that physics/chemistry
    papers never contain.

    Example:
        "sulfur fugacity" →
        "sulfur fugacity fS2 fO2 sulfide saturation pyrrhotite FMQ NNO
         mantle magma melt igneous metamorphic experimental petrology"
    """
    if not query:
        return query

    query_lower = query.lower()
    query_words = set(query_lower.split())

    # Check if the query contains any ambiguous term
    expanded_terms = list(query_words)  # start with original query terms

    found_ambiguous = False
    for term, (synonyms, context) in _DOMAIN_ONTOLOGY.items():
        if term in query_lower:
            found_ambiguous = True
            # Add domain-specific synonyms
            for syn in synonyms:
                if syn.lower() not in query_lower:
                    expanded_terms.append(syn)
            # Add geo context terms
            for ctx in context:
                if ctx.lower() not in query_lower:
                    expanded_terms.append(ctx)
            log.info(
                "Ontology expansion for '%s': +%d synonyms, +%d context terms",
                term,
                len(synonyms),
                len(context),
            )

    if found_ambiguous:
        return " ".join(expanded_terms)

    # No ambiguous term — check if it has strong geo context
    # If yes, return unchanged
    return query




# ── Graduated domain scoring (replaces binary penalty) ────────────────
# Each geo term adds a boost; each non-geo term subtracts.
# Score starts at 1.0 (neutral). Range: [0.0, 2.0].
# >1.0 = strong geo paper (boosted in ranking)
# <0.5 = weak geo signal (filtered for ambiguous queries)
# =0.0 = killed (overwhelming non-geo evidence)

_GEO_BOOST_TERMS: dict[str, float] = {
    # Minerals (strong signal)
    "garnet": 0.3, "pyroxene": 0.3, "olivine": 0.3, "amphibole": 0.3,
    "diamond": 0.3, "coesite": 0.3, "pyrrhotite": 0.3, "pentlandite": 0.3,
    "feldspar": 0.2, "quartz": 0.2, "zircon": 0.2, "monazite": 0.2,
    "magnetite": 0.2, "anhydrite": 0.2, "pyrite": 0.2, "spinel": 0.2,
    "clinopyroxene": 0.3, "orthopyroxene": 0.3, "hornblende": 0.3,
    "biotite": 0.2, "muscovite": 0.2, "chlorite": 0.2, "staurolite": 0.3,
    # Rock types
    "eclogite": 0.3, "peridotite": 0.3, "granulite": 0.3, "amphibolite": 0.3,
    "metapelite": 0.3, "basalt": 0.2, "granite": 0.2, "gneiss": 0.2,
    "schist": 0.2, "mylonite": 0.3, "serpentinite": 0.3, "blueschist": 0.3,
    # Geological concepts
    "mantle": 0.2, "metamorphic": 0.2, "igneous": 0.2, "sedimentary": 0.2,
    "magma": 0.2, "melt": 0.2, "subduction": 0.3, "orogenic": 0.2,
    "crustal": 0.2, "tectonic": 0.2, "fault zone": 0.3, "shear zone": 0.3,
    # Methods / geochemistry
    "experimental petrology": 0.3, "phase equilibrium": 0.3,
    "thermobarometry": 0.3, "geothermobarometry": 0.3,
    "pseudosection": 0.3, "thermocalc": 0.3, "perple_x": 0.3,
    "fmq": 0.3, "nno": 0.3, "iw buffer": 0.3, "qfm": 0.3,
    "redox": 0.2, "sulfide saturation": 0.3, "oxygen fugacity": 0.3,
    "sulfur fugacity": 0.3, "fs2": 0.3, "fo2": 0.3,
    "epma": 0.2, "la-icp-ms": 0.2, "xanes": 0.3, "mossbauer": 0.3,
    "u-pb": 0.3, "ar-ar": 0.3, "geochronolog": 0.3,
    "seismic": 0.2, "paleomagnet": 0.3, "structural geolog": 0.3,
    "hydrothermal": 0.2, "ore deposit": 0.3, "mineraliz": 0.2,
}

_NON_GEO_PENALTY: dict[str, float] = {
    # Statistical mechanics (strong kill)
    "lattice gas": 0.8, "lattice model": 0.8, "lattice qcd": 0.8,
    "lattice gauge": 0.8, "lattice field": 0.8, "lattice dynamics": 0.8,
    "ising model": 0.8, "bose gas": 0.8, "bose-einstein": 0.8,
    "self-avoiding walk": 0.8, "coulomb gas": 0.8,
    "yang-lee": 0.8, "hard-core model": 0.8, "hard-core lattice": 0.8,
    # Physics (moderate kill)
    "partition function": 0.6, "hamiltonian": 0.6,
    "fermion": 0.6, "quark": 0.8, "renormalization": 0.6,
    "condensed matter": 0.5, "quantum field theory": 0.8,
    "many-body": 0.5, "plasma physics": 0.5,
    # Polymer / soft matter
    "polymer": 0.5, "copolymer": 0.5, "monomer": 0.5,
    "polymer adsorption": 0.8, "polymer chain": 0.5,
    # Math
    "graph theory": 0.5, "combinatorial": 0.4,
    "percolation theory": 0.5, "scale-free network": 0.5,
    # Environmental chemistry (not geology)
    "multimedia fugacity": 0.8, "pesticide": 0.5, "pollutant": 0.5,
    "contaminant fate": 0.5, "chemical transport": 0.4,
    # Biology
    "protein folding": 0.5, "dna sequencing": 0.5,
    "cell membrane": 0.5, "enzyme": 0.4, "antibody": 0.5,
    "apoptosis": 0.5, "metabolomics": 0.5,
}


def domain_score(text: str) -> float:
    """Graduated domain relevance score.

    Returns 0.0-2.0:
    - >1.0: strong geo paper (boosted in ranking)
    - 0.5-1.0: moderate geo signal
    - 0.1-0.5: weak geo signal
    - 0.0: killed (overwhelming non-geo signal)

    Replaces binary domain_penalty() for ranking.
    """
    text_lower = text.lower()
    score = 1.0  # neutral start

    for term, boost in _GEO_BOOST_TERMS.items():
        if term in text_lower:
            score += boost

    for term, penalty in _NON_GEO_PENALTY.items():
        if term in text_lower:
            score -= penalty

    return max(0.0, min(2.0, score))


def domain_penalty(paper_text: str) -> float:
    """Binary domain penalty (backward-compatible wrapper).

    Returns 1.0 for geo papers, 0.0 for non-geo.
    Delegates to domain_score() — killed when score <= 0.0.
    """
    score = domain_score(paper_text)
    return 0.0 if score <= 0.0 else 1.0


# =============================================================================
# Hierarchical ontology — parent/child relationships for earth science methods
# =============================================================================

# Tree structure: method → subtypes → specific calibrations
_ONTOLOGY_TREE: dict[str, dict] = {
    "thermometry": {
        "children": {
            "exchange_thermometer": {
                "synonyms": ["Fe-Mg exchange", "solvus", "distribution coefficient"],
                "children": {
                    "garnet_clinopyroxene": {"synonyms": ["garnet-cpx", "Fe-Mg garnet"]},
                    "garnet_biotite": {"synonyms": ["garnet-bt", "GB"]},
                    "two_pyroxene": {"synonyms": ["cpx-opx", "2-pyroxene"]},
                },
            },
            "trace_element_thermometer": {
                "synonyms": ["trace element", "REE", "Ti-in-zircon", "Zr-in-rutile"],
                "children": {},
            },
            "single_mineral_thermometer": {
                "synonyms": ["single-clinopyroxene", "single-amphibole", "single-phase"],
                "children": {},
            },
            "experimental_calibration": {
                "synonyms": ["experimental", "reversed calibration", "phase equilibrium"],
                "children": {},
            },
            "thermodynamic_model": {
                "synonyms": ["thermodynamic", "MELTS", "THERMOCALC", "Perple_X"],
                "children": {},
            },
        },
    },
    "barometry": {
        "children": {
            "exchange_barometer": {
                "synonyms": ["partitioning", "distribution coefficient barometer"],
                "children": {},
            },
            "net_transfer_barometer": {
                "synonyms": ["GASP", "GADS", "GPMB", "reaction barometer"],
                "children": {},
            },
            "single_mineral_barometer": {
                "synonyms": ["single-clinopyroxene barometer", "amphibole barometer"],
                "children": {},
            },
        },
    },
    "geochronology": {
        "children": {
            "U-Pb": {
                "synonyms": ["zircon U-Pb", "ID-TIMS", "SIMS", "LA-ICP-MS"],
                "children": {},
            },
            "Ar-Ar": {
                "synonyms": ["40Ar/39Ar", "laser fusion", "step heating"],
                "children": {},
            },
            "Re-Os": {
                "synonyms": ["Re-Os", "osmium", "sulfide Re-Os"],
                "children": {},
            },
        },
    },
    "isotope_geochemistry": {
        "children": {
            "radiogenic": {
                "synonyms": ["Sr-Nd-Pb", "Hf", "Os", "radiogenic isotope"],
                "children": {},
            },
            "stable": {
                "synonyms": ["d18O", "d13C", "d34S", "dD", "stable isotope"],
                "children": {},
            },
        },
    },
    "structural_analysis": {
        "children": {
            "stress_inversion": {
                "synonyms": ["paleostress", "fault slip", "stress tensor"],
                "children": {},
            },
            "strain_analysis": {
                "synonyms": ["strain ellipsoid", "Fry", "Rf/phi", "strain fringe"],
                "children": {},
            },
        },
    },
}


def get_ontology_tree() -> dict[str, dict]:
    """Return the hierarchical ontology tree."""
    return _ONTOLOGY_TREE


def find_in_tree(query: str) -> tuple[str | None, str | None]:
    """Find where a query term fits in the ontology tree.

    Returns (parent_method, subtype) or (None, None) if not found.
    """
    q_lower = query.lower()
    q_words = q_lower.split()
    for parent, subtree in _ONTOLOGY_TREE.items():
        # Match parent key or common stem (e.g., "thermometry" matches "thermometer")
        parent_stem = parent[:max(5, len(parent) - 3)]
        if parent in q_lower or any(w.startswith(parent_stem) for w in q_words):
            return parent, None
        for child_key, child_data in subtree.get("children", {}).items():
            child_syns = [child_key] + child_data.get("synonyms", [])
            if any(s in q_lower for s in child_syns):
                return parent, child_key
            for grandchild_key, grandchild_data in child_data.get("children", {}).items():
                gc_syns = [grandchild_key] + grandchild_data.get("synonyms", [])
                if any(s in q_lower for s in gc_syns):
                    return parent, grandchild_key
    return None, None


def get_subtree_synonyms(method: str) -> list[str]:
    """Get all synonyms from a method and all its descendants."""
    node = _ONTOLOGY_TREE.get(method)
    if not node:
        return []
    synonyms: list[str] = []
    for child_key, child_data in node.get("children", {}).items():
        synonyms.append(child_key)
        synonyms.extend(child_data.get("synonyms", []))
        for gc_data in child_data.get("children", {}).values():
            synonyms.extend(gc_data.get("synonyms", []))
    return synonyms
