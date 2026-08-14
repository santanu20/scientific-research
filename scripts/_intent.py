"""Intent-aware research query parser for earth science.

Parses natural language queries into structured research intent:
    property + material + method + context → topic template

Architecture:
    Layer 1: Pattern matching (regex) — 80% of queries, instant
    Layer 2: LLM fallback (Ollama) — complex queries, 5-10s
    Layer 3: Embedding classification (BGE) — unknown queries
    Layer 4: Fallback — current ontology expansion

Example:
    "pyroxene thermometry" →
        ResearchIntent(
            property="temperature",
            material="pyroxene",
            method="thermometry",
            topic_key="thermometry",
            material_synonyms=["clinopyroxene", "orthopyroxene", ...],
        )
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("scientific_research.intent")


# =============================================================================
# Data structures
# =============================================================================


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


@dataclass
class ResearchIntent:
    """Structured research intent parsed from a natural-language query."""

    target_property: str | None = None
    material: str | None = None
    method: str | None = None
    context: str | None = None
    goal: str | None = None  # calibration | application | review | identification
    topic_key: str | None = None
    material_synonyms: list[str] = field(default_factory=list)
    method_synonyms: list[str] = field(default_factory=list)
    template: TopicTemplate | None = None
    parser_source: str = "pattern"
    raw_query: str = ""

    @property
    def has_template(self) -> bool:
        return self.template is not None

    @property
    def is_geoscience(self) -> bool:
        return self.topic_key is not None or self.material is not None


# =============================================================================
# Materials dictionary — 175+ earth science materials
# =============================================================================

MATERIALS: dict[str, tuple[list[str], str]] = {
    # ── Silicate minerals (rock-forming) ──────────────────────────────
    "pyroxene": (
        [
            "clinopyroxene",
            "orthopyroxene",
            "augite",
            "diopside",
            "enstatite",
            "pigeonite",
            "hedenbergite",
            "hypersthene",
            "ferrosilite",
            "Cpx",
            "Opx",
            "Cpx-Liq",
            "Cpx-liquid",
            "sub-calcic",
            "omphacite",
        ],
        "silicate",
    ),
    "garnet": (
        ["almandine", "pyrope", "grossular", "spessartine", "andradite", "uvarovite", "Grt"],
        "silicate",
    ),
    "amphibole": (
        [
            "hornblende",
            "glaucophane",
            "tremolite",
            "actinolite",
            "edenite",
            "pargasite",
            "riebeckite",
            "barroisite",
            "cummingtonite",
            "anthophyllite",
            "Amp",
        ],
        "silicate",
    ),
    "olivine": (["forsterite", "fayalite", "Fo", "Fa", "Ol", "Fo-Fa"], "silicate"),
    "feldspar": (
        [
            "plagioclase",
            "alkali feldspar",
            "anorthite",
            "albite",
            "orthoclase",
            "sanidine",
            "anorthoclase",
            "oligoclase",
            "andesine",
            "labradorite",
            "bytownite",
            "Fsp",
        ],
        "silicate",
    ),
    "quartz": (["quartz", "SiO2", "coesite", "stishovite"], "silicate"),
    "mica": (
        ["muscovite", "biotite", "phlogopite", "paragonite", "lepidolite", "muscovite-biotite"],
        "silicate",
    ),
    "chlorite": (["chlorite", "chamosite", "clinochlore"], "silicate"),
    "epidote": (["epidote", "zoisite", "clinozoisite", "piedmontite"], "silicate"),
    "staurolite": (["staurolite"], "silicate"),
    "kyanite": (["kyanite", "disthene"], "silicate"),
    "sillimanite": (["sillimanite", "fibrolite"], "silicate"),
    "andalusite": (["andalusite", "chiastolite"], "silicate"),
    "cordierite": (["cordierite", "iolite"], "silicate"),
    "serpentine": (
        ["serpentine", "serpentinite", "antigorite", "lizardite", "chrysotile"],
        "silicate",
    ),
    "talc": (["talc", "steatite"], "silicate"),
    # ── Accessory minerals ────────────────────────────────────────────
    "zircon": (
        [
            "zircon",
            "ZrSiO4",
            "detrital zircon",
            "zircon grain",
            "metamorphic zircon",
            "igneous zircon",
        ],
        "accessory",
    ),
    "monazite": (["monazite", "monazite-(Ce)", "monazite-(La)"], "accessory"),
    "xenotime": (["xenotime", "xenotime-(Y)"], "accessory"),
    "apatite": (["apatite", "fluorapatite", "chlorapatite", "hydroxylapatite"], "accessory"),
    "titanite": (["titanite", "sphene"], "accessory"),
    "rutile": (["rutile", "TiO2"], "accessory"),
    "perovskite": (["perovskite", "CaTiO3"], "accessory"),
    "allanite": (["allanite", "orthite"], "accessory"),
    "baddeleyite": (["baddeleyite", "ZrO2"], "accessory"),
    # ── Oxide minerals ────────────────────────────────────────────────
    "magnetite": (["magnetite", "Fe3O4", "titanomagnetite"], "oxide"),
    "hematite": (["hematite", "Fe2O3", "specularite"], "oxide"),
    "ilmenite": (["ilmenite", "FeTiO3", "titanomagnetite"], "oxide"),
    "spinel": (["spinel", "magnesiochromite", "pleonaste", "hercynite", "chromite"], "oxide"),
    "chromite": (["chromite", "FeCr2O4", "magnesiochromite"], "oxide"),
    "corundum": (["corundum", "Al2O3", "sapphire", "ruby"], "oxide"),
    "pseudobrookite": (["pseudobrookite", "ulvospinel", "armalcolite"], "oxide"),
    # ── Sulfide minerals ──────────────────────────────────────────────
    "pyrite": (["pyrite", "FeS2", "fools gold"], "sulfide"),
    "pyrrhotite": (["pyrrhotite", "Fe1-xS", "po"], "sulfide"),
    "pentlandite": (["pentlandite", "(Fe,Ni)9S8"], "sulfide"),
    "chalcopyrite": (["chalcopyrite", "CuFeS2"], "sulfide"),
    "galena": (["galena", "PbS"], "sulfide"),
    "sphalerite": (["sphalerite", "ZnS", "wurtzite"], "sulfide"),
    "molybdenite": (["molybdenite", "MoS2"], "sulfide"),
    "arsenopyrite": (["arsenopyrite", "FeAsS"], "sulfide"),
    "bornite": (["bornite", "Cu5FeS4"], "sulfide"),
    "sulfide": (["sulfide", "sulphide", "base metal sulfide"], "sulfide"),
    # ── Carbonate/sulfate/phosphate ───────────────────────────────────
    "calcite": (["calcite", "CaCO3", "micrite"], "carbonate"),
    "dolomite": (["dolomite", "CaMg(CO3)2", "dolostone"], "carbonate"),
    "aragonite": (["aragonite"], "carbonate"),
    "ankerite": (["ankerite", "siderite", "magnesite", "rhodochrosite"], "carbonate"),
    "anhydrite": (
        ["anhydrite", "CaSO4", "gypsum", "barite", "celestite", "barite-celestite"],
        "sulfate",
    ),
    # ── Native elements ───────────────────────────────────────────────
    "gold": (["gold", "Au", "native gold", "electrum"], "native"),
    "diamond": (["diamond", "C", "microdiamond", "nanodiamond"], "native"),
    "graphite": (["graphite", "graphitization"], "native"),
    "platinum": (["platinum", "Pt", "PGE", "PGM", "osmiridium"], "native"),
    # ── Igneous rocks (volcanic) ──────────────────────────────────────
    "basalt": (
        [
            "basalt",
            "basaltic",
            "MORB",
            "mid-ocean ridge basalt",
            "OIB",
            "ocean island basalt",
            "flood basalt",
            "LIP",
            "continental flood basalt",
        ],
        "volcanic",
    ),
    "andesite": (["andesite", "andesitic", "arc andesite"], "volcanic"),
    "dacite": (["dacite", "dacitic"], "volcanic"),
    "rhyolite": (["rhyolite", "rhyolitic", "obsidian", "ignimbrite"], "volcanic"),
    "komatiite": (["komatiite", "komatiitic", "boninite"], "volcanic"),
    "phonolite": (["phonolite", "trachyte", "tephrite"], "volcanic"),
    "carbonatite": (
        ["carbonatite", "nephelinite", "melilitite", "kimberlite", "lamproite", "lamprophyre"],
        "volcanic",
    ),
    "tuff": (["tuff", "tuffaceous", "volcanic ash", "pyroclastic", "ignimbrite"], "volcanic"),
    # ── Igneous rocks (plutonic) ──────────────────────────────────────
    "granite": (
        [
            "granite",
            "granitic",
            "granodiorite",
            "tonalite",
            "trondhjemite",
            "monzonite",
            "syenite",
            "A-type",
            "S-type",
            "I-type",
        ],
        "plutonic",
    ),
    "gabbro": (["gabbro", "gabbroic", "norite", "troctolite"], "plutonic"),
    "diorite": (["diorite", "dioritic", "monzodiorite", "quartz diorite"], "plutonic"),
    "peridotite": (
        [
            "peridotite",
            "lherzolite",
            "harzburgite",
            "dunite",
            "wehrlite",
            "mantle peridotite",
            "mantle xenolith",
            "mantle nodule",
        ],
        "plutonic",
    ),
    "anorthosite": (["anorthosite", "anorthositic", "massif-type anorthosite"], "plutonic"),
    "pegmatite": (["pegmatite", "aplite", "granite pegmatite"], "plutonic"),
    "pyroxenite": (["pyroxenite", "websterite", "olivine websterite"], "plutonic"),
    # ── Metamorphic rocks ─────────────────────────────────────────────
    "eclogite": (
        ["eclogite", "eclogitic", "UHP eclogite", "ultrahigh-pressure eclogite"],
        "metamorphic",
    ),
    "granulite": (["granulite", "granulitic", "charnockite", "enderbite"], "metamorphic"),
    "amphibolite": (["amphibolite", "amphibolitic"], "metamorphic"),
    "gneiss": (
        [
            "gneiss",
            "gneissic",
            "orthogneiss",
            "paragneiss",
            "augen gneiss",
            "migmatite",
            "migmatitic",
            "anatexite",
        ],
        "metamorphic",
    ),
    "schist": (
        [
            "schist",
            "schistose",
            "mica schist",
            "garnet schist",
            "chlorite schist",
            "blueschist",
            "blue schist",
        ],
        "metamorphic",
    ),
    "skarn": (["skarn", "tactite", "calc-silicate"], "metamorphic"),
    "marble": (["marble", "crystalline limestone", "metacarbonate"], "metamorphic"),
    "quartzite": (["quartzite", "quartzitic"], "metamorphic"),
    "mylonite": (
        [
            "mylonite",
            "mylonitic",
            "ultramylonite",
            "cataclasite",
            "pseudotachylite",
            "fault rock",
            "fault gouge",
        ],
        "metamorphic",
    ),
    "hornfels": (["hornfels", "contact metamorphic"], "metamorphic"),
    "metapelite": (
        ["metapelite", "metapelitic", "metasediment", "pelite", "metapsammite"],
        "metamorphic",
    ),
    "metabasite": (["metabasite", "metabasalt", "metavolcanic", "greenstone"], "metamorphic"),
    # ── Sedimentary rocks ─────────────────────────────────────────────
    "sandstone": (
        [
            "sandstone",
            "arenite",
            "wacke",
            "arkose",
            "greywacke",
            "quartz arenite",
            "lithic arenite",
        ],
        "sedimentary",
    ),
    "shale": (
        ["shale", "shaly", "mudstone", "siltstone", "argillite", "claystone", "black shale"],
        "sedimentary",
    ),
    "limestone": (
        [
            "limestone",
            "calcarenite",
            "micrite",
            "chalk",
            "coquina",
            "packstone",
            "wackestone",
            "grainstone",
            "floatstone",
            "rudstone",
        ],
        "sedimentary",
    ),
    "dolostone": (["dolostone", "dolomite rock"], "sedimentary"),
    "chert": (["chert", "cherty", "jasper", "radiolarite", "diatomite"], "sedimentary"),
    "conglomerate": (["conglomerate", "breccia", "fanglomerate", "olistostrome"], "sedimentary"),
    "evaporite": (["evaporite", "halite", "rock salt", "gypsum deposit"], "sedimentary"),
    "coal": (["coal", "coal seam", "lignite", "anthracite", "bituminous"], "sedimentary"),
    "BIF": (["banded iron formation", "BIF", "taconite", "itabirite", "jaspilite"], "sedimentary"),
    "turbidite": (["turbidite", "flysch", "contourite", "debris flow"], "sedimentary"),
    # ── Fluids and melts ──────────────────────────────────────────────
    "melt": (
        [
            "silicate melt",
            "basaltic melt",
            "granitic melt",
            "andesitic melt",
            "magmatic melt",
            "melt inclusion",
        ],
        "fluid",
    ),
    "volatile": (
        [
            "magmatic volatile",
            "H2O",
            "CO2",
            "SO2",
            "H2S",
            "Cl",
            "F",
            "volatile content",
            "degassing",
            "fluid flux",
        ],
        "fluid",
    ),
    "fluid": (
        ["hydrothermal fluid", "metamorphic fluid", "brine", "formation water", "ore fluid"],
        "fluid",
    ),
    "soil": (["soil", "regolith", "paleosol", "saprolite", "laterite"], "regolith"),
    "groundwater": (["groundwater", "aquifer", "pore water", "formation water"], "fluid"),
}


# =============================================================================
# Methods dictionary — 50+ analytical methods with property mapping
# =============================================================================

METHODS: dict[str, dict[str, Any]] = {
    # ── Thermobarometry ───────────────────────────────────────────────
    "thermometry": {
        "property": "temperature",
        "unit": "C",
        "range": (100, 2000),
        "synonyms": [
            "thermometer",
            "calibration",
            "temperature estimate",
            "Fe-Mg exchange",
            "solvus",
            "exchange equilibrium",
            "temperature calculation",
        ],
    },
    "barometry": {
        "property": "pressure",
        "unit": "kbar",
        "range": (0.1, 150),
        "synonyms": [
            "barometer",
            "pressure estimate",
            "pressure calculation",
            "geobarometer",
            "pressure indicator",
        ],
    },
    "thermobarometry": {
        "property": "temperature-pressure",
        "unit": "C-kbar",
        "range": (0, 0),
        "synonyms": [
            "geothermobarometry",
            "P-T estimate",
            "P-T path",
            "pressure-temperature",
            "geothermobarometer",
        ],
    },
    # ── Geochronology ─────────────────────────────────────────────────
    "geochronology": {
        "property": "age",
        "unit": "Ma",
        "range": (0, 4600),
        "synonyms": ["dating", "age determination", "geochronological", "age spectra", "concordia"],
    },
    "U-Pb": {
        "property": "age",
        "unit": "Ma",
        "range": (0, 4600),
        "synonyms": ["U-Pb dating", "zircon U-Pb", "ID-TIMS", "SHRIMP", "concordia", "discordance"],
    },
    "Ar-Ar": {
        "property": "age",
        "unit": "Ma",
        "range": (0.01, 4500),
        "synonyms": ["40Ar/39Ar", "step heating", "Ar-Ar dating", "K-Ar"],
    },
    "Re-Os": {
        "property": "age",
        "unit": "Ma",
        "range": (0, 4500),
        "synonyms": ["Re-Os dating", "osmium isotope", "molybdenite dating"],
    },
    "fission track": {
        "property": "age",
        "unit": "Ma",
        "range": (0.1, 1000),
        "synonyms": ["apatite fission track", "zircon fission track", "AFT", "ZFT", "annealing"],
    },
    "helium": {
        "property": "age",
        "unit": "Ma",
        "range": (0.01, 500),
        "synonyms": ["(U-Th)/He", "apatite He", "zircon He", "AHe", "ZHe"],
    },
    "cosmogenic": {
        "property": "age",
        "unit": "ka",
        "range": (0.001, 10000),
        "synonyms": [
            "cosmogenic nuclide",
            "Be-10",
            "Al-26",
            "Ne-21",
            "exposure age",
            "denudation rate",
        ],
    },
    # ── Geochemistry ──────────────────────────────────────────────────
    "fugacity": {
        "property": "fugacity",
        "unit": "log units",
        "range": (-10, 10),
        "synonyms": [
            "fO2",
            "fS2",
            "oxygen fugacity",
            "sulfur fugacity",
            "redox",
            "redox state",
            "FMQ",
            "NNO",
            "IW",
        ],
    },
    "diffusion": {
        "property": "diffusion",
        "unit": "m2/s",
        "range": (0, 0),
        "synonyms": [
            "diffusivity",
            "geospeedometry",
            "cooling rate",
            "diffusion chronometry",
            "Arrhenius",
        ],
    },
    "isotope": {
        "property": "isotope ratio",
        "unit": "permil",
        "range": (-100, 100),
        "synonyms": [
            "isotope geochemistry",
            "radiogenic isotope",
            "stable isotope",
            "Sr-Nd-Pb",
            "d18O",
            "d13C",
            "d34S",
        ],
    },
    "trace element": {
        "property": "concentration",
        "unit": "ppm",
        "range": (0, 10000),
        "synonyms": [
            "trace element geochemistry",
            "REE",
            "rare earth",
            "partition coefficient",
            "Kd",
            "spider diagram",
        ],
    },
    # ── Structural ────────────────────────────────────────────────────
    "stress": {
        "property": "stress",
        "unit": "MPa",
        "range": (0, 10000),
        "synonyms": ["paleostress", "stress inversion", "differential stress", "stress tensor"],
    },
    "strain": {
        "property": "strain",
        "unit": "ratio",
        "range": (0, 100),
        "synonyms": [
            "strain analysis",
            "finite strain",
            "strain ellipsoid",
            "Fry method",
            "Rf/phi",
        ],
    },
    "anisotropy": {
        "property": "anisotropy",
        "unit": "%",
        "range": (0, 20),
        "synonyms": [
            "seismic anisotropy",
            "SKS splitting",
            "shear wave splitting",
            "lattice preferred orientation",
            "LPO",
        ],
    },
    # ── Geophysics ────────────────────────────────────────────────────
    "tomography": {
        "property": "velocity",
        "unit": "km/s",
        "range": (0.5, 15),
        "synonyms": ["seismic tomography", "velocity model", "travel time", "P-wave", "S-wave"],
    },
    "conductivity": {
        "property": "conductivity",
        "unit": "S/m",
        "range": (0, 1),
        "synonyms": ["electrical conductivity", "magnetotelluric", "resistivity", "MT"],
    },
    "gravity": {
        "property": "gravity",
        "unit": "mGal",
        "range": (-500, 500),
        "synonyms": [
            "gravity survey",
            "gravity anomaly",
            "Bouguer",
            "free-air",
            "gravity gradient",
        ],
    },
    # ── Sedimentology ─────────────────────────────────────────────────
    "provenance": {
        "property": "provenance",
        "unit": "",
        "range": (0, 0),
        "synonyms": [
            "sedimentary provenance",
            "detrital",
            "source rock",
            "framework mode",
            "Dickinson",
        ],
    },
    # ── Volcanology ───────────────────────────────────────────────────
    "viscosity": {
        "property": "viscosity",
        "unit": "Pa s",
        "range": (0, 1e20),
        "synonyms": ["magma viscosity", "melt viscosity", "rheology", "flow law"],
    },
    # ── Paleoclimate ──────────────────────────────────────────────────
    "paleoclimate": {
        "property": "temperature",
        "unit": "C",
        "range": (-50, 100),
        "synonyms": ["paleoclimate", "paleotemperature", "climate proxy", "CO2 proxy", "paleo-CO2"],
    },
    # ── Paleomagnetism ────────────────────────────────────────────────
    "paleomagnetism": {
        "property": "magnetization",
        "unit": "A/m",
        "range": (0, 100),
        "synonyms": [
            "paleomagnetic",
            "natural remanent",
            "NRM",
            "demagnetization",
            "apparent polar wander",
        ],
    },
    # ── Hydrogeology ──────────────────────────────────────────────────
    "permeability": {
        "property": "permeability",
        "unit": "m2",
        "range": (0, 1),
        "synonyms": [
            "hydraulic conductivity",
            "fluid flow",
            "Darcy",
            "aquifer",
            "groundwater flow",
        ],
    },
    "porosity": {
        "property": "porosity",
        "unit": "%",
        "range": (0, 100),
        "synonyms": ["pore space", "void ratio", "bulk porosity", "effective porosity"],
    },
    # ── General petrology ────────────────────────────────────────────
    "inclusions": {
        "target_property": "composition", "unit": "wt%", "range": (0, 100),
        "synonyms": ["melt inclusion", "fluid inclusion", "inclusion study", "inclusions"],
    },
    "petrogenesis": {
        "target_property": "composition",
        "unit": "wt%",
        "range": (0, 100),
        "synonyms": [
            "petrogenesis",
            "petrogenetic",
            "magma evolution",
            "crystallization",
            "differentiation",
            "source",
            "partial melting",
            "fractional crystallization",
        ],
    },
    "metamorphism": {
        "target_property": "grade",
        "unit": "",
        "range": (0, 0),
        "synonyms": [
            "metamorphism",
            "metamorphic",
            "metamorphic grade",
            "facies",
            "metamorphic evolution",
            "P-T-t",
            "metamorphic reaction",
            "paragenesis",
        ],
    },
    "geochemistry": {
        "target_property": "composition",
        "unit": "wt%",
        "range": (0, 100),
        "synonyms": [
            "geochemistry",
            "geochemical",
            "major element",
            "whole-rock",
            "bulk composition",
            "Harker diagram",
        ],
    },
    # ── Additional structural ────────────────────────────────────────
    "kinematics": {
        "target_property": "strain",
        "unit": "degrees",
        "range": (0, 360),
        "synonyms": [
            "kinematics",
            "fault kinematics",
            "slip",
            "rake",
            "movement",
            "slip sense",
            "shear sense",
        ],
    },
    # ── Geothermal ───────────────────────────────────────────────────
    "heat flow": {
        "target_property": "heat flow",
        "unit": "mW/m2",
        "range": (0, 500),
        "synonyms": [
            "heat flow",
            "geothermal gradient",
            "thermal conductivity",
            "surface heat flow",
        ],
    },
    # ── Ore/exploration ──────────────────────────────────────────────
    "mineralization": {
        "target_property": "grade",
        "unit": "ppm",
        "range": (0, 10000),
        "synonyms": [
            "mineralization",
            "ore deposit",
            "ore-forming",
            "hydrothermal alteration",
            "mineral exploration",
            "ore genesis",
            "sulfide mineralization",
        ],
    },
    # ── Sedimentology ────────────────────────────────────────────────
    "stratigraphy": {
        "target_property": "age",
        "unit": "Ma",
        "range": (0, 4000),
        "synonyms": [
            "stratigraphy",
            "sequence stratigraphy",
            "facies analysis",
            "depositional environment",
            "sedimentology",
            "sedimentary facies",
        ],
    },
    # ── Geomorphology ────────────────────────────────────────────────
    "geomorphology": {
        "target_property": "erosion rate",
        "unit": "mm/yr",
        "range": (0, 100),
        "synonyms": [
            "geomorphology",
            "landscape evolution",
            "denudation",
            "incision",
            "glacial erosion",
            "fluvial",
            "weathering rate",
        ],
    },
    # ── Tectonics ────────────────────────────────────────────────────
    "tectonics": {
        "target_property": "displacement",
        "unit": "km",
        "range": (0, 10000),
        "synonyms": [
            "tectonics",
            "plate tectonics",
            "subduction",
            "collision",
            "rifting",
            "orogeny",
            "plate reconstruction",
            "paleogeography",
        ],
    },
    # ── Biogeochemistry ──────────────────────────────────────────────
    "biogeochemistry": {
        "target_property": "flux",
        "unit": "mol/yr",
        "range": (0, 1e15),
        "synonyms": [
            "biogeochemistry",
            "biogeochemical",
            "carbon cycle",
            "nutrient cycling",
            "organic carbon",
            "primary productivity",
        ],
    },
    # ── Petroleum ────────────────────────────────────────────────────
    "petroleum": {
        "target_property": "maturity",
        "unit": "%Ro",
        "range": (0, 5),
        "synonyms": [
            "petroleum",
            "petroleum geology",
            "source rock",
            "reservoir",
            "kerogen",
            "hydrocarbon",
            "oil and gas",
            "basin modeling",
        ],
    },
    # ── Engineering geology ──────────────────────────────────────────
    "geotechnical": {
        "target_property": "strength",
        "unit": "MPa",
        "range": (0, 500),
        "synonyms": [
            "geotechnical",
            "rock mechanics",
            "slope stability",
            "soil mechanics",
            "engineering geology",
            "foundation",
        ],
    },
    # ── Environmental ────────────────────────────────────────────────
    "environmental": {
        "target_property": "concentration",
        "unit": "ppm",
        "range": (0, 10000),
        "synonyms": [
            "environmental geology",
            "contamination",
            "CO2 sequestration",
            "carbon storage",
            "waste disposal",
            "remediation",
        ],
    },
    # ── Planetary ────────────────────────────────────────────────────
    "planetary": {
        "target_property": "composition",
        "unit": "wt%",
        "range": (0, 100),
        "synonyms": [
            "planetary geology",
            "impact crater",
            "meteorite",
            "lunar",
            "martian",
            "mars",
            "exoplanet geochemistry",
        ],
    },
}


# Build reverse lookup: any synonym → canonical material/method
_MATERIAL_LOOKUP: dict[str, str] = {}
for canonical, (syns, _) in MATERIALS.items():
    _MATERIAL_LOOKUP[canonical] = canonical
    for s in syns:
        _MATERIAL_LOOKUP[s.lower()] = canonical

_METHOD_LOOKUP: dict[str, str] = {}
for canonical, info in METHODS.items():
    _METHOD_LOOKUP[canonical] = canonical
    for s in info["synonyms"]:
        _MATERIAL_LOOKUP[s.lower()] = canonical  # reuse
    for s in info["synonyms"]:
        _METHOD_LOOKUP[s.lower()] = canonical


# =============================================================================
# Pattern matching parser — Layer 1 (80% coverage, instant)
# =============================================================================

# Sort by length descending so "thermobarometry" matches before "thermometry"
_sorted_materials = sorted(MATERIALS.keys(), key=len, reverse=True)
_sorted_methods = sorted(METHODS.keys(), key=len, reverse=True)

# Pattern 1: "material method" or "method of material"
_P1 = re.compile(
    r"\b(?P<material>" + "|".join(re.escape(m) for m in _sorted_materials) + r")\s+"
    r"(?P<method>" + "|".join(re.escape(m) for m in _sorted_methods) + r")\b",
    re.IGNORECASE,
)
_P2 = re.compile(
    r"\b(?P<method>" + "|".join(re.escape(m) for m in _sorted_methods) + r")\s+"
    r"(?:of|using|from|in)\s+"
    r"(?P<material>" + "|".join(re.escape(m) for m in _sorted_materials) + r")\b",
    re.IGNORECASE,
)
# Pattern 3: "isotope system" like "U-Pb zircon"
_P3 = re.compile(
    r"\b(?P<method>U-Pb|Ar-Ar|Re-Os|Sm-Nd|Lu-Hf|fission track|cosmogenic|"
    r"(?:U-Th)/He|radiocarbon)\s+"
    r"(?P<material>" + "|".join(re.escape(m) for m in _sorted_materials) + r")\b",
    re.IGNORECASE,
)
# Pattern 4: property-only queries like "oxygen fugacity" or "seismic anisotropy"
_PROPERTY_PATTERNS = {
    "fugacity": re.compile(r"\b(?:oxygen|sulfur|sulphur)\s+fugacit\w*\b", re.IGNORECASE),
    "anisotropy": re.compile(r"\b(?:seismic|mantle)\s+anisotrop\w*\b", re.IGNORECASE),
    "redox": re.compile(r"\b(?:redox\s+state|mantle\s+redox|oxidation\s+state)\b", re.IGNORECASE),
    "ore": re.compile(r"\b(?:ore\s+deposit|mineralization|ore-forming)\b", re.IGNORECASE),
    "volcanic_hazards": re.compile(r"\b(?:volcanic\s+hazard\w*|eruption\s+(?:dynamics|risk))\b", re.IGNORECASE),
    "subduction": re.compile(r"\b(?:subduction|slab\s+(?:rollback|tear|detachment))\b", re.IGNORECASE),
    "mantle_plume": re.compile(r"\b(?:mantle\s+plume|hotspot|large\s+igneous\s+province)\b", re.IGNORECASE),
    "impact": re.compile(r"\b(?:impact\s+crater|meteorite\s+impact|shock\s+metamorph)\b", re.IGNORECASE),
}


def _match_material(query_lower: str) -> str | None:
    """Find the first material mentioned in the query."""
    # Direct canonical match
    for canonical in _sorted_materials:
        if canonical in query_lower:
            return canonical
    # Synonym match
    for syn, canonical in _MATERIAL_LOOKUP.items():
        if len(syn) > 3 and syn in query_lower:
            return canonical
    return None


def _match_method(query_lower: str) -> str | None:
    """Find the first method mentioned in the query."""
    for canonical in _sorted_methods:
        if canonical in query_lower:
            return canonical
    for syn, canonical in _METHOD_LOOKUP.items():
        if len(syn) > 4 and syn in query_lower:
            return canonical
    return None


def _match_context(query_lower: str) -> str | None:
    """Detect geological context from query."""
    contexts = {
        "igneous": ["igneous", "magmatic", "volcanic", "plutonic", "intrusive"],
        "metamorphic": ["metamorphic", "metamorphism", "metapelitic", "metabasic"],
        "mantle": ["mantle", "asthenosphere", "lithospheric mantle", "mantle wedge"],
        "sedimentary": ["sedimentary", "basin", "detrital", "diagenetic"],
        "planetary": ["planetary", "lunar", "martian", "mars", "exoplanet"],
        "structural": ["structural", "tectonic", "fault", "shear", "deformation"],
        "hydrogeological": ["hydrogeological", "groundwater", "aquifer", "contaminant"],
    }
    for context, terms in contexts.items():
        for term in terms:
            if term in query_lower:
                return context
    return None


def _derive_topic_key(material: str | None, method: str | None, query_lower: str) -> str | None:
    """Derive the topic template key from material + method."""
    # Method-based topics
    method_to_topic = {
        "thermometry": "thermometry",
        "barometry": "barometry",
        "thermobarometry": "thermobarometry",
        "geochronology": "geochronology",
        "U-Pb": "geochronology",
        "Ar-Ar": "geochronology",
        "Re-Os": "geochronology",
        "fission track": "thermochronology",
        "helium": "thermochronology",
        "cosmogenic": "thermochronology",
        "fugacity": "fugacity",
        "diffusion": "diffusion_chronometry",
        "isotope": "isotope_geochemistry",
        "trace element": "trace_element_geochemistry",
        "stress": "structural_analysis",
        "strain": "structural_analysis",
        "anisotropy": "seismic_anisotropy",
        "tomography": "seismic_tomography",
        "conductivity": "geoelectrical",
        "gravity": "potential_fields",
        "provenance": "sedimentary_provenance",
        "viscosity": "volcanology",
        "paleoclimate": "paleoclimate",
        "paleomagnetism": "paleomagnetism",
        "permeability": "hydrogeology",
        "porosity": "hydrogeology",
        # Extended coverage
        "petrogenesis": "igneous_petrology",
        "metamorphism": "metamorphic_petrology",
        "geochemistry": "geochemistry_general",
        "kinematics": "structural_analysis",
        "heat flow": "geothermal",
        "mineralization": "ore_genesis",
        "stratigraphy": "sedimentology",
        "geomorphology": "geomorphology",
        "tectonics": "tectonics",
        "biogeochemistry": "biogeochemistry",
        "petroleum": "petroleum_geology",
        "geotechnical": "engineering_geology",
        "environmental": "environmental_geology",
        "planetary": "planetary_geology",
        "inclusions": "igneous_petrology",
    }
    if method and method in method_to_topic:
        return method_to_topic[method]
    # Property-based fallback
    for prop_name, pattern in _PROPERTY_PATTERNS.items():
        if pattern.search(query_lower):
            prop_to_topic = {
                "fugacity": "fugacity",
                "anisotropy": "seismic_anisotropy",
                "redox": "fugacity",
                "ore": "ore_genesis",
                "volcanic_hazards": "volcanology",
                "subduction": "tectonics",
                "mantle_plume": "tectonics",
                "impact": "planetary_geology",
            }
            return prop_to_topic.get(prop_name)
    return None


def parse_intent_pattern(query: str) -> ResearchIntent | None:
    """Layer 1: Parse query using regex pattern matching.

    Handles ~80% of queries. Returns None if no patterns match.
    """
    if not query or not query.strip():
        return None
    q_lower = query.lower().strip()

    material = _match_material(q_lower)
    method = _match_method(q_lower)
    context = _match_context(q_lower)
    topic_key = _derive_topic_key(material, method, q_lower)

    # Need at least material OR method to classify
    if not material and not method and not topic_key:
        return None

    # Build material synonyms
    material_syns: list[str] = []
    if material and material in MATERIALS:
        material_syns = MATERIALS[material][0]

    # Build method synonyms
    method_syns: list[str] = []
    if method and method in METHODS:
        method_syns = METHODS[method]["synonyms"]

    # Derive property from method
    prop = None
    if method and method in METHODS:
        prop = METHODS[method].get("target_property") or METHODS[method].get("property")
    elif topic_key == "fugacity":
        prop = "fugacity"

    # Detect goal from query
    goal = None
    q_lower = query.lower().strip()
    if any(w in q_lower for w in ["calibrat", "experimental calibration", "standardize"]):
        goal = "calibration"
    elif any(w in q_lower for w in ["review", "overview", "state of the art", "summar"]):
        goal = "review"
    elif any(w in q_lower for w in ["identif", "classify", "discriminate", "detect"]):
        goal = "identification"
    elif any(w in q_lower for w in ["appl", "case study", "example", "field study"]):
        goal = "application"

    return ResearchIntent(
        target_property=prop,
        material=material,
        method=method,
        context=context,
        goal=goal,
        topic_key=topic_key,
        material_synonyms=material_syns,
        method_synonyms=method_syns,
        template=None,  # loaded lazily by caller
        parser_source="pattern",
        raw_query=query,
    )


# =============================================================================
# LLM fallback parser — Layer 2 (complex queries, 5-10s)
# =============================================================================

_LLM_PROMPT = """Analyze this earth science research query and extract structured intent.

Query: "{query}"

Respond ONLY in JSON (no prose, no markdown fences):
{{
  "property": "temperature|pressure|age|fugacity|diffusion|anisotropy|strain|stress|composition|density|velocity|none",
  "material": "pyroxene|garnet|zircon|amphibole|olivine|feldspar|quartz|basalt|eclogite|peridotite|none",
  "method": "thermometry|barometry|thermobarometry|geochronology|fugacity|diffusion|isotope|structural|tomography|none",
  "context": "igneous|metamorphic|mantle|sedimentary|planetary|structural|hydrogeological|none",
  "topic": "thermometry|barometry|thermobarometry|geochronology|fugacity|diffusion|isotope_geochemistry|structural_analysis|seismic_anisotropy|seismic_tomography|sedimentary_provenance|volcanology|paleoclimate|paleomagnetism|hydrogeology|ore_genesis|none"
}}

Only include fields you can identify with high confidence. Use "none" for uncertain fields."""


def parse_intent_llm(query: str, timeout: float = 30.0) -> ResearchIntent | None:
    """Layer 2: Use Ollama LLM to parse complex research queries.

    Returns None if Ollama unavailable or parsing fails.
    """
    try:
        import urllib.request

        payload = json.dumps(
            {
                "model": "qwen3.5",
                "messages": [
                    {"role": "user", "content": "/no_think\n" + _LLM_PROMPT.format(query=query)}
                ],
                "stream": False,
                "think": False,
                "options": {"temperature": 0.0, "top_k": 1, "num_predict": 512},
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
            msg = result.get("message", {}).get("content", "")

        # Strip thinking tokens
        if msg.startswith(("/no_think", "<think")):
            msg = msg.split("</think>")[-1].strip() if "</think>" in msg else msg

        # Find JSON in response
        import re as _re

        json_match = _re.search(r"\{[^}]+\}", msg, _re.DOTALL)
        if not json_match:
            return None
        parsed = json.loads(json_match.group())

        material = parsed.get("material", "none")
        method = parsed.get("method", "none")
        if material == "none":
            material = None
        if method == "none":
            method = None

        topic_key = parsed.get("topic", "none")
        if topic_key == "none":
            topic_key = None

        material_syns = MATERIALS.get(material, ([], ""))[0] if material else []
        method_syns = METHODS.get(method, {}).get("synonyms", []) if method else []

        return ResearchIntent(
            target_property=parsed.get("property") if parsed.get("property") != "none" else None,
            material=material,
            method=method,
            context=parsed.get("context") if parsed.get("context") != "none" else None,
            topic_key=topic_key,
            material_synonyms=material_syns,
            method_synonyms=method_syns,
            template=None,
            parser_source="llm",
            raw_query=query,
        )
    except Exception as e:
        log.debug("LLM intent parsing failed: %s", e)
        return None


# =============================================================================
# Main entry point — try pattern, then LLM, then fallback
# =============================================================================

_intent_cache: dict[str, ResearchIntent] = {}


def parse_intent(query: str) -> ResearchIntent:
    """Parse a research query into structured intent.

    Tries pattern matching first (instant), then LLM (5-10s),
    then returns a minimal fallback intent.
    """
    if not query:
        return ResearchIntent(raw_query=query)

    # Cache hit
    if query in _intent_cache:
        return _intent_cache[query]

    # Layer 1: Pattern matching
    intent = parse_intent_pattern(query)

    # Layer 2: LLM fallback
    if intent is None:
        log.debug("Pattern matching failed for '%s', trying LLM...", query[:50])
        intent = parse_intent_llm(query)

    # Layer 3: Fallback
    if intent is None:
        intent = ResearchIntent(
            raw_query=query,
            parser_source="fallback",
        )
        log.debug("Intent parsing fell back to generic for '%s'", query[:50])

    # Load topic template if available
    if intent.topic_key:
        try:
            from _topic_templates import get_template

            intent.template = get_template(intent.topic_key)
            if intent.template:
                log.info(
                    "Intent: property=%s material=%s method=%s topic=%s template=%s (via %s)",
                    intent.target_property,
                    intent.material,
                    intent.method,
                    intent.topic_key,
                    intent.template.name,
                    intent.parser_source,
                )
        except Exception:
            pass

    _intent_cache[query] = intent
    return intent


# =============================================================================
# Intent-driven query expansion
# =============================================================================


def expand_with_intent(intent: ResearchIntent) -> str:
    """Generate a domain-specific search query from parsed intent.

    Uses topic template synonyms + landmark authors + material synonyms.
    Falls back to existing ontology expansion if no template.
    """
    if not intent.is_geoscience:
        return intent.raw_query

    parts = [intent.raw_query]

    # Add material synonyms
    if intent.material_synonyms:
        seen = set(w.lower() for w in parts)
        for syn in intent.material_synonyms[:10]:
            if syn.lower() not in seen:
                parts.append(syn)
                seen.add(syn.lower())

    # Add method/template synonyms
    if intent.template and intent.template.synonyms:
        seen = set(w.lower() for w in parts)
        for syn in intent.template.synonyms:
            if syn.lower() not in seen:
                parts.append(syn)
                seen.add(syn.lower())
    elif intent.method_synonyms:
        seen = set(w.lower() for w in parts)
        for syn in intent.method_synonyms:
            if syn.lower() not in seen:
                parts.append(syn)
                seen.add(syn.lower())

    # Add landmark authors
    if intent.template and intent.template.landmarks:
        for author in intent.template.landmarks:
            parts.append(author)

    expanded = " ".join(parts)
    if expanded != intent.raw_query:
        log.info("Intent expansion: '%s' -> %d terms", intent.raw_query[:40], len(expanded.split()))
    return expanded


# =============================================================================
# Intent-aware filtering
# =============================================================================


def intent_filter(papers: list, intent: ResearchIntent) -> list:
    """Filter papers based on intent-specific relevance criteria.

    - Excludes papers matching template.exclude patterns
    - Does NOT reject papers missing must_have (handles via ranking penalty instead)
    """
    if not intent.has_template or not intent.template.exclude:
        return papers

    exclude_patterns = [ex.lower() for ex in intent.template.exclude]
    filtered = []

    for p in papers:
        title = ""
        abstract = ""
        if hasattr(p, "title"):
            title = (p.title or "").lower()
            abstract = (p.abstract or "").lower()
        elif isinstance(p, dict):
            title = (p.get("title") or "").lower()
            abstract = (p.get("abstract") or "").lower()
        text = f"{title} {abstract}"

        # Exclude check
        excluded = False
        for pattern in exclude_patterns:
            if pattern in text:
                excluded = True
                break

        if not excluded:
            filtered.append(p)

    if len(filtered) < len(papers):
        log.info(
            "Intent filter: %d -> %d (removed %d peripheral)",
            len(papers),
            len(filtered),
            len(papers) - len(filtered),
        )

    return filtered


# =============================================================================
# Intent-aware ranking boost
# =============================================================================


def intent_boost(papers: list, intent: ResearchIntent) -> list[float]:
    """Compute intent-specific boost factors for ranking.

    Boosts papers containing landmark authors and must-have terms.
    Returns list of multipliers (1.0 = no change).
    """
    if not intent.has_template:
        return [1.0] * len(papers)

    template = intent.template
    boosts = []

    for p in papers:
        title = ""
        abstract = ""
        if hasattr(p, "title"):
            title = (p.title or "").lower()
            abstract = (p.abstract or "").lower()
        elif isinstance(p, dict):
            title = (p.get("title") or "").lower()
            abstract = (p.get("abstract") or "").lower()
        text = f"{title} {abstract}"
        boost = 1.0

        # Boost for landmark authors
        for author in template.landmarks:
            if author.lower() in text:
                boost *= 1.5

        # Boost for must-have terms
        for term in template.must_have:
            if term.lower() in text:
                boost *= 1.2

        # Boost for material + property co-occurrence
        if intent.material and intent.target_property:
            mat_in = intent.material.lower() in text
            prop_in = intent.target_property.lower() in text
            if mat_in and prop_in:
                boost *= 1.3

        # Custom boost terms from template
        if template.boost_terms:
            for term, factor in template.boost_terms.items():
                if term.lower() in text:
                    boost *= factor

        boosts.append(boost)

    return boosts
