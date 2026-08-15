"""Topic templates for intent-aware research.

Each template defines:
- synonyms: method-specific vocabulary for search expansion
- landmarks: key author names to search for
- must_have: terms a paper should contain for relevance
- exclude: peripheral concepts to filter out
- report_sections: conceptual sections for report organization
- boost_terms: graduated boost factors for ranking

20 templates covering all major earth science research types.
"""

from __future__ import annotations

from _types import TopicTemplate

_TEMPLATES: dict[str, TopicTemplate] = {}


def _register(t: TopicTemplate) -> TopicTemplate:
    _TEMPLATES[t.name] = t
    return t



# =============================================================================
# 21. Igneous Petrology
# =============================================================================
_register(
    TopicTemplate(
        name="igneous_petrology",
        display_name="Igneous Petrology",
        synonyms=[
            "petrogenesis", "magma evolution", "crystallization", "fractional crystallization",
            "partial melting", "magma differentiation", "liquid line of descent",
            "phase petrology", "experimental petrology", "magma series",
            "tholeiitic", "calc-alkaline", "alkaline", "felsic", "mafic", "ultramafic",
        ],
        landmarks=["Tuttle", "Bowen", "Yoder", "Tilley", "Carmichael", "Ghiorso", "Sack"],
        must_have=["magma", "igneous", "petrogenesis", "volcanic", "plutonic", "crystallization"],
        exclude=["metamorphic", "sedimentary", "structural", "fault", "earthquake"],
        report_sections=[
            "Magma composition and series classification",
            "Fractional crystallization and magma evolution",
            "Partial melting and source characteristics",
            "Phase relations and experimental constraints",
            "Tectonic environment and petrogenetic model",
            "Comparison with global igneous provinces",
        ],
        boost_terms={"experimental": 1.3, "phase diagram": 1.4, "MELTS": 1.3},
        target_property="composition",
        unit="wt%",
        valid_range=(0.0, 100.0),
    )
)

# =============================================================================
# 22. Metamorphic Petrology
# =============================================================================
_register(
    TopicTemplate(
        name="metamorphic_petrology",
        display_name="Metamorphic Petrology",
        synonyms=[
            "metamorphism", "metapelite", "metabasite", "granulite", "eclogite",
            "blueschist", "P-T path", "pseudosection", "phase equilibria",
            "metamorphic grade", "metamorphic facies", "isograd",
            "garnet zoning", "reaction texture", "metamorphic evolution",
        ],
        landmarks=["Spear", "Powell", "Holland", "Connolly", "White", "Kohn", "Florence"],
        must_have=["metamorphic", "metamorphism", "facies", "grade", "P-T", "petrology"],
        exclude=["igneous", "volcanic", "magma", "sedimentary", "basin"],
        report_sections=[
            "Metamorphic facies and grade assignment",
            "P-T evolution and pseudosection modeling",
            "Mineral chemistry and zoning profiles",
            "Reaction textures and metamorphic history",
            "Geochronological constraints on metamorphism",
            "Tectonic interpretation of P-T paths",
        ],
        boost_terms={"pseudosection": 1.4, "Perple_X": 1.3, "THERMOCALC": 1.3, "garnet": 1.2},
        target_property="pressure-temperature",
        unit="kbar/°C",
        valid_range=(0.1, 150.0),
    )
)

# =============================================================================
# 23. Geochemistry (General)
# =============================================================================
_register(
    TopicTemplate(
        name="geochemistry_general",
        display_name="Geochemistry",
        synonyms=[
            "geochemical", "major element", "trace element", "rare earth element",
            "REE pattern", "spider diagram", "normalization", "chondrite",
            "primitive mantle", "bulk earth", "element partitioning",
            "compatibility", "incompatible element", "fluid mobile", "fluid immobile",
        ],
        landmarks=["Sun", "McDonough", "Hofmann", "Rollinson", "White"],
        must_have=["geochem", "element", "composition", "trace", "major", "isotope"],
        exclude=["materials science", "physics", "engineering", "biological"],
        report_sections=[
            "Major element geochemistry",
            "Trace element and REE patterns",
            "Isotopic constraints",
            "Source characteristics and mantle normalization",
            "Fractionation and partitioning",
            "Geodynamic implications",
        ],
        boost_terms={"partition coefficient": 1.3, "normalization": 1.2, "chondrite": 1.2},
        target_property="composition",
        unit="ppm/wt%",
        valid_range=(0.0, 100.0),
    )
)

# =============================================================================
# 24. Geothermal
# =============================================================================
_register(
    TopicTemplate(
        name="geothermal",
        display_name="Geothermal Systems",
        synonyms=[
            "heat flow", "geothermal gradient", "thermal conductivity",
            "heat generation", "radiogenic heat", "thermal model",
            "geothermal energy", "hydrothermal system", "hot spring",
            "fumarole", "geyser", "thermal reservoir",
        ],
        landmarks=["Pollack", "Chapman", "Furlong", "Artemieva", "Hasterok"],
        must_have=["heat", "thermal", "geothermal", "gradient", "conductivity"],
        exclude=["atmospheric", "climate", "weather", "ocean thermal"],
        report_sections=[
            "Heat flow measurements and regional patterns",
            "Thermal conductivity and heat generation",
            "Geothermal gradient modeling",
            "Hydrothermal systems and fluid circulation",
            "Geothermal resource assessment",
            "Tectonic and magmatic heat sources",
        ],
        boost_terms={"heat flow": 1.3, "mW/m2": 1.2, "thermal gradient": 1.3},
        target_property="heat flow",
        unit="mW/m²",
        valid_range=(0.0, 500.0),
    )
)

# =============================================================================
# 25. Sedimentology
# =============================================================================
_register(
    TopicTemplate(
        name="sedimentology",
        display_name="Sedimentology",
        synonyms=[
            "sedimentary", "depositional environment", "stratigraphy",
            "sequence stratigraphy", "facies analysis", "sediment transport",
            "diagenesis", "sedimentary structure", "provenance",
            "basin analysis", "deposition", "sedimentation rate",
        ],
        landmarks=["Reading", "Boggs", "Miall", "Einsele", "Allen"],
        must_have=["sediment", "stratigraph", "deposition", "facies", "basin"],
        exclude=["igneous", "volcanic", "metamorphic", "magma"],
        report_sections=[
            "Depositional environment and facies",
            "Stratigraphic framework and sequence analysis",
            "Sediment provenance and transport",
            "Diagenesis and post-depositional changes",
            "Basin evolution and subsidence history",
            "Sea-level and climatic controls",
        ],
        boost_terms={"sequence stratigraphy": 1.3, "facies": 1.2, "provenance": 1.2},
        target_property="stratigraphy",
        unit="m/Myr",
        valid_range=(0.0, 10000.0),
    )
)

# =============================================================================
# 26. Geomorphology
# =============================================================================
_register(
    TopicTemplate(
        name="geomorphology",
        display_name="Geomorphology",
        synonyms=[
            "landscape evolution", "surface process", "erosion rate",
            "denudation", "tectonic geomorphology", "river profile",
            "channel morphology", " hillslope", "relief", "uplift rate",
            "incision", "weathering", "mass wasting",
        ],
        landmarks=["Summerfield", "Burbank", "Anderson", "Whipple", "Tucker"],
        must_have=["geomorph", "landscape", "erosion", "surface", "relief", "incision"],
        exclude=["igneous", "metamorphic", "geochemistry laboratory"],
        report_sections=[
            "Landscape morphology and relief analysis",
            "Erosion and denudation rates",
            "Tectonic geomorphology and active faulting",
            "River profile analysis and channel dynamics",
            "Weathering and hillslope processes",
            "Long-term landscape evolution models",
        ],
        boost_terms={"erosion rate": 1.3, "uplift": 1.2, "incision": 1.2, "cosmogenic": 1.3},
        target_property="erosion rate",
        unit="mm/yr",
        valid_range=(0.0, 10.0),
    )
)

# =============================================================================
# 27. Tectonics
# =============================================================================
_register(
    TopicTemplate(
        name="tectonics",
        display_name="Tectonics",
        synonyms=[
            "plate tectonics", "tectonic evolution", "orogeny", "mountain building",
            "continental collision", "subduction zone", "rifting", "extension",
            "convergence", "accretionary wedge", "suture", "terrane",
            "basin inversion", "plate boundary", "back-arc",
        ],
        landmarks=["Dewey", "Molnar", "England", "Houseman", "Cloos", "Ernst"],
        must_have=["tectonic", "orogen", "subduction", "collision", "rift", "plate"],
        exclude=["biology", "ecology", "materials science"],
        report_sections=[
            "Regional tectonic framework",
            "Plate kinematics and boundary processes",
            "Orogenic evolution and mountain building",
            "Subduction zone dynamics",
            "Rifting and extensional tectonics",
            "Neotectonics and active deformation",
        ],
        boost_terms={"orogen": 1.2, "subduction": 1.2, "collision": 1.2, "plate reconstruction": 1.3},
        target_property="displacement",
        unit="km/Myr",
        valid_range=(0.0, 1000.0),
    )
)

# =============================================================================
# 28. Biogeochemistry
# =============================================================================
_register(
    TopicTemplate(
        name="biogeochemistry",
        display_name="Biogeochemistry",
        synonyms=[
            "biogeochemical cycle", "carbon cycle", "nutrient cycling",
            "organic geochemistry", "biomarker", "carbon sequestration",
            "microbial", "redox", "isotope fractionation biological",
            "primary productivity", "organic matter", "kerogen",
        ],
        landmarks=["Berner", "Garrels", "Holland", "Kump", "Canfield"],
        must_have=["biogeochem", "organic", "carbon cycle", "nutrient", "microbial"],
        exclude=["igneous petrology", "metamorphic phase", "structural geology"],
        report_sections=[
            "Biogeochemical cycling and element fluxes",
            "Organic geochemistry and biomarkers",
            "Carbon sequestration and storage",
            "Microbial processes and isotope fractionation",
            "Paleoenvironmental reconstruction",
            "Anthropogenic impacts on biogeochemical cycles",
        ],
        boost_terms={"carbon cycle": 1.3, "isotope fractionation": 1.2, "biomarker": 1.3},
        target_property="flux",
        unit="mol/yr",
        valid_range=(0.0, 1e15),
    )
)

# =============================================================================
# 29. Petroleum Geology
# =============================================================================
_register(
    TopicTemplate(
        name="petroleum_geology",
        display_name="Petroleum Geology",
        synonyms=[
            "petroleum", "hydrocarbon", "oil and gas", "reservoir",
            "source rock", "kerogen", "petroleum system", "trap",
            "seal", "migration", "porosity", "permeability reservoir",
            "organic richness", "thermal maturity", "vitrinite reflectance",
        ],
        landmarks=["Tissot", "Welte", "Hunt", "Demaison", "Pepper"],
        must_have=["petroleum", "hydrocarbon", "reservoir", "source rock", "kerogen"],
        exclude=["igneous", "metamorphic", "structural geology fault"],
        report_sections=[
            "Source rock characterization and maturity",
            "Reservoir properties and diagenesis",
            "Petroleum system analysis",
            "Trap and seal mechanisms",
            "Migration pathways and timing",
            "Basin modeling and resource assessment",
        ],
        boost_terms={"vitrinite reflectance": 1.3, "kerogen": 1.2, "source rock": 1.3, "Ro": 1.2},
        target_property="thermal maturity",
        unit="%Ro",
        valid_range=(0.2, 5.0),
    )
)

# =============================================================================
# 30. Engineering Geology
# =============================================================================
_register(
    TopicTemplate(
        name="engineering_geology",
        display_name="Engineering Geology",
        synonyms=[
            "geotechnical", "rock mechanics", "soil mechanics",
            "slope stability", "foundation", "tunnel", "excavation",
            "dam site", "landslide", "rock mass quality", "RMR",
            "subsidence", "compaction", "bearing capacity",
        ],
        landmarks=["Hoek", "Goodman", "Bieniawski", "Barton", "Terzaghi"],
        must_have=["geotechnical", "engineering geology", "rock mechanics", "slope", "foundation"],
        exclude=["pure geochemistry", "paleontology", "stratigraphy regional"],
        report_sections=[
            "Site characterization and rock mass quality",
            "Slope stability and landslide assessment",
            "Foundation design and bearing capacity",
            "Excavation and tunnel support",
            "Geohazard assessment",
            "Geotechnical modeling and monitoring",
        ],
        boost_terms={"RMR": 1.3, "RQD": 1.3, "slope stability": 1.3, "Hoek-Brown": 1.3},
        target_property="rock strength",
        unit="MPa",
        valid_range=(0.0, 500.0),
    )
)

# =============================================================================
# 31. Environmental Geology
# =============================================================================
_register(
    TopicTemplate(
        name="environmental_geology",
        display_name="Environmental Geology",
        synonyms=[
            "contamination", "groundwater quality", "heavy metal",
            "remediation", "landfill", "waste disposal",
            "acid mine drainage", "radioactive waste", "geological storage",
            "CO2 sequestration", "water quality", "pollution",
            "site assessment", "geochemical baseline",
        ],
        landmarks=["Alloway", "Fetter", "Domenico", "Schwartz", "Appelo"],
        must_have=["environmental", "contamination", "groundwater quality", "remediation", "pollut"],
        exclude=["igneous petrology", "pure metamorphic"],
        report_sections=[
            "Contamination sources and pathways",
            "Groundwater quality and monitoring",
            "Remediation strategies",
            "Geochemical baseline and background",
            "Waste disposal and geological storage",
            "Risk assessment and regulatory framework",
        ],
        boost_terms={"remediation": 1.3, "contamination": 1.2, "groundwater": 1.2, "baseline": 1.2},
        target_property="concentration",
        unit="mg/L",
        valid_range=(0.0, 10000.0),
    )
)

# =============================================================================
# 32. Planetary Geology
# =============================================================================
_register(
    TopicTemplate(
        name="planetary_geology",
        display_name="Planetary Geology",
        synonyms=[
            "planetary", "mars", "martian", "lunar", "moon",
            "venus", "mercury", "asteroid", "meteorite", "impact crater",
            "shock metamorphism", "planetary differentiation",
            "planetary mantle", "extraterrestrial", "chondrite",
        ],
        landmarks=["Taylor", "McSween", "Kieffer", "Head", "Solomon", "Zuber"],
        must_have=["planetary", "mars", "lunar", "venus", "meteorite", "impact", "extraterrestrial"],
        exclude=["earth earthquake", "structural geology fault", "petroleum"],
        report_sections=[
            "Planetary surface processes and landforms",
            "Impact cratering and shock metamorphism",
            "Volcanism and magmatic evolution",
            "Meteorite and sample analysis",
            "Planetary interiors and differentiation",
            "Comparative planetology",
        ],
        boost_terms={"Mars": 1.3, "meteorite": 1.2, "impact": 1.2, "lunar": 1.2},
        target_property="composition",
        unit="wt%",
        valid_range=(0.0, 100.0),
    )
)


def get_template(name: str) -> TopicTemplate | None:
    """Get a topic template by name. Returns None if not found."""
    return _TEMPLATES.get(name)


def all_templates() -> dict[str, TopicTemplate]:
    """Return all registered templates."""
    return dict(_TEMPLATES)


# =============================================================================
# 1. Thermometry
# =============================================================================
_register(
    TopicTemplate(
        name="thermometry",
        display_name="Thermometry",
        target_property="temperature",
        unit="°C",
        valid_range=(100, 2000),
        synonyms=[
            "thermometer",
            "calibration",
            "temperature estimate",
            "Fe-Mg exchange",
            "solvus",
            "exchange equilibrium",
            "two-pyroxene",
            "clinopyroxene-liquid",
            "orthopyroxene",
            "single-clinopyroxene",
            "RE-in-clinopyroxene",
            "Ti-in-zircon",
            "Zr-in-rutile",
            "thermometric",
            "temperature calculation",
            "equilibrium temperature",
        ],
        landmarks=[
            "Putirka",
            "Wells",
            "Lindsley",
            "Brey",
            "Kohler",
            "Nimis",
            "Beattie",
            "Woodland",
            "Taylor",
            "Ferry",
            "Watson",
            "Tomkins",
        ],
        must_have=[
            "thermometer",
            "calibration",
            "temperature",
            "thermometry",
            "exchange",
            "equilibrium",
            "solvus",
            "Putirka",
            "Wells",
        ],
        exclude=[
            "asteroid",
            "comet",
            "spectroscopy",
            "dust",
            "ceramic",
            "crystal structure determination",
            "magnetic properties",
            "Raman band assignment",
            "optical properties",
            "dielectric",
            # Public health / industrial — NOT thermometry
            "asbestos",
            "chrysotile",
            "fibrous amphibole",
            "cleavage fragment",
            "mesothelioma",
            "carcinogen",
            "occupational exposure",
            "fiber counting",
            "phase-contrast microscopy",
            # Chemistry / physics — NOT geoscience thermometry
            "xenon",
            "NaXe",
            "noble gas compound",
            "metallic hydrogen",
            # Structural geology / rheology — NOT thermometry
            "dislocation creep",
            "lattice-preferred orientation",
            "LPO",
            "rheolog",
            "deformation mechanism",
            "shear zone",
            "microstructure",
            "texture formation",
            "crystallographic preferred orientation",
            # Other non-thermometry
            "encyclopedia",
            "dictionary",
            "mica-amphibole-peridotite",
            "amphibole-schist",
            "amphibole-peridotite",
        ],
        report_sections=[
            "Two-pyroxene thermometry",
            "Clinopyroxene-liquid thermometry",
            "Single-clinopyroxene methods",
            "Trace-element thermometry",
            "Calibration and uncertainty",
            "Applications",
        ],
        boost_terms={
            "calibration": 1.3,
            "Putirka": 1.5,
            "Wells": 1.4,
            "exchange equilibrium": 1.3,
            "experimental": 1.2,
        },
    )
)

# =============================================================================
# 2. Barometry
# =============================================================================
_register(
    TopicTemplate(
        name="barometry",
        display_name="Barometry",
        target_property="pressure",
        unit="kbar",
        valid_range=(0.1, 150),
        synonyms=[
            "barometer",
            "pressure estimate",
            "geobarometer",
            "GASP",
            "GBPQ",
            "GRAIL",
            "GQBar",
            "aluminosilicate",
            "quartz-coesite",
            "diamond inclusion",
            "pressure calculation",
            "pressure indicator",
        ],
        landmarks=[
            "Ghent",
            "Newton",
            "Koziol",
            "Holdaway",
            "Kohn",
            "Spear",
            "Caddick",
            "Powell",
        ],
        must_have=[
            "barometer",
            "pressure",
            "barometry",
            "GASP",
            "GRAIL",
            "equilibrium",
        ],
        exclude=[
            "asteroid",
            "comet",
            "ceramic",
            "magnetic",
            "crystal structure",
            "optical",
        ],
        report_sections=[
            "Net-transfer barometers",
            "Exchange barometers",
            "Phase-equilibrium barometry",
            "Elastic thermobarometry",
            "Calibration and uncertainty",
            "Applications",
        ],
        boost_terms={
            "GASP": 1.4,
            "GRAIL": 1.4,
            "pseudosection": 1.3,
            "experimental": 1.2,
        },
    )
)

# =============================================================================
# 3. Thermobarometry
# =============================================================================
_register(
    TopicTemplate(
        name="thermobarometry",
        display_name="Thermobarometry",
        target_property="temperature-pressure",
        unit="°C-kbar",
        valid_range=(0, 0),
        synonyms=[
            "geothermobarometry",
            "P-T estimate",
            "P-T path",
            "pressure-temperature",
            "geothermobarometer",
            "pseudosection",
            "phase equilibria",
            "THERMOCALC",
            "Perple_X",
            "multi-equilibrium",
        ],
        landmarks=[
            "Holland",
            "Powell",
            "Spear",
            "Kohn",
            "Vance",
            "Connolly",
            "Stipska",
            "Carson",
        ],
        must_have=[
            "thermobarometry",
            "P-T",
            "pressure-temperature",
            "pseudosection",
            "phase equilibria",
            "geothermobarometry",
        ],
        exclude=[
            "asteroid",
            "comet",
            "ceramic",
            "magnetic",
            "crystal structure",
            "optical",
        ],
        report_sections=[
            "Conventional thermobarometry",
            "Phase-equilibrium modeling",
            "P-T-t paths",
            "Elastic thermobarometry",
            "Uncertainty propagation",
            "Applications",
        ],
        boost_terms={
            "THERMOCALC": 1.4,
            "Perple_X": 1.4,
            "pseudosection": 1.3,
            "P-T path": 1.3,
        },
    )
)

# =============================================================================
# 4. Geochronology
# =============================================================================
_register(
    TopicTemplate(
        name="geochronology",
        display_name="Geochronology",
        target_property="age",
        unit="Ma",
        valid_range=(0, 4600),
        synonyms=[
            "geochronology",
            "dating",
            "age determination",
            "U-Pb",
            "zircon dating",
            "ID-TIMS",
            "SHRIMP",
            "concordia",
            "discordance",
            "age spectra",
            "SIMS",
            "LA-ICP-MS dating",
            "isotope dilution",
        ],
        landmarks=[
            "Ludwig",
            "Williams",
            "Ireland",
            "Nemchin",
            "Gehrels",
            "Dickinson",
            "Pullen",
            "Barth",
            "Scherer",
        ],
        must_have=[
            "geochronology",
            "dating",
            "U-Pb",
            "age",
            "zircon",
            "concordia",
            "isotopic",
        ],
        exclude=[
            "protein",
            "DNA",
            "cell",
            "enzyme",
            "drug",
            "clinical",
            "patient",
        ],
        report_sections=[
            "U-Pb methodology",
            "Concordia and discordance",
            "Age interpretation",
            "Detrital geochronology",
            "Thermal history",
            "Applications",
        ],
        boost_terms={
            "U-Pb": 1.3,
            "zircon": 1.2,
            "concordia": 1.3,
            "SHRIMP": 1.2,
            "detrital": 1.2,
        },
    )
)

# =============================================================================
# 5. Thermochronology (fission track, He)
# =============================================================================
_register(
    TopicTemplate(
        name="thermochronology",
        display_name="Thermochronology",
        target_property="age",
        unit="Ma",
        valid_range=(0.1, 1000),
        synonyms=[
            "thermochronology",
            "fission track",
            "(U-Th)/He",
            "apatite He",
            "zircon He",
            "annealing",
            "closure temperature",
            "cooling age",
            "exhumation",
        ],
        landmarks=[
            "Farley",
            "Reiners",
            "Gallagher",
            "Ketcham",
            "Brandon",
            "Crowley",
            "Ehlers",
        ],
        must_have=[
            "thermochronology",
            "fission track",
            "helium",
            "He dating",
            "cooling",
            "exhumation",
        ],
        exclude=[
            "protein",
            "DNA",
            "drug",
            "clinical",
        ],
        report_sections=[
            "Fission-track thermochronology",
            "(U-Th)/He thermochronology",
            "Thermal modeling",
            "Exhumation and erosion",
            "Applications",
        ],
        boost_terms={
            "apatite": 1.2,
            "closure": 1.2,
            "exhumation": 1.3,
        },
    )
)

# =============================================================================
# 6. Fugacity / Redox
# =============================================================================
_register(
    TopicTemplate(
        name="fugacity",
        display_name="Fugacity and Redox",
        target_property="fugacity",
        unit="log units",
        valid_range=(-10, 10),
        synonyms=[
            "fO2",
            "fS2",
            "oxygen fugacity",
            "sulfur fugacity",
            "redox",
            "redox state",
            "redox buffer",
            "FMQ",
            "NNO",
            "IW",
            "QFM",
            "MH",
            "Fe3+/SigmaFe",
            "ferric iron",
            "ferrous-ferric",
            "sulfide saturation",
            "sulfur speciation",
        ],
        landmarks=[
            "Frost",
            "McCammon",
            "Ballhaus",
            "Wood",
            "Canil",
            "Kress",
            "Carmichael",
            "Osborn",
        ],
        must_have=[
            "fugacity",
            "redox",
            "FMQ",
            "buffer",
            "oxygen fugacity",
            "oxidation",
        ],
        exclude=[
            "lattice gas",
            "lattice model",
            "Ising",
            "Bose gas",
            "partition function",
            "polymer",
            "self-avoiding",
            "Coulomb gas",
            "QCD",
            "fermion",
            "quark",
            "Yang-Lee",
            "Hamiltonian",
            "renormalization",
            "asteroid",
            "comet",
            "dust",
            "ceramic",
            "multimedia fugacity model",
            "pesticide",
        ],
        report_sections=[
            "Redox buffers and calibration",
            "Measurement methods",
            "Mantle oxygen fugacity",
            "Arc magma redox",
            "Ore deposit applications",
        ],
        boost_terms={
            "FMQ": 1.3,
            "redox": 1.2,
            "mantle": 1.2,
            "experimental": 1.2,
        },
    )
)

# =============================================================================
# 7. Diffusion chronometry
# =============================================================================
_register(
    TopicTemplate(
        name="diffusion_chronometry",
        display_name="Diffusion Chronometry",
        target_property="diffusion",
        unit="m²/s",
        valid_range=(0, 0),
        synonyms=[
            "diffusion",
            "diffusivity",
            "geospeedometry",
            "cooling rate",
            "diffusion chronometry",
            "Arrhenius",
            "activation energy",
            "closure temperature",
            "diffusion profile",
        ],
        landmarks=[
            "Dodson",
            "Ganguly",
            "Chakraborty",
            "Cherniak",
            "Watson",
            "Lasaga",
            "Jurewicz",
        ],
        must_have=[
            "diffusion",
            "diffusivity",
            "geospeedometry",
            "cooling rate",
            "closure",
        ],
        exclude=[
            "polymer",
            "membrane",
            "protein",
            "drug delivery",
            "catalyst",
        ],
        report_sections=[
            "Diffusion theory and Arrhenius parameters",
            "Closure temperature",
            "Cooling rate determination",
            "Garnet zoning and diffusion",
            "Applications",
        ],
        boost_terms={
            "geospeedometry": 1.4,
            "cooling rate": 1.3,
            "garnet zoning": 1.3,
            "Dodson": 1.3,
        },
    )
)

# =============================================================================
# 8. Isotope geochemistry
# =============================================================================
_register(
    TopicTemplate(
        name="isotope_geochemistry",
        display_name="Isotope Geochemistry",
        target_property="isotope ratio",
        unit="‰",
        valid_range=(-100, 100),
        synonyms=[
            "isotope geochemistry",
            "radiogenic isotope",
            "stable isotope",
            "Sr-Nd-Pb",
            "Hf",
            "d18O",
            "d13C",
            "d34S",
            "dD",
            "d15N",
            "isotopic composition",
            "fractionation",
        ],
        landmarks=[
            "Hoefs",
            "Valley",
            "Eiler",
            "White",
            "Albarede",
            "Faure",
            "Dickin",
            "Sharp",
        ],
        must_have=[
            "isotope",
            "isotopic",
            "fractionation",
            "d18O",
            "d13C",
            "Sr",
            "Nd",
            "Pb",
        ],
        exclude=[
            "protein",
            "DNA",
            "drug",
            "clinical",
            "asteroid spectroscopy",
        ],
        report_sections=[
            "Radiogenic isotopes",
            "Stable isotopes",
            "Clumped isotopes",
            "Fractionation mechanisms",
            "Tracers and provenance",
        ],
        boost_terms={
            "isotope": 1.2,
            "fractionation": 1.2,
        },
    )
)

# =============================================================================
# 9. Trace element geochemistry
# =============================================================================
_register(
    TopicTemplate(
        name="trace_element_geochemistry",
        display_name="Trace Element Geochemistry",
        target_property="concentration",
        unit="ppm",
        valid_range=(0, 10000),
        synonyms=[
            "trace element",
            "rare earth",
            "REE",
            "partition coefficient",
            "Kd",
            "spider diagram",
            "multi-element",
            "HFSE",
            "LILE",
            "LREE",
            "HREE",
        ],
        landmarks=[
            "McDonough",
            "Sun",
            "Salters",
            "Wood",
            "Bedard",
            "Rollinson",
            "Pearce",
        ],
        must_have=[
            "trace element",
            "REE",
            "partition",
            "rare earth",
            "spider",
        ],
        exclude=[
            "protein",
            "DNA",
            "drug",
            "clinical",
        ],
        report_sections=[
            "Partition coefficients",
            "REE patterns",
            "Mantle source characterization",
            "Mineral-melt equilibria",
            "Applications",
        ],
        boost_terms={
            "partition coefficient": 1.3,
            "REE": 1.2,
            "mantle source": 1.2,
        },
    )
)

# =============================================================================
# 10. Seismic anisotropy
# =============================================================================
_register(
    TopicTemplate(
        name="seismic_anisotropy",
        display_name="Seismic Anisotropy",
        target_property="anisotropy",
        unit="%",
        valid_range=(0, 20),
        synonyms=[
            "seismic anisotropy",
            "SKS splitting",
            "shear wave splitting",
            "P-wave anisotropy",
            "lattice preferred orientation",
            "LPO",
            "azimuthal anisotropy",
            "radial anisotropy",
        ],
        landmarks=[
            "Silver",
            "Savage",
            "Long",
            "Karato",
            "Mainprice",
            "Tommasi",
            "Holt",
        ],
        must_have=[
            "anisotropy",
            "splitting",
            "SKS",
            "shear wave",
            "LPO",
        ],
        exclude=[
            "optical anisotropy",
            "dielectric",
            "magnetic anisotropy",
        ],
        report_sections=[
            "SKS and shear-wave splitting",
            "Lattice preferred orientation",
            "Mantle flow patterns",
            "Crustal anisotropy",
            "Geodynamic implications",
        ],
        boost_terms={
            "SKS": 1.3,
            "splitting": 1.3,
            "mantle flow": 1.2,
        },
    )
)

# =============================================================================
# 11. Seismic tomography
# =============================================================================
_register(
    TopicTemplate(
        name="seismic_tomography",
        display_name="Seismic Tomography",
        target_property="velocity",
        unit="km/s",
        valid_range=(0.5, 15),
        synonyms=[
            "seismic tomography",
            "velocity model",
            "travel time tomography",
            "P-wave tomography",
            "S-wave tomography",
            "mantle structure",
            "body wave",
            "surface wave",
        ],
        landmarks=[
            "Grand",
            "van der Hilst",
            "Zhao",
            "Rawlinson",
            "Boschi",
            "Montelli",
            "Sigloch",
        ],
        must_have=[
            "tomography",
            "velocity",
            "seismic",
            "mantle structure",
        ],
        exclude=[
            "medical imaging",
            "CT scan",
        ],
        report_sections=[
            "Tomographic methods",
            "Mantle structure",
            "Subduction zones",
            "Mantle plumes",
            "Crustal structure",
        ],
        boost_terms={
            "tomography": 1.3,
            "velocity model": 1.2,
            "mantle": 1.1,
        },
    )
)

# =============================================================================
# 12. Structural analysis
# =============================================================================
_register(
    TopicTemplate(
        name="structural_analysis",
        display_name="Structural Analysis",
        target_property="stress-strain",
        unit="MPa",
        valid_range=(0, 10000),
        synonyms=[
            "stress inversion",
            "paleostress",
            "strain analysis",
            "finite strain",
            "fault kinematics",
            "fracture analysis",
            "fold geometry",
            "brittle deformation",
            "ductile shear",
        ],
        landmarks=[
            "Allmendinger",
            "Angelier",
            "Marrett",
            "Twiss",
            "Dunne",
            "Yamaji",
            "Delvaux",
        ],
        must_have=[
            "stress",
            "strain",
            "fault",
            "fracture",
            "structural",
            "deformation",
        ],
        exclude=[
            "materials science",
            "mechanical engineering",
            "protein structure",
        ],
        report_sections=[
            "Stress inversion methods",
            "Strain analysis",
            "Fault kinematics",
            "Microstructures",
            "Tectonic implications",
        ],
        boost_terms={
            "paleostress": 1.3,
            "stress inversion": 1.3,
            "fault": 1.1,
        },
    )
)

# =============================================================================
# 13. Sedimentary provenance
# =============================================================================
_register(
    TopicTemplate(
        name="sedimentary_provenance",
        display_name="Sedimentary Provenance",
        target_property="provenance",
        unit="",
        valid_range=(0, 0),
        synonyms=[
            "sedimentary provenance",
            "detrital",
            "source rock",
            "framework mode",
            "Dickinson",
            "heavy mineral",
            "detrital zircon",
            "paleogeography",
        ],
        landmarks=[
            "Dickinson",
            "Garzanti",
            "Weltje",
            "Fedo",
            "Ingersoll",
            "Basu",
            "Pettijohn",
        ],
        must_have=[
            "provenance",
            "detrital",
            "source",
            "sandstone",
            "sedimentary",
        ],
        exclude=[
            "igneous geochemistry",
            "mantle",
        ],
        report_sections=[
            "Framework mineral modes",
            "Detrital zircon geochronology",
            "Heavy mineral analysis",
            "Paleogeographic reconstruction",
        ],
        boost_terms={
            "detrital": 1.3,
            "provenance": 1.3,
        },
    )
)

# =============================================================================
# 14. Volcanology
# =============================================================================
_register(
    TopicTemplate(
        name="volcanology",
        display_name="Volcanology",
        target_property="eruption dynamics",
        unit="",
        valid_range=(0, 0),
        synonyms=[
            "eruption dynamics",
            "magma chamber",
            "pyroclastic",
            "lava flow",
            "volcanic hazard",
            "gas emission",
            "magma rheology",
            "conduit flow",
        ],
        landmarks=[
            "Houghton",
            "Carey",
            "Sparks",
            "Cashman",
            "Wilson",
            "Head",
            "Papale",
        ],
        must_have=[
            "volcanic",
            "eruption",
            "magma",
            "lava",
            "pyroclastic",
        ],
        exclude=[
            "igneous petrology",
            "metamorphic",
        ],
        report_sections=[
            "Eruption dynamics",
            "Magma chamber processes",
            "Pyroclastic deposits",
            "Volcanic hazards",
            "Monitoring",
        ],
        boost_terms={
            "eruption": 1.2,
            "magma chamber": 1.3,
        },
    )
)

# =============================================================================
# 15. Ore genesis
# =============================================================================
_register(
    TopicTemplate(
        name="ore_genesis",
        display_name="Ore Genesis",
        target_property="mineralization",
        unit="",
        valid_range=(0, 0),
        synonyms=[
            "ore deposit",
            "mineralization",
            "hydrothermal",
            "porphyry copper",
            "epithermal",
            "VMS",
            "SEDEX",
            "orogenic gold",
            "fluid inclusion",
            "ore-forming",
        ],
        landmarks=[
            "Sillitoe",
            "Hedenquist",
            "Richards",
            "Wilkinson",
            "Large",
            "Heinrich",
        ],
        must_have=[
            "ore",
            "deposit",
            "mineralization",
            "hydrothermal",
            "sulfide",
        ],
        exclude=[
            "sedimentary petrology",
            "igneous geochemistry",
        ],
        report_sections=[
            "Porphyry and epithermal systems",
            "VMS and SEDEX deposits",
            "Orogenic gold",
            "Fluid evolution",
            "Exploration implications",
        ],
        boost_terms={
            "porphyry": 1.3,
            "epithermal": 1.3,
            "ore-forming": 1.3,
        },
    )
)

# =============================================================================
# 16. Paleoclimate
# =============================================================================
_register(
    TopicTemplate(
        name="paleoclimate",
        display_name="Paleoclimate",
        target_property="temperature",
        unit="°C",
        valid_range=(-50, 100),
        synonyms=[
            "paleoclimate",
            "paleotemperature",
            "climate proxy",
            "CO2 proxy",
            "paleo-CO2",
            "ice core",
            "foraminifera",
            "tree ring",
            "speleothem",
        ],
        landmarks=[
            "Pagani",
            "Royer",
            "Beerling",
            "Huber",
            "Zachos",
            "Crowley",
            "Hansen",
        ],
        must_have=[
            "paleoclimate",
            "paleotemperature",
            "climate",
            "proxy",
            "paleo",
        ],
        exclude=[
            "igneous",
            "metamorphic",
            "mantle",
        ],
        report_sections=[
            "Temperature proxies",
            "CO2 reconstruction",
            "Ice volume and sea level",
            "Climate modeling",
            "Biotic response",
        ],
        boost_terms={
            "proxy": 1.2,
            "paleoclimate": 1.3,
        },
    )
)

# =============================================================================
# 17. Paleomagnetism
# =============================================================================
_register(
    TopicTemplate(
        name="paleomagnetism",
        display_name="Paleomagnetism",
        target_property="magnetization",
        unit="A/m",
        valid_range=(0, 100),
        synonyms=[
            "paleomagnetic",
            "natural remanent",
            "NRM",
            "demagnetization",
            "apparent polar wander",
            "APWP",
            "magnetostratigraphy",
            "paleolatitude",
        ],
        landmarks=[
            "Tauxe",
            "Kirschvink",
            "Butler",
            "McElhinny",
            "McFadden",
            "Opdyke",
        ],
        must_have=[
            "paleomagnetic",
            "remanent",
            "magnetization",
            "paleolatitude",
        ],
        exclude=[
            "rock magnetism only",
            "magnetic properties",
        ],
        report_sections=[
            "Natural remanent magnetization",
            "Demagnetization techniques",
            "Apparent polar wander",
            "Magnetostratigraphy",
            "Plate reconstruction",
        ],
        boost_terms={
            "paleomagnetic": 1.3,
            "remanent": 1.2,
        },
    )
)

# =============================================================================
# 18. Hydrogeology
# =============================================================================
_register(
    TopicTemplate(
        name="hydrogeology",
        display_name="Hydrogeology",
        target_property="flow rate",
        unit="m/s",
        valid_range=(0, 100),
        synonyms=[
            "groundwater",
            "aquifer",
            "hydraulic conductivity",
            "fluid flow",
            "contaminant transport",
            "recharge",
            "well hydraulics",
            "Darcy flow",
        ],
        landmarks=[
            "Freeze",
            "Cherry",
            "Domenico",
            "Bear",
            "Schwartz",
            "Delleur",
            "Fitts",
        ],
        must_have=[
            "groundwater",
            "aquifer",
            "hydrogeolog",
            "hydraulic",
            "flow",
        ],
        exclude=[
            "igneous",
            "metamorphic",
            "mantle",
            "mineral chemistry",
        ],
        report_sections=[
            "Groundwater flow",
            "Aquifer characterization",
            "Contaminant transport",
            "Recharge and discharge",
            "Modeling",
        ],
        boost_terms={
            "groundwater": 1.2,
            "aquifer": 1.2,
        },
    )
)

# =============================================================================
# 19. Geoelectrical
# =============================================================================
_register(
    TopicTemplate(
        name="geoelectrical",
        display_name="Geoelectrical Methods",
        target_property="conductivity",
        unit="S/m",
        valid_range=(0, 1),
        synonyms=[
            "electrical conductivity",
            "magnetotelluric",
            "MT",
            "resistivity",
            "impedance tensor",
            "phase tensor",
            "tipper",
            "EM",
        ],
        landmarks=[
            "Chave",
            "Jones",
            "Simpson",
            "Bahr",
            "Caldwell",
            "Berdichevsky",
        ],
        must_have=[
            "conductivity",
            "resistivity",
            "magnetotelluric",
            "electrical",
        ],
        exclude=[
            "materials science",
            "semiconductor",
        ],
        report_sections=[
            "Magnetotelluric methods",
            "Electrical resistivity",
            "Mantle conductivity",
            "Crustal structure",
            "Mineral exploration",
        ],
        boost_terms={
            "magnetotelluric": 1.3,
            "MT": 1.2,
        },
    )
)

# =============================================================================
# 20. Potential fields (gravity + magnetic)
# =============================================================================
_register(
    TopicTemplate(
        name="potential_fields",
        display_name="Gravity and Magnetic Surveys",
        target_property="anomaly",
        unit="mGal",
        valid_range=(-500, 500),
        synonyms=[
            "gravity survey",
            "gravity anomaly",
            "Bouguer",
            "free-air",
            "aeromagnetic",
            "magnetic anomaly",
            "gravity gradient",
            "potential field",
        ],
        landmarks=[
            "Blakely",
            "Gunn",
            "Hinze",
            "Nabighian",
            "Hansen",
        ],
        must_have=[
            "gravity",
            "magnetic anomaly",
            "Bouguer",
            "aeromagnetic",
        ],
        exclude=[
            "paleomagnetic",
            "rock magnetism",
        ],
        report_sections=[
            "Gravity methods",
            "Magnetic methods",
            "Data processing",
            "Crustal structure",
            "Mineral exploration",
        ],
        boost_terms={
            "gravity": 1.2,
            "aeromagnetic": 1.3,
        },
    )
)
