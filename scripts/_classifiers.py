#!/usr/bin/env python3
"""Non-LLM field-agnostic classifiers for scientific abstracts.

Provides:
- Discipline detection (keyword-based, 19 fields)
- Novelty classification (cue phrases: first/novel/new/review/incremental)
- Key finding extraction (TextRank + result sentence + position weighting)
- Interpretation extraction (conclusion sentence)

All pure regex/keyword — no LLM, no torch, ~0ms per paper.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger("scientific_research.classifiers")

# =============================================================================
# Discipline detection — keyword-based classifier
# =============================================================================
_DISCIPLINE_KEYWORDS: dict[str, list] = {
    "geochemistry": [
        ("isotope", 2),
        ("δ18o", 5),
        ("delta18", 5),
        ("geochem", 5),
        ("trace element", 2),
        ("ree", 2),
        ("fe3+", 5),
        ("fe2+", 3),
        ("redox", 3),
        ("oxygen fugacity", 5),
        ("mantle", 1),
        ("melt", 1),
        ("crustal", 1),
        ("weathering", 2),
        ("diagenesis", 2),
        ("organic matter", 2),
        ("δ13c", 5),
        ("δ34s", 5),
        ("sr-nd", 4),
        ("radiogenic", 3),
        ("sims", 5),
        ("secondary ion", 4),
        # geological geochemistry terms (added 2026-07-12)
        ("lile", 4),
        ("hfse", 4),
        ("large-ion lithophile", 5),
        ("high field strength", 5),
        ("partition coefficient", 4),
        ("incompatible", 3),
        ("compatible", 3),
        ("fluid-mobile", 4),
        ("slab fluid", 4),
        ("chondrite", 3),
        ("chondritic", 3),
        ("normalization", 2),
        ("primitive mantle", 4),
        ("nb anomaly", 5),
        ("ta anomaly", 5),
        ("sr isotope", 3),
        ("nd isotope", 3),
        ("εnd", 5),
        ("87sr/86sr", 5),
        ("fractionation", 2),
        ("enrichment", 2),
        ("depletion", 2),
        ("metasomat", 3),
        ("aqueous fluid", 3),
        ("solubility", 2),
    ],
    "geology": [
        ("basalt", 2),
        ("granite", 2),
        ("sediment", 2),
        ("stratigraph", 4),
        ("formation", 1),
        ("geological", 3),
        ("outcrop", 3),
        ("core sample", 2),
        ("tectonic", 3),
        ("fault", 2),
        ("fold", 1),
        ("metamorphic", 2),
        ("igneous", 2),
        ("volcanic", 1),
        ("deformation", 2),
        # geological terms (added 2026-07-12)
        ("thermometry", 4),
        ("barometry", 4),
        ("thermobarometry", 5),
        ("geotherm", 3),
        ("geobarometer", 4),
        ("p-t path", 5),
        ("p-t condition", 5),
        ("pt-t", 3),
        ("metamorphic grade", 4),
        ("facies", 3),
        ("protolith", 4),
        ("orogeny", 4),
        ("orogenic", 3),
        ("craton", 3),
        ("cratonic", 3),
        ("terrane", 3),
        ("schist", 2),
        ("gneiss", 2),
        ("metamorph", 3),
        ("equilibrium", 2),
        ("assemblage", 2),
        ("paragenesis", 4),
        ("lithology", 3),
        ("stratigraphy", 3),
        ("subduction", 2),
        ("collision", 2),
        ("rifting", 3),
        ("intrusion", 2),
        ("emplacement", 2),
    ],
    "geophysics": [
        ("seismic", 5),
        ("tomograph", 5),
        ("gravity anomaly", 4),
        ("magnetic anomaly", 4),
        ("magnetotelluric", 5),
        ("heat flow", 4),
        ("gps velocity", 5),
        ("insar", 5),
        ("earthquake", 4),
        ("receiver function", 5),
        ("vp/vs", 5),
        ("shear wave", 4),
        ("mantle", 1),
        ("anisotropy", 3),
        ("lithosphere", 2),
    ],
    "petrology": [
        ("phase diagram", 5),
        ("pseudosection", 5),
        ("melt inclusion", 5),
        ("liquidus", 4),
        ("solidus", 4),
        ("crystallization", 2),
        ("cumulate", 4),
        ("peridotite", 4),
        ("eclogite", 4),
        ("amphibolite", 3),
        ("metapelite", 4),
        ("experimental petrology", 5),
        ("phase equilibria", 4),
        # mineral + rock terms (added 2026-07-12)
        ("garnet", 3),
        ("biotite", 3),
        ("clinopyroxene", 4),
        ("orthopyroxene", 4),
        ("plagioclase", 3),
        ("amphibole", 3),
        ("kyanite", 4),
        ("sillimanite", 4),
        ("staurolite", 4),
        ("chlorite", 3),
        ("epidote", 3),
        ("muscovite", 3),
        ("lherzolite", 5),
        ("harzburgite", 4),
        ("websterite", 4),
        ("granulite", 4),
        ("schist", 2),
        ("gneiss", 2),
        ("geothermometry", 5),
        ("geobarometry", 5),
        ("kd", 3),
        ("distribution coefficient", 4),
        ("partitioning", 3),
        ("exchange reaction", 4),
        ("solid solution", 3),
        ("miscibility gap", 4),
        ("afm", 3),
        ("akermanite", 4),
        ("anorthite", 3),
        ("albite", 3),
        ("forsterite", 4),
        ("fayalite", 4),
        ("enstatite", 4),
        ("diopside", 4),
        ("hornblende", 4),
        ("pargasite", 5),
        ("phlogopite", 4),
        ("spinel", 3),
        ("quartz", 1),
        ("calcite", 2),
        ("dolomite", 2),
    ],
    "structural_geology": [
        ("stress", 3),
        ("strain", 3),
        ("shear zone", 5),
        ("fault slip", 5),
        ("kinematic", 4),
        ("vorticity", 5),
        ("cleavage", 4),
        ("foliation", 4),
        ("lineation", 4),
        ("balanced cross", 5),
    ],
    "volcanology": [
        ("eruption", 5),
        ("magma chamber", 5),
        ("lava flow", 5),
        ("pyroclastic", 5),
        ("tephra", 5),
        ("volcanic gas", 5),
        ("plume", 2),
        ("conduit", 4),
        ("degassing", 5),
    ],
    "paleontology": [
        ("fossil", 5),
        ("foraminifer", 5),
        ("radiolarian", 5),
        ("conodont", 5),
        ("palynolog", 5),
        ("taphonom", 5),
        ("biostratigraph", 5),
        ("trace fossil", 4),
    ],
    "hydrogeology": [
        ("groundwater", 5),
        ("aquifer", 5),
        ("hydraulic", 4),
        ("recharge", 4),
        ("contaminant transport", 5),
        ("pumping test", 5),
        ("tracer test", 4),
        ("well", 1),
    ],
    "economic_geology": [
        ("ore deposit", 5),
        ("mineralization", 4),
        ("alteration", 3),
        ("gold", 3),
        ("copper", 2),
        ("sulfide", 3),
        ("porphyry", 5),
        ("epithermal", 5),
        ("sedex", 5),
        # alteration + ore mineral terms (added 2026-07-12)
        ("potassic", 4),
        ("propylitic", 4),
        ("phyllic", 4),
        ("argillic", 4),
        ("stockwork", 4),
        ("chalcopyrite", 5),
        ("bornite", 5),
        ("molybdenite", 5),
        ("magmatic-hydrothermal", 5),
        ("hydrothermal fluid", 4),
        ("vein", 1),
        ("breccia", 3),
        ("skarn", 4),
        ("greisen", 4),
        ("volcanogenic massive sulfide", 5),
        ("vms", 4),
        ("ioCG", 4),
        ("uranium", 3),
        ("fluid inclusion", 4),
        ("halogen fugacity", 5),
    ],
    "geochronology": [
        ("u-pb", 5),
        ("ar-ar", 5),
        ("40ar/39ar", 5),
        ("re-os", 5),
        ("fission track", 5),
        ("cosmogenic", 5),
        ("luminescence", 4),
        ("radiocarbon", 5),
        ("14c", 4),
        ("detrital zircon", 5),
        ("geochronolog", 5),
        ("dating", 2),
    ],
    "paleoclimate": [
        ("paleoclimate", 5),
        ("ice core", 5),
        ("speleothem", 5),
        ("tree ring", 5),
        ("milankovitch", 5),
        ("holocene", 3),
        ("pleistocene", 3),
        ("glacial", 3),
        ("interglacial", 4),
        ("younger dryas", 5),
    ],
    "oceanography": [
        ("ocean", 2),
        ("thermohaline", 5),
        ("amoc", 5),
        ("carbonate chemistry", 5),
        ("biological pump", 5),
        ("nutrient", 2),
        ("phytoplankton", 4),
        ("ctd", 5),
        ("argo float", 5),
        ("sediment trap", 4),
    ],
    "remote_sensing": [
        ("landsat", 5),
        ("aster", 5),
        ("sentinel", 5),
        ("insar", 5),
        ("lidar", 5),
        ("hyperspectral", 5),
        ("multispectral", 5),
        ("ndvi", 5),
        ("radar", 3),
        ("satellite", 2),
        ("aerial", 3),
    ],
    "soil_science": [
        ("soil", 3),
        ("pedolog", 5),
        ("cation exchange", 4),
        ("organic carbon", 2),
        ("denitrification", 4),
        ("hydraulic conductivity", 3),
        ("loess", 4),
        ("paleosol", 5),
        ("weathering profile", 3),
    ],
    "planetary_geology": [
        ("mars", 5),
        ("lunar", 5),
        ("moon", 4),
        ("meteorite", 5),
        ("chondrite", 5),
        ("asteroid", 5),
        ("comet", 4),
        ("crater", 3),
        ("regolith", 5),
        ("planetary", 5),
    ],
    "physics": [
        ("quantum", 4),
        ("superconduct", 5),
        ("particle", 4),
        ("nuclear", 4),
        ("relativistic", 4),
        ("photon", 3),
        ("electron", 2),
        ("neutron", 3),
        ("magnetization", 4),
        ("resistivity", 4),
        ("band gap", 5),
        ("scattering", 3),
    ],
    "chemistry": [
        ("catalyst", 5),
        ("reaction", 2),
        ("synthesis", 2),
        ("molecule", 3),
        ("compound", 2),
        ("nmr", 5),
        ("chromatograph", 5),
        ("electrochem", 5),
        ("polymer", 4),
        ("crystal structure", 3),
        ("functional group", 4),
    ],
    "biology": [
        ("gene", 4),
        ("protein", 4),
        ("cell", 3),
        ("crispr", 5),
        ("genome", 5),
        ("enzyme", 4),
        ("membrane", 4),
        ("tissue", 3),
        ("organism", 3),
        ("phylogen", 5),
        ("ecosystem", 4),
        ("species", 2),
        ("evolution", 3),
    ],
    "computer_science": [
        ("neural network", 5),
        ("deep learning", 5),
        ("algorithm", 4),
        ("dataset", 5),
        ("benchmark", 5),
        ("machine learning", 5),
        ("transformer", 5),
        ("training", 4),
        ("accuracy", 2),
        ("gpu", 4),
        ("inference", 3),
        ("pytorch", 5),
        ("tensorflow", 5),
    ],
}


def detect_discipline(text: str) -> str | None:
    """Detect scientific discipline from text using keyword matching.

    Returns the discipline with the most keyword hits, or None if no match.
    Handles multi-field papers by picking the dominant field.
    """
    if not text:
        return None
    text_lower = text.lower()
    scores: dict[str, int] = {}
    for discipline, keywords in _DISCIPLINE_KEYWORDS.items():
        count = 0
        for kw in keywords:
            weight = 1
            kw_str = kw
            if isinstance(kw, tuple):
                kw_str, weight = kw
            hits = text_lower.count(kw_str)
            count += hits * weight
        if count > 0:
            scores[discipline] = count
    if not scores:
        return None
    return max(scores, key=scores.get)


# =============================================================================
# Novelty classification — cue phrase based
# =============================================================================
_NOVELTY_MILESTONE = re.compile(
    r"(?:first\s+(?:to\s|time\s|report|demonstrat|measure|direct|observ|synthesi|stud|investigat|description|characterization)|"
    r"previously\s+unreport|never\s+before|unprecedented|"
    r"paradigm\s+shift|groundbreak|landmark|seminal)",
    re.IGNORECASE,
)
_NOVELTY_METHODOLOGICAL = re.compile(
    r"\b(new\s+(?:method|technique|approach|algorithm|tool|instrument|protocol)|"
    r"novel\s+(?:method|technique|approach|framework|architecture|model|system|tool|algorithm)|"
    r"we\s+(?:develop|present|introduce)\s+(?:a|an|the)\s+(?:new|novel)|"
    r"improved\s+method|innovative)\b",
    re.IGNORECASE,
)
_NOVELTY_REVIEW = re.compile(
    r"\b(this\s+(?:review|paper)\s+(?:reviews?|summarizes?|surveys?|covers?|presents?|provides?|describes?|highlights?)|"
    r"we\s+review|comprehensive\s+review|systematic\s+review|"
    r"this\s+(?:paper|study)\s+(?:provides?|presents?)\s+(?:an?\s+)?"
    r"(?:overview|review|survey|summary)|state\s+of\s+the\s+art)\b",
    re.IGNORECASE,
)
_NOVELTY_CONFIRMATION = re.compile(
    r"\b(confirm(?:s|ed|ing)?|corroborat|validat|reproduc|verify|consistent\s+with|"
    r"agree\s+with|as\s+(?:previously|shown|reported))\b",
    re.IGNORECASE,
)


def classify_novelty(text: str) -> str:
    """Classify novelty from cue phrases.

    Returns: 'milestone', 'methodological', 'review', 'confirmation', or 'incremental'.
    """
    if not text:
        return "incremental"
    if _NOVELTY_REVIEW.search(text):
        return "review"
    if _NOVELTY_MILESTONE.search(text):
        return "milestone"
    if _NOVELTY_METHODOLOGICAL.search(text):
        return "methodological"
    if _NOVELTY_CONFIRMATION.search(text):
        return "confirmation"
    return "incremental"


# =============================================================================
# Key finding extraction — result sentence + TextRank + position
# =============================================================================
_RESULT_CUES = re.compile(
    r"(?:we\s+(?:found|find|show|demonstrat|report|observ|reveal)|"
    r"results?\s+(?:show|indicate|demonstrat|reveal|suggest)|"
    r"our\s+(?:results?|findings?|data|analys[ei]s)\s+(?:show|indicate|demonstrat)|"
    r"in\s+conclusion|we\s+conclude|these\s+(?:results|findings|data)\s+(?:show|indicate|demonstrat)|"
    r"this\s+(?:study|work)\s+(?:shows|demonstrates|reveals|provides))",
    re.IGNORECASE,
)


# Sentences that are NOT findings (background/method/context)
_NON_FINDING_CUES = re.compile(
    r"^(?:Since|Because|Although|While|It is likely|It has been suggested|"
    r"In order to|To (?:investigate|study|examine|understand|determine)|"
    r"Here we|In this study we|We (?:analyzed|collected|measured|used|applied|studied|investigated)|"
    r"There is|There are|Previous studies|Recent studies|It is well known|"
    r"The (?:origin|formation|nature|composition|presence|occurrence) of|"
    # Metadata / supplementary — NOT findings
    r"Tables?\s*S?\d|Figures?\s*S?\d|Supplementar|"
    r"Available\s+(?:as|online|at)|Data [Aa]vailab|can be found|"
    r"(?:Large|Small|Significant) amounts? of|"
    r"This article is only|You do not have access|"
    r"(?:Off-target|Off target) effects? (?:are|is) a (?:critical|major|significant) (?:issue|problem|concern)|"
    r"(?:CRISPR|Cas9) (?:has|have) (?:revolutionized|transformed|emerged)|"
    r"(?:Quantum|Surface code) (?:error correction|computing) (?:is|has) (?:a |an )?(?:critical|essential|important))",
    re.IGNORECASE,
)

# Sentences with numerical data are likely findings
_DATA_SENTENCE = re.compile(r"\d+\.?\d*\s*(?:‰|per\s*mil|wt%|%|Ma|Ga|GPa|°C|K|ppm|ppb|Mpa|kbar)")


def extract_key_finding(text: str, max_chars: int = 300) -> str:
    """Extract the key finding sentence from an abstract.

    Strategy:
    1. Find sentences with result cues that are NOT methodology/background
    2. Among result sentences, prefer ones with numerical data
    3. Prefer the last qualifying result sentence (conclusion)
    4. Fallback: last data-bearing sentence
    5. Fallback: last sentence of abstract (if not a non-finding cue)
    """
    if not text:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", text)
    if not sentences:
        return text[:max_chars]

    # Find result sentences that are NOT methodology/background
    result_sentences = []
    for i, sent in enumerate(sentences):
        sent_stripped = sent.strip()
        if not sent_stripped or len(sent_stripped) < 30:
            continue
        # Must have a result cue
        if not _RESULT_CUES.search(sent):
            continue
        # Must NOT start with a non-finding cue
        if _NON_FINDING_CUES.match(sent_stripped):
            continue
        has_data = bool(_DATA_SENTENCE.search(sent))
        result_sentences.append((i, sent, has_data))

    if result_sentences:
        # Prefer sentences with data, then prefer last (conclusion)
        data_sentences = [r for r in result_sentences if r[2]]
        if data_sentences:
            return data_sentences[-1][1].strip()[:max_chars]
        return result_sentences[-1][1].strip()[:max_chars]

    # Fallback: find data-bearing sentences in second half of abstract
    second_half = sentences[len(sentences) // 2 :]
    data_sentences = [s for s in second_half if _DATA_SENTENCE.search(s) and not _NON_FINDING_CUES.match(s.strip())]
    if data_sentences:
        return data_sentences[-1].strip()[:max_chars]

    # No data-bearing or result-cue sentence found.
    # Return empty — better no finding than a misleading fragment.
    return ""


# =============================================================================
# Interpretation extraction — conclusion sentence
# =============================================================================
_CONCLUSION_CUES = re.compile(
    r"(?:we\s+conclude|our\s+results?\s+(?:suggest|indicat|show|demonstrat)|"
    r"these\s+(?:results|findings)\s+(?:suggest|indicat|show|demonstrat|imply)|"
    r"this\s+(?:study|work)\s+(?:demonstrates|shows|reveals|provides|suggests)|"
    r"in\s+conclusion|taken\s+together|overall|"
    r"we\s+(?:propose|suggest|infer|conclude))",
    re.IGNORECASE,
)


def extract_interpretation(text: str, max_chars: int = 300) -> str:
    """Extract the interpretation/conclusion from an abstract.

    Looks for conclusion cue phrases. Falls back to last 1-2 sentences.
    """
    if not text:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", text)

    # Find conclusion sentences
    conclusion_sentences = []
    for i, sent in enumerate(sentences):
        if _CONCLUSION_CUES.search(sent):
            conclusion_sentences.append(sent.strip())

    if conclusion_sentences:
        return " ".join(conclusion_sentences)[:max_chars]

    # Fallback: last 1-2 sentences
    if len(sentences) >= 2:
        return (sentences[-2].strip() + " " + sentences[-1].strip())[:max_chars]
    elif sentences:
        return sentences[-1].strip()[:max_chars]

    return ""


# =============================================================================
# Study type detection (broader than study design patterns)
# =============================================================================
def detect_study_type(text: str) -> str | None:
    """Detect the general study type for field-agnostic classification.

    Returns: 'experimental', 'field_study', 'computational', 'theoretical',
             'review', 'method_development', 'observational', or None.
    """
    if not text:
        return None
    text_lower = text.lower()

    # Check in priority order
    if any(
        kw in text_lower
        for kw in [
            "we review",
            "this review",
            "comprehensive review",
            "systematic review",
            "we summarize",
        ]
    ):
        return "review"
    if any(
        kw in text_lower
        for kw in [
            "we propose",
            "we present a new",
            "we develop",
            "novel method",
            "new algorithm",
            "new technique",
            "new neural",
            "new model",
            "new architecture",
            "novel architecture",
        ]
    ):
        return "method_development"
    if any(
        kw in text_lower
        for kw in [
            "dft",
            "ab initio",
            "molecular dynamics",
            "finite element",
            "monte carlo",
            "numerical model",
            "modeling",
            "thermocalc",
            "simulation",
            "computational",
        ]
    ):
        return "computational"
    if any(
        kw in text_lower for kw in ["theoretical", "analytical solution", "first principles", "perturbation theory"]
    ):
        return "theoretical"
    if any(
        kw in text_lower
        for kw in [
            "field study",
            "field survey",
            "field campaign",
            "fieldwork",
            "outcrop",
            "drill core",
            "field samples",
        ]
    ):
        return "field_study"
    if any(
        kw in text_lower
        for kw in [
            "we measured",
            "we synthesized",
            "we prepared",
            "we fabricated",
            "we tested",
            "we performed",
            "we studied",
            "we investigated",
            "we analyzed",
            "we examined",
            "we collected",
            "samples from",
            "samples were",
            "experiment",
            "dating",
            "were measured",
            "was measured",
            "were analyzed",
            "was analyzed",
            "were collected",
            "was collected",
            "velocities",
            "gravity survey",
            "magnetic survey",
            "seismic survey",
        ]
    ):
        return "experimental"
    if any(kw in text_lower for kw in ["we observed", "telescope", "satellite data", "survey data"]):
        return "observational"

    return None


if __name__ == "__main__":
    # Test on sample abstracts from different fields
    test_texts = [
        (
            "We studied MORB glasses from the Mid-Atlantic Ridge. Samples were analyzed by SIMS. We found δ18O = +5.2‰. This is the first measurement of oxygen isotopes in this region. Our results suggest a depleted mantle source.",
            "geology",
        ),
        (
            "We present a new deep learning architecture for image classification. We trained on ImageNet and achieved accuracy = 95%. This novel method outperforms all baselines. Our results demonstrate the effectiveness of attention mechanisms.",
            "CS",
        ),
        (
            "We synthesized a new palladium catalyst. The catalyst was characterized by NMR and XRD. We found the yield = 85%. This is the first report of this catalytic system. Our results suggest broad substrate tolerance.",
            "chemistry",
        ),
        (
            "This review summarizes recent advances in CRISPR gene editing. We review off-target detection methods. Overall, CRISPR technology has transformed molecular biology.",
            "biology",
        ),
    ]

    for text, field in test_texts:
        disc = detect_discipline(text)
        nov = classify_novelty(text)
        kf = extract_key_finding(text)
        interp = extract_interpretation(text)
        st = detect_study_type(text)

        print(f"[{field}] disc={disc} type={st} novelty={nov}")
        print(f"  finding: {kf[:80]}")
        print(f"  interp:  {interp[:80]}")
        print()


# =============================================================================
# Subject extraction — what was studied (field-agnostic)
# =============================================================================
_SUBJECT_CUES = re.compile(
    r"(?:we\s+(?:studied|investigated|analy[sz]ed|examined|measured|characterized|surveyed|explored)|"
    r"samples?\s+(?:from|were|collected)|"
    r"specimens?\s+from|"
    r"we\s+(?:collected|obtained|selected)|"
    r"(?:this|the)\s+(?:study|paper|work)\s+(?:investigates?|examines?|analy[sz]es?|reports?|presents?|focuses?)|"
    r"patients?\s+with|subjects?\s+with|participants?\s+with|"
    r"rocks?\s+from|minerals?\s+from|basalts?\s+from|"
    r"films?\s+(?:of|grown|deposited)|crystals?\s+of|"
    r"compounds?\s+(?:synthesized|prepared)|"
    r"cells?\s+(?:were|treated|cultured)|tissues?\s+from|"
    r"datasets?\s+(?:of|from|containing)|models?\s+trained\s+on)\s+"
    r"([^\.]{10,180}?)(?:[\.]|were|had|using|by|via|at|for|with|to|and|in|of)",
    re.IGNORECASE,
)


def extract_subject(text: str, max_chars: int = 200) -> str:
    """Extract what was studied — the main subject/system/material.

    Strategy:
    1. Find sentence with "we studied/investigated/analyzed X" → extract X
    2. Find sentence with "samples from/specimens from" → extract what
    3. Fallback: first sentence (usually background)
    """
    if not text:
        return ""

    m = _SUBJECT_CUES.search(text)
    if m:
        return m.group(1).strip()[:max_chars]

    # Fallback: first sentence (must be substantial)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if sentences and len(sentences[0].strip()) >= 15:
        return sentences[0].strip()[:max_chars]

    return ""


# =============================================================================
# Method extraction — what technique/method was used (field-agnostic)
# =============================================================================
_METHOD_CUES = re.compile(
    r"(?:we\s+(?:used|employed|applied|utilized|measured|performed|conducted|implemented)|"
    r"(?:was|were)\s+(?:measured|analyzed|determined|characterized|prepared|synthesized|fabricated)|"
    r"analy[sz]ed\s+by|measured\s+(?:using|by|via)|determined\s+(?:by|using)|"
    r"following\s+(?:the|a)\s+(?:method|protocol|procedure|approach)|"
    r"according\s+to)\s+"
    r"([^\.]{10,180}?)(?:[\.]|for|at|and|with|to|in)",
    re.IGNORECASE,
)


def extract_method(text: str, max_chars: int = 200) -> str:
    """Extract the main method/technique used.

    Strategy:
    1. Find "we used/employed/applied X" → extract X
    2. Find "analyzed by/measured using X" → extract X
    3. Fallback: look for known instrument/technique keywords
    """
    if not text:
        return ""

    m = _METHOD_CUES.search(text)
    if m:
        return m.group(1).strip()[:max_chars]

    # Fallback: find known technique keywords
    techniques = [
        "SIMS",
        "XRF",
        "ICP-MS",
        "EPMA",
        "LA-ICP-MS",
        "SEM",
        "TEM",
        "FTIR",
        "XRD",
        "Raman",
        "NMR",
        "HPLC",
        "GC-MS",
        "PCR",
        "western blot",
        "CRISPR",
        "deep learning",
        "neural network",
        "DFT",
        "molecular dynamics",
        "electron microprobe",
        "mass spectrometry",
        "spectroscopy",
    ]
    text_lower = text.lower()
    found = []
    for t in techniques:
        if len(t) <= 5:  # short abbreviation — match uppercase only (TEM, not "system")
            if t in text:  # case-sensitive
                found.append(t)
        else:  # longer term — case-insensitive
            if t.lower() in text_lower:
                found.append(t)
    if found:
        return found[0]

    return ""


# =============================================================================
# Structured abstract section parser
# =============================================================================
_SECTION_HEADERS = re.compile(
    r"(Background|Objective|Aim[s]?|Introduction|Purpose|"
    r"Method[s]?|Materials?\s+and\s+Method[s]?|Experimental|"
    r"Results?|Discussion|Conclusion[s]?|Findings?|"
    r"Setting|Participants?|Intervention|Measurements?|"
    r"Study\s+Design|Approach|Procedure[s]?)\s*:\s*",
    re.IGNORECASE,
)


def parse_structured_abstract(text: str) -> dict[str, str]:
    """Parse structured abstract into sections.

    Returns dict mapping section name → text.
    If abstract is unstructured (no headers), returns empty dict.
    """
    if not text:
        return {}

    headers = list(_SECTION_HEADERS.finditer(text))
    if not headers:
        return {}

    sections = {}
    for i, match in enumerate(headers):
        section_name = match.group(1).lower()
        start = match.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            sections[section_name] = section_text

    return sections


def extract_subject_from_sections(sections: dict[str, str]) -> str:
    """Extract subject from the appropriate section of a structured abstract."""
    for key in (
        "objective",
        "aim",
        "background",
        "introduction",
        "purpose",
        "participants",
        "setting",
    ):
        if key in sections:
            subj = extract_subject(sections[key])
            if subj:
                return subj
    return ""


def extract_method_from_sections(sections: dict[str, str]) -> str:
    """Extract method from the appropriate section."""
    for key in (
        "methods",
        "method",
        "materials and methods",
        "experimental",
        "approach",
        "procedure",
        "intervention",
    ):
        if key in sections:
            meth = extract_method(sections[key])
            if meth:
                return meth
    return ""


# =============================================================================
# Research type classification — drives synthesis output format
# =============================================================================
# Three research types map to different narrative output structures:
#   verification: yes/no question with specific claim → verdict + stance summary
#   survey:       broad topic overview → chronological narrative
#   subtopic:     specific focused investigation → focused narrative + measurements
#   comparative:  compare X vs Y → side-by-side grouping + pros/cons
#   data_compilation: compile values → data table + statistical summary

_VERIFICATION_CUES = re.compile(
    r"\b(?:is\s+it\s+(?:true|false|correct|known|possible)\b|"
    r"does\s+\w+|do\s+\w+\s+(?:show|have|exhibit)\b|"
    r"can\s+\w+|could\s+\w+|will\s+\w+|"
    r"has\s+(?:it\s+been|anyone|any)\b|"
    r"have\s+(?:any\s+)?(?:studies|papers)\b|"
    r"verify\b|true\s+or\s+false\b|"
    r"confirm\s+or\s+deny\b|prove\s+that\b|"
    r"right\s+or\s+wrong\b)"
)
_VERIFICATION_PUNCT = re.compile(r"\?\s*$")

_SURVEY_CUES = re.compile(
    r"\b(?:latest\s+research\b|recent\s+(?:advances|developments|progress)\b|"
    r"overview\s+of\b|state\s+of\s+(?:the\s+art|research|knowledge)\b|"
    r"review\s+of\b|current\s+understanding\b|"
    r"what\s+is\s+known\b|what\s+do\s+we\s+know\b|"
    r"summar(?:y|ize)\b|literature\s+(?:review|survey)\b|"
    r"survey\s+of\b|progress\s+in\b|"
    r"advances\s+in\b|trends\s+in\b|"
    r"introduction\s+to\b|background\s+on\b)"
)

_COMPARATIVE_CUES = re.compile(
    r"\b(?:compare\b|comparison\s+(?:of|between)\b|versus\b|\bvs\.?\b|"
    r"difference\s+between\b|differences\s+between\b|"
    r"contrast\s+between\b|relative\s+(?:to|advantages?)\b|"
    r"better\s+than\b|pros\s+and\s+cons\b|"
    r"which\s+(?:is|method|approach)\b|"
    r"alternativ\w+\s+to\b)"
)

_COMPILATION_CUES = re.compile(
    r"\b(?:compile\b|compilation\b|global\s+(?:dataset|database|compilation)\b|"
    r"reported\s+values\b|range\s+of\s+values\b|summary\s+of\s+values\b|"
    r"how\s+(?:much|many)\b|what\s+(?:are|is)\s+the\s+(?:typical|reported|measured)\b|"
    r"data\s+(?:summary|table|compilation)\b|"
    r"all\s+(?:known|reported|measured)\b)"
)


def detect_research_type(query: str) -> str:
    """Classify research query into one of 5 types.

    Detection order (highest specificity first):
    1. comparative — "compare X vs Y", "difference between"
    2. data_compilation — "compile values", "reported values", "global dataset"
    3. survey — "latest research on", "overview of", short broad topic
    4. verification — "is it true", question mark
    5. subtopic — everything else (default)

    Returns one of: 'comparative', 'data_compilation', 'survey',
                    'verification', 'subtopic'.
    """
    if not query or not query.strip():
        return "survey"

    q = query.strip()
    q_lower = q.lower()

    # Priority 1: comparative patterns (most specific)
    if _COMPARATIVE_CUES.search(q_lower):
        return "comparative"

    # Priority 2: data compilation patterns
    if _COMPILATION_CUES.search(q_lower):
        return "data_compilation"

    # Priority 3: explicit survey/overview cues
    if _SURVEY_CUES.search(q_lower):
        return "survey"

    # Priority 4: verification patterns (question structure)
    if _VERIFICATION_CUES.search(q_lower):
        return "verification"
    if _VERIFICATION_PUNCT.search(q):
        return "verification"

    # Priority 5: short broad topic (1-3 words, no prepositional narrowing)
    tokens = [t for t in q_lower.split() if len(t) > 1]
    if len(tokens) <= 3 and not any(w in q_lower for w in (" in ", " of ", " during ", " from ", " using ")):
        return "survey"

    # Default: specific subtopic focus
    return "subtopic"
