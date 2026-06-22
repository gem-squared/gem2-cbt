"""Generate synthetic CBT datasets: concept / context / task boundaries.

Each example is self-contained:
    { "level", "contract", "text", "label" (1=compatible,0=incompatible), "meta" }
Labels follow papers/notes.md section 4.

Run (v1):  python scripts/generate_synthetic_data.py --n_per_level 600 --out data/processed
Run (v2):  python scripts/generate_synthetic_data.py --version v2 --n_per_level 3000 --out data/processed_v2
"""
import argparse, json, os, random
from collections import defaultdict

# ----------------------------------------------------------------------------
# CONCEPT: same glyph, contract (domain|unit) fixes the admissible concept.
# ----------------------------------------------------------------------------
# glyph -> domain -> (admissible_concept, unit)
CONCEPT_TABLE = {
    "rho":   {"physics": ("mass_density", "kg/m^3"), "EE": ("resistivity", "ohm*m")},
    "sigma": {"statistics": ("standard_deviation", "unitless"),
              "EE": ("conductivity", "S/m"), "mechanics": ("stress", "Pa")},
    "mu":    {"statistics": ("mean", "unitless"),
              "physics": ("friction_coefficient", "unitless"),
              "EM": ("permeability", "H/m")},
    "lambda":{"optics": ("wavelength", "m"), "statistics": ("rate", "1/s"),
              "linalg": ("eigenvalue", "unitless")},
    "k":     {"thermal": ("thermal_conductivity", "W/(m*K)"),
              "mechanics": ("spring_constant", "N/m"),
              "chemistry": ("rate_constant", "1/s")},
    "T":     {"thermodynamics": ("temperature", "K"),
              "mechanics": ("period", "s"), "linalg": ("transformation", "unitless")},
}
CONCEPT_TEMPLATES = [
    "In {domain}, {glyph} means {concept}.",
    "In {domain}, {glyph} denotes {concept}.",
    "Within {domain}, the symbol {glyph} refers to {concept}.",
    "Under {domain} conventions, {glyph} is the {concept}.",
    "{glyph} (in {domain}) represents {concept}.",
]
# neutral fillers that do NOT change the label, used to expand surface variety
CONCEPT_PREFIX = ["", "Note: ", "Recall that ", "By convention, ", "As stated, ", "Here, "]
CONCEPT_SUFFIX = ["", " (standard usage)", " in most textbooks", " as commonly written",
                  " per the convention", " in this section"]

def gen_concept(n, rng):
    rows = []
    glyphs = list(CONCEPT_TABLE.keys())
    # pool of all concepts to pick plausible-but-wrong ones
    all_concepts = sorted({c for d in CONCEPT_TABLE.values() for (c, _u) in d.values()})
    for _ in range(n):
        glyph = rng.choice(glyphs)
        domains = CONCEPT_TABLE[glyph]
        domain = rng.choice(list(domains.keys()))
        admissible, unit = domains[domain]
        contract = f"{domain}|{unit}"
        tmpl = rng.choice(CONCEPT_TEMPLATES)
        if rng.random() < 0.5:  # compatible
            concept, label = admissible, 1
        else:                   # incompatible: a different valid concept of SOME domain
            wrong = rng.choice([c for c in all_concepts if c != admissible])
            concept, label = wrong, 0
        core = tmpl.format(domain=domain, glyph=glyph, concept=concept)
        pre, suf = rng.choice(CONCEPT_PREFIX), rng.choice(CONCEPT_SUFFIX)
        if suf:  # splice suffix before the trailing period
            core = core[:-1] + suf + "."
        text = pre + core
        rows.append({"level": "concept", "contract": contract, "text": text,
                     "label": label,
                     "meta": {"glyph": glyph, "domain": domain, "concept": concept}})
    return rows

# ----------------------------------------------------------------------------
# CONTEXT: role/argument structure. contract = "role-preserve".
# ----------------------------------------------------------------------------
AGENTS = ["a person", "a child", "a dog", "a worker", "a robot", "a woman", "a boy"]
PATIENTS = ["an umbrella", "a book", "a cup", "a phone", "a rope", "a basket", "a kite"]
RELATIONS = [("holding", "holds"), ("carrying", "carries"),
             ("pushing", "pushes"), ("lifting", "lifts")]

def _np_swap_article(np_phrase):
    return np_phrase  # keep simple

def gen_context(n, rng):
    rows = []
    for _ in range(n):
        a = rng.choice(AGENTS); p = rng.choice(PATIENTS)
        ger, third = rng.choice(RELATIONS)
        base = f"{a} {ger} {p}".capitalize()
        contract = "role-preserve"
        if rng.random() < 0.5:  # compatible paraphrase (roles preserved)
            variant = rng.choice([
                f"{a} who is {ger} {p}".capitalize(),
                f"{a} that {third} {p}".capitalize(),
                f"{p} held by {a}".capitalize() if ger == "holding"
                    else f"{a} {ger} {p}".capitalize(),
            ])
            label = 1
        else:  # incompatible: swap agent/patient
            variant = rng.choice([
                f"{p} {ger} {a}".capitalize(),
                f"{p} that {third} {a}".capitalize(),
            ])
            label = 0
        text = f"Base: {base}. Test: {variant}."
        rows.append({"level": "context", "contract": contract, "text": text,
                     "label": label, "meta": {"agent": a, "patient": p, "rel": ger}})
    return rows

# ----------------------------------------------------------------------------
# TASK: output must use only provided facts. contract = "facts-only".
# ----------------------------------------------------------------------------
NAMES = ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace"]
CITIES = ["Seoul", "Tokyo", "Paris", "Berlin", "Boston", "Lima", "Cairo"]
YEARS = [2015, 2017, 2018, 2019, 2020, 2021, 2022]
EXTRAS = ["and became a famous researcher", "and won a national award",
          "after starting a successful company", "and later moved to Mars",
          "where she met a celebrity", "and published three books"]

def gen_task(n, rng):
    rows = []
    task = "Summarize the source in one sentence using ONLY provided facts."
    contract = "facts-only"
    for _ in range(n):
        name = rng.choice(NAMES); city = rng.choice(CITIES); year = rng.choice(YEARS)
        source = f"{name} moved to {city} in {year}."
        if rng.random() < 0.5:  # compatible: faithful
            output = rng.choice([
                f"{name} moved to {city} in {year}.",
                f"In {year}, {name} moved to {city}.",
                f"{name} relocated to {city} in {year}.",
            ])
            label = 1
        else:  # incompatible: adds unlicensed fact
            extra = rng.choice(EXTRAS)
            output = f"{name} moved to {city} in {year} {extra}."
            label = 0
        text = f"Task: {task} Source: {source} Output: {output}"
        rows.append({"level": "task", "contract": contract, "text": text,
                     "label": label, "meta": {"name": name, "city": city, "year": year}})
    return rows

# ============================================================================
# V2 GENERATORS — counterfactual, load-bearing, cue-free, family-split
# ============================================================================

# --------------------------------------------------------------------------
# V2 CONCEPT: qualitative descriptors; label set by contract, not text.
# Family = (symbol, qualifier_id). Same text appears under HIGH and LOW
# contracts with OPPOSITE labels → contract must carry the information.
# --------------------------------------------------------------------------
V2_SYMBOLS = ["ρ", "μ", "σ", "κ", "α", "η", "ε", "χ", "β", "λ", "φ", "τ"]

# Qualifier groups: HIGH-group compatible with high-valued contracts,
# LOW-group compatible with low-valued contracts.
V2_QUALIFIERS_HIGH = [
    "very high", "high", "notably high", "well above the baseline",
    "above the reference threshold", "substantially elevated",
    "toward the upper end of the range", "significantly above average",
]
V2_QUALIFIERS_LOW = [
    "very low", "low", "notably low", "well below the baseline",
    "below the reference threshold", "substantially reduced",
    "toward the lower end of the range", "significantly below average",
]
V2_QUALIFIERS_MID = [
    "moderate", "near the baseline", "within normal bounds",
    "at an intermediate level", "close to the reference value",
    "neither particularly high nor low",
]

# Two contract poles per symbol (high-valued / low-valued)
V2_CONTRACT_HIGH = [
    "physics/high-density", "mechanics/high-stiffness", "optics/high-refractive-index",
    "acoustics/high-impedance", "chemistry/high-concentration",
    "thermal/high-conductivity", "EM/high-permeability",
    "materials/high-hardness", "fluid/high-viscosity",
    "nuclear/high-cross-section", "signal/high-amplitude", "optics/high-reflectance",
]
V2_CONTRACT_LOW = [
    "electrical/low-resistivity", "mechanics/low-friction",
    "optics/low-absorption", "chemistry/low-solubility",
    "thermal/low-diffusivity", "EM/low-permittivity",
    "materials/low-porosity", "fluid/low-turbulence",
    "nuclear/low-attenuation", "signal/low-noise", "optics/low-scattering",
    "mechanics/low-damping",
]

V2_CONCEPT_TEMPLATES = [
    "In this system, {sym} is {qual}.",
    "The measured value of {sym} is {qual}.",
    "Observation: {sym} registers as {qual}.",
    "{sym} reads {qual} under these conditions.",
    "The parameter {sym} is found to be {qual}.",
    "Here, {sym} is {qual}.",
    "The quantity {sym} is {qual} in this context.",
    "Measurement indicates {sym} is {qual}.",
    "{sym} is characterized as {qual}.",
    "The recorded {sym} is {qual}.",
    "Analysis shows {sym} is {qual}.",
    "Results confirm {sym} is {qual}.",
]


def gen_concept_v2(n_per_level, rng):
    """Generate concept counterfactual pairs.
    For each (symbol, qualifier) family, the SAME text is compatible under a
    high-valued contract and incompatible under a low-valued contract (or vice
    versa). The domain word is absent from the text — only the contract carries it.
    Family key = (symbol, qualifier_group, qualifier_text).
    """
    rows = []
    # Build symbol→(high_contract, low_contract) assignments deterministically
    sym_contracts = {}
    for i, sym in enumerate(V2_SYMBOLS):
        sym_contracts[sym] = (
            V2_CONTRACT_HIGH[i % len(V2_CONTRACT_HIGH)],
            V2_CONTRACT_LOW[i % len(V2_CONTRACT_LOW)],
        )

    # Enumerate all (symbol, qualifier, template) combinations
    all_families = []
    for sym in V2_SYMBOLS:
        high_c, low_c = sym_contracts[sym]
        for qgroup, compat_high, compat_low in [
            ("HIGH", True, False),   # HIGH qualifier: compat under high-valued, incompat under low
            ("LOW",  False, True),   # LOW qualifier: incompat under high-valued, compat under low
            ("MID",  True, False),   # MID qualifier: compat under high-valued, incompat under low-valued
        ]:
            qlist = (V2_QUALIFIERS_HIGH if qgroup == "HIGH"
                     else V2_QUALIFIERS_LOW if qgroup == "LOW"
                     else V2_QUALIFIERS_MID)
            for qual in qlist:
                for tmpl in V2_CONCEPT_TEMPLATES:
                    text = tmpl.format(sym=sym, qual=qual)
                    family_key = f"{sym}|{qgroup}|{qual}"
                    # Two rows: one per contract pole
                    rows.append({"level": "concept", "contract": high_c,
                                 "text": text, "label": 1 if compat_high else 0,
                                 "family": family_key,
                                 "meta": {"sym": sym, "qual": qual, "qgroup": qgroup,
                                          "contract_pole": "high"}})
                    rows.append({"level": "concept", "contract": low_c,
                                 "text": text, "label": 1 if compat_low else 0,
                                 "family": family_key,
                                 "meta": {"sym": sym, "qual": qual, "qgroup": qgroup,
                                          "contract_pole": "low"}})
    rng.shuffle(rows)
    # Trim or cycle to hit n_per_level
    if len(rows) >= n_per_level:
        return rows[:n_per_level]
    # If short, cycle with shuffled re-picks
    extra = []
    while len(rows) + len(extra) < n_per_level:
        extra.extend(rng.sample(rows, min(len(rows), n_per_level - len(rows) - len(extra))))
    return (rows + extra)[:n_per_level]


# --------------------------------------------------------------------------
# V2 CONTEXT: 5 contracts, each with a distinct transformation type.
# Incompatibles violate the specific contract, NOT just role-swap.
# --------------------------------------------------------------------------
V2_CONTEXT_CONTRACTS = [
    "agent-patient-preserve",
    "attribute-preserve",
    "location-preserve",
    "temporal-preserve",
    "negation-scope-preserve",
]

# Templates: (base, compat_variant, incompat_variant)
# Family key = (template_id, contract)
_APR_TRIPLES = [  # agent-patient-preserve  (kept for smoke test reference)
    ("A person held an umbrella.",
     "An umbrella was held by a person.",
     "An umbrella held a person."),
    ("A child carried a book.",
     "A book was carried by a child.",
     "A book carried a child."),
    ("A worker pushed a cart.",
     "A cart was pushed by a worker.",
     "A cart pushed a worker."),
    ("A dog pulled a sled.",
     "A sled was pulled by a dog.",
     "A sled pulled a dog."),
    ("A robot lifted a crate.",
     "A crate was lifted by a robot.",
     "A crate lifted a robot."),
    ("A woman moved a chair.",
     "A chair was moved by a woman.",
     "A chair moved a woman."),
    ("A boy threw a ball.",
     "A ball was thrown by a boy.",
     "A ball threw a boy."),
    ("A guard opened a gate.",
     "A gate was opened by a guard.",
     "A gate opened a guard."),
    ("A teacher held a marker.",
     "A marker was held by a teacher.",
     "A marker held a teacher."),
    ("A driver steered a truck.",
     "A truck was steered by a driver.",
     "A truck steered a driver."),
]
_ATTR_TRIPLES = [  # attribute-preserve
    ("The red car moved quickly.",
     "The fast red vehicle moved quickly.",
     "The blue car moved quickly."),
    ("The tall tree swayed gently.",
     "The large tall tree swayed slowly.",
     "The short tree swayed gently."),
    ("The cold water flowed fast.",
     "The fast-flowing cold water moved quickly.",
     "The warm water flowed fast."),
    ("The heavy stone sank deep.",
     "The deep-sinking heavy stone fell fast.",
     "The light stone sank deep."),
    ("The bright lamp lit the room.",
     "The room was lit by the bright lamp.",
     "The dim lamp lit the room."),
    ("The sharp knife cut the rope.",
     "The rope was cut by the sharp knife.",
     "The blunt knife cut the rope."),
    ("The soft pillow rested on the bed.",
     "The bed held the soft pillow.",
     "The hard pillow rested on the bed."),
    ("The loud alarm rang at dawn.",
     "At dawn the loud alarm rang out.",
     "The quiet alarm rang at dawn."),
]
_LOC_TRIPLES = [  # location-preserve
    ("She ran to the store.",
     "She sprinted toward the shop.",
     "She ran to the park."),
    ("He walked to the office.",
     "He went on foot to the workplace.",
     "He walked to the gym."),
    ("They traveled to the mountains.",
     "They headed toward the highland region.",
     "They traveled to the coast."),
    ("The bird flew to the nest.",
     "The bird headed for its nest.",
     "The bird flew to the tree."),
    ("The package arrived at the warehouse.",
     "The shipment reached the storage facility.",
     "The package arrived at the office."),
    ("A letter was sent to the embassy.",
     "A letter was mailed to the diplomatic building.",
     "A letter was sent to the ministry."),
    ("The train stopped at the station.",
     "The train halted at the terminal.",
     "The train stopped at the depot."),
    ("Guests checked in at the hotel.",
     "Visitors registered at the inn.",
     "Guests checked in at the hostel."),
]
_TEMP_TRIPLES = [  # temporal-preserve
    ("He will arrive tomorrow.",
     "He arrives the following day.",
     "He arrived yesterday."),
    ("She will finish the report soon.",
     "She is about to complete the report.",
     "She finished the report last week."),
    ("They will meet next Monday.",
     "They are scheduled to meet the coming Monday.",
     "They met last Monday."),
    ("The event will start at noon.",
     "The event is set to begin at midday.",
     "The event started this morning."),
    ("Rain will fall overnight.",
     "Overnight rain is expected.",
     "Rain fell overnight last week."),
    ("The system will reboot in an hour.",
     "The system is rebooting within the hour.",
     "The system rebooted an hour ago."),
    ("Orders will ship by Friday.",
     "Shipment is expected before the weekend.",
     "Orders shipped last Friday."),
    ("The bridge will open next spring.",
     "The bridge is set to open in spring.",
     "The bridge opened last spring."),
]
_NEG_TRIPLES = [  # negation-scope-preserve
    ("Not all birds can fly.",
     "Some birds are unable to fly.",
     "All birds cannot fly."),
    ("Not every student passed.",
     "Some students did not pass.",
     "Every student did not pass."),
    ("Not all doors were locked.",
     "Some doors remained unlocked.",
     "All doors were not locked."),
    ("Not every rule applies here.",
     "Some rules do not apply here.",
     "Every rule does not apply here."),
    ("Not all servers responded.",
     "Some servers failed to respond.",
     "All servers did not respond."),
    ("Not every claim was verified.",
     "Some claims were not verified.",
     "Every claim was not verified."),
    ("Not all lights were on.",
     "Some lights were off.",
     "All lights were not on."),
    ("Not every test passed.",
     "Some tests did not pass.",
     "Every test did not pass."),
]

_CONTEXT_TRIPLES = {
    "agent-patient-preserve": _APR_TRIPLES,
    "attribute-preserve": _ATTR_TRIPLES,
    "location-preserve": _LOC_TRIPLES,
    "temporal-preserve": _TEMP_TRIPLES,
    "negation-scope-preserve": _NEG_TRIPLES,
}

# --------------------------------------------------------------------------
# Combinatorial pools for programmatic context generation (scales to ≥3000)
# --------------------------------------------------------------------------
_CTX_AGENTS   = ["a person", "a child", "a dog", "a worker", "a robot",
                  "a woman", "a boy", "a guard", "a teacher", "a driver",
                  "a farmer", "a nurse", "a pilot", "a clerk"]
_CTX_PATIENTS = ["an umbrella", "a book", "a cup", "a phone", "a rope",
                  "a basket", "a kite", "a crate", "a marker", "a cart",
                  "a lamp", "a chair", "a bag", "a box"]
# (gerund, past-participle, 3rd-person)
_CTX_VERBS    = [("holding", "held", "holds"),
                  ("carrying", "carried", "carries"),
                  ("pushing", "pushed", "pushes"),
                  ("lifting", "lifted", "lifts"),
                  ("moving", "moved", "moves"),
                  ("pulling", "pulled", "pulls"),
                  ("dragging", "dragged", "drags"),
                  ("grabbing", "grabbed", "grabs"),
                  ("dropping", "dropped", "drops"),
                  ("placing", "placed", "places")]

_CTX_SUBJECTS  = ["the car", "the stone", "the knife", "the lamp", "the tree",
                   "the river", "the bag", "the tower", "the door", "the ship",
                   "the clock", "the flag"]
# (adj_compat, adj_incompat) — contrasting attribute pairs
_CTX_ATTR_PAIRS = [("red", "blue"), ("tall", "short"), ("cold", "warm"),
                    ("heavy", "light"), ("bright", "dim"), ("sharp", "blunt"),
                    ("soft", "hard"), ("loud", "quiet"), ("fast", "slow"),
                    ("clean", "dirty"), ("old", "new"), ("smooth", "rough")]
_CTX_ATTR_VERBS = [("moved quickly", "moved fast"), ("stood still", "remained still"),
                    ("shone brightly", "glowed brightly"), ("ran fast", "moved rapidly"),
                    ("sank deep", "fell deep"), ("cut cleanly", "sliced neatly"),
                    ("rested there", "stayed there"), ("rang out", "sounded loud")]

_CTX_LOC_SUBJ  = ["she", "he", "they", "the bird", "the package",
                   "the letter", "the train", "the guests", "the car",
                   "the team"]
_CTX_LOC_VERBS = [("ran to", "sprinted toward"),
                   ("walked to", "went on foot to"),
                   ("traveled to", "headed toward"),
                   ("flew to", "headed for"),
                   ("arrived at", "reached"),
                   ("sent to", "mailed to"),
                   ("stopped at", "halted at"),
                   ("checked in at", "registered at")]
# MERGED location pool: compatible and incompatible draw from same set,
# preventing trivial cues from disjoint location vocabularies.
_CTX_LOCS      = ["the store", "the office", "the mountains", "the nest",
                   "the warehouse", "the embassy", "the station", "the hotel",
                   "the library", "the clinic", "the school", "the harbor",
                   "the park", "the gym", "the coast", "the tree",
                   "the headquarters", "the ministry", "the depot", "the hostel",
                   "the cafe", "the arena", "the factory", "the airport"]

# Shared person-name pool for agent-patient-preserve.
# Using the same names as both agents and patients ensures name-bigrams
# appear in BOTH compatible and incompatible across the dataset, preventing
# animate-vs-inanimate or naming-convention cues.
_APR_ACTORS = ["Alice", "Bob", "Carol", "David", "Eve", "Frank",
               "George", "Helen", "Ivan", "Julia", "Karl", "Lena"]
# Verb pairs: (past_tense, past_synonym) — active construction, no passive.
_APR_VERB_PAIRS = [
    ("helped", "assisted"), ("called", "contacted"), ("met", "encountered"),
    ("hired", "employed"), ("taught", "trained"), ("followed", "trailed"),
    ("replaced", "succeeded"), ("praised", "commended"), ("guided", "led"),
    ("supported", "aided"),
]

_CTX_TEMP_SUBJ = ["he", "she", "they", "the event", "the rain",
                   "the system", "the orders", "the bridge",
                   "the team", "the report", "the flight", "the show"]
# (future_phrase, future_syn, past_phrase, past_syn)
# past_syn used for past-base compatible pairs (adds temporal balance).
_CTX_TEMP_ACTS = [
    ("will arrive {t}", "arrives {t_syn}", "arrived {t_past}", "appeared {t_past}"),
    ("will finish {t}", "is about to finish {t_syn}", "finished {t_past}", "completed {t_past}"),
    ("will meet {t}", "is scheduled to meet {t_syn}", "met {t_past}", "gathered {t_past}"),
    ("will start {t}", "is set to begin {t_syn}", "started {t_past}", "commenced {t_past}"),
    ("will fall {t}", "is expected {t_syn}", "fell {t_past}", "dropped {t_past}"),
    ("will reboot {t}", "is rebooting {t_syn}", "rebooted {t_past}", "restarted {t_past}"),
    ("will ship {t}", "is expected before {t_syn}", "shipped {t_past}", "was sent {t_past}"),
    ("will open {t}", "is set to open {t_syn}", "opened {t_past}", "became available {t_past}"),
]
# (future_time, future_syn, past_time, past_syn_time)
_CTX_FUTURE_TIMES = [
    ("tomorrow", "the following day", "yesterday", "the day before"),
    ("soon", "shortly", "last week", "some weeks ago"),
    ("next Monday", "the coming Monday", "last Monday", "that Monday"),
    ("at noon", "at midday", "this morning", "earlier that day"),
    ("overnight", "during the night", "last night", "the night before"),
    ("in an hour", "within the hour", "an hour ago", "earlier"),
    ("by Friday", "before the weekend", "last Friday", "that Friday"),
    ("next spring", "in spring", "last spring", "that spring"),
]

_CTX_NEG_NOUNS = ["birds", "students", "doors", "rules", "servers",
                   "claims", "lights", "tests", "drivers", "files",
                   "sensors", "nodes", "tasks", "users", "packets",
                   "agents", "modules", "signals", "records", "flags",
                   "threads", "queries", "tokens", "entries", "layers"]
_CTX_NEG_PREDS = ["can fly", "passed", "were locked", "apply here", "responded",
                   "were verified", "were on", "completed", "were updated",
                   "succeeded", "were tested", "were active"]


def gen_context_v2(n_per_level, rng):
    """Generate context examples programmatically from word pools.
    Each (agent, patient, verb), (subject, attr, verb), etc. = unique family.
    Target: n_per_level total context rows across 5 contracts (~n_per_level/5 each).
    Family key encodes all distinguishing fields.
    """
    per_contract = max(1, n_per_level // 5)
    rows = []

    # 1. agent-patient-preserve (person names for both roles)
    # Using the same name pool for agents and patients: any name appears as BOTH
    # subject (compatible) and object (incompatible) across different families,
    # breaking any name-bigram ↔ label correlation.
    pool = []
    for ai in range(len(_APR_ACTORS)):
        for pi in range(len(_APR_ACTORS)):
            if ai == pi:
                continue
            for vi, (v1, v2) in enumerate(_APR_VERB_PAIRS):
                agent, patient = _APR_ACTORS[ai], _APR_ACTORS[pi]
                base     = f"{agent} {v1} {patient}."
                compat   = f"{agent} {v2} {patient}."   # synonym verb, same roles
                incompat = f"{patient} {v1} {agent}."   # same verb, swapped roles
                fk = f"apr|{ai}|{pi}|{vi}"
                pool.append((base, compat, incompat, fk))
    rng.shuffle(pool)
    for base, compat, incompat, fk in pool[:per_contract // 2 + 1]:
        rows.append({"level": "context", "contract": "agent-patient-preserve",
                     "text": f"Base: {base} Test: {compat}", "label": 1,
                     "family": fk, "meta": {}})
        rows.append({"level": "context", "contract": "agent-patient-preserve",
                     "text": f"Base: {base} Test: {incompat}", "label": 0,
                     "family": fk, "meta": {}})

    # 2. attribute-preserve (symmetric pairs)
    # Generate BOTH directions of each adj pair: (adj_c → compat, adj_i → incompat)
    # AND (adj_i → compat, adj_c → incompat). This ensures each adjective appears
    # in BOTH compatible and incompatible contexts across the dataset, eliminating
    # the adj-polarity bigram cue.
    pool = []
    for si, subj in enumerate(_CTX_SUBJECTS):
        for ai, (adj_c, adj_i) in enumerate(_CTX_ATTR_PAIRS):
            for vi, (verb_c, verb_syn) in enumerate(_CTX_ATTR_VERBS):
                s = subj[4:]  # strip "the "
                # Direction 0: adj_c is base attribute
                base0     = f"The {adj_c} {s} {verb_c}."
                compat0   = f"The {adj_c} {s} {verb_syn}."
                incompat0 = f"The {adj_i} {s} {verb_c}."
                pool.append((base0, compat0, incompat0, f"attr|{si}|{ai}|{vi}|0"))
                # Direction 1: adj_i is base attribute (symmetric — breaks polarity cue)
                base1     = f"The {adj_i} {s} {verb_c}."
                compat1   = f"The {adj_i} {s} {verb_syn}."
                incompat1 = f"The {adj_c} {s} {verb_c}."
                pool.append((base1, compat1, incompat1, f"attr|{si}|{ai}|{vi}|1"))
    rng.shuffle(pool)
    for base, compat, incompat, fk in pool[:per_contract // 2 + 1]:
        rows.append({"level": "context", "contract": "attribute-preserve",
                     "text": f"Base: {base} Test: {compat}", "label": 1,
                     "family": fk, "meta": {}})
        rows.append({"level": "context", "contract": "attribute-preserve",
                     "text": f"Base: {base} Test: {incompat}", "label": 0,
                     "family": fk, "meta": {}})

    # 3. location-preserve (merged location pool)
    # Both compatible and incompatible draw from the same _CTX_LOCS pool,
    # so location-bigrams appear in both labels (no disjoint-pool cue).
    pool = []
    for si, subj in enumerate(_CTX_LOC_SUBJ):
        for vi, (va, vb) in enumerate(_CTX_LOC_VERBS):
            for li, la in enumerate(_CTX_LOCS):
                # lb: a different location from the same pool
                other_locs = [l for l in _CTX_LOCS if l != la]
                lb = other_locs[li % len(other_locs)]
                base     = f"{subj} {va} {la}.".capitalize()
                compat   = f"{subj} {vb} {la}.".capitalize()
                incompat = f"{subj} {va} {lb}.".capitalize()
                fk = f"loc|{si}|{vi}|{li}"
                pool.append((base, compat, incompat, fk))
    rng.shuffle(pool)
    for base, compat, incompat, fk in pool[:per_contract // 2 + 1]:
        rows.append({"level": "context", "contract": "location-preserve",
                     "text": f"Base: {base} Test: {compat}", "label": 1,
                     "family": fk, "meta": {}})
        rows.append({"level": "context", "contract": "location-preserve",
                     "text": f"Base: {base} Test: {incompat}", "label": 0,
                     "family": fk, "meta": {}})

    # 4. temporal-preserve (balanced tenses)
    # Generates BOTH future-base pairs (compat=future_syn, incompat=past) AND
    # past-base pairs (compat=past_syn, incompat=future). Past-tense markers
    # therefore appear in BOTH compatible (past→past) and incompatible (future→past),
    # breaking the future-only-base tense-cue.
    pool = []
    for si, subj in enumerate(_CTX_TEMP_SUBJ):
        for ai, act_tmpl in enumerate(_CTX_TEMP_ACTS):
            for ti, (t_fut, t_syn, t_past, t_past_syn) in enumerate(_CTX_FUTURE_TIMES):
                future_ph  = act_tmpl[0].format(t=t_fut)
                fut_syn_ph = act_tmpl[1].format(t_syn=t_syn)
                past_ph    = act_tmpl[2].format(t_past=t_past)
                past_syn_ph = act_tmpl[3].format(t_past=t_past_syn)
                # Future-base pair
                pool.append((
                    f"{subj} {future_ph}.".capitalize(),
                    f"{subj} {fut_syn_ph}.".capitalize(),
                    f"{subj} {past_ph}.".capitalize(),
                    f"temp|{si}|{ai}|{ti}|f"
                ))
                # Past-base pair (mirror: compat=past_syn, incompat=future)
                pool.append((
                    f"{subj} {past_ph}.".capitalize(),
                    f"{subj} {past_syn_ph}.".capitalize(),
                    f"{subj} {future_ph}.".capitalize(),
                    f"temp|{si}|{ai}|{ti}|p"
                ))
    rng.shuffle(pool)
    for base, compat, incompat, fk in pool[:per_contract // 2 + 1]:
        rows.append({"level": "context", "contract": "temporal-preserve",
                     "text": f"Base: {base} Test: {compat}", "label": 1,
                     "family": fk, "meta": {}})
        rows.append({"level": "context", "contract": "temporal-preserve",
                     "text": f"Base: {base} Test: {incompat}", "label": 0,
                     "family": fk, "meta": {}})

    # 5. negation-scope-preserve
    pool = []
    for ni, noun in enumerate(_CTX_NEG_NOUNS):
        for pi, pred in enumerate(_CTX_NEG_PREDS):
            # "Not all X Y." → compat: "Some X do not Y." incompat: "All X do not Y."
            pred_bare = pred  # already bare-ish; strip aux for "do not X"
            base     = f"Not all {noun} {pred}."
            compat   = f"Some {noun} do not {pred_bare}."
            incompat = f"All {noun} do not {pred_bare}."
            fk = f"neg|{ni}|{pi}"
            pool.append((base, compat, incompat, fk))
    rng.shuffle(pool)
    for base, compat, incompat, fk in pool[:per_contract // 2 + 1]:
        rows.append({"level": "context", "contract": "negation-scope-preserve",
                     "text": f"Base: {base} Test: {compat}", "label": 1,
                     "family": fk, "meta": {}})
        rows.append({"level": "context", "contract": "negation-scope-preserve",
                     "text": f"Base: {base} Test: {incompat}", "label": 0,
                     "family": fk, "meta": {}})

    rng.shuffle(rows)
    return rows[:n_per_level]


# --------------------------------------------------------------------------
# V2 TASK: 5 contracts, LENGTH-MATCHED incompatibles (wrong city/year),
# NO appended-clause cue ("and ...", "also ...", "became ...").
# Family key = (source_id, contract).
# --------------------------------------------------------------------------
V2_TASK_CONTRACTS = [
    "facts-only",
    "paraphrase-only",
    "extract-entity",
    "answer-if-supported",
    "refuse-if-unknown",
]

_NAMES = ["Alice", "Bob", "Carol", "David", "Eva", "Frank", "Grace",
          "Hana", "Ivan", "Julia", "Karl", "Lena", "Marco", "Nina"]
# MERGED pools: compatible and incompatible both draw from same vocabulary.
# Disjoint pools created trivial bigram cues (Lima/Cairo only in incompatible).
_CITIES = ["Seoul", "Tokyo", "Paris", "Berlin", "Boston", "Lima", "Cairo", "Lagos", "Rome", "Oslo"]
_YEARS  = [2010, 2012, 2013, 2015, 2016, 2017, 2018, 2019, 2021, 2022]
_ROLES = ["researcher", "engineer", "designer", "manager", "analyst"]

# Paraphrase pairs (compat, incompat-same-length)
_PARA_PAIRS = [
    ("relocated to {city} in {year}", "was transferred to {city} by {year}"),
    ("moved to {city} in {year}", "left for {city} around {year}"),
    ("settled in {city} in {year}", "arrived at {city} during {year}"),
    ("transferred to {city} in {year}", "was relocated near {city} in {year}"),
    ("went to {city} in {year}", "traveled to {city} around {year}"),
]


def _make_task_rows(name, city_a, city_b, year_a, year_b, role, src_id, rng, para_override=None):
    """Generate one source → rows for all 5 task contracts."""
    rows = []
    source = f"{name} worked as a {role} in {city_a} starting in {year_a}."
    family_base = f"src{src_id}"

    # facts-only: compatible = faithful restatement; incompatible = wrong city (length-matched)
    compat_fo  = f"{name} worked as a {role} in {city_a} starting in {year_a}."
    incompat_fo = f"{name} worked as a {role} in {city_b} starting in {year_a}."
    for text, lab in [(f"Source: {source} Output: {compat_fo}", 1),
                      (f"Source: {source} Output: {incompat_fo}", 0)]:
        rows.append({"level": "task", "contract": "facts-only", "text": text,
                     "label": lab, "family": f"{family_base}|facts-only",
                     "meta": {"src_id": src_id, "contract": "facts-only"}})

    # paraphrase-only: compatible = paraphrase; incompatible = wrong year (length-matched)
    para_c, para_i = para_override if para_override else rng.choice(_PARA_PAIRS)
    compat_po   = f"{name} {para_c.format(city=city_a, year=year_a)}."
    incompat_po = f"{name} {para_c.format(city=city_a, year=year_b)}."
    for text, lab in [(f"Source: {source} Output: {compat_po}", 1),
                      (f"Source: {source} Output: {incompat_po}", 0)]:
        rows.append({"level": "task", "contract": "paraphrase-only", "text": text,
                     "label": lab, "family": f"{family_base}|paraphrase-only",
                     "meta": {"src_id": src_id, "contract": "paraphrase-only"}})

    # extract-entity: compatible = correct extraction; incompatible = wrong city (length-matched)
    compat_ee   = f"Name: {name}. City: {city_a}. Year: {year_a}. Role: {role}."
    incompat_ee = f"Name: {name}. City: {city_b}. Year: {year_a}. Role: {role}."
    for text, lab in [(f"Source: {source} Extract: {compat_ee}", 1),
                      (f"Source: {source} Extract: {incompat_ee}", 0)]:
        rows.append({"level": "task", "contract": "extract-entity", "text": text,
                     "label": lab, "family": f"{family_base}|extract-entity",
                     "meta": {"src_id": src_id, "contract": "extract-entity"}})

    # answer-if-supported: compatible = answers only what source supports;
    # incompatible = wrong year in answer (length-matched, no extra clause)
    compat_as   = f"Yes, {name} worked in {city_a} starting {year_a}."
    incompat_as = f"Yes, {name} worked in {city_a} starting {year_b}."
    for text, lab in [(f"Source: {source} Answer: {compat_as}", 1),
                      (f"Source: {source} Answer: {incompat_as}", 0)]:
        rows.append({"level": "task", "contract": "answer-if-supported", "text": text,
                     "label": lab, "family": f"{family_base}|answer-if-supported",
                     "meta": {"src_id": src_id, "contract": "answer-if-supported"}})

    # refuse-if-unknown: compatible = refuses to answer unknown fact;
    # incompatible = fabricates unknown fact (wrong city used as fabricated detail)
    unknown_q = f"What is {name}'s salary?"
    compat_ru   = f"The source does not state {name}'s salary."
    incompat_ru = f"The source shows {name} earned a salary in {city_b}."
    for text, lab in [(f"Source: {source} Query: {unknown_q} Response: {compat_ru}", 1),
                      (f"Source: {source} Query: {unknown_q} Response: {incompat_ru}", 0)]:
        rows.append({"level": "task", "contract": "refuse-if-unknown", "text": text,
                     "label": lab, "family": f"{family_base}|refuse-if-unknown",
                     "meta": {"src_id": src_id, "contract": "refuse-if-unknown"}})
    return rows


def gen_task_v2(n_per_level, rng):
    """Exhaustively enumerate (name, city_a, year_a, role) combinations → unique rows."""
    import itertools
    all_rows = []
    src_id = 0
    for name, city_a, year_a, role in itertools.product(_NAMES, _CITIES, _YEARS, _ROLES):
        # city_b/year_b from SAME pool as city_a/year_a (different value) —
        # prevents trivial bigram cue from disjoint pools.
        other_cities = [c for c in _CITIES if c != city_a]
        other_years  = [y for y in _YEARS  if y != year_a]
        city_b = other_cities[src_id % len(other_cities)]
        year_b = other_years[src_id % len(other_years)]
        para = _PARA_PAIRS[src_id % len(_PARA_PAIRS)]
        all_rows.extend(_make_task_rows(name, city_a, city_b, year_a, year_b, role, src_id, rng,
                                        para_override=para))
        src_id += 1
    # Dedupe on text within this level
    seen, uniq = set(), []
    for r in all_rows:
        if r["text"] not in seen:
            seen.add(r["text"]); uniq.append(r)
    rng.shuffle(uniq)
    return uniq[:n_per_level]


# --------------------------------------------------------------------------
# V2 FAMILY-SPLIT: assign each (level, family) to train or test as a unit.
# Asserts ZERO surface-text leakage between splits.
# --------------------------------------------------------------------------
def family_split_v2(rows, test_frac, rng):
    """Assign counterfactual families to train or test entirely.
    No example from a family appears in both splits.
    Returns (train_rows, test_rows).
    """
    by_family = defaultdict(list)
    for r in rows:
        by_family[r["family"]].append(r)
    families = sorted(by_family.keys())
    rng.shuffle(families)
    n_test_fam = max(1, int(len(families) * test_frac))
    test_fams = set(families[:n_test_fam])
    train_rows, test_rows = [], []
    for r in rows:
        if r["family"] in test_fams:
            test_rows.append(r)
        else:
            train_rows.append(r)
    # Assert zero text leakage: no (level, text) in both splits
    train_texts = set((r["level"], r["text"]) for r in train_rows)
    test_texts  = set((r["level"], r["text"]) for r in test_rows)
    overlap = train_texts & test_texts
    assert len(overlap) == 0, (
        f"FAMILY SPLIT VIOLATION: {len(overlap)} (level,text) pairs appear in "
        f"both train and test. Family grouping failed.")
    return train_rows, test_rows


# --------------------------------------------------------------------------
# V2 MAIN
# --------------------------------------------------------------------------
def run_v2(args):
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    rows = (gen_concept_v2(args.n_per_level, rng)
            + gen_context_v2(args.n_per_level, rng)
            + gen_task_v2(args.n_per_level, rng))
    # Dedupe on (level, contract, text)
    seen, uniq = set(), []
    for r in rows:
        key = (r["level"], r["contract"], r["text"])
        if key not in seen:
            seen.add(key); uniq.append(r)
    train, test = family_split_v2(uniq, args.test_frac, rng)
    # Shuffle within each split
    rng.shuffle(train); rng.shuffle(test)
    for name, split in [("train", train), ("test", test)]:
        path = os.path.join(args.out, f"{name}.jsonl")
        with open(path, "w") as f:
            for r in split:
                # Write without 'family' key (internal split artifact)
                out_r = {k: v for k, v in r.items() if k != "family"}
                f.write(json.dumps(out_r) + "\n")
        pos = sum(r["label"] for r in split)
        print(f"{name}: {len(split)} examples, {pos} compatible "
              f"({pos/max(len(split),1):.2%})")
    from collections import Counter
    c = Counter((r["level"], r["contract"]) for r in uniq)
    print("v2 per (level,contract):")
    for (lv, co), n in sorted(c.items()):
        print(f"  {lv}/{co}: {n}")
    print(f"dataset_version: v2  seed: {args.seed}  "
          f"n_per_level: {args.n_per_level}  total: {len(uniq)}")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1", choices=["v1", "v2"])
    ap.add_argument("--n_per_level", type=int, default=600)
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test_frac", type=float, default=0.15)
    args = ap.parse_args()

    if args.version == "v2":
        run_v2(args)
        return

    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)

    rows = (gen_concept(args.n_per_level, rng)
            + gen_context(args.n_per_level, rng)
            + gen_task(args.n_per_level, rng))
    # dedupe on (level,contract,text) to avoid train/test leakage of identical strings
    seen, uniq = set(), []
    for r in rows:
        key = (r["level"], r["contract"], r["text"])
        if key in seen:
            continue
        seen.add(key); uniq.append(r)
    rng.shuffle(uniq)
    n_test = int(len(uniq) * args.test_frac)
    test, train = uniq[:n_test], uniq[n_test:]

    for name, split in [("train", train), ("test", test)]:
        path = os.path.join(args.out, f"{name}.jsonl")
        with open(path, "w") as f:
            for r in split:
                f.write(json.dumps(r) + "\n")
        pos = sum(r["label"] for r in split)
        print(f"{name}: {len(split)} examples, {pos} compatible "
              f"({pos/max(len(split),1):.2%})")
    # report per-level counts
    from collections import Counter
    c = Counter((r["level"], r["label"]) for r in uniq)
    print("per (level,label):", dict(c))

if __name__ == "__main__":
    main()
