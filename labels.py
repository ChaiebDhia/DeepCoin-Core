_LABELS_EN = {
    "historian":    "Historical Analysis",
    "validator":    "Forensic Validation",
    "investigator": "Visual Investigation",
    "report_title": "NUMISMATIC ANALYSIS REPORT",
    "cnn_result":   "CNN CLASSIFICATION RESULT",
    "expert_nar":   "EXPERT NARRATIVE"
}

_LABELS_FR = {
    "historian":    "Analyse Historique",
    "validator":    "Validation Forensic",
    "investigator": "Investigation Visuelle",
    "report_title": "RAPPORT D'ANALYSE NUMISMATIQUE",
    "cnn_result":   "RÉSULTAT DE CLASSIFICATION CNN",
    "expert_nar":   "RÉCIT D'EXPERT"
}


# ── Text-cleaning patterns ───────────────────────────────────────────────────
# Matches [CONTEXT 1], [CONTEXT 2], [CONTEXT CNN], [CONTEXT N], etc.
# These are internal RAG citation markers injected into the LLM prompt so that
# Gemini can produce grounded answers.  They must NEVER appear in the final PDF.
_RE_CONTEXT = re.compile(r"\[CONTEXT\s*(?:\d+|CNN|N)?\s*(?:—[^\]]*)?\]", re.I)

# Markdown patterns produced by some LLM responses when the model ignores the
# "plain prose only" instruction.  Order matters — bold before italic.
_MD_PATTERNS = [
    (re.compile(r"\*{3}(.+?)\*{3}"),    r"\1"),   # ***bold-italic*** → text
    (re.compile(r"\*{2}(.+?)\*{2}"),    r"\1"),   # **bold** → text
    (re.compile(r"\*(.+?)\*"),          r"\1"),   # *italic* → text
    (re.compile(r"_{2}(.+?)_{2}"),      r"\1"),   # __bold__ → text
    (re.compile(r"_(.+?)_"),            r"\1"),   # _italic_ → text
    (re.compile(r"`{1,3}(.+?)`{1,3}"), r"\1"),   # `code` / ```code``` → text
    (re.compile(r"(?m)^\s*#{1,6}\s+"),  ""),     # ## Heading at line-start → stripped
    (re.compile(r"#{2,}\s*"),           ""),     # inline ## / #### (LLM artefact)
]

# Greek → Latin transliteration (dict-based)
_GREEK_MAP: dict = {
    "Α":"A",  "Β":"B",  "Γ":"G",  "Δ":"D",  "Ε":"E",  "Ζ":"Z",
    "Η":"E",  "Θ":"TH", "Ι":"I",  "Κ":"K",  "Λ":"L",  "Μ":"M",
    "Ν":"N",  "Ξ":"X",  "Ο":"O",  "Π":"P",  "Ρ":"R",  "Σ":"S",
    "Τ":"T",  "Υ":"Y",  "Φ":"PH", "Χ":"KH", "Ψ":"PS", "Ω":"O",
    "α":"a",  "β":"b",  "γ":"g",  "δ":"d",  "ε":"e",  "ζ":"z",
    "η":"e",  "θ":"th", "ι":"i",  "κ":"k",  "λ":"l",  "μ":"m",
    "ν":"n",  "ξ":"x",  "ο":"o",  "π":"p",  "ρ":"r",  "σ":"s",
    "ς":"s",  "τ":"t",  "υ":"y",  "φ":"ph", "χ":"kh", "ψ":"ps",
    "ω":"o",
}

# Typographic characters that NFD decomposition cannot resolve to ASCII.
# These appear frequently in LLM output (curly quotes, dashes, ligatures)
# and would silently become '?' via latin-1 encode("replace").
# Mapped explicitly BEFORE the NFD step so the result is always readable.
_TYPO_MAP: dict = {
    "\u2018": "'",   # LEFT  SINGLE QUOTATION MARK  '
    "\u2019": "'",   # RIGHT SINGLE QUOTATION MARK  '  ← commonest source of '?'
    "\u201A": ",",   # SINGLE LOW-9 QUOTATION MARK  ‚
    "\u201C": '"',   # LEFT  DOUBLE QUOTATION MARK  "
    "\u201D": '"',   # RIGHT DOUBLE QUOTATION MARK  "
    "\u201E": '"',   # DOUBLE LOW-9 QUOTATION MARK  „
    "\u2013": "-",   # EN DASH                       –
    "\u2014": "-",   # EM DASH                       —
    "\u2015": "-",   # HORIZONTAL BAR                ―
    "\u2026": "...", # HORIZONTAL ELLIPSIS           …
    "\u00DF": "ss",  # LATIN SMALL LETTER SHARP S    ß  (German: Geiß → Geiss)
    "\u00C6": "AE",  # LATIN CAPITAL LIGATURE AE     Æ
    "\u00E6": "ae",  # LATIN SMALL   LIGATURE AE     æ
    "\u0152": "OE",  # LATIN CAPITAL LIGATURE OE     Œ
    "\u0153": "oe",  # LATIN SMALL   LIGATURE OE     œ
    "\u00D8": "O",   # LATIN CAPITAL LETTER O STROKE Ø
    "\u00F8": "o",   # LATIN SMALL   LETTER O STROKE ø
    "\u00D0": "D",   # LATIN CAPITAL LETTER ETH      Ð
    "\u00F0": "d",   # LATIN SMALL   LETTER ETH      ð
    "\u00DE": "TH",  # LATIN CAPITAL LETTER THORN    Þ
    "\u00FE": "th",  # LATIN SMALL   LETTER THORN    þ
}


def _s(text: str) -> str:
    """
    Sanitise a string for PDF output through six ordered steps.

    Step 1 — Strip RAG citation markers  ([CONTEXT 1], [CONTEXT CNN], …)
             These are internal prompt tokens that must never appear in print.
    Step 2 — Strip Markdown syntax  (**, *, ##, `code`, …)
             Some LLM responses contain inline Markdown even when instructed
             not to.  We extract just the visible text between the markers.
    Step 3 — Collapse excess whitespace left by stripping.
    Step 4 — Typographic character normalisation (_TYPO_MAP)
             Curly quotes, em/en dashes, ligatures, sharp-s, etc. that NFD
             decomposition cannot resolve to latin-1.  These are the most
             frequent source of '?' in LLM-generated text.
    Step 5 — Greek → Latin transliteration  (Σ→S, Α→A, …)
             fpdf2's built-in fonts use latin-1 encoding, which excludes the
             Greek Unicode block (U+0370–U+03FF).
    Step 6 — Unicode NFD decomposition + combining-mark removal
             Converts accented characters (ō, é, ñ, …) to their ASCII base
             letter by decomposing to NFD form and stripping combining marks
             (Unicode category 'Mn').  Handles the long tail of accented
             Latin chars not covered by _TYPO_MAP.
    Step 7 — latin-1 encode/decode  (final safety net for any remaining
             characters outside the font's supported range).
    """
    t = str(text)

    # Step 1 — remove [CONTEXT N] / [CONTEXT CNN] citation markers
    t = _RE_CONTEXT.sub("", t)

    # Step 2 — strip Markdown formatting, keep visible text
    for pattern, repl in _MD_PATTERNS:
        t = pattern.sub(repl, t)

    # Step 3a — strip Corpus Nummorum HTML navigation artefacts
    #   corpus-nummorum.eu embeds "go to the NLP result of this description"
    #   links and "| legend Design" / "| legend Legend" section labels as
    #   inline text in the scraped HTML.  These are never numismatic content.
    t = re.sub(r"go to the NLP result of this description", "", t, flags=re.I)
    t = re.sub(r"\|\s*legend\s+(Design|Legend)\s*", " / ", t, flags=re.I)
    t = re.sub(r"^legend\s+(Design|Legend)\s+", "", t, flags=re.I)  # at field start
    t = re.sub(r"\s*\|\s*Legend:\s*", " / Legend: ", t)  # | Legend: x -> / Legend: x
    t = re.sub(r"\s*\|\s*$", "", t)  # trailing " | " left after stripping

    # Step 3b — normalise German date notation from corpus-nummorum.eu
    #   The KB stores dates in German: "500-400 v.Chr." = "500-400 BC"
    t = re.sub(r"\bv\.\s*Chr\.", "BC",  t)
    t = re.sub(r"\bn\.\s*Chr\.", "AD",  t)

    # Step 3b — collapse runs of spaces / clean up leftover punctuation gaps
    t = re.sub(r"  +", " ", t).strip()

    # Step 4 — typographic character normalisation
    t = "".join(_TYPO_MAP.get(c, c) for c in t)

    # Step 5 — Greek transliteration
    t = "".join(_GREEK_MAP.get(c, c) for c in t)

    # Step 6 — decompose accented chars, strip combining diacritics
    t = unicodedata.normalize("NFD", t)
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")

    # Step 7 — final latin-1 safety net
    return t.encode("latin-1", "replace").decode("latin-1")


def _safe(text: str) -> str:
    """
    Minimal latin-1 safety encode for filenames, timestamps, and other
    system-generated strings that contain NO Markdown and NO Greek.

    WHY NOT use _s() here:
        _s() applies Markdown-stripping patterns including _italic_ which
        matches underscores in filenames like 'CN_type_1015_cn_coin.jpg',
        silently eating the underscores and producing 'CNtype1015cncoin.jpg'.
        _safe() skips all Markdown processing — it only ensures the string
        is latin-1 encodeable so fpdf2 can render it without crashing.
    """
    return str(text).encode("latin-1", "replace").decode("latin-1")


def _conf_color(conf: float) -> tuple:
    """
    Return an RGB colour tuple for a confidence value.

    Thresholds mirror the Gatekeeper routing thresholds so the badge colour
    matches the routing decision intuitively:
        Green  (> 85%)  — high confidence, historian route
        Amber  (40-85%) — medium confidence, validator route
        Red    (< 40%)  — low confidence, investigator route
    """
    if conf > 0.85:
        return _C_GREEN
    if conf > 0.40:
        return _C_AMBER
    return _C_RED_DK


def _enrich_label(type_id, include_date: bool = False) -> str:
    """
    Return a user-friendly coin description for a CN type ID.

    Format (no date):   "Material Denomination - Mint"
    Format (with date): "Material Denomination - Mint, Date"
    Example: "Silver Drachm - Maroneia, c.365-330 BC"

    WHAT: Combines the material, denomination, mint (and optionally date) from
          the RAG knowledge base into a single readable string.

    WHY no date in the stripe label:
        The result stripe uses a 50-char truncation limit for the coin name.
        Adding the date would overflow it for most coins.  The date is only
        appended when include_date=True, which is used by the top-5 table to
        differentiate coins of the same denomination from the same mint
        (e.g. CN 1015, 1017, 864 are all Silver Drachm - Maroneia but with
        different date ranges).

    WHY: The raw CN type number (e.g. 532, 1015) is an opaque database key
         that means nothing to a museum visitor or researcher reading the PDF.
         An enriched string gives an immediate, self-explanatory description.

    HOW: Calls get_rag_engine().get_by_id() which is an in-memory dict lookup
         (zero I/O, sub-millisecond).  Falls back gracefully to "CN {type_id}".
    """
    try:
        from src.core.rag_engine import get_rag_engine
        rec = get_rag_engine().get_by_id(int(type_id))
        if not rec:
            return f"CN {type_id}"
        mat   = (rec.get("material",     "") or "").strip().title()
        denom = (rec.get("denomination", "") or "").strip().title()
        mint  = (rec.get("mint",         "") or "").strip()
        date  = (rec.get("date",         "") or "").strip()
        # Strip parenthetical qualifiers from denom: "Large Denomination (Bronze)" -> "Large Denomination"
        denom = re.sub(r'\s*\([^)]*\)', '', denom).strip()
        # Strip archaeological period appended to date: "c. 500-450 BC Archaic Period" -> "c. 500-450 BC"
        date = re.sub(
            r'\s+(Archaic|Classical|Hellenistic|Roman|Byzantine|Early|Late|Middle)\b.*',
            '', date, flags=re.IGNORECASE).strip()
        # Filter denominations that are scraped field names, not real values
        _BAD_DENOMS = {"material", "type", "region", "date", "mint",
                       "period", "denomination", "weight", "diameter",
                       "obverse", "reverse", "legend", "authority"}
        # Before filtering, rescue metal name hidden in compound bad-denom
        # e.g. denom="Material bronze" -> mat empty -> extract "Bronze" from denom
        _METAL_WORDS = {"bronze", "silver", "gold", "electrum", "billon", "copper", "lead"}
        _denom_words = denom.lower().split()
        if not mat and _denom_words:
            for _w in _denom_words:
                if _w in _METAL_WORDS:
                    mat = _w.title()
                    break
        # Also catch compound denominations whose first word is a bad key
        # e.g. "Material Bronze" -> split()[0] == "material" -> filter out
        if denom.lower() in _BAD_DENOMS or (_denom_words and _denom_words[0] in _BAD_DENOMS):
            denom = ""
        # NOTE: use module-level re — no local import needed
        parts = " ".join(p for p in (mat, denom) if p)
        base  = f"{parts} - {mint}" if (parts and mint) else (parts or mint or f"CN {type_id}")
        if include_date and date and len(date) <= 30:
            return f"{base}, {date}"
        return base
    except Exception:
        return f"CN {type_id}"


def _basename(path: str) -> str:
    """
    Return just the original filename from a path, stripping the UUID prefix.

    WHY strip UUID:
        classify.py saves uploads as '{uuid}_{original_filename}' to prevent
        name collisions.  The UUID is internal bookkeeping — the header of the
        PDF should show the human-readable original filename, not a 36-char
        identifier that means nothing to a reader.

    UUID format: 8-4-4-4-12 hex chars + underscore = 37 leading characters.
    Example: '2240d431-f93c-4fc1-b8b9-96fece4bab9d_coin.jpg' -> 'coin.jpg'
    """
    name = path.replace("\\", "/").split("/")[-1] if path else "N/A"
    # Strip UUID prefix: exactly 36 hex+dash chars followed by underscore
    # Uses module-level re — no local import needed.
    name = re.sub(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_", "", name)
    return name


# ═══════════════════════════════════════════════════════════════════════════════
# Main class
# ═══════════════════════════════════════════════════════════════════════════════
