"""Scripted follow-up questions (design: Accuracy strategy 3, "Ask the customer,
deterministically").

When a field is missing or uncertain, the chat asks a scripted question with
constrained choices taken from the gazetteer, e.g. "Was the FIR filed at
Kotwali Nagar, Ghaziabad or Kotwali, Meerut?". A structured answer resolves most
gaps without staff time. There is no generative AI anywhere in this dialogue:
it is a fixed state machine over field statuses.
"""

from __future__ import annotations

from .gazetteer.model import Gazetteer

QUESTIONS = {
    "district": ("In which district is the case (or the property / incident)?",
                 "मामला किस जिले का है?"),
    "police_station": ("At which police station was the FIR filed?", "एफआईआर किस थाने में दर्ज हुई?"),
    "court": ("In which court is the case going on?", "मामला किस न्यायालय में चल रहा है?"),
    "case_type": ("Which of these best describes your matter?", "आपका मामला किस प्रकार का है?"),
}


def follow_up_questions(fields: dict[str, dict], gaz: Gazetteer, taxonomy_tops: list[str],
                        language: str = "en") -> list[dict]:
    """Build questions for fields whose status is review or missing.
    `fields` is the serialised FusionResult.fields mapping."""
    lang_idx = 1 if language == "hi" else 0
    out = []
    district = fields.get("district", {}).get("value")
    for name in ("district", "police_station", "court"):
        f = fields.get(name) or {}
        if f.get("status") not in ("review", "missing"):
            continue
        if name == "court" and f.get("status") == "missing":
            continue  # a court is derived from rules when nobody names one
        choices = [v for v in [f.get("value")] + [a[0] for a in f.get("alternatives", [])] if v]
        if not choices and name != "district" and district:
            choices = gaz.descendants_of_type(district, name)[:5]
        if not choices and name == "district":
            choices = gaz.by_type["district"][:8]
        out.append({"field": name, "question": QUESTIONS[name][lang_idx],
                    "choices": [{"id": c, "label": gaz.name(c)} for c in dict.fromkeys(choices)],
                    "typeahead": f"/typeahead?type={name}" + (f"&within={district}" if district and name != "district" else "")})
    ct = fields.get("case_type") or {}
    if ct.get("status") in ("review", "missing"):
        out.append({"field": "case_type", "question": QUESTIONS["case_type"][lang_idx],
                    "choices": [{"id": t, "label": t.replace("_", " ").title()} for t in taxonomy_tops]})
    return out
