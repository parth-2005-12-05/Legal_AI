"""Command-line entry point.

    python -m caseway demo                 run sample cases end to end and print the results
    python -m caseway process FILE [--chat TEXT] [--language hi|en] [--fee-band N]
                                           run one document through the pipeline
    python -m caseway audit-verify         check the audit log hash chain
    python -m caseway typeahead TYPE PREFIX [--within ID]

All commands work fully offline against the local var/ folder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ROOT
from .pipeline import CaseWay


def _print(obj) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def _brief(summary: dict) -> dict:
    keep = {k: summary[k] for k in ("case_id", "status", "partner_review", "needs_review") if k in summary}
    if "fields" in summary:
        keep["fields"] = {k: f"{v['display']}  [{v['status']} {v['confidence']:.2f}]"
                          for k, v in summary["fields"].items() if v["value"]}
    if "case_type" in summary:
        ct = summary["case_type"]
        keep["case_type"] = f"{ct['primary_top']}/{ct['primary_sub']} ({ct['tier']}, {ct['probability']:.2f})"
    if "match" in summary:
        keep["match"] = summary["match"]
    return keep


def demo(app: CaseWay) -> None:
    samples = ROOT / "data" / "samples"
    print("== Case 1: FIR upload + chat, urgent bail with a conflict-of-interest hit ==")
    c1 = app.create_case("My brother was arrested yesterday in a cheating case by Kotwali police in Ghaziabad. "
                         "We need bail.", "hi", 3)
    app.upload(c1, "fir.txt", (samples / "fir_ghaziabad.txt").read_bytes())
    _print(_brief(app.process(c1)))
    print("\n-> senior lawyer clears it; with partner_review off it goes straight to the customer")
    app.senior_decision(c1, "senior_lawyer:demo", True, "conflicted lawyers excluded; urgent availability confirmed")
    _print(app.customer_view(c1))

    print("\n== Case 2: chat only, consumer matter, no flags ==")
    c2 = app.create_case("The builder has not given possession of my flat in Indirapuram, Ghaziabad "
                         "for 3 years. I want a refund.", "en", 2)
    _print(_brief(app.process(c2)))

    print("\n== Case 3: Hindi chat, classifier unsure -> paralegal review with follow-up questions ==")
    c3 = app.create_case("मेरी पत्नी गुजारा भत्ता मांग रही है, मामला परिवार न्यायालय गाजियाबाद में है", "hi")
    s3 = app.process(c3)
    _print(_brief(s3) | {"follow_up_questions": s3.get("follow_up_questions", [])})
    print("\n-> paralegal confirms the case type; matching runs on verified facts")
    _print(_brief(app.submit_field_review(c3, "paralegal:demo", {"case_type": "family/maintenance"})))

    ok, bad = app.audit.verify()
    print(f"\nAudit chain intact: {ok}  anchor: {app.audit.anchor()[:16]}...")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="caseway", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo")
    pp = sub.add_parser("process")
    pp.add_argument("file")
    pp.add_argument("--chat", default="")
    pp.add_argument("--language", default="hi")
    pp.add_argument("--fee-band", type=int)
    sub.add_parser("audit-verify")
    pt = sub.add_parser("typeahead")
    pt.add_argument("type")
    pt.add_argument("prefix")
    pt.add_argument("--within")
    args = p.parse_args(argv)

    app = CaseWay()
    if args.cmd == "demo":
        demo(app)
    elif args.cmd == "process":
        cid = app.create_case(args.chat, args.language, args.fee_band)
        app.upload(cid, Path(args.file).name, Path(args.file).read_bytes())
        _print(app.process(cid))
    elif args.cmd == "audit-verify":
        ok, bad = app.audit.verify()
        _print({"intact": ok, "first_bad_seq": bad, "anchor": app.audit.anchor()})
        return 0 if ok else 1
    elif args.cmd == "typeahead":
        _print(app.typeahead(args.type, args.prefix, args.within))
    return 0
