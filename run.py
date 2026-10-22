#!/usr/bin/env python3
"""Run the regulatory-exposure pipeline step by step.

  python run.py download-fr            # Federal Register: final and proposed rules (full text)
  python run.py download-10k           # SEC EDGAR: 10-K filings, one per firm-year
  python run.py extract                # GPT-5.6 Terra locates and cleans Item 1 and Item 1A
  python run.py embed                  # text-embedding-3-large: rules and 10-K sections
  python run.py domains                # clustering into 20 domains + GPT-5.6 Terra labels
  python run.py score [--dry-run]      # GPT-5.6 Terra rates rules and 10-K sections (0-100)
  python run.py link                   # exposure, similarity, firm-year and firm-agency-year measures
  python run.py all                    # everything above, in order
  python run.py endpoints              # which OpenAI API endpoint and model each step uses

Every step is resumable: finished items are skipped when a step is run again.
"""
import argparse

from regexposure import domains, edgar, embed, extract, fedreg, link, llm, score
from regexposure.settings import load_config

STEPS = ["download-fr", "download-10k", "extract", "embed", "domains", "score", "link"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=STEPS + ["all", "endpoints"])
    ap.add_argument("--config", default=None, help="path to config.yaml (default: repository root)")
    ap.add_argument("--corpus", choices=["rules", "sections", "both"], default="both", help="embed / score: which corpus")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N items (pilot runs)")
    ap.add_argument("--dry-run", action="store_true", help="score: count tokens and estimate cost, no API calls")
    a = ap.parse_args()
    cfg = load_config(a.config)
    corpora = ["rules", "sections"] if a.corpus == "both" else [a.corpus]
    if a.step == "endpoints":
        print(llm.describe_endpoints(cfg))
        return

    for step in (STEPS if a.step == "all" else [a.step]):
        print(f"=== {step} ===")
        if step in llm.STEP_ENDPOINTS and not (step == "score" and a.dry_run):
            ep, key = llm.STEP_ENDPOINTS[step]
            print(f"OpenAI API: POST {ep} ({getattr(cfg.openai, key)})")
        if step == "download-fr":
            fedreg.download(cfg)
        elif step == "download-10k":
            edgar.download(cfg)
        elif step == "extract":
            extract.run(cfg, limit=a.limit)
        elif step == "embed":
            for c in corpora:
                embed.run(cfg, c)
        elif step == "domains":
            domains.run(cfg)
        elif step == "score":
            for c in corpora:
                score.run(cfg, c, limit=a.limit, dry_run=a.dry_run)
        elif step == "link":
            link.run(cfg)


if __name__ == "__main__":
    main()
