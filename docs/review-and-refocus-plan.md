# AI Bias Laboratory — Review & Refocus Plan

**Status:** Draft for Peter's approval · 2026-07-06 · Reviewer: Fable
**Peter's sharpened vision:** *generate a test set + accompanying prompts to test bias in an AI model — as a standalone tool AND integratable in a corporate environment.*
**Reviewed:** README, REQUIREMENTS.md (12 FR / 8 NFR), DEVELOPMENT_PLAN.md (4 phases/16 wks), repo structure, 7 UI screenshots.

---

## 1. Verdict in one paragraph

The repo contains a good product idea wrapped in four other products. Peter's vision — **test-set + prompt generation → run against any model → scored evidence** — is exactly FR-F1+F2+F5, and those are the right core. But the current scope also promises propagation visualization, 4 mitigation techniques, LIME/SHAP, AIA workflows, GPAI red-teaming and a multi-tenant SaaS — a 12-feature platform that a solo builder will never finish, and that buries the differentiator. The plan-vs-reality drift confirms it: README says "Coming Soon", screenshots show a built dashboard, the planned `src/bias_lab` + `tests/` + `pyproject.toml` structure doesn't exist (there's a flat `app/`), and there are **zero tests for a product whose entire value is the correctness of its numbers**. Refocus on the core loop, restructure once, and this becomes both shippable and the strongest piece of the professional portfolio.

## 2. The refocused product

**One sentence:** *"Bias Lab genereert een op jouw use-case toegesneden bias-testset met prompts, draait die tegen elk AI-model (API of lokaal), en levert gescoorde, reproduceerbare evidence — standalone via UI/CLI, of als geautomatiseerde gate in je corporate CI/MLOps-pipeline."*

**Core loop (= the product):**
1. **Describe** — user describes the model's use-case/domain (e.g. "CV-screening, NL, financial sector") + selects protected attributes.
2. **Generate** — LLM-assisted generation of a paired/counterfactual test set + prompts (name-swap, attribute-swap, template variants; seeded, versioned, reviewable — the human can edit before running).
3. **Run** — execute against the target model via connectors (OpenAI/Anthropic/Azure/local HF/generic REST), seeded and logged.
4. **Score** — established metrics via **Fairlearn/Aequitas as the computation engine** (never reimplement statistics — wrap reference implementations; that IS the validation story).
5. **Report** — JSON (machine, CI), JUnit-XML (pipeline gates), PDF (compliance/audit), each with config-hash + seed for reproducibility.

**Two consumption modes (Peter's requirement):**
- **Standalone:** Streamlit UI + CLI, runs local/self-hosted ("your HR data never leaves your infrastructure" — the same trust story as the rest of the portfolio, and decisive for exactly the corporate buyers this targets).
- **Corporate integration:** pure Python package + REST API + CI-gate mode (`bias-lab run --config … --fail-under …`) — *"gitleaks for bias"*: a pipeline step that fails the build when bias scores regress. This replaces the Phase-4 multi-tenant SaaS ambition: corporates won't upload protected-attribute data to a startup SaaS, but they WILL pip-install an auditable package inside their perimeter. Cheaper to build, easier to sell, consistent with self-host positioning.

**Regulatory tailwind (use it in the pitch):** EU AI Act Art. 10 requires bias examination for high-risk systems (obligations from **2 Dec 2027** — the market's preparation window is now), and the 2026 Digital Omnibus added **Art. 4a**, explicitly permitting special-category data for bias detection under strict conditions. Bias Lab should *implement* those Art. 4a safeguards (access controls, deletion, no third-party transfer, necessity documentation) as product features — that's a checklist competitors will scramble for.

## 3. Requirements assessment (what to change)

| # | Finding | Action |
|---|---------|--------|
| R1 | F1/F2/F5/F10 = the core; keep, sharpen with acceptance criteria (per metric: formula, reference implementation, tolerance) | Rewrite as v2 requirements around the core loop |
| R2 | F3 (propagation), F4 (mitigation sim), F8 (LIME/SHAP), F11 (GPAI red-team), F12 (AIA) | Move to explicit **backlog/later** — valuable, not core; each is a product on its own |
| R3 | F9 (EU AI Act risk classification) duplicates eu-ai-act-compliance-checker | **Boundary decision:** Bias Lab = evidence *generation*; the checker = classification/assessment. Share one Act knowledge base (versioned YAML, post-Omnibus — the checker's golden-set work feeds this) |
| R4 | NF3 (SaaS multi-tenant, billing) | Replace with **NF-Corporate: package + API + CI-gate + SSO-ready logs**; SaaS is a later decision, not architecture-now |
| R5 | NF4 GDPR "Phase 2" | Pull to Phase 1 + add Art. 4a safeguard set; local-first processing is the default |
| R6 | New NFR missing: **test-set quality** — generated sets need their own validation (balance, grammaticality, no leakage of the attribute into confounds) + human-review step before use | Add; this is the credibility of the whole product |
| R7 | New NFR missing: **model-connector contract** (timeouts, retries, cost caps, rate limits) — corporate runs = thousands of prompts | Add |

## 4. Development plan assessment

The 4-phase/16-week plan is well-written but (a) sequenced UI-first instead of validation-first, (b) Phase 4 builds the wrong thing (SaaS), (c) 16 weeks solo alongside FamilyLink and the day job is ~2× optimistic (portfolio-review rule), and (d) the repo never executed the plan's own week-1 items (pyproject, tests, pre-commit). Revised phasing:

| Phase | Content | Exit criteria | Effort |
|---|---|---|---|
| **0. Restructure & truth-up** | `src/bias_lab/` package layout per the plan's own diagram; pyproject; move metrics out of Streamlit into pure module; README truth-up (fix quick-start, status per feature); CI skeleton; screenshots → docs/img | `pip install -e .` works; CI green; README matches reality | 2–3 dgn |
| **1. Metric validation harness** (the moat) | Fixtures with hand-computed statistical parity/equalized odds/disparate impact; cross-check vs Fairlearn+Aequitas on 2 reference datasets (Adult, COMPAS); seed-reproducibility regression test | All metrics match reference implementations within tolerance, in CI — publish this table in the README as a sales asset | 1 wk |
| **2. Test-set generator v1** | Counterfactual/paired prompt generation (NL+EN), 5 attributes, seeded + versioned, human-review UI step, quality checks (R6) | Peter generates a CV-screening test set (100+ paired prompts) and approves quality | 2 wkn |
| **3. Runner + connectors** | OpenAI/Anthropic/HF-local/generic REST connectors with cost caps; batch execution; JSON/JUnit output; CLI `bias-lab run` | End-to-end: generated set → 2 different models → comparable scored reports; CI-gate demo fails a build on injected bias | 2 wkn |
| **4. Reports & pilot** | PDF report (config-hash, seeds, Art. 4a documentation section); run a real pilot on one of Peter's own models (eu-ai-act checker's LLM!) | Pilot report passes Peter's own professional review; decision point: corporate pilot vs. further build | 1–2 wkn |

**Dogfooding opportunity (do this):** Bias Lab's first real target = the **eu-ai-act-compliance-checker's own LLM** (Groq→xAI comparison is already planned — add a bias dimension to that same measurement cycle). One stone, three birds: real pilot, portfolio synergy, and marketing story.

## 5. Immediate next steps

1. Peter: approve refocus (core loop + corporate = package/CI, SaaS deferred) and the F3/F4/F8/F11/F12 → backlog move.
2. Fable: rewrite REQUIREMENTS.md v2 + acceptance criteria after approval.
3. Moshe: Phase 0+1 as one mandate (restructure + validation harness) — after FamilyLink fase 0 in the queue, or as filler task; Phase 1 is the highest-ROI day-work in this repo per the original portfolio review.
4. Boundary ADR with eu-ai-act-compliance-checker (shared Act knowledge base) — 30 min, prevents double maintenance.
