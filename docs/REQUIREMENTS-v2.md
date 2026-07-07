# AI Bias Laboratory — Requirements v2.0 (Refocused)

**Status:** Approved refocus 2026-07-06 (Peter) · Supersedes REQUIREMENTS.md v1
**Product in one sentence:** genereer een op de use-case toegesneden bias-testset met prompts, draai die tegen elk AI-model, en lever gescoorde, reproduceerbare evidence — standalone (UI/CLI, self-hosted) én als CI-gate in een corporate pipeline.
**Scope rule:** everything serves the core loop (Describe → Generate → Run → Score → Report). Features that don't are backlog (§D).

---

## 0. Positioning — why this exists (the deployer's burden)

Bias Lab tests **AI systems as deployed**, not foundation models in isolation. The distinction that matters is not "own model vs. top-tier model" — it is **who carries the legal liability**, and that is almost never the model builder.

- **The obligation sits with the deployer.** Under the EU AI Act, the Art. 10 bias-examination duty for a high-risk system falls on the organisation that *deploys* it. An insurer running GPT-4 or Claude behind CV-screening or claims triage cannot hide behind "OpenAI already tested the model". It must show that in *its* use-case, on *its* population, in *its* language, the system shows no prohibited bias. Model builders test general fairness; the deployer must prove **contextual** fairness. That gap is the market.
- **Bias lives in the integration, not only the model.** What runs in production is model + system prompt + RAG context + fine-tuning + thresholds + use-case framing. Any of those layers can introduce or amplify bias, and the model builder never saw your prompt. Bias Lab tests the **composed system as it runs live** — which no one else can do for you.
- **"We tested it" is not audit evidence.** Vendor fairness reports are not accepted by an auditor or supervisor as compliance evidence for *your* system. You need reproducible, independent, in-context evidence with config-hash and seed — exactly what the core loop produces. The vendor claim is not a substitute.
- **Sharpest case: a top-tier model used as an assessment/decision engine.** Any tool that uses a foundation model to judge, classify or support decisions (e.g. Peter's own eu-ai-act-compliance-checker, or any RAG/agent system) inherits that model's bias *inside the judgement function itself* — a blind spot no one tests, because the builder never evaluated the model as "compliance classifier for the Dutch insurance sector". This is both a product use-case and Bias Lab's first dogfood pilot (§E).

**Primary customer:** the organisation with an Art. 10 obligation deploying a model — bought or built — in a high-risk context. Since nearly every corporate deployer buys its LLM yet still carries the full burden of proof, this is a larger and more legally compelling market than "test your own model".

---

## A. Functional requirements — core loop

### Describe
- **BL-FR-01 Use-case intake.** User describes the target model's task, domain, language (NL/EN) and selects protected attributes (gender, race/ethnicity, age, religion, disability — extensible).
  *AC: intake completable in <5 min via UI or a single YAML config (CI mode); config is versioned and hashed.*

### Generate
- **BL-FR-02 Test-set generation.** LLM-assisted generation of **paired/counterfactual prompts** tailored to the use-case: attribute-swap pairs (identical context, only the protected attribute varies), name-swap variants, and template-based sets. Seeded and reproducible.
  *AC: same seed + config ⇒ byte-identical test set; ≥100 pairs per attribute in <10 min; every pair traceable to its template + swap.*
- **BL-FR-03 Human review gate.** Generated sets are reviewable/editable before use; approval is recorded (who, when, set-hash).
  *AC: no test set can run against a target model without an approved status (CI mode: pre-approved sets referenced by hash).*
- **BL-FR-04 Set quality checks (automated).** Balance across attributes, duplicate detection, attribute-leakage check (the swapped attribute must be the only meaningful difference), grammaticality sampling.
  *AC: quality report attached to every set; generation fails hard when balance or leakage thresholds are violated.*
- **BL-FR-05 Benchmark import.** BBQ, CrowS-Pairs, WinoBias loadable as standard sets alongside generated ones (v1 F2, retained).
  *AC: benchmark results reproducible against published baselines for at least one open reference model.*

### Run
- **BL-FR-06 Model connectors.** OpenAI-compatible APIs (incl. xAI/Groq), Anthropic, local HuggingFace, generic REST. Uniform interface; batch execution.
  *AC: adding a connector touches one module; a run against 2 different providers yields structurally identical result files.*
- **BL-FR-07 Run controls.** Cost caps (max tokens/requests/€), rate limiting, retries with backoff, resumable runs, full request/response logging (redacted where needed).
  *AC: a run interrupted at 50% resumes without re-querying completed items; hitting the cost cap stops cleanly with a partial report.*

### Score
- **BL-FR-08 Metrics via reference implementations.** Statistical parity difference, equalized odds, disparate impact ratio computed **through Fairlearn/Aequitas** (wrap, never reimplement). Paired-prompt scoring: response-divergence metrics (refusal-rate delta, sentiment/outcome delta per pair).
  *AC: every metric cross-validated in CI against hand-computed fixtures AND reference-library output on Adult + COMPAS datasets, within documented tolerance.*
- **BL-FR-09 Thresholds & verdicts.** Configurable pass/fail thresholds per metric (e.g. disparate impact < 0.8 = fail) producing a single run verdict for CI gating.
  *AC: `bias-lab run --fail-under` exits non-zero on violation; thresholds recorded in the report.*

### Report
- **BL-FR-10 Outputs.** JSON (machine), JUnit-XML (CI), PDF (compliance) — all containing: config hash, seed, set hash + approval, model + version, metrics + verdicts, Art. 4a documentation section (why processing was necessary, safeguards applied).
  *AC: two runs with identical inputs produce identical JSON (timestamps excluded); PDF readable by a non-technical compliance officer (tested on one).*

## B. Non-functional requirements

- **BL-NFR-01 Deployment: local-first.** Full functionality self-hosted (Hetzner/on-prem); no phone-home. "Your data never leaves your infrastructure" is a product claim — make it verifiably true (documented egress: only the model APIs the user configures).
- **BL-NFR-02 Corporate integration.** Pure Python package (`pip install bias-lab`), CLI, REST API (FastAPI) — no multi-tenant SaaS in scope. Audit-grade logging (who ran what against which model), SSO-ready (reverse-proxy auth acceptable v1).
- **BL-NFR-03 Privacy & Art. 4a (Omnibus 2026).** When real datasets with special-category data are processed for bias detection: strict access controls, no third-party transfer, deletion after correction/retention expiry, necessity documentation — implemented as product features, surfaced in the PDF report. GDPR posture is Phase 1, not Phase 2.
- **BL-NFR-04 Reproducibility.** Every artifact (set, run, report) carries seed + config-hash; a regression test re-runs a logged config and compares (v1 NF5, now testable).
- **BL-NFR-05 Validation transparency.** The metric-validation table (our output vs. Fairlearn/Aequitas on reference data) is published in the README/product — QA as sales asset.
- **BL-NFR-06 Readability.** Reports understandable for legal/compliance (v1 NF6, retained); NL and EN.
- **BL-NFR-07 Regulatory evolvability.** EU AI Act knowledge (Art. 10, Art. 4a, timelines) in versioned YAML, **shared with eu-ai-act-compliance-checker** (single knowledge base — boundary ADR).

## C. Product boundary (ADR to record)

**Bias Lab = evidence generation** (test, measure, prove). **eu-ai-act-compliance-checker = assessment** (classify, gap-analyse, report obligations). Shared component: the Act knowledge base. Bias Lab's reports are *inputs* to the checker's Art. 10 evidence requirements — that's the integration, not feature overlap.

## D. Backlog (moved out of scope, deliberate — v1 features F3/F4/F8/F11/F12)

Bias propagation visualization · mitigation simulation (4 techniques) · LIME/SHAP explainability · GPAI adversarial red-teaming · AIA/NIST workflow · SaaS multi-tenancy & billing. Revisit only after the core loop has a paying or piloting user.

## E. Traceability

IDs (BL-FR-xx / BL-NFR-xx) referenced in commits, issues and tests. Phased delivery per docs/review-and-refocus-plan.md §4 (Phase 0 restructure → 1 validation harness → 2 generator → 3 runner → 4 reports & pilot). First pilot target: the eu-ai-act-compliance-checker's own LLM, in the same measurement cycle as the Groq→xAI comparison.
