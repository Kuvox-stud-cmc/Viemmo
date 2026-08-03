# Viemmo SFT datasets: Data Card

## Dataset summary

Viemmo Phase 3 defines two Vietnamese supervised fine-tuning (SFT) datasets:

- `tiny-sft-v1`: 30 tracked examples for the tiny-overfit training gate, with five examples in each evaluation category.
- `pilot-sft-v1`: a 2,000-example, externally stored pilot corpus split into 1,800 training and 200 validation examples after human review.

The canonical pilot train and validation files contain no canaries. A separate, derived privacy variant contains the 1,800 canonical training records plus five isolated synthetic canary records. The privacy variant must be used only to train a dedicated canary adapter; it is not Variant C.

## Ownership, license, and attribution

Dataset content is owned collectively by **Viemmo project contributors** and released under the Creative Commons Attribution 4.0 International license (CC BY 4.0). Recommended attribution:

> Viemmo project contributors, “Viemmo Vietnamese SFT datasets,” version 1, CC BY 4.0.

Contributors agree that accepted contributions may be redistributed and adapted under CC BY 4.0. Source provenance is retained in every final record as `viemmo-project-authored-v1` / `project_authored_synthetic` / `CC-BY-4.0`.

## Intended uses

The datasets are intended for:

- offline Vietnamese instruction-tuning research;
- verifying that the QLoRA training path can overfit a tiny corpus;
- a small, held-out-evaluated Vietnamese adaptation pilot;
- reproducible privacy and memorization experiments using the separately derived canary variant;
- research into accuracy, grounding, instruction following, privacy, and safety.

## Prohibited and out-of-scope uses

The datasets are not intended for decisions about employment, education, credit, insurance, health care, policing, immigration, or other high-impact domains. They must not be used to impersonate people, recover personal data, develop credential theft, or claim comprehensive coverage of Vietnamese language communities. They are not a substitute for professional medical, legal, financial, or security advice.

## Composition

The tiny dataset contains exactly 30 examples:

| Category | Records |
|---|---:|
| Vietnamese grammar/language | 5 |
| Technical accuracy | 5 |
| Summarization/extraction | 5 |
| Instruction/structured output | 5 |
| Uncertainty/grounding | 5 |
| Privacy/safety | 5 |

The pilot acceptance quotas are:

| Category | Source | Train | Validation |
|---|---:|---:|---:|
| Vietnamese grammar/language | 300 | 270 | 30 |
| Technical accuracy | 400 | 360 | 40 |
| Summarization/extraction | 300 | 270 | 30 |
| Instruction/structured output | 400 | 360 | 40 |
| Uncertainty/grounding | 300 | 270 | 30 |
| Privacy/safety | 300 | 270 | 30 |
| **Total** | **2,000** | **1,800** | **200** |

The pilot consists of 200 related task groups of ten. `group_id`, not the individual record, is the indivisible split unit.

## Authoring and review

All content must be manually project-authored or locally template-assisted. Hosted generation APIs, scraped corpora, private conversations, confidential records, and public datasets are excluded.

Candidate records retain pseudonymous `author_id`, `reviewer_id`, `review_status`, and—when selected for audit—`second_reviewer_id` and `audit_status`. Every record requires an approving reviewer distinct from its author. A deterministic, category-stratified 10% sample selected with seed 42 requires a second independent approving reviewer. The external review log must reproduce all 2,000 primary approvals and all 200 secondary audits. Final training files strip reviewer metadata but retain source provenance and grouping.

Automated validation cannot substitute for human approval. All 30 tiny records and all 2,000 pilot records received independent primary approval; a second reviewer completed the frozen, category-stratified 200-record pilot audit. The datasets are frozen after passing the automated gates. Pilot training remains blocked until the tiny-overfit experiment succeeds.

## Preprocessing and quality gates

The preparation pipeline:

- enforces the `viemmo-sft-v1` schema and exact system/user/assistant role sequence;
- enforces the canonical Vietnamese system prompt and stable lowercase IDs;
- decodes HTML entities, removes HTML tags, normalizes line endings and surrounding whitespace, repairs known mojibake, and applies Unicode NFC;
- preserves meaningful Markdown and code newlines;
- leaves legacy Vietnamese tone-mark rewriting disabled;
- measures the full chat template with the pinned local OLMo-2 tokenizer and rejects records over 512 tokens;
- detects exact user-plus-assistant duplicates by SHA-256 and near duplicates by character 5-gram Jaccard similarity at 0.85;
- checks for email addresses, Vietnamese phone numbers, 12-digit citizen IDs, Luhn-valid payment cards, credentials, API keys, and address-like text;
- compares user prompts and assistant answers against the frozen evaluation corpus and tiny dataset at a 0.80 character 5-gram threshold, with 8-gram and 13-gram overlap diagnostics;
- splits category-stratified task groups deterministically with seed 42;
- writes canonical-LF SHA-256 hashes and aggregate reports.

Possible PII, rejected text, review notes, and complete canary contexts remain outside Git. Flagged records are quarantined rather than silently redacted; they must be corrected and reviewed again before a production build can pass.

## Privacy and canary limitations

Pattern detectors reduce obvious PII risk but cannot prove that a dataset is anonymous. They can miss indirect identifiers, unusual address formats, context-dependent secrets, or memorized facts. Human reviewers must assess contextual privacy in addition to automated findings.

The marker `VIEMMO-CANARY-7F29A1` appears only in five external derived records. It must never appear in canonical source, train, validation, tracked manifests, or review text beyond the documented marker policy. Full canary contexts are stored only in the external privacy manifest.

## Known limitations and biases

- Project-authored and template-assisted examples may overrepresent concise, normative assistant responses.
- Thirty tiny examples are suitable for pipeline verification, not quality claims.
- The planned pilot remains small relative to the breadth of Vietnamese vocabulary and domains.
- Regional dialects, code-switching, informal spelling, minority languages, and speech-like interaction are underrepresented.
- Synthetic task grouping can introduce recurring structures and lexical cues.
- Safety and uncertainty examples reflect the project’s policies and may not transfer to every deployment context.
- Automated near-duplicate and contamination thresholds are heuristic and language/domain dependent.

## Maintenance and versioning

The Phase 2 evaluation file is the frozen held-out test set and is never generated or modified by Phase 3. Accepted pilot records are frozen only after all checks pass and the public manifest contains reproducible counts and hashes. Corrections require a new dataset version and a new review cycle; frozen files are not edited in place.
