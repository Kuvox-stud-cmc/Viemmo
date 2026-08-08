# Viemmo dataset workspace

Only the small `tiny-sft-v1` training set, public manifests, licenses, and aggregate reports are tracked in Git. The 2,000-record pilot source, review history, quarantine, final splits, and full canary contexts live under `LLM_STORAGE_ROOT`.

## Canonical record

```json
{
  "schema_version": "viemmo-sft-v1",
  "id": "pilot-tech-001-01",
  "category": "technical_accuracy",
  "group_id": "pilot-tech-001",
  "source": {
    "id": "viemmo-project-authored-v1",
    "type": "project_authored_synthetic",
    "license": "CC-BY-4.0"
  },
  "messages": [
    {"role": "system", "content": "Bạn là một trợ lý tiếng Việt chính xác và thận trọng."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

Authoring records additionally require `author_id`, `reviewer_id`, and `review_status`. Records in the seed-42 audit sample also require `second_reviewer_id` and `audit_status`. Reviewer IDs must be pseudonymous, and all decisions must also be present in the external review log. Final train/validation files strip these fields.

## External layout

```text
Viemmo-storage/datasets/pilot-sft-v1/
├── source/
│   ├── project-authored-v1.jsonl
│   └── review-log.jsonl
├── quarantine/
├── train.jsonl
├── validation.jsonl
└── privacy/
    ├── train-canary.jsonl
    └── canary-manifest.json
```

Do not commit anything from this external directory. `quarantine/` may contain rejected text or possible PII and therefore requires the same access controls as source authoring data.

## Validation and reproduction

During authoring, run the automated tiny checks using only the pinned local tokenizer:

```powershell
python scripts/prepare_dataset.py validate --allow-unreviewed
```

For final validation, place 30 primary approval events in
`$env:LLM_STORAGE_ROOT/datasets/tiny-sft-v1/review-log.jsonl`; each event must
contain `record_id`, `stage: primary`, `decision: approved`, and distinct
`author_id`/`reviewer_id`. Then run `python scripts/prepare_dataset.py validate`
without the authoring-only flag.

Build the pilot after human authoring and review are complete:

```powershell
$env:LLM_STORAGE_ROOT = "D:\Viemmo\Viemmo-storage"
python scripts/prepare_dataset.py build --config configs/datasets/pilot-sft-v1.yaml
```

The original 2,000-record review draft can be reproduced without a hosted API:

```powershell
python scripts/generate_pilot_drafts.py `
  --output "$env:LLM_STORAGE_ROOT/datasets/pilot-sft-v1/source/project-authored-v1.jsonl" `
  --tokenizer "$env:LLM_STORAGE_ROOT/upstream/OLMo-2-0425-1B-Instruct"
```

Generation always writes `review_status: pending` and `reviewer_id:
reviewer-unassigned`. Reviewers should correct or replace a bad record in its
existing category/group slot instead of simply deleting it, because production
acceptance requires the exact quotas and ten records per group.

The build fails if any record is unapproved, outside quotas, over 512 tokens, possible PII, a duplicate, contaminated by evaluation/tiny data, or assigned to an incorrect audit sample. A failure that creates `quarantine/rejected.jsonl` must be corrected in source and independently reviewed again; the pipeline never silently redacts or accepts rejected records.

The validation split uses whole task groups and seed 42. The canary-derived train file is deterministic and contains five additional records, while canonical train and validation remain canary-free.

## Contribution and attribution

By contributing accepted dataset content, contributors agree to license it under CC BY 4.0 as part of “Viemmo project contributors.” Record the contributor through the project’s pseudonymous author ID and retain the private attribution mapping according to project policy. Recommended public attribution is:

> Viemmo project contributors, “Viemmo Vietnamese SFT datasets,” version 1, CC BY 4.0.

See [LICENSE-CC-BY-4.0.txt](LICENSE-CC-BY-4.0.txt) and the project [Data Card](../docs/data-card.md).
