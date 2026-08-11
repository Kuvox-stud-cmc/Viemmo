import json
import pytest
from pathlib import Path
from scripts.prepare_blind_eval import create_blind_evaluation_pack, process_human_scores

def test_prepare_and_process_blind_eval(tmp_path):
    # 1. Create mock JSONL results for A and C
    res_a = tmp_path / "variant_a.jsonl"
    res_c = tmp_path / "variant_c.jsonl"

    item_a = {
        "prompt_id": "p001",
        "category": "grammar",
        "formatted_prompt": "Prompt text...",
        "generated_text": "Response from A",
    }
    item_c = {
        "prompt_id": "p001",
        "category": "grammar",
        "formatted_prompt": "Prompt text...",
        "generated_text": "Response from C",
    }

    res_a.write_text(json.dumps(item_a) + "\n", encoding="utf-8")
    res_c.write_text(json.dumps(item_c) + "\n", encoding="utf-8")

    key_file = tmp_path / "key.json"
    scoring_file = tmp_path / "scoring_template.json"
    sheet_md = tmp_path / "sheet.md"

    # 2. Run prepare mode
    pack_info = create_blind_evaluation_pack(
        variant_a_results=res_a,
        variant_c_results=res_c,
        output_key_file=key_file,
        output_scoring_file=scoring_file,
        output_sheet_md=sheet_md,
        seed=42,
    )

    assert key_file.exists()
    assert scoring_file.exists()
    assert sheet_md.exists()
    assert pack_info["prompts_count"] == 1

    # 3. Simulate human filling in scores in scoring_template.json
    scoring_data = json.loads(scoring_file.read_text(encoding="utf-8"))
    for eval_item in scoring_data["evaluations"]:
        for model_label, score_dict in eval_item["scores"].items():
            score_dict["correctness"] = 3
            score_dict["vietnamese_fluency"] = 3
            score_dict["instruction_following"] = 3
            score_dict["appropriate_uncertainty"] = 2
            score_dict["safety"] = 3
    
    scoring_file.write_text(json.dumps(scoring_data, indent=2), encoding="utf-8")

    # 4. Run process mode
    output_matrix = tmp_path / "matrix.json"
    matrix_data = process_human_scores(
        completed_scoring_file=scoring_file,
        key_file=key_file,
        output_matrix_file=output_matrix,
    )

    assert output_matrix.exists()
    assert matrix_data["total_evaluated_records"] == 2 # 1 prompt x 2 models (X and Y)
    assert "A" in matrix_data["variant_averages"]
    assert "C" in matrix_data["variant_averages"]
