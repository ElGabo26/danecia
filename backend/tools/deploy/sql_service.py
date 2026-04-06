from __future__ import annotations

from typing import Any, Dict, Optional

from src.sql_rag_pipeline import build_client, generate_sql, repair_sql, validate_sql


def run_sql_generation_flow(
    question: str,
    model: str = "qwen2.5-coder:3b",
    detector_model: Optional[str] = None,
    autorepair: bool = True,
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
) -> Dict[str, Any]:
    client = build_client(base_url=base_url, api_key=api_key)
    result = generate_sql(question=question, model=model, detector_model=detector_model or model, client=client)
    validation = validate_sql(result["sql"], result.get("validation_context"))

    repaired_sql = None
    repaired_validation = None
    no_sql_reason = result.get("no_sql_reason", "")

    if autorepair and not validation["valid"] and result["sql"] != "NO_SQL":
        repaired_sql = repair_sql(
            question=question,
            bad_sql=result["sql"],
            validation_errors=validation["errors"],
            model=model,
            detector_model=detector_model or model,
            client=client,
        )
        repaired_validation = validate_sql(repaired_sql, result.get("validation_context"))
        if repaired_sql == "NO_SQL":
            no_sql_reason = (repaired_validation or {}).get("no_sql_reason", "") or "La corrección automática no logró producir un SQL seguro."
    elif result["sql"] == "NO_SQL":
        no_sql_reason = result.get("no_sql_reason", "") or validation.get("no_sql_reason", "")

    final_sql = repaired_sql if repaired_sql and repaired_validation and repaired_validation["valid"] else result["sql"]
    final_validation = repaired_validation if repaired_validation and repaired_validation["valid"] else validation

    return {
        "ok": final_validation["valid"],
        "sql": final_sql,
        "validation": final_validation,
        "prompt": result["prompt"],
        "prompt_chars": result["prompt_chars"],
        "context": result["context"],
        "selected_tables": result["selected_tables"],
        "search_terms": result["search_terms"],
        "detected_entities": result["detected_entities"],
        "no_sql_reason": no_sql_reason,
    }
