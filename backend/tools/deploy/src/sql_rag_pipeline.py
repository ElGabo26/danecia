from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .makeContext import get_context
from .makeContextCorrection import makeContextCorrection

MAX_PROMPT_CHARS = 3000
DEFAULT_ENTITY_MODEL = "qwen2.5-coder:3b"
BASE_DIR = Path(__file__).resolve().parent.parent
CATALOG_DIR = BASE_DIR / "catalog"


def build_client(base_url: str = "http://localhost:11434/v1", api_key: str = "ollama"):
    from openai import OpenAI
    return OpenAI(base_url=base_url, api_key=api_key)


def _extract_json_block(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "{}"
    match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    return match.group(1) if match else "{}"


def _llm_generate(prompt: str, model: str, client: object, system_prompt: str) -> str:
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def _sanitize_sales_aliases(sql: str, question: str, explicit_autoconsumo: bool) -> str:
    if sql.strip() == "NO_SQL" or explicit_autoconsumo:
        return sql
    if "venta" not in question.lower():
        return sql
    replacements = [
        (r"\bAS\s+AUTOCONSUMOS\b", "AS VENTAS"),
        (r"\bAS\s+AUTOCONSUMO\b", "AS VENTAS"),
        (r"\bAS\s+VENTA_AUTOCONSUMO\b", "AS VENTAS"),
        (r"\bAS\s+TOTAL_AUTOCONSUMO\b", "AS TOTAL_VENTAS"),
        (r"\bAS\s+TOTAL_AUTOCONSUMOS\b", "AS TOTAL_VENTAS"),
    ]
    fixed = sql
    for pattern, repl in replacements:
        fixed = re.sub(pattern, repl, fixed, flags=re.IGNORECASE)
    return fixed


def build_prompt(question: str, context: Dict[str, Any], max_chars: int = MAX_PROMPT_CHARS) -> str:
    if context["detected_entities"]:
        entities_txt = "; ".join(
            f"{e['entity_type']}={e['value']} solo_en={','.join(col.split('.')[-1] for col in e['columns'][:4])}"
            for e in context["detected_entities"]
        )
        entity_rule = (
            "Solo si hay entidad explícita usa semejanza con MAYÚSCULAS: UPPER(col) LIKE '%VALOR%'. "
            "Restringe la búsqueda a las columnas candidatas de la entidad detectada. "
            f"Entidades:{entities_txt}."
        )
    else:
        entity_rule = "No se detectaron entidades categóricas explícitas; no uses LIKE ni filtros de texto semejantes."

    base = (
        "Genera solo SQL MySQL/SingleStore. Solo SELECT. No inventes tablas, columnas ni joins. "
        f"Si existe en la lista  de Joins permitidos realiza JOIN con DIM_FECHA, en otro caso utiliza las  columnas, mes  y anio  ppara  filtar los valores"
        "Los meses y  anios  siempre son numero enteros"
        f"{entity_rule} Usa columnas NOM_/DESC_ para mostrar nombres conocidos. "
        "Si el pedido es de ventas, evita alias AUTOCONSUMO salvo petición explícita; usa VENTAS o TOTAL_VENTAS. "
        "Si el contexto no alcanza, responde NO_SQL."
    )
    prompt = f"{base}\nCTX:{context['context_text']}\nQ:{question}\nSQL:"
    if len(prompt) > max_chars:
        print("contextoexcedido")
        excess = len(prompt) - max_chars
        reduced_ctx = context["context_text"][:-excess - 3] + "..." if excess + 3 < len(context["context_text"]) else context["context_text"][: max(0, max_chars - len(base) - len(question) - 20)]
        prompt = f"{base}\nCTX:{reduced_ctx}\nQ:{question}\nSQL:"
    print(prompt[:max_chars])
    return prompt[:max_chars]


def _infer_no_sql_reason(question: str, context: Dict[str, Any], validation_errors: Optional[List[str]] = None) -> str:
    q_low = question.lower().strip()
    if validation_errors:
        first = validation_errors[0]
        if "tabla" in first.lower() or "join" in first.lower():
            return "La consulta usa tablas o joins fuera del catálogo permitido."
        if "columna" in first.lower():
            return "La consulta usa columnas fuera del contexto permitido."
        if "fecha" in first.lower() or "dim_fecha" in first.lower():
            return "La consulta no respetó la regla de filtrado temporal con DIM_FECHA."
        if "like" in first.lower() or "semejanza" in first.lower():
            return "La consulta aplicó búsqueda textual sin una entidad categórica válida."
        return "La consulta no pasó la revisión de consistencia del esquema."
    if context["detected_entities"]:
        values = ", ".join(entity["value"] for entity in context["detected_entities"])
        return f"No fue posible mapear con seguridad {values} a columnas y joins válidos."
    if "estados financieros" in q_low and "plantaciones" in q_low:
        return "La solicitud es ambigua; falta precisar la métrica o la estructura exacta del informe financiero."
    return "No fue posible construir una consulta confiable solo con el contexto disponible."


def _review_sql_with_llm(sql: str, question: str, context: Dict[str, Any], client: object, model: str) -> Dict[str, Any]:
    schema_catalog = json.loads((CATALOG_DIR / "schema_catalog.json").read_text(encoding="utf-8"))
    join_whitelist = json.loads((CATALOG_DIR / "join_whitelist.json").read_text(encoding="utf-8"))
    selected_tables = set(context["selected_tables"])
    allowed_joins = [j for j in join_whitelist["joins"] if j["left_table"] in selected_tables and j["right_table"] in selected_tables]
    schema_subset = {
        "tables": [t for t in schema_catalog["tables"] if t["table_name"] in selected_tables]
    }
    review_prompt = (
        "Revisa si el SQL cumple el catálogo y responde solo JSON válido con claves: "
        "valid(bool), errors(list[str]), tables(list[str]), columns(list[str]), like_columns(list[str]), like_values(list[str]). "
        "Reglas: solo SELECT; no inventar tablas, columnas ni joins; usar solo las tablas del contexto; "
        "si hay entidades detectadas, LIKE solo en columnas candidatas NOM_/DESC_; si no hay entidades, no usar LIKE; "
        "usar DIM_FECHA para filtrar fechas cuando el dominio sea ventas o el contexto lo exija; una sola sentencia. "
        f"Question:{question}\n"
        f"ContextSummary:{context['context_text']}\n"
        f"DetectedEntities:{json.dumps(context['detected_entities'], ensure_ascii=False)}\n"
        f"AllowedLikeColumns:{json.dumps(context['allowed_like_columns'], ensure_ascii=False)}\n"
        f"AllowedJoins:{json.dumps(allowed_joins, ensure_ascii=False)}\n"
        f"SchemaSubset:{json.dumps(schema_subset, ensure_ascii=False)[:5000]}\n"
        f"SQL:{sql}"
    )
    try:
        raw = _llm_generate(review_prompt[:11000], model=model, client=client, system_prompt="Responde solo JSON válido.")
        payload = json.loads(_extract_json_block(raw))
        payload.setdefault("valid", False)
        payload.setdefault("errors", [])
        payload.setdefault("tables", [])
        payload.setdefault("columns", [])
        payload.setdefault("like_columns", [])
        payload.setdefault("like_values", [])
        return payload
    except Exception as exc:
        return {
            "valid": False,
            "errors": [f"No se pudo revisar el SQL con LLM: {exc}"],
            "tables": [],
            "columns": [],
            "like_columns": [],
            "like_values": [],
        }


def generate_sql(question: str, model: str = DEFAULT_ENTITY_MODEL, client: Optional[object] = None, temperature: float = 0.0, detector_model: Optional[str] = None) -> Dict[str, Any]:
    client = client or build_client()
    context = get_context(question=question, client=client, detector_model=detector_model or model, max_chars=3000)
    prompt = build_prompt(question, context, max_chars=MAX_PROMPT_CHARS)
    sql = _llm_generate(prompt, model=model, client=client, system_prompt="Devuelve únicamente SQL")
    sql = _sanitize_sales_aliases(sql, question, context["explicit_autoconsumo"])
    
    return {
        "question": question,
        "context": context["context_text"],
        "prompt": prompt,
        "prompt_chars": len(prompt),
        "selected_tables": context["selected_tables"],
        "search_terms": context["search_terms"],
        "detected_entities": context["detected_entities"],
        "validation_context": context,
        "sql": sql,
        
    }
    
def correct_sql(
    question: str,
    sql: str,
    error,
    model: str = DEFAULT_ENTITY_MODEL,
    client: Optional[object] = None
) -> Dict[str, Any]:
    client = client or build_client()

    error_text = str(error)
    
    print("ingreso al context")
    print(error_text)
    print(sql)
    context = makeContextCorrection(sql, error_text)
    prompt = context["context_text"]
    print("PROMPT GENERADO")
    corrected_sql = _llm_generate(
        prompt,
        model=model,
        client=client,
        system_prompt="Devuelve únicamente SQL"
    )

    return {
        "question": question,
        "context": context["context_text"],
        "prompt": prompt,
        "prompt_chars": len(prompt),
        "selected_tables": context["tables_used"],
        "search_terms": [],
        "detected_entities": [],
        "validation_context": context,
        "sql": corrected_sql,
    }

def validate_sql(sql: str, validation_context: Optional[Dict[str, Any]] = None, question: str = "", client: Optional[object] = None, model: str = DEFAULT_ENTITY_MODEL) -> Dict[str, Any]:
    context = validation_context or {}
    raw = (sql or "").strip()
    if raw == "NO_SQL":
        return {
            "valid": False,
            "errors": [],
            "tables": [],
            "columns": [],
            "sql": raw,
            "like_columns": [],
            "like_values": [],
            "no_sql_reason": context.get("no_sql_reason") or _infer_no_sql_reason(question, context),
        }
    if client is None:
        client = build_client()
    review = _review_sql_with_llm(raw, question=question, context=context, client=client, model=model)
    return {
        "valid": bool(review.get("valid", False)),
        "errors": review.get("errors", []),
        "tables": review.get("tables", []),
        "columns": review.get("columns", [])[:60],
        "sql": raw,
        "like_columns": review.get("like_columns", []),
        "like_values": review.get("like_values", []),
        "no_sql_reason": "" if review.get("valid") else _infer_no_sql_reason(question, context, review.get("errors", [])),
    }


def repair_sql(question: str, bad_sql: str, validation_errors: List[str], model: str = DEFAULT_ENTITY_MODEL, client: Optional[object] = None, detector_model: Optional[str] = None) -> str:
    client = client or build_client()
    context = get_context(question=question, client=client, detector_model=detector_model or model, max_chars=2200)
    if context["detected_entities"]:
        entity_rule = "; ".join(
            f"{e['entity_type']}={e['value']} solo_en={','.join(col.split('.')[-1] for col in e['columns'][:4])}"
            for e in context["detected_entities"]
        )
    else:
        entity_rule = "sin entidades explícitas; no uses LIKE"
    prompt = (
        "Corrige el SQL usando solo el contexto. Solo SELECT. "
        f"{entity_rule}. Usa columnas NOM_/DESC_ para nombres legibles. "
        "La búsqueda semejante solo aplica cuando hay entidad categórica explícita y debe usar MAYÚSCULAS. "
        "Si es un informe de ventas, no uses aliases AUTOCONSUMO salvo petición explícita. "
        "Si no puedes corregirlo con seguridad, devuelve NO_SQL. "
        f"Errores:{' | '.join(validation_errors)} CTX:{context['context_text']} SQL_Original:{bad_sql}\nSQL:"
    )
    fixed = _llm_generate(prompt[:MAX_PROMPT_CHARS], model=model, client=client, system_prompt="Devuelve únicamente SQL o NO_SQL.")
    return _sanitize_sales_aliases(fixed, question, context["explicit_autoconsumo"])
