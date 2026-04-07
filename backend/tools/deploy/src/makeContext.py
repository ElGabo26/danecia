from __future__ import annotations

import json
import re
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional


def get_context(question: str, client: Optional[object] = None, detector_model: str = "qwen2.5-coder:3b", max_chars: int = 2200) -> Dict[str, Any]:
    """Construye el contexto desde catálogo, whitelist, reglas, glosario y ejemplos.

    Retorna un paquete listo para prompt y validación.
    Ahora también agrega las descripciones de las columnas seleccionadas de las tablas del dominio.
    """
    base_dir = Path(__file__).resolve().parent.parent
    catalog_dir = base_dir / "catalog"
    examples_dir = base_dir / "examples"

    load_json = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
    schema_catalog = load_json(catalog_dir / "schema_catalog.json")
    join_whitelist = load_json(catalog_dir / "join_whitelist.json")
    business_rules = load_json(catalog_dir / "business_rules.json")
    business_glossary = load_json(catalog_dir / "business_glossary.json")
    sql_examples = [json.loads(line) for line in (examples_dir / "sql_examples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    tables_by_name = {t["table_name"].upper(): t for t in schema_catalog["tables"]}

    strip_accents = lambda v: "".join(ch for ch in unicodedata.normalize("NFKD", v) if not unicodedata.combining(ch))
    upper_ascii = lambda v: strip_accents(v).upper()
    tokens = re.findall(r"[a-zA-ZáéíóúÁÉÍÓÚñÑ0-9_]+", question.lower())
    q_low = question.lower()
    current_date = date.today().isoformat()
    explicit_autoconsumo = "autoconsumo" in q_low

    # 1) Dominio, métricas y dimensiones desde glosario
    domain_scores = {k: 0 for k in business_glossary.get("domains", {})}
    for dom, words in business_glossary.get("domains", {}).items():
        for w in words:
            if w in q_low:
                domain_scores[dom] += 2
    domain = max(domain_scores.items(), key=lambda x: x[1])[0] if domain_scores else "ventas"
    if max(domain_scores.values(), default=0) == 0:
        domain = "ventas"

    metrics = sorted({col for word, cols in business_glossary.get("metrics", {}).items() if word in q_low for col in cols})
    dimensions = sorted({col for word, cols in business_glossary.get("dimensions", {}).items() if word in q_low for col in cols})
    if not metrics and domain == "ventas":
        metrics = ["VENTA_AUTOCONSUMO"]

    # 2) Índice de columnas NOM_/DESC_ por tipo de entidad a partir del catálogo
    entity_column_index = {k: [] for k in business_glossary.get("entity_types", {})}
    for table in schema_catalog["tables"]:
        tname = table["table_name"].upper()
        for col in table.get("columns", []):
            cname = str(col["name"]).upper()
            if not (cname.startswith("NOM_") or cname.startswith("DESC_")):
                continue
            for entity_type, hints in business_glossary.get("entity_types", {}).items():
                if any(h in cname for h in hints):
                    entity_column_index[entity_type].append(f"{tname}.{cname}")

    # 3) Detector de entidades categóricas (LLM) solo si hay una pista explícita
    command_words = {upper_ascii(w) for w in business_glossary.get("command_words", [])}
    entity_hints = business_glossary.get("entity_types", {})
    should_detect = any(hint.lower() in q_low for hints in entity_hints.values() for hint in hints)
    detected_entities = []
    if client is not None and should_detect:
        detector_prompt = (
            "Detecta solo entidades categóricas explícitas y devuelve JSON puro como lista de objetos "
            "{entity_type,value}. Tipos válidos: marca,familia,empresa,unidad_negocio,cuenta,cliente,producto,plantacion,canal,ruta,zona,agencia,material,origen. "
            "Ignora verbos de instrucción, palabras genéricas y dominios como INFORME, VENTAS, PLANTACIONES, ESTADOS FINANCIEROS. "
            f"Pregunta:{question}"
        )
        try:
            response = client.chat.completions.create(
                model=detector_model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": "Responde solo JSON válido."},
                    {"role": "user", "content": detector_prompt[:1200]},
                ],
            )
            raw = (response.choices[0].message.content or "[]").strip()
            match = re.search(r"(\[.*\]|\{.*\})", raw, flags=re.DOTALL)
            payload = json.loads(match.group(1) if match else "[]")
            payload = payload if isinstance(payload, list) else []
            generic_ban = {"INFORME", "VENTAS", "VENTA", "PLANTACIONES", "PLANTACION", "ESTADOS", "FINANCIEROS", "GENERA", "MUESTRA", "DAME", "TRAE"}
            seen = set()
            for item in payload:
                entity_type = str(item.get("entity_type", "")).strip().lower()
                value = upper_ascii(str(item.get("value", "")).strip())
                if entity_type not in entity_hints or not value or value in generic_ban or value in command_words:
                    continue
                cols = entity_column_index.get(entity_type, [])
                key = (entity_type, value)
                if cols and key not in seen:
                    seen.add(key)
                    detected_entities.append({
                        "entity_type": entity_type,
                        "raw_text": str(item.get("value", "")).strip(),
                        "value": value,
                        "columns": cols,
                        "confidence": "media",
                        "source": "llm_detector",
                    })
        except Exception:
            detected_entities = []

    search_terms = [e["value"] for e in detected_entities]

    # 4) Filtro temporal por defecto
    date_filter_sql = f"DF.FECHA BETWEEN '2025-01-01' AND '{current_date}'"

    # 5) Selección de tablas
    def table_score(table: Dict[str, Any]) -> int:
        score = 0
        tname = table["table_name"].upper()
        cols = {str(c["name"]).upper() for c in table.get("columns", [])}
        desc = str(table.get("description", "")).lower()
        if domain in str(table.get("domain", "")).lower():
            score += 20
        if tname.endswith("DIM_FECHA"):
            score += 35
        if domain == "ventas" and tname.endswith("FAC_VENTA_TOTAL"):
            score += 18
        if domain == "finanzas" and "FAC_ESTRESULTADOS" in tname:
            score += 18
        if domain == "presupuesto" and tname.endswith("FAC_SGA_PRESUPUESTO_EXT"):
            score += 18
        if domain == "kardex" and tname.endswith("FAC_V_MQRY_KARDEX_AGRICOLA"):
            score += 18
        score += sum(8 for m in metrics if m.upper() in cols)
        score += sum(5 for d in dimensions if d.upper() in cols)
        score += sum(1 for tok in tokens if tok in desc or tok.upper() in tname)
        for entity in detected_entities:
            if any(ref.startswith(tname + ".") for ref in entity["columns"]):
                score += 20
        if any(c.startswith(("NOM_", "DESC_")) for c in cols):
            score += 2
        return score

    scored_tables = sorted(schema_catalog["tables"], key=table_score, reverse=True)
    selected_tables = []
    seen_tables = set()
    for table in scored_tables:
        s = table_score(table)
        if s <= 0 and len(selected_tables) >= 2:
            continue
        tname = table["table_name"].upper()
        if tname not in seen_tables:
            seen_tables.add(tname)
            selected_tables.append(table)
        if len(selected_tables) >= 6:
            break
    if "DDM_ERP.DIM_FECHA" not in seen_tables and "DDM_ERP.DIM_FECHA" in tables_by_name:
        selected_tables.append(tables_by_name["DDM_ERP.DIM_FECHA"])
        seen_tables.add("DDM_ERP.DIM_FECHA")
    for entity in detected_entities:
        for ref in entity["columns"]:
            tname = ref.rsplit(".", 1)[0]
            if tname not in seen_tables and tname in tables_by_name:
                selected_tables.append(tables_by_name[tname])
                seen_tables.add(tname)

    # 6) Joins, reglas, ejemplos y columnas de nombres
    selected_table_names = [t["table_name"] for t in selected_tables]
    selected_joins = [
        j for j in join_whitelist.get("joins", [])
        if j["left_table"].upper() in seen_tables and j["right_table"].upper() in seen_tables
    ][:12]

    selected_rules = [
        item["rule"] for item in business_rules.get("rules", [])
        if item.get("domain", "global") in {"global", domain}
    ]
    selected_rules.append(
        "Solo aplicar búsqueda semejante con UPPER(columna) LIKE cuando exista entidad categórica explícita detectada."
        if detected_entities else
        "No aplicar búsqueda por semejanza porque no se detectaron entidades categóricas explícitas."
    )

    preferred_name_columns = {}
    allowed_like_columns = []
    selected_column_descriptions = {}
    metric_set = {m.upper() for m in metrics}
    dimension_set = {d.upper() for d in dimensions}
    for table in selected_tables:
        tname = table["table_name"].upper()
        all_columns = table.get("columns", [])
        preferred_name_columns[tname] = [
            str(c["name"]).upper() for c in all_columns
            if str(c["name"]).upper().startswith(("NOM_", "DESC_"))
        ][:12]

        keep_cols = preferred_name_columns[tname][:4]
        keep_cols += [str(c["name"]).upper() for c in all_columns if str(c["name"]).upper() in metric_set][:3]
        keep_cols += [str(c["name"]).upper() for c in all_columns if str(c["name"]).upper() in dimension_set][:3]
        keep_cols = list(dict.fromkeys([c for c in keep_cols if c]))[:8]

        if domain in str(table.get("domain", "")).lower() or tname.endswith("DIM_FECHA"):
            descriptions = []
            for col in all_columns:
                cname = str(col.get("name", "")).upper()
                if cname in keep_cols:
                    cdesc = str(col.get("description", "") or "Sin descripción")
                    descriptions.append({"name": cname, "description": cdesc})
            if descriptions:
                selected_column_descriptions[tname] = descriptions

    for entity in detected_entities:
        allowed_like_columns.extend([ref.split(".")[-1] for ref in entity["columns"]])
    allowed_like_columns = sorted(dict.fromkeys(allowed_like_columns))

    ranked_examples = []
    for ex in sql_examples:
        score = 0
        if ex.get("domain") == domain:
            score += 5
        if any(tok in ex.get("question", "").lower() for tok in tokens[:5]):
            score += 2
        ranked_examples.append((score, ex))
    examples = [ex for score, ex in sorted(ranked_examples, key=lambda x: x[0], reverse=True)[:2] if score > 0]

    # 7) Contexto compacto con descripción de columnas seleccionadas
    table_bits = []
    column_desc_bits = []
    for table in selected_tables:
        tname = table["table_name"].upper()
        cols = [str(c["name"]).upper() for c in table.get("columns", [])]
        keep_cols = preferred_name_columns[tname][:4] + [c for c in cols if c.upper() in metric_set][:3] + [c for c in cols if c.upper() in dimension_set][:3]
        keep_cols = list(dict.fromkeys([c for c in keep_cols if c]))[:8]
        if keep_cols:
            table_bits.append(f"{tname}[{','.join(keep_cols)}]")
        else:
            table_bits.append(tname)
        if tname in selected_column_descriptions:
            desc_fragments = [f"{item['name']}:{item['description']}" for item in selected_column_descriptions[tname][:6]]
            if desc_fragments:
                column_desc_bits.append(f"{tname}=>{' | '.join(desc_fragments)}")

    join_bits = [f"{j['left_table']}->{j['right_table']} ON {j['condition']}" for j in selected_joins[:8]]
    example_bits = [re.sub(r"\s+", " ", ex.get("sql", ""))[:220] for ex in examples]
    entity_bits = [f"{e['entity_type']}:{e['value']}=>{','.join(c.split('.')[-1] for c in e['columns'][:3])}" for e in detected_entities] or ["sin_entidades_explicitas"]

    context_text = (
        f"dom={domain}; fechas={date_filter_sql}; tablas={' | '.join(table_bits)}; "
        f"desc_cols={' || '.join(column_desc_bits)}; joins={' | '.join(join_bits)}; "
        f"entidades={' | '.join(entity_bits)}; reglas={' | '.join(selected_rules[:6])}; ejemplos={' | '.join(example_bits)}"
    )
    context_text = re.sub(r"\s+", " ", context_text).strip()
    if len(context_text) > max_chars:
        context_text = context_text[: max_chars - 3] + "..."

    return {
        "domain": domain,
        "context_text": context_text,
        "selected_tables": selected_table_names,
        "selected_joins": selected_joins,
        "selected_rules": selected_rules[:10],
        "examples": examples,
        "search_terms": search_terms,
        "detected_entities": detected_entities,
        "allowed_like_columns": allowed_like_columns,
        "preferred_name_columns": preferred_name_columns,
        "selected_column_descriptions": selected_column_descriptions,
        "date_filter_sql": date_filter_sql,
        "explicit_autoconsumo": explicit_autoconsumo,
        "metrics": metrics,
        "dimensions": dimensions,
    }
