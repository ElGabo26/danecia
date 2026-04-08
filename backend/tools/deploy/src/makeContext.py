from __future__ import annotations

import json
import re
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_context(question: str, client: Optional[object] = None, detector_model: str = "qwen2.5-coder:3b", max_chars: int = 2200) -> Dict[str, Any]:
    """Construye el contexto desde catálogo, whitelist, reglas, glosario y ejemplos.

    Prioridad del contexto:
    1. Instrucciones generales
    2. Instrucciones del dominio
    3. Descripción de las tablas pertenecientes al dominio
    4. Joins válidos
    5. Columnas utilizables

    El recorte se realiza desde el final, preservando el orden de prioridad.
    """
    base_dir = Path(__file__).resolve().parent.parent
    catalog_dir = base_dir / "catalog"
    examples_dir = base_dir / "examples"

    def load_json(p: Path) -> Dict[str, Any]:
        return json.loads(Path(p).read_text(encoding="utf-8"))

    schema_catalog = load_json(catalog_dir / "schema_catalog.json")
    join_whitelist = load_json(catalog_dir / "join_whitelist.json")
    business_rules = load_json(catalog_dir / "business_rules.json")
    business_glossary = load_json(catalog_dir / "business_glossary.json")
    sql_examples = [
        json.loads(line)
        for line in (examples_dir / "sql_examples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tables_by_name = {t["table_name"].upper(): t for t in schema_catalog["tables"]}

    def strip_accents(v: str) -> str:
        return "".join(ch for ch in unicodedata.normalize("NFKD", v) if not unicodedata.combining(ch))

    def upper_ascii(v: str) -> str:
        return strip_accents(v).upper()

    def clean_text(v: Any) -> str:
        txt = re.sub(r"\s+", " ", str(v or "")).strip()
        return txt if txt and txt.lower() not in {"nan", "none", "null"} else "Sin descripción"

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

    # 2) Índice de columnas NOM_/DESC_ por tipo de entidad
    entity_column_index: Dict[str, List[str]] = {k: [] for k in business_glossary.get("entity_types", {})}
    for table in schema_catalog["tables"]:
        tname = table["table_name"].upper()
        for col in table.get("columns", []):
            cname = str(col["name"]).upper()
            if not (cname.startswith("NOM_") or cname.startswith("DESC_")):
                continue
            for entity_type, hints in business_glossary.get("entity_types", {}).items():
                if any(h in cname for h in hints):
                    entity_column_index[entity_type].append(f"{tname}.{cname}")

    # 3) Detector de entidades categóricas (LLM) solo si hay pista explícita
    command_words = {upper_ascii(w) for w in business_glossary.get("command_words", [])}
    entity_hints = business_glossary.get("entity_types", {})
    should_detect = any(hint.lower() in q_low for hints in entity_hints.values() for hint in hints)
    detected_entities: List[Dict[str, Any]] = []
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
    date_filter_sql = f"DF.FECHA BETWEEN '2025-01-01' AND '{current_date}'"

    # 4) Selección de tablas
    def is_domain_table(table: Dict[str, Any]) -> bool:
        table_domain = str(table.get("domain", "")).lower()
        tname = table["table_name"].upper()
        if domain in table_domain:
            return True
        if domain == "ventas" and tname.endswith("FAC_VENTA_TOTAL"):
            return True
        if domain == "finanzas" and "FAC_ESTRESULTADOS" in tname:
            return True
        if domain == "presupuesto" and tname.endswith("FAC_SGA_PRESUPUESTO_EXT"):
            return True
        if domain == "kardex" and tname.endswith("FAC_V_MQRY_KARDEX_AGRICOLA"):
            return True
        return False

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
    selected_tables: List[Dict[str, Any]] = []
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

    selected_table_names = [t["table_name"] for t in selected_tables]

    # 5) Joins válidos para tablas seleccionadas
    selected_joins = [
        j for j in join_whitelist.get("joins", [])
        if j["left_table"].upper() in seen_tables and j["right_table"].upper() in seen_tables
    ][:12]

    # 6) Reglas por prioridad
    general_rules = [
        item["rule"] for item in business_rules.get("rules", [])
        if item.get("domain", "global") == "global"
    ]
    domain_rules = [
        item["rule"] for item in business_rules.get("rules", [])
        if item.get("domain") == domain
    ]
    domain_rules.append(
        "Solo aplicar búsqueda semejante con UPPER(columna) LIKE cuando exista entidad categórica explícita detectada."
        if detected_entities else
        "No aplicar búsqueda por semejanza porque no se detectaron entidades categóricas explícitas."
    )
    domain_rules.append(f"Usar siempre JOIN con DIM_FECHA para filtrar fechas: {date_filter_sql}.")
    if domain == "ventas" and not explicit_autoconsumo:
        domain_rules.append("En informes de ventas no exponer AUTOCONSUMO como encabezado; usar VENTAS o TOTAL_VENTAS.")

    preferred_name_columns: Dict[str, List[str]] = {}
    allowed_like_columns: List[str] = []
    metric_set = {m.upper() for m in metrics}
    dimension_set = {d.upper() for d in dimensions}
    selected_column_descriptions: List[str] = []
    domain_table_descriptions: List[str] = []
    usable_columns_by_table: Dict[str, List[str]] = {}

    for table in selected_tables:
        tname = table["table_name"].upper()
        cols_meta = table.get("columns", [])
        cols = [str(c["name"]).upper() for c in cols_meta]
        preferred_name_columns[tname] = [
            str(c["name"]).upper() for c in cols_meta
            if str(c["name"]).upper().startswith(("NOM_", "DESC_"))
        ][:12]

        selected_col_names = preferred_name_columns[tname][:4]
        selected_col_names += [c for c in cols if c in metric_set][:3]
        selected_col_names += [c for c in cols if c in dimension_set][:3]
        selected_col_names = list(dict.fromkeys([c for c in selected_col_names if c]))[:10]
        usable_columns_by_table[tname] = selected_col_names

        if is_domain_table(table):
            tdesc = clean_text(table.get("description", ""))
            domain_table_descriptions.append(f"{tname}={tdesc}")
            desc_map = {str(c.get("name", "")).upper(): clean_text(c.get("description", "")) for c in cols_meta}
            for cname in selected_col_names:
                if cname in desc_map:
                    selected_column_descriptions.append(f"{tname}.{cname}={desc_map[cname]}")

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

    entity_bits = [
        f"{e['entity_type']}:{e['value']}=>{','.join(c.split('.')[-1] for c in e['columns'][:3])}"
        for e in detected_entities
    ] or ["sin_entidades_explicitas"]
    example_bits = [re.sub(r"\s+", " ", ex.get("sql", ""))[:180] for ex in examples]
    join_bits = [f"{j['left_table']}->{j['right_table']} ON {j['condition']}" for j in selected_joins[:8]]
    column_bits = [f"{t}[{','.join(cols)}]" for t, cols in usable_columns_by_table.items() if cols]

    # BLOQUES PRIORIZADOS
    general_instructions_block = "instr_general=" + " | ".join(general_rules[:6])
    domain_instructions_block = "instr_dominio=" + " | ".join(domain_rules[:6])
    table_descriptions_block = "tablas_dominio=" + " | ".join(domain_table_descriptions[:8])
    joins_block = "joins_validos=" + " | ".join(join_bits) if join_bits else "joins_validos=sin_joins_detectados"
    columns_block = "columnas_utilizables=" + " | ".join(column_bits) if column_bits else "columnas_utilizables=sin_columnas_priorizadas"

    # Bloque complementario no prioritario
    extra_block_parts = []
    if selected_column_descriptions:
        extra_block_parts.append("col_desc=" + " | ".join(selected_column_descriptions))
    extra_block_parts.append(f"dom={domain}")
    extra_block_parts.append(f"fechas={date_filter_sql}")
    extra_block_parts.append("entidades=" + " | ".join(entity_bits))
    if example_bits:
        extra_block_parts.append("ejemplos=" + " | ".join(example_bits))
    secondary_block = "; ".join(extra_block_parts)

    priority_blocks = [
        re.sub(r"\s+", " ", general_instructions_block).strip(),
        re.sub(r"\s+", " ", domain_instructions_block).strip(),
        re.sub(r"\s+", " ", table_descriptions_block).strip(),
        re.sub(r"\s+", " ", joins_block).strip(),
        re.sub(r"\s+", " ", columns_block).strip(),
    ]
    priority_text = "; ".join([b for b in priority_blocks if b and not b.endswith("=")])
    secondary_block = re.sub(r"\s+", " ", secondary_block).strip()

    if len(priority_text) >= max_chars:
        # Mantener orden y recortar únicamente dentro del último tramo disponible.
        context_text = priority_text[: max_chars - 3] + "..."
    else:
        remaining = max_chars - len(priority_text) - (2 if secondary_block else 0)
        if secondary_block and remaining > 0:
            if len(secondary_block) > remaining:
                secondary_block = secondary_block[: max(0, remaining - 3)] + "..."
            context_text = priority_text + "; " + secondary_block
        else:
            context_text = priority_text

    return {
        "domain": domain,
        "context_text": context_text,
        "selected_tables": selected_table_names,
        "selected_joins": selected_joins,
        "general_instructions": general_rules,
        "domain_instructions": domain_rules,
        "domain_table_descriptions": domain_table_descriptions,
        "usable_columns_by_table": usable_columns_by_table,
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
