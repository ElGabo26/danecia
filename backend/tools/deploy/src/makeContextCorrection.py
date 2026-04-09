
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def makeContextCorrection(
    sql_query: str,
    error_message: str,
    schema_catalog_path: Optional[str] = None,
    join_whitelist_path: Optional[str] = None,
    valores_rag_json_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Genera contexto de corrección para una consulta SQL fallida.

    Parámetros
    ----------
    sql_query : str
        Consulta SQL a revisar.
    error_message : str
        Error devuelto por el motor SQL.
    schema_catalog_path : str | None
        Ruta a schema_catalog_updated.json o schema_catalog.json.
    join_whitelist_path : str | None
        Ruta a join_whitelist.json.
    valores_rag_json_path : str | None
        Ruta a valoresRag_descripciones.json.

    Retorna
    -------
    Dict[str, Any]
        Diccionario con:
        - tables_used
        - columns_used_by_table
        - where_conditions
        - detected_error_type
        - correction_context
        - context_text
    """
    base_dir = Path(__file__).resolve().parent

    if schema_catalog_path is None:
        cand = [
            base_dir / "catalog" / "schema_catalog.json",
        ]
        schema_catalog_path = next((str(p) for p in cand if p.exists()), None)

    if join_whitelist_path is None:
        cand = [
            base_dir / "join_whitelist.json",
            base_dir / "catalog" / "join_whitelist.json",
        ]
        join_whitelist_path = next((str(p) for p in cand if p.exists()), None)

    if valores_rag_json_path is None:
        cand = [
            base_dir / "valoresRag_descripciones.json",
            base_dir / "catalog" / "valoresRag_descripciones.json",
        ]
        valores_rag_json_path = next((str(p) for p in cand if p.exists()), None)

    if not schema_catalog_path:
        raise FileNotFoundError("No se encontró schema_catalog.json o schema_catalog_updated.json")
    if not join_whitelist_path:
        raise FileNotFoundError("No se encontró join_whitelist.json")
    if not valores_rag_json_path:
        raise FileNotFoundError("No se encontró valoresRag_descripciones.json")

    with open(schema_catalog_path, "r", encoding="utf-8") as f:
        schema_catalog = json.load(f)
    with open(join_whitelist_path, "r", encoding="utf-8") as f:
        join_whitelist = json.load(f)
    with open(valores_rag_json_path, "r", encoding="utf-8") as f:
        valores_rag = json.load(f)

    sql = sql_query.strip()
    sql_norm = re.sub(r"\s+", " ", sql).strip()
    sql_up = sql_norm.upper()
    err_low = (error_message or "").lower()

    # ---------- Índices ----------
    table_catalog: Dict[str, Dict[str, Any]] = {}
    table_short_index: Dict[str, str] = {}
    column_index_by_table: Dict[str, Dict[str, Dict[str, Any]]] = {}
    global_column_to_tables: Dict[str, List[str]] = {}

    for table in schema_catalog.get("tables", []):
        tname = str(table.get("table_name", "")).upper()
        if not tname:
            continue
        table_catalog[tname] = table
        table_short_index[tname.split(".")[-1]] = tname
        column_index_by_table[tname] = {}
        for col in table.get("columns", []):
            cname = str(col.get("name", "")).upper()
            if not cname:
                continue
            column_index_by_table[tname][cname] = col
            global_column_to_tables.setdefault(cname, []).append(tname)

    values_index: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for item in valores_rag:
        tname = str(item.get("table_name", "")).upper()
        cname = str(item.get("column_name", "")).upper()
        if not tname or not cname:
            continue
        values_index.setdefault(tname, {})[cname] = item

    # ---------- Helpers ----------
    def clean_identifier(token: str) -> str:
        return token.strip().strip(",;()").strip("`").strip('"').upper()

    def table_full_name(token: str) -> Optional[str]:
        token = clean_identifier(token)
        if token in table_catalog:
            return token
        return table_short_index.get(token)

    def split_sql_sections(query: str) -> Dict[str, str]:
        sections = {"select": "", "from": "", "where": "", "group_by": "", "order_by": ""}
        q = re.sub(r"\s+", " ", query).strip()
        up = q.upper()

        patterns = {
            "select": r"\bSELECT\b(.*?)\bFROM\b",
            "from": r"\bFROM\b(.*?)(\bWHERE\b|\bGROUP BY\b|\bORDER BY\b|\bLIMIT\b|$)",
            "where": r"\bWHERE\b(.*?)(\bGROUP BY\b|\bORDER BY\b|\bLIMIT\b|$)",
            "group_by": r"\bGROUP BY\b(.*?)(\bORDER BY\b|\bLIMIT\b|$)",
            "order_by": r"\bORDER BY\b(.*?)(\bLIMIT\b|$)",
        }
        for key, pattern in patterns.items():
            m = re.search(pattern, q, flags=re.IGNORECASE | re.DOTALL)
            if m:
                sections[key] = m.group(1).strip()
        return sections

    def extract_alias_map(query: str) -> Dict[str, str]:
        alias_map: Dict[str, str] = {}
        table_pattern = re.compile(
            r"(?:FROM|JOIN)\s+([A-Z0-9_\.]+)(?:\s+(?:AS\s+)?([A-Z0-9_]+))?",
            re.IGNORECASE,
        )
        for tbl, alias in table_pattern.findall(query):
            full = table_full_name(tbl)
            if not full:
                continue
            alias_map[clean_identifier(tbl)] = full
            alias_map[full.split(".")[-1]] = full
            if alias:
                alias_map[clean_identifier(alias)] = full
        return alias_map

    def extract_tables(query: str) -> List[str]:
        tables: List[str] = []
        for tbl, _alias in re.findall(r"(?:FROM|JOIN)\s+([A-Z0-9_\.]+)(?:\s+(?:AS\s+)?[A-Z0-9_]+)?", query, re.IGNORECASE):
            full = table_full_name(tbl)
            if full and full not in tables:
                tables.append(full)
        return tables

    def split_expressions(expr_block: str) -> List[str]:
        parts, current, level = [], [], 0
        for ch in expr_block:
            if ch == "(":
                level += 1
            elif ch == ")":
                level = max(0, level - 1)
            if ch == "," and level == 0:
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
            else:
                current.append(ch)
        tail = "".join(current).strip()
        if tail:
            parts.append(tail)
        return parts

    def extract_columns(query: str, alias_map: Dict[str, str], tables_used: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        results: Dict[str, List[Dict[str, Any]]] = {t: [] for t in tables_used}
        seen = set()

        # alias.columna
        for alias, col in re.findall(r"\b([A-Z_][A-Z0-9_]*)\.([A-Z_][A-Z0-9_]*)\b", query, re.IGNORECASE):
            alias_u, col_u = clean_identifier(alias), clean_identifier(col)
            table = alias_map.get(alias_u)
            if not table:
                continue
            meta = column_index_by_table.get(table, {}).get(col_u, {"name": col_u, "description": ""})
            key = (table, col_u)
            if key not in seen:
                seen.add(key)
                results.setdefault(table, []).append({
                    "column_name": col_u,
                    "description": meta.get("description", ""),
                    "source": "qualified_reference",
                })

        # columnas sin alias: se asignan si son únicas
        tokens = re.findall(r"\b([A-Z_][A-Z0-9_]*)\b", query, re.IGNORECASE)
        reserved = {
            "SELECT","FROM","LEFT","RIGHT","INNER","OUTER","JOIN","ON","WHERE","AND","OR","NOT","IN","LIKE","AS",
            "SUM","AVG","COUNT","MIN","MAX","CASE","WHEN","THEN","ELSE","END","DISTINCT","GROUP","BY","ORDER","LIMIT",
            "IS","NULL","BETWEEN","CAST","DATE","YEAR","MONTH","DAY","WITH","UNION","ALL","DESC","ASC","RTRIM","LTRIM",
            "COALESCE","ABS","ROUND","CURRENT_TIMESTAMP"
        }
        for tok in tokens:
            tok_u = clean_identifier(tok)
            if tok_u in reserved or tok_u.isdigit():
                continue
            owners = [t for t in global_column_to_tables.get(tok_u, []) if t in tables_used]
            if len(owners) == 1:
                table = owners[0]
                key = (table, tok_u)
                if key not in seen:
                    seen.add(key)
                    meta = column_index_by_table.get(table, {}).get(tok_u, {"name": tok_u, "description": ""})
                    results.setdefault(table, []).append({
                        "column_name": tok_u,
                        "description": meta.get("description", ""),
                        "source": "unqualified_reference",
                    })
        return results

    def extract_where_conditions(where_clause: str, alias_map: Dict[str, str]) -> List[Dict[str, Any]]:
        if not where_clause:
            return []
        normalized = re.sub(r"\s+", " ", where_clause).strip()
        parts = re.split(r"\s+(AND|OR)\s+", normalized, flags=re.IGNORECASE)

        conditions: List[Dict[str, Any]] = []
        connector = None
        for part in parts:
            p = part.strip()
            if not p:
                continue
            if p.upper() in {"AND", "OR"}:
                connector = p.upper()
                continue

            col_match = re.search(r"\b([A-Z_][A-Z0-9_]*)\.([A-Z_][A-Z0-9_]*)\b", p, re.IGNORECASE)
            if col_match:
                alias_u = clean_identifier(col_match.group(1))
                col_u = clean_identifier(col_match.group(2))
                table = alias_map.get(alias_u)
            else:
                col_u = None
                table = None
                # intento simple de columna no calificada
                m2 = re.search(r"\b([A-Z_][A-Z0-9_]*)\b\s*(=|<>|!=|>|<|>=|<=|LIKE|IN|BETWEEN)\s*", p, re.IGNORECASE)
                if m2:
                    col_u = clean_identifier(m2.group(1))
                    owners = global_column_to_tables.get(col_u, [])
                    if len(owners) == 1:
                        table = owners[0]

            operator = None
            for op in [" BETWEEN ", " LIKE ", " IN ", ">=", "<=", "<>", "!=", "=", ">", "<", " IS "]:
                if op.strip() in p.upper():
                    operator = op.strip()
                    break

            values = re.findall(r"'([^']*)'", p)
            if not values:
                nums = re.findall(r"\b\d+(?:\.\d+)?\b", p)
                values = nums[:3]

            conditions.append({
                "connector_before": connector,
                "raw_condition": p,
                "table_name": table,
                "column_name": col_u,
                "operator": operator,
                "values_in_condition": values,
            })
            connector = None
        return conditions

    def detect_error_type(message: str) -> str:
        m = message.lower()
        if "doesn't exist" in m or "unknown column" in m or "invalid identifier" in m or "columna" in m and "no existe" in m:
            return "missing_column"
        if "syntax" in m or "parse" in m or "you have an error in your sql syntax" in m:
            return "syntax_error"
        if "unknown table" in m or "doesn't exist" in m and "table" in m or "tabla" in m and "no existe" in m:
            return "missing_table"
        if "join" in m and ("invalid" in m or "unknown" in m):
            return "join_error"
        if "0 rows" in m or "no rows" in m or "no data" in m or "empty set" in m or "no retorn" in m or "ningun valor" in m or "ningún valor" in m:
            return "no_rows"
        return "generic_error"

    def columns_for_table(table_name: str) -> List[Dict[str, Any]]:
        table = table_catalog.get(table_name, {})
        cols = []
        for c in table.get("columns", []):
            cols.append({
                "column_name": str(c.get("name", "")).upper(),
                "description": c.get("description", ""),
                "type": c.get("type", ""),
            })
        return cols

    def valid_joins_for_tables(tables_used: List[str]) -> List[Dict[str, Any]]:
        allowed = []
        used = set(tables_used)
        for j in join_whitelist.get("joins", []):
            lt = str(j.get("left_table", "")).upper()
            rt = str(j.get("right_table", "")).upper()
            if lt in used and rt in used:
                allowed.append({
                    "left_table": lt,
                    "right_table": rt,
                    "join_type": j.get("join_type", "LEFT JOIN"),
                    "condition": j.get("condition") or j.get("on"),
                    "reason": j.get("reason", ""),
                })
        return allowed

    def value_hints_for_where(where_conditions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        hints = []
        for cond in where_conditions:
            table = cond.get("table_name")
            column = cond.get("column_name")
            if not table or not column:
                continue
            value_meta = values_index.get(table, {}).get(column)
            if not value_meta:
                continue
            hints.append({
                "table_name": table,
                "column_name": column,
                "description_from_values": value_meta.get("column_description_from_values", ""),
                "existing_values": value_meta.get("values", []),
                "used_values_in_query": cond.get("values_in_condition", []),
            })
        return hints

    def error_focus_tables(message: str, tables_used: List[str], columns_map: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        m = message.upper()
        focused = []
        # Si menciona explícitamente una tabla
        for t in tables_used:
            short = t.split(".")[-1]
            if short in m or t in m:
                focused.append(t)

        # Si menciona una columna inexistente, asociarla si existe en consulta por tabla
        col_matches = re.findall(r"(?:UNKNOWN COLUMN|COLUMN|COLUMNA)\s+'?([A-Z_][A-Z0-9_]*)", m)
        for col in col_matches:
            col_u = clean_identifier(col)
            for t, cols in columns_map.items():
                if any(c["column_name"] == col_u for c in cols) and t not in focused:
                    focused.append(t)
        return focused or tables_used[:2]

    sections = split_sql_sections(sql_norm)
    alias_map = extract_alias_map(sql_norm)
    tables_used = extract_tables(sql_norm)
    columns_used_map = extract_columns(sql_norm, alias_map, tables_used)
    where_conditions = extract_where_conditions(sections.get("where", ""), alias_map)
    error_type = detect_error_type(error_message)

    correction_context: Dict[str, Any] = {"error_message": error_message, "error_type": error_type}

    if error_type == "missing_column":
        focus_tables = error_focus_tables(error_message, tables_used, columns_used_map)
        correction_context["tables_for_review"] = focus_tables
        correction_context["available_columns_by_table"] = {
            t: columns_for_table(t) for t in focus_tables
        }

    elif error_type in {"syntax_error", "join_error", "missing_table"}:
        correction_context["valid_joins"] = valid_joins_for_tables(tables_used)
        correction_context["tables_for_review"] = [
            {"table_name": t, "description": table_catalog.get(t, {}).get("description", "")}
            for t in tables_used
        ]

    elif error_type == "no_rows":
        correction_context["where_value_hints"] = value_hints_for_where(where_conditions)
        correction_context["where_columns_review"] = [
            {
                "table_name": c.get("table_name"),
                "column_name": c.get("column_name"),
                "operator": c.get("operator"),
                "used_values_in_query": c.get("values_in_condition", []),
            }
            for c in where_conditions if c.get("column_name")
        ]

    else:
        correction_context["tables_for_review"] = [
            {"table_name": t, "description": table_catalog.get(t, {}).get("description", "")}
            for t in tables_used
        ]
        correction_context["columns_used_by_table"] = columns_used_map
        correction_context["valid_joins"] = valid_joins_for_tables(tables_used)

    # Contexto textual compacto para LLM corrector
    table_bits = []
    for t in tables_used:
        used_cols = columns_used_map.get(t, [])
        cols_txt = ", ".join(
            f"{c['column_name']} ({c.get('description','')})".strip()
            for c in used_cols[:8]
        )
        table_bits.append(f"{t}: {cols_txt}")

    where_bits = []
    for c in where_conditions[:10]:
        frag = f"{c.get('raw_condition')}"
        where_bits.append(frag)

    ctx_lines = [
        f"ERROR_TIPO: {error_type}",
        f"ERROR: {error_message}",
        f"TABLAS_USADAS: {' | '.join(tables_used) if tables_used else 'ninguna_detectada'}",
        f"COLUMNAS_USADAS: {' | '.join(table_bits) if table_bits else 'ninguna_detectada'}",
        f"WHERE_USADOS: {' | '.join(where_bits) if where_bits else 'sin_where'}",
    ]

    if correction_context.get("available_columns_by_table"):
        for t, cols in correction_context["available_columns_by_table"].items():
            cols_txt = " ; ".join(f"{c['column_name']}: {c.get('description','')}" for c in cols[:40])
            ctx_lines.append(f"COLUMNAS_VALIDAS_{t}: {cols_txt}")

    if correction_context.get("valid_joins"):
        joins_txt = " | ".join(
            f"{j['join_type']} {j['right_table']} ON {j['condition']}"
            for j in correction_context["valid_joins"][:20]
        )
        ctx_lines.append(f"JOINS_VALIDOS: {joins_txt}")

    if correction_context.get("where_value_hints"):
        for hint in correction_context["where_value_hints"][:10]:
            vals = hint.get("existing_values", [])
            vals_txt = ", ".join(map(str, vals[:50]))
            ctx_lines.append(
                f"VALORES_EXISTENTES_{hint['table_name']}.{hint['column_name']}: {vals_txt}"
            )
            if hint.get("description_from_values"):
                ctx_lines.append(
                    f"DESC_{hint['table_name']}.{hint['column_name']}: {hint['description_from_values']}"
                )

    return {
        "tables_used": tables_used,
        "columns_used_by_table": columns_used_map,
        "where_conditions": where_conditions,
        "detected_error_type": error_type,
        "correction_context": correction_context,
        "context_text": "\n".join(ctx_lines),
    }
