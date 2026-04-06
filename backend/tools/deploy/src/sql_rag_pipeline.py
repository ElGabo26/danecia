from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
CATALOG_DIR = BASE_DIR / "catalog"
EXAMPLES_DIR = BASE_DIR / "examples"
MAX_PROMPT_CHARS = 3000
DEFAULT_DATE_START = "2025-01-01"
CURRENT_DATE = date.today().isoformat()
BUSINESS_NAME_PREFIXES = ("NOM_", "DESC_")
DEFAULT_ENTITY_MODEL = "qwen2.5-coder:3b"

COMMAND_WORDS = {
    "genera", "generar", "genere", "muestra", "mostrar", "muéstrame", "muestrame", "dame",
    "trae", "consulta", "obtén", "obten", "obtener", "quiero", "necesito", "requiero",
    "arma", "prepara", "elabora"
}
STOPWORDS = {
    "de", "la", "el", "los", "las", "del", "por", "para", "y", "o", "en", "a", "un", "una",
    "con", "sin", "que", "como", "cómo", "cuanto", "cuántos", "cual", "cuál", "mes", "año", "fecha",
    "hasta", "desde", "hoy", "actual", "totales", "total", "ventas", "venta", "informe", "reporte",
    "reportes", "estado", "estados", "financiero", "financieros", "buscar", "busca", "string", "strings",
    "nombres", "nombre", "similar", "semejante", "parecido", "datos", "tabla", "tablas", "sql",
    "marca", "marcas", "familia", "familias", "empresa", "empresas", "compania", "companias", "compañia",
    "compañias", "unidad", "unidades", "negocio", "cuenta", "cuentas", "producto", "productos", "cliente",
    "clientes", "plantacion", "plantación", "plantaciones", "canal", "canales", "ruta", "rutas", "zona",
    "zonas", "agencia", "agencias", "material", "materiales", "origen", "orígenes", "origenes"
}.union(COMMAND_WORDS)
GENERIC_ENTITY_VALUES = {
    "MARCA", "MARCAS", "FAMILIA", "FAMILIAS", "EMPRESA", "EMPRESAS", "COMPANIA", "COMPANIAS", "COMPAÑIA",
    "COMPAÑIAS", "UNIDAD", "UNIDADES", "UNIDAD DE NEGOCIO", "UNIDADES DE NEGOCIO", "NEGOCIO", "NEGOCIOS",
    "CUENTA", "CUENTAS", "PRODUCTO", "PRODUCTOS", "CLIENTE", "CLIENTES", "PLANTACION", "PLANTACIONES",
    "CANAL", "CANALES", "RUTA", "RUTAS", "ZONA", "ZONAS", "AGENCIA", "AGENCIAS", "MATERIAL", "MATERIALES",
    "ORIGEN", "ORIGENES", "ORÍGENES"
}
EXPLICIT_AUTOCONSUMO_TERMS = {"autoconsumo", "autoconsumos", "venta_autoconsumo", "venta autoconsumo"}
REPORT_REQUEST_PATTERNS = (
    "genera el informe", "genera informe", "genera el reporte", "genera reporte",
    "muestra el informe", "muestra informe", "dame el informe", "dame informe",
    "estados financieros", "estado financiero"
)
DEFAULT_ENTITY_TYPES = {
    "marca": ["MARCA"],
    "familia": ["FAMILIA"],
    "empresa": ["COMPANIA", "EMPRESA", "CIA", "COMPANIA"],
    "unidad_negocio": ["UNID_NEGOCIO", "UNIDAD_NEGOCIO", "NEGOCIO"],
    "cuenta": ["CUENTA", "CTA"],
    "cliente": ["CLIENTE"],
    "producto": ["PRODUCTO", "MATERIAL", "ITEM", "ARTICULO", "ARTÍCULO"],
    "plantacion": ["PLANTACION", "PLANTACIÓN"],
    "canal": ["CANAL"],
    "ruta": ["RUTA"],
    "zona": ["ZONA"],
    "agencia": ["AGENCIA"],
    "material": ["MATERIAL"],
    "origen": ["ORIGEN"],
}
ENTITY_PATTERN_MAP = {
    "marca": r"(?:marca)\s+(?:de\s+|del\s+|la\s+|el\s+)?([a-zA-Z0-9áéíóúñÑ_\- ]{2,60})",
    "familia": r"(?:familia)\s+(?:de\s+|del\s+|la\s+|el\s+)?([a-zA-Z0-9áéíóúñÑ_\- ]{2,60})",
    "empresa": r"(?:empresa|compa(?:ñ|n)ia|cia)\s+(?:de\s+|del\s+|la\s+|el\s+)?([a-zA-Z0-9áéíóúñÑ_\- ]{2,60})",
    "unidad_negocio": r"(?:unidad\s+de\s+negocio|unidad)\s+(?:de\s+|del\s+|la\s+|el\s+)?([a-zA-Z0-9áéíóúñÑ_\- ]{2,60})",
    "cuenta": r"(?:cuenta)\s+(?:de\s+|del\s+|la\s+|el\s+)?([a-zA-Z0-9áéíóúñÑ_\- ]{2,60})",
    "cliente": r"(?:cliente)\s+(?:de\s+|del\s+|la\s+|el\s+)?([a-zA-Z0-9áéíóúñÑ_\- ]{2,60})",
    "producto": r"(?:producto)\s+(?:de\s+|del\s+|la\s+|el\s+)?([a-zA-Z0-9áéíóúñÑ_\- ]{2,60})",
    "plantacion": r"(?:plantaci(?:ó|o)n)\s+(?:de\s+|del\s+|la\s+|el\s+)?([a-zA-Z0-9áéíóúñÑ_\- ]{2,60})",
    "canal": r"(?:canal)\s+(?:de\s+|del\s+|la\s+|el\s+)?([a-zA-Z0-9áéíóúñÑ_\- ]{2,60})",
    "ruta": r"(?:ruta)\s+(?:de\s+|del\s+|la\s+|el\s+)?([a-zA-Z0-9áéíóúñÑ_\- ]{2,60})",
    "zona": r"(?:zona)\s+(?:de\s+|del\s+|la\s+|el\s+)?([a-zA-Z0-9áéíóúñÑ_\- ]{2,60})",
    "agencia": r"(?:agencia)\s+(?:de\s+|del\s+|la\s+|el\s+)?([a-zA-Z0-9áéíóúñÑ_\- ]{2,60})",
    "material": r"(?:material)\s+(?:de\s+|del\s+|la\s+|el\s+)?([a-zA-Z0-9áéíóúñÑ_\- ]{2,60})",
    "origen": r"(?:origen)\s+(?:de\s+|del\s+|la\s+|el\s+)?([a-zA-Z0-9áéíóúñÑ_\- ]{2,60})",
}


def _strip_accents(value: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", value) if not unicodedata.combining(ch))


def _upper_ascii(value: str) -> str:
    return _strip_accents(value).upper()


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-ZáéíóúÁÉÍÓÚñÑ0-9_]+", text.lower())


def _question_tokens(question: str) -> List[str]:
    return [t for t in _tokenize(question) if len(t) > 2 and t not in STOPWORDS and not re.fullmatch(r"20\d{2}", t)]


def _looks_like_instruction_phrase(text: str) -> bool:
    low = text.lower().strip()
    return any(pattern in low for pattern in REPORT_REQUEST_PATTERNS) or low in COMMAND_WORDS


def _clean_candidate_phrase(value: str) -> str:
    value = re.sub(r"\b(?:desde|hasta|en|por|con|sin|y|o|del|de la|de el|para)\b.*$", "", value, flags=re.IGNORECASE).strip()
    value = re.sub(r"\s+", " ", value)
    return value.strip(" .,:;-")


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


SCHEMA_CATALOG = load_json(CATALOG_DIR / "schema_catalog.json")
JOIN_WHITELIST = load_json(CATALOG_DIR / "join_whitelist.json")
BUSINESS_RULES = load_json(CATALOG_DIR / "business_rules.json")
BUSINESS_GLOSSARY = load_json(CATALOG_DIR / "business_glossary.json")
TABLES_BY_NAME = {t["table_name"]: t for t in SCHEMA_CATALOG["tables"]}
ENTITY_TYPES = BUSINESS_GLOSSARY.get("entity_types", DEFAULT_ENTITY_TYPES)


def load_examples() -> List[Dict[str, Any]]:
    with open(EXAMPLES_DIR / "sql_examples.jsonl", "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


SQL_EXAMPLES = load_examples()


@dataclass
class CategoricalEntity:
    entity_type: str
    raw_text: str
    value: str
    columns: List[str]
    confidence: str = "media"
    source: str = "heuristic"


@dataclass
class NormalizedIntent:
    question: str
    domain: str
    metrics: List[str]
    dimensions: List[str]
    filters: List[str]
    date_filter_sql: str
    date_filter_label: str
    search_terms: List[str]
    detected_entities: List[CategoricalEntity]
    explicit_autoconsumo: bool = False
    requires_dim_fecha: bool = True


@dataclass
class ContextPackage:
    domain: str
    tables: List[Dict[str, Any]]
    joins: List[Dict[str, Any]]
    rules: List[str]
    examples: List[Dict[str, Any]]
    context_text: str
    selected_table_names: List[str]
    preferred_name_columns: Dict[str, List[str]]
    search_terms: List[str]
    detected_entities: List[CategoricalEntity]
    allowed_like_columns: List[str]


@dataclass
class JoinUsage:
    left_table: str
    right_table: str
    join_type: str
    condition: str


def _build_entity_column_index() -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = {k: [] for k in ENTITY_TYPES}
    for table in SCHEMA_CATALOG["tables"]:
        tname = table["table_name"].upper()
        for column in table["columns"]:
            cname = str(column["name"]).upper()
            if not cname.startswith(BUSINESS_NAME_PREFIXES):
                continue
            ref = f"{tname}.{cname}"
            for entity_type, keys in ENTITY_TYPES.items():
                if any(key in cname for key in keys):
                    index.setdefault(entity_type, []).append(ref)
    return {k: sorted(dict.fromkeys(v)) for k, v in index.items()}


ENTITY_COLUMN_INDEX = _build_entity_column_index()


def _normalize_business_term(term: str) -> str:
    return _upper_ascii(re.sub(r"\s+", " ", term.strip()))


def _is_generic_entity_value(value: str) -> bool:
    upper = _normalize_business_term(value)
    return not upper or upper in GENERIC_ENTITY_VALUES or upper in {w.upper() for w in COMMAND_WORDS}


def _has_explicit_autoconsumo(question: str) -> bool:
    q_low = question.lower()
    return any(term in q_low for term in EXPLICIT_AUTOCONSUMO_TERMS)


def _domain_from_question(q_low: str) -> str:
    domain = "ventas"
    for candidate, words in BUSINESS_GLOSSARY.get("domains", {}).items():
        if any(word in q_low for word in words):
            domain = candidate
            break
    return domain


def _normalize_date_filter(q_low: str) -> Tuple[str, str]:
    full_dates = re.findall(r"\b20\d{2}-\d{2}-\d{2}\b", q_low)
    years = re.findall(r"\b20\d{2}\b", q_low)
    if len(full_dates) >= 2:
        start, end = sorted(full_dates[:2])
        return f"DF.FECHA BETWEEN '{start}' AND '{end}'", f"rango explícito {start} a {end}"
    if len(full_dates) == 1:
        one = full_dates[0]
        return f"DF.FECHA = '{one}'", f"fecha explícita {one}"
    if years:
        year = years[0]
        end = CURRENT_DATE if CURRENT_DATE.startswith(year) else f"{year}-12-31"
        return f"DF.FECHA BETWEEN '{year}-01-01' AND '{end}'", f"año explícito {year}"
    return f"DF.FECHA BETWEEN '{DEFAULT_DATE_START}' AND '{CURRENT_DATE}'", f"rango por defecto {DEFAULT_DATE_START} a {CURRENT_DATE}"


def _dedupe_entities(entities: Sequence[CategoricalEntity]) -> List[CategoricalEntity]:
    seen = set()
    output: List[CategoricalEntity] = []
    for entity in entities:
        key = (entity.entity_type, entity.value)
        if key in seen:
            continue
        seen.add(key)
        cols = sorted(dict.fromkeys(entity.columns))
        output.append(CategoricalEntity(
            entity_type=entity.entity_type,
            raw_text=entity.raw_text,
            value=entity.value,
            columns=cols,
            confidence=entity.confidence,
            source=entity.source,
        ))
    return output


def _heuristic_detect_entities(question: str) -> List[CategoricalEntity]:
    entities: List[CategoricalEntity] = []
    q_low = question.lower()

    for entity_type, pattern in ENTITY_PATTERN_MAP.items():
        for match in re.findall(pattern, q_low, flags=re.IGNORECASE):
            raw = _clean_candidate_phrase(match)
            value = _normalize_business_term(raw)
            if _is_generic_entity_value(value) or _looks_like_instruction_phrase(value.lower()):
                continue
            entities.append(CategoricalEntity(
                entity_type=entity_type,
                raw_text=raw,
                value=value,
                columns=ENTITY_COLUMN_INDEX.get(entity_type, []),
                confidence="alta",
                source="heuristic",
            ))

    # quoted strings inherit nearest explicit entity type when available
    for raw in re.findall(r"[\"“”']([^\"“”']{2,60})[\"“”']", question):
        cleaned = _clean_candidate_phrase(raw)
        value = _normalize_business_term(cleaned)
        if _is_generic_entity_value(value):
            continue
        prefix = question[:question.find(raw)].lower()[-50:]
        inferred_type = None
        for entity_type in ENTITY_PATTERN_MAP:
            hint = entity_type.replace("_", " ")
            if hint in prefix:
                inferred_type = entity_type
                break
        if inferred_type:
            entities.append(CategoricalEntity(
                entity_type=inferred_type,
                raw_text=cleaned,
                value=value,
                columns=ENTITY_COLUMN_INDEX.get(inferred_type, []),
                confidence="media",
                source="quote",
            ))

    return _dedupe_entities(entities)


def _should_run_entity_detector(question: str) -> bool:
    q_low = question.lower()
    if any(key.replace("_", " ") in q_low for key in ENTITY_TYPES):
        return True
    if re.search(r"[\"“”']([^\"“”']{2,60})[\"“”']", question):
        return True
    return any(word in q_low for word in ("marca", "familia", "empresa", "compañ", "unidad de negocio", "unidad", "cuenta", "cliente", "producto"))


def _extract_json_block(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    return match.group(0) if match else text


def _llm_detect_entities(question: str, client: Optional[object] = None, model: Optional[str] = None) -> List[CategoricalEntity]:
    if client is None or not _should_run_entity_detector(question):
        return []
    model = model or DEFAULT_ENTITY_MODEL
    prompt = (
        "Detecta solo entidades categóricas explícitas en la pregunta. Devuelve JSON puro como lista con objetos "
        "{entity_type,value}. Tipos válidos: marca,familia,empresa,unidad_negocio,cuenta,cliente,producto,plantacion,canal,ruta,zona,agencia,material,origen. "
        "No devuelvas verbos como GENERA, MUESTRA, DAME ni términos genéricos como PLANTACIONES, VENTAS o INFORME. "
        f"Pregunta:{question}"
    )
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": "Responde solo JSON válido."},
                {"role": "user", "content": prompt[:900]},
            ],
        )
        raw = response.choices[0].message.content or "[]"
        payload = json.loads(_extract_json_block(raw))
        entities: List[CategoricalEntity] = []
        for item in payload if isinstance(payload, list) else []:
            entity_type = str(item.get("entity_type", "")).strip().lower()
            value_raw = _clean_candidate_phrase(str(item.get("value", "")).strip())
            value = _normalize_business_term(value_raw)
            if entity_type not in ENTITY_TYPES:
                continue
            if _is_generic_entity_value(value):
                continue
            entities.append(CategoricalEntity(
                entity_type=entity_type,
                raw_text=value_raw,
                value=value,
                columns=ENTITY_COLUMN_INDEX.get(entity_type, []),
                confidence="media",
                source="llm_detector",
            ))
        return _dedupe_entities(entities)
    except Exception:
        return []


def detect_categorical_entities(question: str, client: Optional[object] = None, detector_model: Optional[str] = None) -> List[CategoricalEntity]:
    heuristic = _heuristic_detect_entities(question)
    llm_entities = _llm_detect_entities(question, client=client, model=detector_model)
    merged = _dedupe_entities([*heuristic, *llm_entities])
    return [entity for entity in merged if entity.columns]


def normalize_intent(question: str, client: Optional[object] = None, detector_model: Optional[str] = None) -> NormalizedIntent:
    q = question.strip()
    q_low = q.lower()
    domain = _domain_from_question(q_low)

    metrics: List[str] = []
    dimensions: List[str] = []
    for word, cols in BUSINESS_GLOSSARY.get("metrics", {}).items():
        if word in q_low:
            metrics.extend(cols)
    for word, cols in BUSINESS_GLOSSARY.get("dimensions", {}).items():
        if word in q_low:
            dimensions.extend(cols)

    explicit_autoconsumo = _has_explicit_autoconsumo(q)
    if not metrics and domain == "ventas":
        metrics.append("VENTA_AUTOCONSUMO")

    detected_entities = detect_categorical_entities(q, client=client, detector_model=detector_model)
    search_terms = [entity.value for entity in detected_entities]
    date_filter_sql, date_filter_label = _normalize_date_filter(q_low)

    filters = [date_filter_label, "Siempre usar JOIN con DIM_FECHA para filtrar fechas"]
    if detected_entities:
        resume = ", ".join(f"{e.entity_type}:{e.value}" for e in detected_entities)
        filters.append(f"Entidades categóricas detectadas: {resume}")
        filters.append("Solo cuando exista entidad explícita usar UPPER(columna) LIKE '%VALOR%' sobre sus columnas candidatas")
    else:
        filters.append("No se detectaron entidades categóricas explícitas; no aplicar búsqueda textual semejante")

    return NormalizedIntent(
        question=q,
        domain=domain,
        metrics=sorted(set(metrics)),
        dimensions=sorted(set(dimensions)),
        filters=filters,
        date_filter_sql=date_filter_sql,
        date_filter_label=date_filter_label,
        search_terms=search_terms,
        detected_entities=detected_entities,
        explicit_autoconsumo=explicit_autoconsumo,
        requires_dim_fecha=True,
    )


def _score_table(table: Dict[str, Any], intent: NormalizedIntent) -> int:
    score = 0
    tname = table["table_name"].upper()
    columns = {str(c["name"]).upper() for c in table["columns"]}
    desc = str(table.get("description", "")).lower()
    domain = str(table.get("domain", "")).lower()

    if intent.domain and intent.domain in domain:
        score += 18
    if tname.endswith("DIM_FECHA"):
        score += 25
    if tname.endswith("FAC_VENTA_TOTAL") and intent.domain == "ventas":
        score += 16

    for metric in intent.metrics:
        if metric.upper() in columns:
            score += 10
    for dim in intent.dimensions:
        if dim.upper() in columns:
            score += 7

    for token in _question_tokens(intent.question):
        if token in desc or token.upper() in tname:
            score += 2

    for entity in intent.detected_entities:
        candidate_tables = {ref.rsplit(".", 1)[0] for ref in entity.columns}
        if tname in candidate_tables:
            score += 20
            break

    name_cols = [c for c in columns if c.startswith(BUSINESS_NAME_PREFIXES)]
    if name_cols:
        score += 2
    return score


def _preferred_name_columns(table: Dict[str, Any]) -> List[str]:
    cols = [str(c["name"]).upper() for c in table["columns"] if str(c["name"]).upper().startswith(BUSINESS_NAME_PREFIXES)]
    return cols[:12]


def _select_tables(intent: NormalizedIntent, max_tables: int = 6) -> List[Dict[str, Any]]:
    scored = sorted(
        (( _score_table(table, intent), table) for table in SCHEMA_CATALOG["tables"]),
        key=lambda item: item[0],
        reverse=True,
    )
    selected: List[Dict[str, Any]] = []
    seen = set()
    for score, table in scored:
        tname = table["table_name"]
        if score <= 0 and len(selected) >= 2:
            continue
        if tname in seen:
            continue
        selected.append(table)
        seen.add(tname)
        if len(selected) >= max_tables:
            break

    fecha = TABLES_BY_NAME.get("DDM_ERP.DIM_FECHA")
    if fecha and fecha["table_name"] not in seen:
        selected.append(fecha)
        seen.add(fecha["table_name"])

    for entity in intent.detected_entities:
        for ref in entity.columns:
            tname = ref.rsplit(".", 1)[0]
            if tname not in seen and tname in TABLES_BY_NAME:
                selected.append(TABLES_BY_NAME[tname])
                seen.add(tname)

    return selected[:max_tables + 1]


def _select_joins(selected_tables: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    allowed_tables = {table["table_name"].upper() for table in selected_tables}
    joins: List[Dict[str, Any]] = []
    for join in JOIN_WHITELIST["joins"]:
        left = join["left_table"].upper()
        right = join["right_table"].upper()
        if left in allowed_tables and right in allowed_tables:
            joins.append(join)
    return joins[:12]


def _select_rules(intent: NormalizedIntent) -> List[str]:
    rules = []
    for item in BUSINESS_RULES.get("rules", []):
        domain = item.get("domain", "global")
        if domain in {"global", intent.domain}:
            rules.append(item["rule"])
    if intent.detected_entities:
        rules.append("Solo aplicar búsqueda semejante cuando exista entidad explícita detectada; usar MAYÚSCULAS.")
    else:
        rules.append("No aplicar búsqueda por semejanza si no hay marca, familia, empresa, unidad de negocio u otra entidad explícita.")
    return rules[:10]


def _select_examples(intent: NormalizedIntent) -> List[Dict[str, Any]]:
    ranked: List[Tuple[int, Dict[str, Any]]] = []
    q_low = intent.question.lower()
    for example in SQL_EXAMPLES:
        score = 0
        if example.get("domain") == intent.domain:
            score += 5
        if any(token in example.get("question", "").lower() for token in _question_tokens(q_low)[:4]):
            score += 2
        ranked.append((score, example))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [example for score, example in ranked[:2] if score > 0]


def _build_context_text(intent: NormalizedIntent, tables: Sequence[Dict[str, Any]], joins: Sequence[Dict[str, Any]], rules: Sequence[str], examples: Sequence[Dict[str, Any]]) -> Tuple[str, Dict[str, List[str]], List[str]]:
    preferred_name_columns: Dict[str, List[str]] = {}
    allowed_like_columns: List[str] = []
    table_bits: List[str] = []
    for table in tables:
        tname = table["table_name"].upper()
        cols = [str(c["name"]).upper() for c in table["columns"]]
        preferred = _preferred_name_columns(table)
        preferred_name_columns[tname] = preferred
        metric_cols = [c for c in cols if c in {m.upper() for m in intent.metrics}][:3]
        keep_cols = preferred[:4] + metric_cols[:3] + [c for c in cols if c in {d.upper() for d in intent.dimensions}][:3]
        keep_cols = list(dict.fromkeys([c for c in keep_cols if c]))[:8]
        if keep_cols:
            table_bits.append(f"{tname}[{','.join(keep_cols)}]")

    for entity in intent.detected_entities:
        allowed_like_columns.extend([ref.rsplit('.', 1)[1] for ref in entity.columns])

    join_bits = [f"{j['left_table']}->{j['right_table']} ON {j['condition']}" for j in joins[:8]]
    rule_bits = rules[:6]
    example_bits = [re.sub(r"\s+", " ", ex.get("sql", ""))[:220] for ex in examples]
    entity_bits = [f"{e.entity_type}:{e.value}=>{','.join(c.split('.')[-1] for c in e.columns[:3])}" for e in intent.detected_entities]
    if not entity_bits:
        entity_bits = ["sin_entidades_explicitas"]

    context_text = (
        f"dom={intent.domain}; fechas={intent.date_filter_sql}; tablas={' | '.join(table_bits)}; "
        f"joins={' | '.join(join_bits)}; entidades={' | '.join(entity_bits)}; "
        f"reglas={' | '.join(rule_bits)}; ejemplos={' | '.join(example_bits)}"
    )
    context_text = re.sub(r"\s+", " ", context_text).strip()
    if len(context_text) > 1900:
        context_text = context_text[:1897] + "..."
    return context_text, preferred_name_columns, sorted(dict.fromkeys(allowed_like_columns))


def select_relevant_context(intent: NormalizedIntent) -> ContextPackage:
    tables = _select_tables(intent)
    joins = _select_joins(tables)
    rules = _select_rules(intent)
    examples = _select_examples(intent)
    context_text, preferred_name_columns, allowed_like_columns = _build_context_text(intent, tables, joins, rules, examples)
    return ContextPackage(
        domain=intent.domain,
        tables=tables,
        joins=joins,
        rules=rules,
        examples=examples,
        context_text=context_text,
        selected_table_names=[table["table_name"] for table in tables],
        preferred_name_columns=preferred_name_columns,
        search_terms=intent.search_terms,
        detected_entities=intent.detected_entities,
        allowed_like_columns=allowed_like_columns,
    )


def build_prompt(question: str, intent: NormalizedIntent, context: ContextPackage, max_chars: int = MAX_PROMPT_CHARS) -> str:
    if context.detected_entities:
        entities_txt = "; ".join(
            f"{e.entity_type}={e.value} solo_en={','.join(col.split('.')[-1] for col in e.columns[:4])}" for e in context.detected_entities
        )
        entity_rule = (
            "Solo si hay entidad explícita usa semejanza con MAYÚSCULAS: UPPER(col) LIKE '%VALOR%'. "
            "Restringe la búsqueda a las columnas candidatas de la entidad detectada. "
            f"Entidades:{entities_txt}."
        )
    else:
        entity_rule = "No se detectaron entidades categóricas explícitas; no uses LIKE ni filtros de texto semejantes."

    base = (
        "Genera solo SQL MySQL/SingleStore o NO_SQL. Solo SELECT. No inventes tablas, columnas ni joins. "
        f"Siempre haz JOIN con DDM_ERP.DIM_FECHA y filtra con {intent.date_filter_sql}. "
        f"{entity_rule} Usa columnas NOM_/DESC_ para mostrar nombres conocidos. "
        "Si el pedido es de ventas, evita alias AUTOCONSUMO salvo petición explícita; usa VENTAS o TOTAL_VENTAS. "
        "Si el contexto no alcanza, responde NO_SQL."
    )
    prompt = f"{base}\nCTX:{context.context_text}\nQ:{question}\nSQL:"
    if len(prompt) > max_chars:
        excess = len(prompt) - max_chars
        reduced_ctx = context.context_text[:-excess - 3] + "..." if excess + 3 < len(context.context_text) else context.context_text[: max(0, max_chars - len(base) - len(question) - 20)]
        prompt = f"{base}\nCTX:{reduced_ctx}\nQ:{question}\nSQL:"
    return prompt[:max_chars]


def _sanitize_sales_aliases(sql: str, question: str) -> str:
    if sql.strip() == "NO_SQL" or _has_explicit_autoconsumo(question):
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


def _infer_no_sql_reason(question: str, intent: NormalizedIntent, context: ContextPackage, sql: Optional[str] = None, validation_errors: Optional[List[str]] = None) -> str:
    q_low = question.lower().strip()
    if sql and sql.strip() == "NO_SQL":
        if not context.selected_table_names:
            return "No se encontró un contexto de tablas suficiente para construir la consulta."
        if intent.detected_entities:
            values = ", ".join(entity.value for entity in intent.detected_entities)
            return f"No fue posible mapear con seguridad {values} a joins y columnas válidas del contexto."
        if "estados financieros" in q_low and "plantaciones" in q_low:
            return "La solicitud sigue siendo ambigua; falta precisar la métrica o estructura del informe financiero."
        return "No fue posible construir una consulta confiable solo con las tablas, columnas y joins permitidos."
    if validation_errors:
        first = validation_errors[0]
        if "Tablas no permitidas" in first:
            return "La consulta intentó usar tablas fuera del catálogo permitido."
        if "Columnas no permitidas" in first:
            return "La consulta intentó usar columnas fuera del catálogo permitido."
        if "JOIN no permitido" in first:
            return "La consulta propuso joins que no están aprobados en el contexto."
        if "LIKE" in first:
            return "La consulta aplicó búsqueda de texto en columnas no autorizadas o sin una entidad explícita detectada."
        if "DIM_FECHA" in first or "FECHA" in first:
            return "La consulta no respetó la regla obligatoria de filtrar fechas con DIM_FECHA."
        return "La consulta no pasó la validación del esquema permitido."
    return "No fue posible generar SQL confiable con el contexto actual."


def build_client(base_url: str = "http://localhost:11434/v1", api_key: str = "ollama"):
    from openai import OpenAI
    return OpenAI(base_url=base_url, api_key=api_key)


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


def generate_sql(question: str, model: str = DEFAULT_ENTITY_MODEL, client: Optional[object] = None, temperature: float = 0.0, detector_model: Optional[str] = None) -> Dict[str, Any]:
    client = client or build_client()
    intent = normalize_intent(question, client=client, detector_model=detector_model or model)
    context = select_relevant_context(intent)
    prompt = build_prompt(question, intent, context, max_chars=MAX_PROMPT_CHARS)
    sql = _llm_generate(prompt, model=model, client=client, system_prompt="Devuelve únicamente SQL o NO_SQL.")
    sql = _sanitize_sales_aliases(sql, question)
    no_sql_reason = _infer_no_sql_reason(question, intent, context, sql=sql if sql.strip() == "NO_SQL" else None)
    validation_context = {
        "detected_entities": [asdict(entity) for entity in context.detected_entities],
        "allowed_like_columns": context.allowed_like_columns,
        "requires_similarity": bool(context.detected_entities),
        "similarity_values": [entity.value for entity in context.detected_entities],
        "no_sql_reason": no_sql_reason,
    }
    return {
        "question": question,
        "intent": {
            **asdict(intent),
            "detected_entities": [asdict(entity) for entity in intent.detected_entities],
        },
        "context": context.context_text,
        "prompt": prompt,
        "prompt_chars": len(prompt),
        "selected_tables": context.selected_table_names,
        "preferred_name_columns": context.preferred_name_columns,
        "search_terms": context.search_terms,
        "detected_entities": [asdict(entity) for entity in context.detected_entities],
        "validation_context": validation_context,
        "sql": sql,
        "no_sql_reason": no_sql_reason if sql.strip() == "NO_SQL" else "",
    }


def _extract_tables(sql: str) -> List[str]:
    tables = re.findall(r"(?:FROM|JOIN)\s+([A-Z0-9_\.]+)", sql, flags=re.IGNORECASE)
    output: List[str] = []
    for table in tables:
        value = table.upper()
        if "." not in value:
            value = f"DDM_ERP.{value}"
        if value not in output:
            output.append(value)
    return output


def _extract_columns(sql: str) -> List[str]:
    refs = re.findall(r"\b([A-Z][A-Z0-9_]*)\.([A-Z][A-Z0-9_]*)\b", sql, flags=re.IGNORECASE)
    output: List[str] = []
    for prefix, col in refs:
        if prefix.upper() == "DDM_ERP":
            continue
        output.append(col.upper())
    return output


def _extract_like_columns(sql: str) -> List[str]:
    patterns = [
        r"UPPER\s*\(\s*[A-Z][A-Z0-9_]*\.([A-Z][A-Z0-9_]*)\s*\)\s+LIKE",
        r"\b[A-Z][A-Z0-9_]*\.([A-Z][A-Z0-9_]*)\s+LIKE",
    ]
    cols: List[str] = []
    for pattern in patterns:
        cols.extend(re.findall(pattern, sql, flags=re.IGNORECASE))
    return [col.upper() for col in cols]


def _extract_like_values(sql: str) -> List[str]:
    values = re.findall(r"LIKE\s+'%([^%']+)%'", sql, flags=re.IGNORECASE)
    return [_upper_ascii(v.strip()) for v in values]


def _extract_join_usages(sql: str) -> List[JoinUsage]:
    pattern = re.compile(
        r"(LEFT|INNER|RIGHT|FULL)?\s*JOIN\s+([A-Z0-9_\.]+)(?:\s+([A-Z][A-Z0-9_]*))?\s+ON\s+(.+?)(?=\s+(?:LEFT|INNER|RIGHT|FULL)?\s*JOIN\s+|\s+WHERE\s+|\s+GROUP\s+BY\s+|\s+ORDER\s+BY\s+|\s+LIMIT\s+|$)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    from_match = re.search(r"FROM\s+([A-Z0-9_\.]+)(?:\s+([A-Z][A-Z0-9_]*))?", sql, flags=re.IGNORECASE)
    current_left = from_match.group(1).upper() if from_match else ""
    if current_left and "." not in current_left:
        current_left = f"DDM_ERP.{current_left}"
    usages: List[JoinUsage] = []
    for join_type, right_table, _, condition in pattern.findall(sql):
        rt = right_table.upper()
        if "." not in rt:
            rt = f"DDM_ERP.{rt}"
        usages.append(JoinUsage(
            left_table=current_left,
            right_table=rt,
            join_type=((join_type or "LEFT").upper() + " JOIN"),
            condition=" ".join(condition.split()),
        ))
        current_left = rt
    return usages


def validate_sql(sql: str, validation_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    errors: List[str] = []
    raw = sql.strip()
    validation_context = validation_context or {}

    if raw == "NO_SQL":
        return {
            "valid": False,
            "errors": ["LLM devolvió NO_SQL"],
            "sql": raw,
            "tables": [],
            "columns": [],
            "no_sql_reason": validation_context.get("no_sql_reason", "El modelo no encontró una consulta suficientemente segura con el contexto permitido."),
        }

    if not re.match(r"^\s*SELECT\b", raw, flags=re.IGNORECASE):
        errors.append("La sentencia no inicia con SELECT.")

    blocked = re.findall(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE)\b", raw, flags=re.IGNORECASE)
    if blocked:
        errors.append(f"Palabras bloqueadas detectadas: {sorted(set(x.upper() for x in blocked))}")
    if raw.count(";") > 1:
        errors.append("Múltiples sentencias detectadas.")
    if "--" in raw or "/*" in raw:
        errors.append("No se permiten comentarios SQL.")

    allowed_tables = {t["table_name"].upper() for t in SCHEMA_CATALOG["tables"]}
    tables = _extract_tables(raw)
    bad_tables = [t for t in tables if t not in allowed_tables]
    if bad_tables:
        errors.append(f"Tablas no permitidas: {bad_tables}")

    allowed_columns = {str(c["name"]).upper() for t in SCHEMA_CATALOG["tables"] for c in t["columns"]}
    columns = _extract_columns(raw)
    bad_columns = [c for c in columns if c not in allowed_columns]
    if bad_columns:
        errors.append(f"Columnas no permitidas: {sorted(set(bad_columns))[:10]}")

    if " LIMIT " not in f" {raw.upper()} ":
        errors.append("Falta LIMIT 1000 o un límite explícito.")

    if "DDM_ERP.DIM_FECHA" not in tables:
        errors.append("Siempre se debe hacer JOIN con DDM_ERP.DIM_FECHA.")
    if not re.search(r"\b(?:DF|DIM_FECHA)\.FECHA\b\s*(?:=|>=|<=|>|<|BETWEEN)", raw, flags=re.IGNORECASE):
        errors.append("El filtro de fechas debe aplicarse sobre DF.FECHA o DIM_FECHA.FECHA.")

    whitelist_conditions = {(j["right_table"].upper(), " ".join(j["condition"].upper().split())) for j in JOIN_WHITELIST["joins"]}
    for usage in _extract_join_usages(raw):
        normalized = (usage.right_table.upper(), " ".join(usage.condition.upper().split()))
        if normalized not in whitelist_conditions:
            errors.append(f"JOIN no permitido: {usage.right_table} ON {usage.condition}")

    allowed_like_columns = {c.upper() for c in validation_context.get("allowed_like_columns", [])}
    detected_entities = validation_context.get("detected_entities", [])
    like_columns = _extract_like_columns(raw)
    like_values = _extract_like_values(raw)
    if not detected_entities and like_columns:
        errors.append("No se autorizó búsqueda textual semejante para esta solicitud.")
    if detected_entities:
        if like_columns and any(col not in allowed_like_columns for col in like_columns):
            errors.append(f"LIKE aplicado sobre columnas no autorizadas: {sorted(set(col for col in like_columns if col not in allowed_like_columns))}")
        if any(v != v.upper() for v in like_values):
            errors.append("Los términos usados en LIKE deben estar en MAYÚSCULAS.")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "tables": tables,
        "columns": columns[:40],
        "sql": raw,
        "like_columns": like_columns,
        "like_values": like_values,
    }


def repair_sql(question: str, bad_sql: str, validation_errors: List[str], model: str = DEFAULT_ENTITY_MODEL, client: Optional[object] = None, detector_model: Optional[str] = None) -> str:
    client = client or build_client()
    intent = normalize_intent(question, client=client, detector_model=detector_model or model)
    context = select_relevant_context(intent)
    if context.detected_entities:
        entity_rule = "; ".join(f"{e.entity_type}={e.value} solo_en={','.join(col.split('.')[-1] for col in e.columns[:4])}" for e in context.detected_entities)
    else:
        entity_rule = "sin entidades explícitas; no uses LIKE"
    prompt = (
        "Corrige el SQL usando solo el contexto. Solo SELECT. Siempre usa JOIN con DIM_FECHA y filtra con "
        f"{intent.date_filter_sql}. {entity_rule}. Usa columnas NOM_/DESC_ para nombres legibles. "
        "La búsqueda semejante solo aplica cuando hay entidad categórica explícita y debe usar MAYÚSCULAS. "
        "Si es un informe de ventas, no uses aliases AUTOCONSUMO salvo petición explícita. "
        "Si no puedes corregirlo con seguridad, devuelve NO_SQL. "
        f"Errores:{' | '.join(validation_errors)} CTX:{context.context_text} SQL_Original:{bad_sql} Q:{question} SQL:"
    )[:MAX_PROMPT_CHARS]
    return _sanitize_sales_aliases(
        _llm_generate(prompt, model=model, client=client, system_prompt="Devuelve únicamente SQL corregido o NO_SQL."),
        question,
    )


if __name__ == "__main__":
    client = build_client()
    question = "ventas por marca Toni en 2025"
    result = generate_sql(question, client=client)
    print(result["prompt_chars"])
    print(result["detected_entities"])
    print(result["prompt"])
    print(result["sql"])
    print(validate_sql(result["sql"], result["validation_context"]))
