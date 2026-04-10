import json
from typing import List, Dict, Any, Optional


def cargar_jsonl(path_jsonl: str) -> List[Dict[str, Any]]:
    registros = []
    with open(path_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            registros.append(json.loads(line))
    return registros


def _normalizar_tabla(nombre: str) -> str:
    return str(nombre).strip().upper()


def _normalizar_texto(valor: Any) -> str:
    return str(valor).strip().lower()


def _lista_desde_valor(valor: Any) -> List[Any]:
    if valor is None:
        return []
    if isinstance(valor, list):
        return valor
    return [valor]


def _registro_aplica_a_dominio(registro: Dict[str, Any], dominio: str) -> bool:
    dominio_norm = _normalizar_texto(dominio)

    posibles_claves = [
        "dominio",
        "nombre_dominio",
        "dominio_metrica",
        "dominio_regla",
        "_section",
    ]

    for clave in posibles_claves:
        if clave in registro and _normalizar_texto(registro[clave]) == dominio_norm:
            return True

    return False


def _extraer_nombre_tabla_desde_registro(registro: Dict[str, Any]) -> Optional[str]:
    posibles_claves = [
        "tabla",
        "nombre_tabla",
        "table_name",
        "tabla_principal",
    ]
    for clave in posibles_claves:
        if clave in registro and registro[clave]:
            return _normalizar_tabla(registro[clave])
    return None


def _extraer_columnas_desde_registro_tabla(registro: Dict[str, Any]) -> List[Dict[str, Any]]:
    posibles_claves = [
        "columnas",
        "lista_columnas",
        "Columnas",
        "columns",
    ]

    for clave in posibles_claves:
        if clave in registro and isinstance(registro[clave], list):
            columnas = []
            for col in registro[clave]:
                if isinstance(col, dict):
                    columnas.append({
                        "nombre": col.get("nombre") or col.get("columna") or col.get("name"),
                        "descripcion": col.get("descripcion") or col.get("description"),
                        "tipo_dato": col.get("tipo_dato") or col.get("tipo") or col.get("data_type")
                    })
                else:
                    columnas.append({
                        "nombre": str(col),
                        "descripcion": None,
                        "tipo_dato": None
                    })
            return columnas

    # Variante plana: una fila por columna
    if any(k in registro for k in ["nombre_columna", "columna"]):
        return [{
            "nombre": registro.get("nombre_columna") or registro.get("columna"),
            "descripcion": registro.get("descripcion_columna") or registro.get("descripcion"),
            "tipo_dato": registro.get("tipo_dato") or registro.get("tipo")
        }]

    return []


def _join_menciona_tablas(join_registro: Dict[str, Any], tablas_objetivo: List[str]) -> bool:
    tablas_norm = {_normalizar_tabla(t) for t in tablas_objetivo}

    # caso 1: dos extremos explícitos
    posibles_izq = [
        join_registro.get("tabla_izquierda"),
        join_registro.get("tabla_origen"),
        join_registro.get("left_table"),
    ]
    posibles_der = [
        join_registro.get("tabla_derecha"),
        join_registro.get("tabla_destino"),
        join_registro.get("right_table"),
    ]

    izq = {_normalizar_tabla(x) for x in posibles_izq if x}
    der = {_normalizar_tabla(x) for x in posibles_der if x}

    if (izq & tablas_norm) and (der & tablas_norm):
        return True

    # caso 2: lista de tablas
    for clave in ["tablas", "tables", "tablas_involucradas"]:
        if clave in join_registro:
            tablas_join = {_normalizar_tabla(x) for x in _lista_desde_valor(join_registro[clave])}
            if len(tablas_join & tablas_norm) >= 2:
                return True

    # caso 3: texto libre
    texto = json.dumps(join_registro, ensure_ascii=False).upper()
    apariciones = sum(1 for t in tablas_norm if t in texto)
    return apariciones >= 2


def extraer_contexto_dominio_tablas(
    dominio: str,
    tablas_objetivo: List[str],
    path_tablas_jsonl: str ="tablas.jsonl",
    path_whitelist_joins_jsonl: str="whitelist_joins.jsonl",
    path_reglas_jsonl: str= "Reglas.jsonl",
    path_metricas_jsonl: Optional[str] = "metricas.jsonl"
) -> Dict[str, Any]:
    """
    Extrae métricas, joins y columnas por tabla para un dominio y una lista de tablas.

    Retorna:
    {
        "metricas": [...],
        "joins": [...],
        "tablas": {
            "TABLA_X": [ {nombre, descripcion, tipo_dato}, ... ]
        }
    }
    """
    tablas_objetivo_norm = [_normalizar_tabla(t) for t in tablas_objetivo]

    tablas_data = cargar_jsonl(path_tablas_jsonl)
    joins_data = cargar_jsonl(path_whitelist_joins_jsonl)
    reglas_data = cargar_jsonl(path_reglas_jsonl)
    metricas_data = cargar_jsonl(path_metricas_jsonl) if path_metricas_jsonl else []

    resultado = {
        "metricas": [],
        "joins": [],
        "tablas": {tabla: [] for tabla in tablas_objetivo_norm}
    }

    # -------------------------------------------------
    # 1) TABLAS Y COLUMNAS
    # -------------------------------------------------
    columnas_por_tabla = {tabla: [] for tabla in tablas_objetivo_norm}

    for reg in tablas_data:
        if not _registro_aplica_a_dominio(reg, dominio):
            continue

        nombre_tabla = _extraer_nombre_tabla_desde_registro(reg)
        if not nombre_tabla or nombre_tabla not in columnas_por_tabla:
            continue

        columnas = _extraer_columnas_desde_registro_tabla(reg)
        if columnas:
            columnas_por_tabla[nombre_tabla].extend(columnas)

    # eliminar duplicados de columnas por nombre
    for tabla, columnas in columnas_por_tabla.items():
        seen = set()
        limpias = []
        for col in columnas:
            nombre_col = col.get("nombre")
            key = _normalizar_texto(nombre_col)
            if nombre_col and key not in seen:
                seen.add(key)
                limpias.append(col)
        resultado["tablas"][tabla] = limpias

    # -------------------------------------------------
    # 2) JOINS
    # -------------------------------------------------
    joins_filtrados = []

    for reg in joins_data:
        if not _registro_aplica_a_dominio(reg, dominio):
            # si el archivo de joins no trae dominio, igual evaluamos
            pass

        if _join_menciona_tablas(reg, tablas_objetivo_norm):
            joins_filtrados.append(reg)

    # deduplicar joins por serialización
    seen_joins = set()
    joins_limpios = []
    for j in joins_filtrados:
        key = json.dumps(j, sort_keys=True, ensure_ascii=False)
        if key not in seen_joins:
            seen_joins.add(key)
            joins_limpios.append(j)

    resultado["joins"] = joins_limpios

    # -------------------------------------------------
    # 3) MÉTRICAS
    # -------------------------------------------------
    metricas_filtradas = []

    # 3.1 desde metricas.jsonl si existe
    for reg in metricas_data:
        if not _registro_aplica_a_dominio(reg, dominio):
            continue

        texto_reg = json.dumps(reg, ensure_ascii=False).upper()
        if any(tabla in texto_reg for tabla in tablas_objetivo_norm):
            metricas_filtradas.append(reg)
            continue

        # también incluir si usa columnas de las tablas objetivo
        columnas_tablas = {
            _normalizar_texto(col["nombre"])
            for tabla in tablas_objetivo_norm
            for col in resultado["tablas"].get(tabla, [])
            if col.get("nombre")
        }

        columnas_metrica = set()
        for clave in ["columnas_utilizadas", "columnas", "campos", "fields"]:
            if clave in reg:
                for c in _lista_desde_valor(reg[clave]):
                    if isinstance(c, dict):
                        columnas_metrica.add(_normalizar_texto(c.get("nombre")))
                    else:
                        columnas_metrica.add(_normalizar_texto(c))

        if columnas_tablas & columnas_metrica:
            metricas_filtradas.append(reg)

    # 3.2 respaldo: extraer métricas desde reglas si no hubo metricas.jsonl
    if not metricas_filtradas:
        for reg in reglas_data:
            if not _registro_aplica_a_dominio(reg, dominio):
                continue

            texto_reg = json.dumps(reg, ensure_ascii=False).upper()
            if any(tabla in texto_reg for tabla in tablas_objetivo_norm):
                metrica_inferida = {
                    "origen": "reglas",
                    "dominio": dominio,
                    "descripcion": reg.get("descripcion") or reg.get("regla") or reg.get("detalle"),
                    "registro_original": reg
                }
                metricas_filtradas.append(metrica_inferida)

    # deduplicar métricas
    seen_metricas = set()
    metricas_limpias = []
    for m in metricas_filtradas:
        key = json.dumps(m, sort_keys=True, ensure_ascii=False)
        if key not in seen_metricas:
            seen_metricas.add(key)
            metricas_limpias.append(m)

    resultado["metricas"] = metricas_limpias

    return resultado