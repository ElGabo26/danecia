from .embeddings_ejemplos import buscar_match_pregunta, buscar_tablas
from .identificar_dominio import identificar_dominio, cargar_jsonl
from .addings  import extraer_contexto_dominio_tablas
from pathlib import Path

from typing import Dict, Any, List
ROOT="/ia/deploy/danecia/backend/tools/context"

def construir_prompt_contexto(contexto: Dict[str, Any], max_columnas_por_tabla: int = 20, max_joins: int = 10, max_metricas: int = 10) -> str:
    """
    Convierte la salida de extraer_contexto_dominio_tablas en un prompt corto.
    
    Estructura esperada de contexto:
    {
        "metricas": [...],
        "joins": [...],
        "tablas": {
            "TABLA": [ {"nombre": ..., "descripcion": ..., "tipo_dato": ...}, ... ]
        }
    }
    """

    partes: List[str] = []

    # 1) Tablas y columnas
    tablas = contexto.get("tablas", {})
    if tablas:
        bloques_tablas = []
        for tabla, columnas in tablas.items():
            cols = []
            for col in (columnas or [])[:max_columnas_por_tabla]:
                nombre = col.get("nombre", "")
                tipo = col.get("tipo_dato")
                desc = col.get("descripcion")

                if tipo and desc:
                    cols.append(f"{nombre}({tipo}: {desc})")
                elif tipo:
                    cols.append(f"{nombre}({tipo})")
                elif desc:
                    cols.append(f"{nombre}({desc})")
                else:
                    cols.append(nombre)

            bloques_tablas.append(f"{tabla}: " + ", ".join(cols))
        partes.append("TABLAS:\n" + "\n".join(bloques_tablas))

    # 2) Joins
    joins = contexto.get("joins", [])
    if joins:
        bloques_joins = []
        for j in joins[:max_joins]:
            if isinstance(j, dict):
                izq = j.get("tabla_izquierda") or j.get("tabla_origen") or j.get("left_table")
                der = j.get("tabla_derecha") or j.get("tabla_destino") or j.get("right_table")
                cond = j.get("condicion") or j.get("join_condition") or j.get("on")
                tipo = j.get("tipo_join") or j.get("join_type")

                if izq or der or cond:
                    texto = f"{izq} -> {der}"
                    if cond:
                        texto += f" ON {cond}"
                    if tipo:
                        texto += f" [{tipo}]"
                    bloques_joins.append(texto)
                else:
                    bloques_joins.append(str(j))
            else:
                bloques_joins.append(str(j))

        partes.append("JOINS:\n" + "\n".join(bloques_joins))

    # 3) Métricas
    metricas = contexto.get("metricas", [])
    if metricas:
        bloques_metricas = []
        for m in metricas[:max_metricas]:
            if isinstance(m, dict):
                nombre = (
                    m.get("nombre")
                    or m.get("metrica")
                    or m.get("descripcion")
                    or "metrica_sin_nombre"
                )
                calculo = m.get("calculo")
                columnas = m.get("columnas_utilizadas") or m.get("columnas") or m.get("campos")

                linea = str(nombre)
                if columnas:
                    if isinstance(columnas, list):
                        linea += f" | cols: {', '.join(map(str, columnas))}"
                    else:
                        linea += f" | cols: {columnas}"
                if calculo:
                    linea += f" | calc: {calculo}"

                bloques_metricas.append(linea)
            else:
                bloques_metricas.append(str(m))

        partes.append("METRICAS:\n" + "\n".join(bloques_metricas))

    # 4) Instrucción final compacta
    partes.append(
        "USA SOLO las tablas, columnas, joins y métricas indicadas. "
        "No inventes columnas ni joins. "
        "Si falta información, responde que no existe suficiente contexto."
    )

    return "\n\n".join(partes)

def makeContext(question,limit):
    base = Path(".")
    catalogo = cargar_jsonl("/ia/deploy/danecia/backend/tools/context/dominios_catalogo.jsonl")
    dominio0=identificar_dominio(question,catalogo)
    dominio=dominio0["dominio_predicho"]
    if not dominio:
        return None
    ejemplo=buscar_match_pregunta(question)["mejor_match"]
    tablas = buscar_tablas(ROOT+"/dominios.jsonl",dominio)
    context=extraer_contexto_dominio_tablas(dominio,tablas)
    promptContext=construir_prompt_contexto(context)
    instrucciones="teniendo en  cuenta este contexto \n"
    if ejemplo["registro_recuperado"]:
        return {"context_text":instrucciones +  f"replica  este  ejemplo {ejemplo["registro_recuperado"]}"}
    return {"context_text":instrucciones + promptContext}
    
    
    
    
    
    