
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

def _normalizar_texto(texto: str) -> str:
    texto = str(texto).lower().strip()
    texto = "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )
    texto = re.sub(r"[^a-z0-9\s_]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

def cargar_jsonl(path_jsonl: str) -> List[Dict]:
    registros = []
    with open(path_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            registros.append(json.loads(line))
    return registros

def _match_terms(texto: str, valores, peso: int, campo: str):
    score = 0
    evidencias = []
    if not isinstance(valores, list):
        valores = [valores]

    for termino in valores:
        termino_norm = _normalizar_texto(termino)
        if termino_norm and termino_norm in texto:
            score += peso
            evidencias.append({
                "campo": campo,
                "termino": termino,
                "peso": peso
            })
    return score, evidencias

def identificar_dominio(
    pregunta: str,
    catalogo_jsonl: List[Dict],
) -> Dict:
    texto = _normalizar_texto(pregunta)
    resultados = []

    for dominio in catalogo_jsonl:
        score = 0
        evidencias = []

        campos_pesos = [
            ("dominio", 1),
            ("descripcion", 1),
            ("palabras_clave", 3),
            ("sinonimos", 2),
            ("metricas_asociadas", 2),
            ("tablas_principales", 1),
            ("columnas_clave", 1),
            ("intenciones_validas", 1),
        ]

        for campo, peso in campos_pesos:
            if campo in dominio:
                s, e = _match_terms(texto, dominio[campo], peso, campo)
                score += s
                evidencias.extend(e)

        resultados.append({
            "dominio": dominio.get("dominio"),
            "score": score,
            "evidencias": evidencias
        })

    resultados.sort(key=lambda x: x["score"], reverse=True)
    mejor = resultados[0] if resultados else {"dominio": None, "score": 0, "evidencias": []}
    

    ambiguo = False
    

    if mejor["score"] == 0:
        ambiguo = True
        
    

    
    return {
        "pregunta": pregunta,
        "dominio_predicho": None if ambiguo else mejor["dominio"],
        "score": mejor["score"],
        "ambiguo": ambiguo,
    }
    
    

if __name__ == "__main__":
    base = Path(".")
    catalogo = cargar_jsonl(str(base / "dominios_catalogo.jsonl"))
    ejemplos = cargar_jsonl(str(base / "ejemplos_dominios.jsonl"))
    pregunta = "dame el informe de Ebitda por empresa y consolidado"

    resultado = identificar_dominio(pregunta, catalogo, ejemplos)
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
