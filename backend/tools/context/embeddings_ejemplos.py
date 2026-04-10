
import json
from typing import List, Dict, Any, Optional
import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

ROOT="/ia/deploy/danecia/backend/tools/context"
def cargar_jsonl(path_jsonl: str) -> List[Dict[str, Any]]:
    registros = []
    with open(path_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            registros.append(json.loads(line))
    return registros


def extraer_pregunta(registro: Dict[str, Any]) -> Optional[str]:
    """
    Busca la pregunta en distintas variantes de llave.
    Prioriza 'Pregunta' y luego 'pregunta'.
    """
    pregunta = registro.get("Pregunta") or registro.get("pregunta")
    if pregunta is None:
        return None
    pregunta = str(pregunta).strip()
    return pregunta if pregunta else None


def construir_embeddings_preguntas(
    path_jsonl: str,
    output_npz: str = "preguntas_index.npz",
    output_meta_json: str = "preguntas_meta.json",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> Dict[str, Any]:
    """
    Genera embeddings con un chunk por pregunta.
    Cada registro del JSONL produce un embedding usando únicamente el texto de la pregunta.

    Parámetros
    ----------
    path_jsonl : str
        Ruta al archivo JSONL de ejemplos.
    output_npz : str
        Archivo .npz donde se guardan embeddings y textos.
    output_meta_json : str
        Archivo .json con el registro completo asociado a cada embedding.
    model_name : str
        Modelo de sentence-transformers.

    Retorna
    -------
    dict
        Resumen del proceso.
    """
    registros = cargar_jsonl(path_jsonl)

    preguntas = []
    meta = []

    for i, registro in enumerate(registros):
        pregunta = extraer_pregunta(registro)
        if not pregunta:
            continue

        preguntas.append(pregunta)
        meta.append({
            "idx": len(meta),
            "pregunta": pregunta,
            "registro_completo": registro
        })

    if not preguntas:
        raise ValueError("No se encontraron preguntas válidas en el archivo JSONL.")

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        preguntas,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    np.savez_compressed(
        output_npz,
        embeddings=np.array(embeddings, dtype=np.float32),
        preguntas=np.array(preguntas, dtype=object)
    )

    with open(output_meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {
        "total_registros": len(registros),
        "total_preguntas_vectorizadas": len(preguntas),
        "output_npz": output_npz,
        "model_name": model_name
    }


def similitud_coseno(v1: np.ndarray, v2: np.ndarray) -> float:
    denom = (norm(v1) * norm(v2))
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


def buscar_match_pregunta(
    pregunta_consulta: str,
    index_npz: str = ROOT+"/preguntas_index.npz",
    meta_json: str = ROOT+"/preguntas_meta.json",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Busca la(s) pregunta(s) más similar(es) a una nueva pregunta
    y recupera la información original del JSONL.

    Parámetros
    ----------
    pregunta_consulta : str
        Nueva pregunta a comparar.
    index_npz : str
        Archivo .npz con embeddings.
    meta_json : str
        Archivo .json con el registro completo por embedding.
    model_name : str
        Modelo de sentence-transformers.
    top_k : int
        Número de coincidencias a devolver.

    Retorna
    -------
    dict
        Resultado con ranking de matches y recuperación del registro completo.
    """
    if not pregunta_consulta or not str(pregunta_consulta).strip():
        raise ValueError("La pregunta de consulta está vacía.")

    data = np.load(index_npz, allow_pickle=True)
    embeddings = data["embeddings"]
    preguntas = data["preguntas"]

    with open(meta_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    model = SentenceTransformer(model_name)
    q_emb = model.encode(
        [pregunta_consulta],
        normalize_embeddings=True,
        show_progress_bar=False
    )[0]

    scores = []
    for i, emb in enumerate(embeddings):
        score = similitud_coseno(q_emb, emb)
        scores.append((i, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    top_scores = scores[:top_k]

    resultados = []
    for idx, score in top_scores:
        resultados.append({
            "rank": len(resultados) + 1,
            "score": round(float(score), 6),
            "pregunta_match": str(preguntas[idx]),
            "registro_recuperado": meta[idx]["registro_completo"]["sql"]
        })

    mejor = resultados[0] if resultados else None

    return {
        "pregunta_consulta": pregunta_consulta,
        "top_k": top_k,
        "mejor_match": mejor,
    }

def buscar_tablas(
    path_jsonl: str,
    dominio: str,
) -> List[str]:
    base=cargar_jsonl(path_jsonl)
    print(dominio)
    result=list(map(lambda x: x["tablas_principales"] if x["nombre_dominio"].strip()==dominio else None,base))
    tablas=[i  for i in result if  i ]
    
    return tablas[0]


if __name__ == "__main__":
    # Ejemplo de uso:
    # 1) Construir embeddings desde el JSONL
    print(buscar_tablas("./dominios.jsonl",'finanzas'))
    