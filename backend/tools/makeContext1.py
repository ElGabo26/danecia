import os
import json
import re
import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer


# =========================================================
# CONFIGURACIÓN
# =========================================================
RAG_PATH = "rag"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K_POR_ARCHIVO = 5


# =========================================================
# CARGA DEL MODELO DE EMBEDDINGS
# =========================================================
embed_model = SentenceTransformer(EMBED_MODEL_NAME)


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================
def embed_query(query: str) -> np.ndarray:
    """Convierte una consulta en embedding."""
    return embed_model.encode([query])[0]


def normalizar_vector(v: np.ndarray) -> np.ndarray:
    """Normaliza un vector."""
    return v / (norm(v) + 1e-10)


def normalizar_matriz(m: np.ndarray) -> np.ndarray:
    """Normaliza una matriz fila por fila."""
    return m / (norm(m, axis=1, keepdims=True) + 1e-10)


def encontrar_pares_rag(rag_path: str):
    """
    Busca pares válidos de archivos .npz y .json dentro de la carpeta rag.
    """
    archivos = os.listdir(rag_path)
    npz_files = sorted([f for f in archivos if f.lower().endswith(".npz")])
    json_files = set([f for f in archivos if f.lower().endswith(".json")])

    pares_validos = []

    for npz_file in npz_files:
        npz_path = os.path.join(rag_path, npz_file)
        base_name = os.path.splitext(npz_file)[0]

        candidatos_json = [
            f"{base_name}.json",
            f"{base_name.replace('_index', '_meta')}.json",
            f"{base_name.replace('index', 'meta')}.json",
            f"{base_name.replace('_embeddings', '_meta')}.json",
            f"{base_name.replace('embeddings', 'meta')}.json",
        ]
        candidatos_json = [c for c in candidatos_json if c in json_files]

        if not candidatos_json:
            print(f"[ADVERTENCIA] No se encontró JSON asociado para: {npz_file}")
            continue

        par_encontrado = False
        for json_file in candidatos_json:
            json_path = os.path.join(rag_path, json_file)

            try:
                data = np.load(npz_path)

                if "embeddings" not in data:
                    print(f"[ADVERTENCIA] El archivo {npz_file} no contiene la clave 'embeddings'")
                    continue

                embeddings = data["embeddings"]

                with open(json_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                if not isinstance(meta, list):
                    print(f"[ADVERTENCIA] El archivo {json_file} no contiene una lista")
                    continue

                if len(embeddings) != len(meta):
                    print(
                        f"[ADVERTENCIA] No coincide el tamaño entre {npz_file} "
                        f"({len(embeddings)} embeddings) y {json_file} ({len(meta)} metadatos)"
                    )
                    continue

                pares_validos.append((npz_path, json_path))
                par_encontrado = True
                break

            except Exception as e:
                print(f"[ERROR] No se pudo validar el par {npz_file} - {json_file}: {e}")

        if not par_encontrado:
            print(f"[ADVERTENCIA] No se encontró un par válido para {npz_file}")

    return pares_validos


def inferir_nombre_documento(item: dict, npz_path: str, json_path: str) -> str:
    """
    Intenta obtener el nombre real del archivo origen del chunk.
    Si no existe en el metadata, usa el nombre del json o npz.
    """
    posibles_campos = [
        "doc", "document", "documento", "source", "source_file",
        "file", "filename", "archivo", "origen"
    ]

    for campo in posibles_campos:
        valor = item.get(campo)
        if valor:
            return os.path.basename(str(valor))

    # fallback: usar nombre del json/meta
    return os.path.basename(json_path)


def cargar_indices_por_archivo(rag_path: str):
    """
    Carga cada archivo de embeddings y metadatos por separado.
    Devuelve una lista de índices independientes.
    """
    pares = encontrar_pares_rag(rag_path)

    if not pares:
        raise FileNotFoundError(
            f"No se encontraron pares válidos de archivos .npz y .json en la carpeta: {rag_path}"
        )

    indices = []

    for npz_path, json_path in pares:
        try:
            data = np.load(npz_path)
            embeddings = data["embeddings"]

            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            meta_enriquecida = []
            for idx, item in enumerate(meta):
                if not isinstance(item, dict):
                    item = {"text": str(item)}

                item = item.copy()
                item["_source_npz"] = os.path.basename(npz_path)
                item["_source_json"] = os.path.basename(json_path)
                item["_document_name"] = inferir_nombre_documento(item, npz_path, json_path)
                item["_chunk_id"] = idx
                meta_enriquecida.append(item)

            indice = {
                "npz_path": npz_path,
                "json_path": json_path,
                "npz_name": os.path.basename(npz_path),
                "json_name": os.path.basename(json_path),
                "document_name": os.path.basename(json_path),
                "embeddings": embeddings,
                "embeddings_norm": normalizar_matriz(embeddings),
                "meta": meta_enriquecida,
            }

            indices.append(indice)
            print(f"[OK] Cargado: {os.path.basename(npz_path)} + {os.path.basename(json_path)} -> {len(meta)} chunks")

        except Exception as e:
            print(f"[ERROR] No se pudo cargar {npz_path} y {json_path}: {e}")

    if not indices:
        raise ValueError("No se pudo cargar ningún índice válido.")

    print(f"[OK] Total de archivos índice cargados: {len(indices)}")
    return indices


# =========================================================
# CARGA GLOBAL POR ARCHIVO
# =========================================================
INDICES_RAG = cargar_indices_por_archivo(RAG_PATH)


# =========================================================
# RECUPERACIÓN DE CONTEXTO
# =========================================================
def retrieve_context_per_file(question: str, k_per_file: int = 5):
    """
    Busca el top-k dentro de cada archivo por separado.
    Devuelve una lista con resultados agrupados por archivo.
    """
    q_vec = normalizar_vector(embed_query(question))
    resultados_por_archivo = []

    for indice in INDICES_RAG:
        sims = indice["embeddings_norm"] @ q_vec
        top_idx = np.argsort(-sims)[:k_per_file]

        resultados_archivo = []
        for i in top_idx:
            item = indice["meta"][i]
            resultados_archivo.append({
                "score": float(sims[i]),
                "text": item.get("text", ""),
                "metadata": item,
                "document_name": item.get("_document_name", indice["document_name"]),
                "source_npz": indice["npz_name"],
                "source_json": indice["json_name"],
            })

        resultados_por_archivo.append({
            "file_name": indice["json_name"],
            "npz_name": indice["npz_name"],
            "document_name": indice["document_name"],
            "results": resultados_archivo,
        })

    return resultados_por_archivo


def retrieve_context_flat(question: str, k_per_file: int = 5):
    """
    Devuelve una lista plana con todos los resultados top-k por archivo.
    """
    grouped = retrieve_context_per_file(question, k_per_file=k_per_file)
    flat = []

    for grupo in grouped:
        flat.extend(grupo["results"])

    return flat


# =========================================================
# FORMATEO DE CONTEXTO
# =========================================================
def format_context_for_prompt(grouped_results) -> str:
    """
    Construye el contexto incluyendo el nombre del archivo
    y el top 5 de chunks encontrados por cada archivo.
    """
    bloques = []

    for grupo in grouped_results:
        encabezado = (
            f"==============================\n"
            f"ARCHIVO INDICE: {grupo['npz_name']}\n"
            f"ARCHIVO METADATA: {grupo['file_name']}\n"
            f"TOP CHUNKS ENCONTRADOS EN ESTE ARCHIVO:\n"
            f"=============================="
        )

        chunks_formateados = [encabezado]

        for pos, r in enumerate(grupo["results"], start=1):
            texto = r["text"].strip()
            doc_name = r["document_name"]

            bloque_chunk = (
                f"[TOP {pos} | score={r['score']:.4f}]\n"
                f"Archivo origen del chunk: {doc_name}\n"
                f"Fuente embeddings: {r['source_npz']}\n"
                f"Fuente metadatos: {r['source_json']}\n"
                f"Contenido:\n{texto}\n"
            )
            chunks_formateados.append(bloque_chunk)

        bloques.append("\n".join(chunks_formateados))

    return "\n\n".join(bloques)


# =========================================================
# LIMPIEZA DE RESPUESTA
# =========================================================
def limpiar_respuesta_deepseek(texto: str) -> str:
    """
    Elimina tags <think>...</think> y deja solo el texto limpio.
    """
    texto_limpio = re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL)
    texto_limpio = texto_limpio.replace("<think>", "").replace("</think>", "")
    return texto_limpio.strip()


# =========================================================
# PROMPT
# =========================================================
with open("/mnt/deploy/danecia/backend/tools/instrucciones.txt", "r", encoding="utf-8") as f:
    instrucciones = f.read()


def build_prompt(question: str, k_per_file: int = TOP_K_POR_ARCHIVO) -> str:
    """
    Construye el prompt agregando el top-k de cada archivo
    e incluyendo el nombre del archivo origen de cada chunk.
    """
    grouped_results = retrieve_context_per_file(question, k_per_file=k_per_file)
    contexto = format_context_for_prompt(grouped_results)

    prompt = f"""
{instrucciones}

TOMAN EN CUENTA EL SIGUIENTE CONTEXTO:
{contexto}

""".strip()

    return prompt


# =========================================================
# PRUEBA RÁPIDA
# =========================================================
if __name__ == "__main__":
    pregunta = "¿Qué información existe sobre clientes y ventas?"

    resultados_agrupados = retrieve_context_per_file(pregunta, k_per_file=5)

    print("\n=== RESULTADOS POR ARCHIVO ===")
    for grupo in resultados_agrupados:
        print("\n--------------------------------------------------")
        print(f"Archivo índice   : {grupo['npz_name']}")
        print(f"Archivo metadata : {grupo['file_name']}")
        print("--------------------------------------------------")

        for i, r in enumerate(grupo["results"], start=1):
            print(f"\n[{i}] score={r['score']:.4f}")
            print(f"Archivo origen del chunk: {r['document_name']}")
            print(f"Fuente NPZ : {r['source_npz']}")
            print(f"Fuente JSON: {r['source_json']}")
            print(r["text"][:500])

    print("\n=== PROMPT GENERADO ===")
    print(build_prompt(pregunta, k_per_file=5))