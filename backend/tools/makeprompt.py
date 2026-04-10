from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from .context.makeContext import makeContext

MAX_PROMPT_CHARS = 3000
DEFAULT_ENTITY_MODEL = "qwen2.5-coder:3b"
BASE_DIR = Path(__file__).resolve().parent.parent
CATALOG_DIR = BASE_DIR / "catalog"


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



def build_prompt(question: str, context: Dict[str, Any], max_chars: int = MAX_PROMPT_CHARS) -> str:
    base = (
        "Genera solo SQL MySQL/SingleStore. Solo SELECT. No inventes tablas, columnas ni joins. "
        f"Si existe en la lista  de Joins permitidos realiza JOIN con DIM_FECHA, en otro caso utiliza las  columnas, mes  y anio  ppara  filtar los valores"
        "Los meses y  anios  siempre son numero enteros"
        f"Usa columnas NOM_/DESC_ para mostrar nombres conocidos. "
    )
    prompt = f"{base}\nCTX:{context['context_text']}\nQ:{question}\nSQL:"
    if len(prompt) > max_chars:
        print("contextoexcedido")
        excess = len(prompt) - max_chars
        reduced_ctx = context["context_text"][:-excess - 3] + "..." if excess + 3 < len(context["context_text"]) else context["context_text"][: max(0, max_chars - len(base) - len(question) - 20)]
        prompt = f"{base}\nCTX:{reduced_ctx}\nQ:{question}\nSQL:"
    print(prompt[:max_chars])
    return prompt[:max_chars]


def generate_sql(question: str, model: str = DEFAULT_ENTITY_MODEL, client: Optional[object] = None, temperature: float = 0.0, detector_model: Optional[str] = None) -> Dict[str, Any]:
    client = client or build_client()
    context = makeContext(question=question, limit=3000)
    prompt = build_prompt(question, context, max_chars=MAX_PROMPT_CHARS)
    sql = _llm_generate(prompt, model=model, client=client, system_prompt=" ")
    
    
    return {
        "question": question,
        "context": context["context_text"],
        "prompt": prompt,
        "prompt_chars": len(prompt),
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
    print(prompt)
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
