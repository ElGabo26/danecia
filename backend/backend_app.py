import json
import os
from typing import Any

import pandas as pd
from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS
from openai import OpenAI

from tools.DataService import DataService
from tools.deploy.sql_service import run_correct_sql_generation_flow, run_sql_generation_flow
from tools.makeConsulta import getData


app = Flask(__name__)
CORS(app)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "300"))

client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key=OLLAMA_API_KEY,
    timeout=OLLAMA_TIMEOUT,
)

MODELO_LOCAL = os.getenv("MODELO_LOCAL", "qwen2.5-coder:3b")
MODELO_RESPONSE = os.getenv("MODELO_RESPONSE", "llama3-chatqa:latest")
MAX_SQL_RETRIES = int(os.getenv("MAX_SQL_RETRIES", "1"))

service = DataService()


def sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def normalize_error_message(error_value: Any) -> str:
    if error_value is None:
        return "Error desconocido al ejecutar SQL."
    if isinstance(error_value, str):
        text = error_value.strip()
        return text or "Error desconocido al ejecutar SQL."
    try:
        return str(error_value)
    except Exception:
        return "Error desconocido al ejecutar SQL."


@app.route('/analizar', methods=['POST'])
def analizar():
    payload = request.get_json(silent=True) or {}
    pregunta = payload.get("prompt") or request.form.get("prompt") or request.values.get("prompt") or "General"
    pregunta = str(pregunta).strip() or "General"
    pregunta = pregunta.upper()

    @stream_with_context
    def generate():
        try:
            yield sse_event({"stage": "inicio", "message": "Solicitud recibida"})
            yield sse_event({"stage": "llm_sql", "message": "Generando consulta SQL"})

            generation = run_sql_generation_flow(
                question=pregunta,
                model=MODELO_LOCAL,
                detector_model=MODELO_RESPONSE,
                base_url=OLLAMA_BASE_URL,
                api_key=OLLAMA_API_KEY,
            )
            sql_query = str(generation.get("sql", "")).strip()
            
            if not sql_query:
                raise ValueError("El generador SQL no retornó una consulta válida.")

            yield sse_event({"stage": "db", "message": "Consultando base de datos"})
            query_result = getData(service, sql_query)
            attempts = 0
            last_error_message = ""

            while attempts < MAX_SQL_RETRIES and not isinstance(query_result, pd.DataFrame):
                last_error_message = normalize_error_message(query_result)
                yield sse_event({
                    "stage": "correccion",
                    "message": f"Corrigiendo consulta: {last_error_message}",
                })
                print(type(sql_query))
                correction = run_correct_sql_generation_flow(
                    question=pregunta,
                    sql=sql_query,
                    error=last_error_message,
                    model=MODELO_LOCAL,
                    detector_model=MODELO_RESPONSE,
                    base_url=OLLAMA_BASE_URL,
                    api_key=OLLAMA_API_KEY,
                )
                corrected_sql = str(correction.get("sql", "")).strip()
                if not corrected_sql:
                    break

                sql_query = corrected_sql
                yield sse_event({"stage": "db", "message": "Reintentando consulta a la base de datos"})
                query_result = getData(service, sql_query)
                attempts += 1

            if not isinstance(query_result, pd.DataFrame):
                error_message = normalize_error_message(query_result)
                yield sse_event({
                    "stage": "error",
                    "message": f"No fue posible ejecutar la consulta SQL: {error_message}",
                    "sql": sql_query,
                })
                return

            if query_result.empty:
                resultado = "No se han encontrado datos"
                yield sse_event({"stage": "fin", "message": resultado, "resultado": resultado, "sql": sql_query})
                return

            yield sse_event({
                "stage": "datos",
                "message": f"Datos obtenidos correctamente: {query_result.shape[0]} filas y {query_result.shape[1]} columnas",
            })

            data_text = query_result.to_json(orient="records", force_ascii=False)
            final_prompt = (
                f"Responde la siguiente pregunta:\n{pregunta}\n"
                f"Solo en base a los siguientes datos adjuntos: {data_text}"
            )

            yield sse_event({"stage": "respuesta_final", "message": "Generando respuesta final"})

            try:
                response = client.chat.completions.create(
                    model=MODELO_RESPONSE,
                    messages=[
                        {"role": "system", "content": data_text},
                        {"role": "user", "content": final_prompt},
                    ],
                    temperature=0.0,
                )
                result = response.choices[0].message.content or data_text
            except Exception:
                result = data_text

            yield sse_event({
                "stage": "fin",
                "message": "Proceso completado",
                "resultado": result,
                "sql": sql_query,
            })

        except Exception as e:
            yield sse_event({"stage": "error", "message": f"Error inesperado en backend: {str(e)}"})

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
        },
    )


@app.route('/', methods=['GET'])
def probar_conexion():
    return "✅ Backend operativo"


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"ok": True, "modelo_sql": MODELO_LOCAL, "modelo_respuesta": MODELO_RESPONSE})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
