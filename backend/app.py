from flask import Flask, jsonify, Response, request, stream_with_context
from flask_cors import CORS
from openai import OpenAI
import json
import pandas as pd

from tools.DataService import DataService
from tools.makeResponse import getResponse
from tools.makeConsulta import getData

app = Flask(__name__)
CORS(app)

# --- CONFIGURACIÓN LOCAL (OLLAMA) ---
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

MODELO_LOCAL = "qwen2.5-coder:3b"
MODELO_RESPONSE = "llama3-chatqa:latest"

service = DataService()


def sse_event(payload: dict) -> str:
    """Formatea un evento SSE estándar."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.route('/analizar', methods=['POST'])
def analizar():
    # Soporta JSON y form-data para evitar incompatibilidades con el frontend.
    payload = request.get_json(silent=True) or {}
    pregunta = (
        payload.get("prompt")
        or request.form.get("prompt")
        or request.values.get("prompt")
        or "General"
    )

    pregunta = str(pregunta).strip() or "General"

    @stream_with_context
    def generate():
        try:
            yield sse_event({"stage": "inicio", "message": "Solicitud recibida"})

            print("pensando")
            print(pregunta)

            yield sse_event({"stage": "llm_sql", "message": "Generando consulta SQL"})
            r1 = getResponse(pregunta, client, MODELO_LOCAL, 0.1)

            if not r1:
                r1 = "NO SELECT;"
                yield sse_event(
                    {
                        "stage": "llm_sql",
                        "message": "No se generó SQL válida, se usará valor por defecto",
                    }
                )

            yield sse_event({"stage": "db", "message": "Consultando base de datos"})
            d = getData(service, r1)
            limit = 0

            while limit <= 3 and not isinstance(d, pd.DataFrame):
                yield sse_event(
                    {
                        "stage": "correccion",
                        "message": f"Corrigiendo consulta. Intento {limit + 1}",
                    }
                )

                pregunta1 = f"""Corrige tu respuesta tomando en cuenta el siguiente error:
{d}"""
                r1 = getResponse(pregunta1, client, MODELO_LOCAL, 0.1)
                print(r1)

                yield sse_event(
                    {"stage": "db", "message": "Reintentando consulta a la base de datos"}
                )
                d = getData(service, r1)
                limit += 1

            if not isinstance(d, pd.DataFrame):
                resultado = "No se han encontrado datos"
                yield sse_event(
                    {"stage": "fin", "message": resultado, "resultado": resultado}
                )
                return

            print(d.shape)

            yield sse_event(
                {
                    "stage": "datos",
                    "message": f"Datos obtenidos correctamente: {d.shape[0]} filas y {d.shape[1]} columnas",
                }
            )

            data_text = d.to_json(orient="records", force_ascii=False)
            final_prompt = f"""Responde la siguiente pregunta:
{pregunta}, solo en base a los siguientes datos adjuntos: {data_text}"""

            print(final_prompt)

            yield sse_event(
                {"stage": "respuesta_final", "message": "Generando respuesta final"}
            )

            try:
                response = client.chat.completions.create(
                    model=MODELO_RESPONSE,
                    messages=[
                        {"role": "system", "content": data_text},
                        {"role": "user", "content": final_prompt},
                    ],
                    temperature=0.1,
                )
                result = response.choices[0].message.content
                print(result)
            except Exception as e:
                print("llama error")
                print(str(e))
                result = data_text

            yield sse_event(
                {"stage": "fin", "message": "Proceso completado", "resultado": result}
            )

        except Exception as e:
            error_msg = f"Error inesperado en backend: {str(e)}"
            print(error_msg)
            yield sse_event({"stage": "error", "message": error_msg})

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
    return jsonify(
        {
            "ok": True,
            "modelo_sql": MODELO_LOCAL,
            "modelo_respuesta": MODELO_RESPONSE,
        }
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
