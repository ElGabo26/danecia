import json
import os

import pandas as pd
from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS
from openai import OpenAI

from tools.DataService import DataService
from tools.makeprompt import generate_sql
from tools.makeConsulta import getData

app = Flask(__name__)
CORS(app)

client = OpenAI(
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
    timeout=float(os.getenv("OLLAMA_TIMEOUT", "300")),
)

MODELO_LOCAL = os.getenv("MODELO_LOCAL", "qwen2.5-coder:3b")
MODELO_RESPONSE = os.getenv("MODELO_RESPONSE", "llama3-chatqa:latest")
MAX_SQL_RETRIES = int(os.getenv("MAX_SQL_RETRIES", "3"))

service = DataService()


def sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


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

            r0 = generate_sql(pregunta)
            r1 = r0.get('sql', '')
            print(r1)
            yield sse_event({"stage": "db", "message": "Consultando base de datos"})
            d = getData(service, r1)
            print(d)
            
            if not isinstance(d, pd.DataFrame):

                resultado = f"Datos no  encontrados, se  generó la  siguiente consulta: \n {r1}"
                yield sse_event({"stage": "fin", "message": resultado, "resultado": resultado})
                return

            yield sse_event({
                "stage": "datos",
                "message": f"Datos obtenidos correctamente: {d.shape[0]} filas y {d.shape[1]} columnas",
            })

            data_text = d.to_json(orient="records", force_ascii=False)
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
                result = response.choices[0].message.content
                print(result,'holi')
            except Exception:
                result = data_text
                print(result)
                
            print(result)
            yield sse_event({"stage": "fin", "message": result, "resultado": result})

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
