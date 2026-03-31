import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
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

service= DataService()


# --- Endpoint Principal ---
@app.route('/analizar', methods=['POST'])
def analizar():
    pregunta = request.form.get("prompt", "General")
    r1 = getResponse(pregunta, client, MODELO_LOCAL, 0.1)
    d = getData(service, r1)
    limit = 0

    if not isinstance(d, pd.DataFrame):
        while limit <= 3 and not isinstance(d, pd.DataFrame):
            pregunta1 = f"""Corrige tu respuesta tomando en cuenta el siguiente error:
{d}"""
            r1 = getResponse(pregunta1, client, MODELO_LOCAL, 0.1)
            d = getData(service, r1)
            limit += 1

    if not isinstance(d, pd.DataFrame):
        return jsonify({"resultado": "No se han encontrado datos"})

    data_text = d.to_json(orient="records", force_ascii=False)
    final_prompt = f"""Responde la siguiente pregunta:
{pregunta}, solo en base a los siguientes datos adjuntos"""

    response = client.chat.completions.create(
        model=MODELO_RESPONSE,
        messages=[
            {"role": "system", "content": data_text},
            {"role": "user", "content": final_prompt}
        ],
        temperature=0.1,
    )
    result = response.choices[0].message.content
    return jsonify({"resultado": result})


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
    app.run(host='0.0.0.0', port=5000, debug=True)
