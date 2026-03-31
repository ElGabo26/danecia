import json
import os
import uuid
from pathlib import Path

import requests
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

CHAT_FILE = DATA_DIR / "chat_history.json"

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:5000/analizar")
FRONTEND_HOST = os.getenv("FRONTEND_HOST", "0.0.0.0")
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "8001"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "180"))

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)


def save_history(data: dict) -> None:
    with open(CHAT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def create_default_history() -> dict:
    return {
        "active_chat_id": None,
        "chats": []
    }


def create_chat(title: str = "Nuevo chat") -> dict:
    return {
        "id": str(uuid.uuid4()),
        "title": title,
        "preview": "",
        "messages": []
    }


def normalize_history(data) -> dict:
    default_data = create_default_history()

    if data is None:
        return default_data

    # Caso correcto
    if isinstance(data, dict):
        if "active_chat_id" not in data:
            data["active_chat_id"] = None

        if "chats" not in data or not isinstance(data["chats"], list):
            data["chats"] = []

        # Normalizar cada chat
        normalized_chats = []
        for chat in data["chats"]:
            if not isinstance(chat, dict):
                continue

            normalized_chat = {
                "id": chat.get("id", str(uuid.uuid4())),
                "title": chat.get("title", "Nuevo chat"),
                "preview": chat.get("preview", ""),
                "messages": chat.get("messages", [])
            }

            if not isinstance(normalized_chat["messages"], list):
                normalized_chat["messages"] = []

            normalized_chats.append(normalized_chat)

        data["chats"] = normalized_chats
        return data

    # Compatibilidad con formato antiguo: lista de chats
    if isinstance(data, list):
        chats = []
        for chat in data:
            if not isinstance(chat, dict):
                continue

            normalized_chat = {
                "id": chat.get("id", str(uuid.uuid4())),
                "title": chat.get("title", "Nuevo chat"),
                "preview": chat.get("preview", ""),
                "messages": chat.get("messages", [])
            }

            if not isinstance(normalized_chat["messages"], list):
                normalized_chat["messages"] = []

            chats.append(normalized_chat)

        return {
            "active_chat_id": chats[0]["id"] if chats else None,
            "chats": chats
        }

    return default_data


def load_history() -> dict:
    default_data = create_default_history()

    if not CHAT_FILE.exists():
        save_history(default_data)
        return default_data

    try:
        with open(CHAT_FILE, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        data = normalize_history(raw_data)
        save_history(data)
        return data

    except Exception:
        save_history(default_data)
        return default_data


def get_active_chat(data: dict) -> dict:
    if not isinstance(data, dict):
        data = create_default_history()

    chats = data.get("chats", [])
    active_id = data.get("active_chat_id")

    if not chats:
        new_chat = create_chat()
        data["chats"] = [new_chat]
        data["active_chat_id"] = new_chat["id"]
        save_history(data)
        return new_chat

    for chat in chats:
        if chat.get("id") == active_id:
            return chat

    data["active_chat_id"] = chats[0]["id"]
    save_history(data)
    return chats[0]


def find_chat_by_id(data: dict, chat_id: str):
    chats = data.get("chats", [])
    for chat in chats:
        if chat.get("id") == chat_id:
            return chat
    return None


@app.route("/", methods=["GET"])
def index():
    data = load_history()
    active_chat = get_active_chat(data)

    return render_template(
        "index.html",
        chats=data.get("chats", []),
        active_chat=active_chat
    )


@app.route("/api/chats", methods=["GET"])
def list_chats():
    data = load_history()
    active_chat = get_active_chat(data)

    return jsonify({
        "ok": True,
        "chats": data.get("chats", []),
        "active_chat_id": active_chat.get("id")
    })


@app.route("/api/chats", methods=["POST"])
def new_chat():
    data = load_history()
    chat = create_chat()

    data["chats"].insert(0, chat)
    data["active_chat_id"] = chat["id"]
    save_history(data)

    return jsonify({
        "ok": True,
        "chat": chat,
        "active_chat_id": chat["id"]
    })


@app.route("/api/chats/<chat_id>", methods=["GET"])
def get_chat(chat_id):
    data = load_history()
    chat = find_chat_by_id(data, chat_id)

    if chat is None:
        return jsonify({
            "ok": False,
            "error": "Chat no encontrado"
        }), 404

    data["active_chat_id"] = chat_id
    save_history(data)

    return jsonify({
        "ok": True,
        "chat": chat
    })


@app.route("/api/chats/<chat_id>/message", methods=["POST"])
def send_message(chat_id):
    data = load_history()
    chat = find_chat_by_id(data, chat_id)

    if chat is None:
        return jsonify({
            "ok": False,
            "error": "Chat no encontrado"
        }), 404

    payload = request.get_json(silent=True) or {}
    prompt = str(payload.get("message", "")).strip()

    if not prompt:
        return jsonify({
            "ok": False,
            "error": "El mensaje está vacío"
        }), 400

    # Guardar mensaje del usuario
    chat["messages"].append({
        "role": "user",
        "content": prompt
    })

    if not chat.get("title") or chat["title"] == "Nuevo chat":
        chat["title"] = prompt[:40] + ("..." if len(prompt) > 40 else "")

    chat["preview"] = prompt[:80] + ("..." if len(prompt) > 80 else "")
    data["active_chat_id"] = chat_id
    save_history(data)

    answer = "No se recibió respuesta del backend."

    try:
        response = requests.post(
            BACKEND_URL,
            json={"prompt": prompt},
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()

        result = response.json()

        # Compatibilidad con distintas claves de respuesta
        answer = (
            result.get("respuesta")
            or result.get("answer")
            or result.get("response")
            or result.get("resultado")
            or "No se recibió respuesta del backend."
        )

    except requests.exceptions.Timeout:
        answer = "Error: el backend tardó demasiado en responder."
    except requests.exceptions.ConnectionError:
        answer = "Error: no se pudo conectar con el backend."
    except requests.exceptions.HTTPError as e:
        try:
            error_json = response.json()
            backend_msg = error_json.get("error") or error_json.get("message") or str(e)
            answer = f"Error HTTP del backend: {backend_msg}"
        except Exception:
            answer = f"Error HTTP del backend: {str(e)}"
    except Exception as e:
        answer = f"Error inesperado: {str(e)}"

    # Guardar respuesta del asistente
    chat["messages"].append({
        "role": "assistant",
        "content": answer
    })

    save_history(data)

    return jsonify({
        "ok": True,
        "chat": chat
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "frontend",
        "backend_url": BACKEND_URL
    })


if __name__ == "__main__":
    app.run(
        host=FRONTEND_HOST,
        port=FRONTEND_PORT,
        debug=False
    )