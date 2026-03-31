import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import requests
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

CHAT_FILE = DATA_DIR / "chat_history.json"

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:5000/analizar")
FRONTEND_HOST = os.getenv("FRONTEND_HOST", "0.0.0.0")
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "8000"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "180"))

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)


def current_time_str() -> str:
    return datetime.now().strftime("%H:%M")


def create_default_history() -> dict:
    return {
        "active_chat_id": None,
        "chats": []
    }


def save_history(data: dict) -> None:
    with open(CHAT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def create_chat(title: str = "Nuevo chat") -> dict:
    return {
        "id": str(uuid.uuid4()),
        "title": title,
        "preview": "",
        "messages": []
    }


def normalize_message(msg) -> dict:
    if not isinstance(msg, dict):
        return {
            "role": "assistant",
            "content": str(msg),
            "time": current_time_str()
        }

    return {
        "role": msg.get("role", "assistant"),
        "content": str(msg.get("content", "")),
        "time": msg.get("time", current_time_str())
    }


def normalize_chat(chat) -> dict:
    if not isinstance(chat, dict):
        return create_chat()

    raw_messages = chat.get("messages")
    if raw_messages is None:
        raw_messages = chat.get("message", [])

    if not isinstance(raw_messages, list):
        raw_messages = []

    return {
        "id": chat.get("id", str(uuid.uuid4())),
        "title": chat.get("title", "Nuevo chat"),
        "preview": chat.get("preview", ""),
        "messages": [normalize_message(msg) for msg in raw_messages]
    }


def normalize_history(data) -> dict:
    default_data = create_default_history()

    if data is None:
        return default_data

    if isinstance(data, list):
        chats = [normalize_chat(chat) for chat in data]
        return {
            "active_chat_id": chats[0]["id"] if chats else None,
            "chats": chats
        }

    if isinstance(data, dict):
        chats = data.get("chats", [])
        if not isinstance(chats, list):
            chats = []

        normalized = {
            "active_chat_id": data.get("active_chat_id"),
            "chats": [normalize_chat(chat) for chat in chats]
        }

        if normalized["active_chat_id"] and not any(
            chat["id"] == normalized["active_chat_id"] for chat in normalized["chats"]
        ):
            normalized["active_chat_id"] = normalized["chats"][0]["id"] if normalized["chats"] else None

        return normalized

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
    for chat in data.get("chats", []):
        if chat.get("id") == chat_id:
            return chat
    return None


@app.route("/", methods=["GET"])
def index():
    data = load_history()

    requested_chat_id = request.args.get("chat_id", "").strip()
    if requested_chat_id:
        target = find_chat_by_id(data, requested_chat_id)
        if target is not None:
            data["active_chat_id"] = requested_chat_id
            save_history(data)

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
        return jsonify({"ok": False, "error": "Chat no encontrado"}), 404

    data["active_chat_id"] = chat_id
    save_history(data)

    return jsonify({"ok": True, "chat": chat})


@app.route("/api/chats/<chat_id>/message", methods=["POST"])
@app.route("/api/chats/<chat_id>/messages", methods=["POST"])
def send_message(chat_id):
    data = load_history()
    chat = find_chat_by_id(data, chat_id)

    if chat is None:
        return jsonify({"ok": False, "error": "Chat no encontrado"}), 404

    payload = request.get_json(silent=True) or {}

    prompt = str(
        payload.get("message")
        or payload.get("prompt")
        or payload.get("content")
        or payload.get("text")
        or ""
    ).strip()

    if not prompt:
        return jsonify({"ok": False, "error": "El mensaje está vacío"}), 400

    user_message = {
        "role": "user",
        "content": prompt,
        "time": current_time_str()
    }
    chat["messages"].append(user_message)

    if not chat.get("title") or chat["title"] == "Nuevo chat":
        chat["title"] = prompt[:40] + ("..." if len(prompt) > 40 else "")

    chat["preview"] = prompt[:80] + ("..." if len(prompt) > 80 else "")
    data["active_chat_id"] = chat_id
    save_history(data)

    answer = "No se recibió respuesta del backend."

    try:
        response = requests.post(
            BACKEND_URL,
            data={"prompt": prompt},
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()

        result = response.json()

        answer = (
            result.get("resultado")
            or result.get("respuesta")
            or result.get("answer")
            or result.get("response")
            or "No se recibió respuesta del backend."
        )

    except requests.exceptions.Timeout:
        answer = "Error: el backend tardó demasiado en responder."
    except requests.exceptions.ConnectionError as e:
        answer = f"Error: no se pudo conectar con el backend en {BACKEND_URL}. Detalle: {e}"
    except requests.exceptions.HTTPError as e:
        try:
            error_json = response.json()
            backend_msg = error_json.get("error") or error_json.get("message") or str(e)
            answer = f"Error HTTP del backend: {backend_msg}"
        except Exception:
            answer = f"Error HTTP del backend: {str(e)}"
    except Exception as e:
        answer = f"Error inesperado: {str(e)}"

    assistant_message = {
        "role": "assistant",
        "content": answer,
        "time": current_time_str()
    }
    chat["messages"].append(assistant_message)
    save_history(data)

    return jsonify({
        "ok": True,
        "chat": chat,
        "user_message": user_message,
        "assistant_message": assistant_message
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
