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
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "8000"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "180"))

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)


def load_history():
    if not CHAT_FILE.exists():
        return {"active_chat_id": None, "chats": []}

    try:
        with open(CHAT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"active_chat_id": None, "chats": []}


def save_history(data):
    with open(CHAT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def create_chat(title="Nuevo chat"):
    return {
        "id": str(uuid.uuid4()),
        "title": title,
        "preview": "",
        "message": []
    }


def get_active_chat(data):
    active_id = data.get("active_chat_id")
    chats = data.get("chats", [])

    if not chats:
        new_chat = create_chat()
        data["chats"] = [new_chat]
        data["active_chat_id"] = new_chat["id"]
        save_history(data)
        return new_chat

    for chat in chats:
        if chat["id"] == active_id:
            return chat

    data["active_chat_id"] = chats[0]["id"]
    save_history(data)
    return chats[0]


@app.route("/", methods=["GET"])
def index():
    data = load_history()
    active_chat = get_active_chat(data)
    return render_template(
        "index.html",
        chats=data["chats"],
        active_chat=active_chat
    )


@app.route("/api/chats", methods=["GET"])
def list_chats():
    data = load_history()
    active_chat = get_active_chat(data)
    return jsonify({
        "chats": data["chats"],
        "active_chat_id": active_chat["id"]
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

    for chat in data["chats"]:
        if chat["id"] == chat_id:
            data["active_chat_id"] = chat_id
            save_history(data)
            return jsonify({"ok": True, "chat": chat})

    return jsonify({"ok": False, "error": "Chat no encontrado"}), 404


@app.route("/api/chats/<chat_id>/message", methods=["POST"])
def send_message(chat_id):
    data = load_history()
    payload = request.get_json(silent=True) or {}
    prompt = (payload.get("message") or "").strip()

    if not prompt:
        return jsonify({"ok": False, "error": "El mensaje está vacío"}), 400

    target_chat = None
    for chat in data["chats"]:
        if chat["id"] == chat_id:
            target_chat = chat
            break

    if target_chat is None:
        return jsonify({"ok": False, "error": "Chat no encontrado"}), 404

    target_chat["message"].append({
        "role": "user",
        "content": prompt
    })

    if not target_chat["title"] or target_chat["title"] == "Nuevo chat":
        target_chat["title"] = prompt[:40] + ("..." if len(prompt) > 40 else "")

    target_chat["preview"] = prompt[:80] + ("..." if len(prompt) > 80 else "")
    data["active_chat_id"] = chat_id
    save_history(data)

    try:
        response = requests.post(
            BACKEND_URL,
            json={"prompt": prompt},
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        result = response.json()

        answer = (
            result.get("respuesta")
            or result.get("answer")
            or result.get("response")
            or "No se recibió respuesta del backend."
        )

    except requests.exceptions.Timeout:
        answer = "Error: el backend tardó demasiado en responder."
    except requests.exceptions.ConnectionError:
        answer = "Error: no se pudo conectar con el backend."
        print(prompt)
    except requests.exceptions.HTTPError as e:
        answer = f"Error HTTP del backend: {str(e)}"
    except Exception as e:
        answer = f"Error inesperado: {str(e)}"

    target_chat["message"].append({
        "role": "assistant",
        "content": answer
    })

    save_history(data)

    return jsonify({
        "ok": True,
        "chat": target_chat
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