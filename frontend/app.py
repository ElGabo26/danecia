
import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import requests
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
CHAT_FILE = DATA_DIR / "chat_history.json"
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:5000")
TIMEOUT_SECONDS = int(os.getenv("BACKEND_TIMEOUT", "180"))


def _default_chat():
    return {
        "id": str(uuid.uuid4()),
        "title": "Nuevo chat",
        "preview": "Sin mensajes todavía",
        "messages": [
            {
                "role": "assistant",
                "content": "Hola. Bienvenido al asistente de Grupo Danec.",
                "time": datetime.now().strftime("%H:%M"),
            }
        ],
    }


def load_history():
    if not CHAT_FILE.exists():
        chats = [_default_chat()]
        save_history(chats)
        return chats

    try:
        with CHAT_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and data:
            return data
    except Exception:
        pass

    chats = [_default_chat()]
    save_history(chats)
    return chats


def save_history(chats):
    with CHAT_FILE.open("w", encoding="utf-8") as f:
        json.dump(chats, f, ensure_ascii=False, indent=2)


def get_chat(chats, chat_id):
    return next((chat for chat in chats if chat["id"] == chat_id), None)


def build_preview(messages):
    valid = [m for m in messages if m.get("role") == "assistant" and m.get("content")]
    if valid:
        return valid[-1]["content"][:90]
    valid = [m for m in messages if m.get("content")]
    return valid[-1]["content"][:90] if valid else "Sin mensajes todavía"


@app.route("/")
def index():
    chats = load_history()
    active_chat_id = request.args.get("chat_id") or chats[0]["id"]
    active_chat = get_chat(chats, active_chat_id) or chats[0]
    return render_template(
        "index.html",
        chats=chats,
        active_chat=active_chat,
        backend_url=BACKEND_URL,
    )


@app.route("/api/chats", methods=["GET"])
def list_chats():
    return jsonify({"ok": True, "chats": load_history()})


@app.route("/api/chats/new", methods=["POST"])
def new_chat():
    chats = load_history()
    chat = _default_chat()
    chats.insert(0, chat)
    save_history(chats)
    return jsonify({"ok": True, "chat": chat})


@app.route("/api/chats/<chat_id>/messages", methods=["POST"])
def send_message(chat_id):
    payload = request.get_json(silent=True) or {}
    prompt = (payload.get("prompt") or "").strip()

    if not prompt:
        return jsonify({"ok": False, "error": "El prompt está vacío."}), 400

    chats = load_history()
    chat = get_chat(chats, chat_id)
    if not chat:
        return jsonify({"ok": False, "error": "No se encontró el chat solicitado."}), 404

    current_time = datetime.now().strftime("%H:%M")
    user_message = {"role": "user", "content": prompt, "time": current_time}
    chat["messages"].append(user_message)

    if chat["title"] == "Nuevo chat":
        chat["title"] = prompt[:40]

    try:
        response = requests.post(
            f"{BACKEND_URL.rstrip('/')}/analizar",
            data={"prompt": prompt},
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        body = response.json()
        assistant_text = body.get("resultado") or "No se recibió contenido desde el backend."
    except requests.RequestException as exc:
        assistant_text = f"No fue posible conectar con el backend: {exc}"
    except ValueError:
        assistant_text = "El backend respondió con un formato no válido."

    assistant_message = {
        "role": "assistant",
        "content": assistant_text,
        "time": datetime.now().strftime("%H:%M"),
    }
    chat["messages"].append(assistant_message)
    chat["preview"] = build_preview(chat["messages"])
    save_history(chats)

    return jsonify(
        {
            "ok": True,
            "chat": chat,
            "user_message": user_message,
            "assistant_message": assistant_message,
        }
    )


@app.route("/health", methods=["GET"])
def health():
    frontend_ok = True
    backend_ok = False
    backend_status = "No verificado"

    try:
        response = requests.get(BACKEND_URL, timeout=5)
        backend_ok = response.ok
        backend_status = response.text
    except requests.RequestException as exc:
        backend_status = str(exc)

    return jsonify(
        {
            "ok": frontend_ok,
            "backend_ok": backend_ok,
            "backend_url": BACKEND_URL,
            "backend_status": backend_status,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
