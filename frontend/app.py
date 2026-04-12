import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

import requests
from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    stream_with_context,
    url_for,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

CHAT_FILE = DATA_DIR / "chat_history.json"
FEEDBACK_FILE = DATA_DIR / "feedback_history.json"

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:5000/analizar")
FRONTEND_HOST = os.getenv("FRONTEND_HOST", "0.0.0.0")
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "8000"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))
STREAM_CONNECT_TIMEOUT = int(os.getenv("STREAM_CONNECT_TIMEOUT", "30"))
SECRET_KEY = os.getenv("SECRET_KEY", "danec-demo-secret")

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = SECRET_KEY


# ------------------------------
# Helpers generales
# ------------------------------
def current_time_str() -> str:
    return datetime.now().strftime("%H:%M")


def normalize_username(raw: str) -> str:
    clean = re.sub(r"\s+", " ", str(raw or "").strip())
    return clean[:60]


def user_key(username: str) -> str:
    return normalize_username(username).casefold()


def create_chat(title: str = "Nuevo chat") -> dict:
    return {"id": str(uuid.uuid4()), "title": title, "preview": "", "messages": []}


def create_default_user_history() -> dict:
    return {"active_chat_id": None, "chats": []}


def create_default_store() -> dict:
    return {"users": {}}


def create_default_feedback_store() -> dict:
    return {"items": []}


def normalize_message(msg) -> dict:
    if not isinstance(msg, dict):
        return {"role": "assistant", "content": str(msg), "time": current_time_str()}

    normalized = {
        "role": msg.get("role", "assistant"),
        "content": str(msg.get("content", "")),
        "time": msg.get("time", current_time_str()),
    }
    if "feedback_id" in msg:
        normalized["feedback_id"] = msg.get("feedback_id")
    return normalized


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
        "title": str(chat.get("title", "Nuevo chat"))[:120],
        "preview": str(chat.get("preview", ""))[:240],
        "messages": [normalize_message(msg) for msg in raw_messages],
    }


def normalize_user_history(data) -> dict:
    default_data = create_default_user_history()
    if data is None:
        return default_data

    if isinstance(data, list):
        chats = [normalize_chat(chat) for chat in data]
        return {"active_chat_id": chats[0]["id"] if chats else None, "chats": chats}

    if isinstance(data, dict):
        chats = data.get("chats", [])
        if not isinstance(chats, list):
            chats = []
        normalized = {
            "active_chat_id": data.get("active_chat_id"),
            "chats": [normalize_chat(chat) for chat in chats],
        }
        if normalized["active_chat_id"] and not any(
            chat["id"] == normalized["active_chat_id"] for chat in normalized["chats"]
        ):
            normalized["active_chat_id"] = normalized["chats"][0]["id"] if normalized["chats"] else None
        return normalized

    return default_data


def migrate_legacy_store(raw_data) -> dict:
    if isinstance(raw_data, dict) and isinstance(raw_data.get("users"), dict):
        store = create_default_store()
        for _, payload in raw_data["users"].items():
            username = normalize_username(payload.get("username", ""))
            if not username:
                continue
            store["users"][user_key(username)] = {
                "username": username,
                "history": normalize_user_history(payload.get("history")),
            }
        return store

    if isinstance(raw_data, list) or (isinstance(raw_data, dict) and "chats" in raw_data):
        legacy_history = normalize_user_history(raw_data)
        username = "General"
        return {
            "users": {
                user_key(username): {
                    "username": username,
                    "history": legacy_history,
                }
            }
        }

    return create_default_store()


def load_store() -> dict:
    default_store = create_default_store()
    if not CHAT_FILE.exists():
        save_store(default_store)
        return default_store

    try:
        with open(CHAT_FILE, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        store = migrate_legacy_store(raw_data)
        save_store(store)
        return store
    except Exception:
        save_store(default_store)
        return default_store


def save_store(store: dict) -> None:
    with open(CHAT_FILE, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)


def load_feedback_store() -> dict:
    default_store = create_default_feedback_store()
    if not FEEDBACK_FILE.exists():
        save_feedback_store(default_store)
        return default_store
    try:
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or not isinstance(data.get("items"), list):
            data = default_store
        return data
    except Exception:
        save_feedback_store(default_store)
        return default_store


def save_feedback_store(store: dict) -> None:
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)


def list_users(store: dict) -> list[dict]:
    users = []
    for key, payload in store.get("users", {}).items():
        users.append({
            "key": key,
            "username": payload.get("username", ""),
            "chat_count": len(payload.get("history", {}).get("chats", [])),
        })
    return sorted(users, key=lambda x: x["username"].casefold())


def get_user_record(store: dict, username: str, create_if_missing: bool = False):
    username = normalize_username(username)
    if not username:
        return None

    key = user_key(username)
    record = store.get("users", {}).get(key)
    if record is None and create_if_missing:
        record = {"username": username, "history": create_default_user_history()}
        store.setdefault("users", {})[key] = record
    return record


def get_current_username() -> str | None:
    return normalize_username(session.get("username", "")) or None


def require_user():
    username = get_current_username()
    if not username:
        return None, redirect(url_for("login_page"))

    store = load_store()
    record = get_user_record(store, username, create_if_missing=False)
    if record is None:
        session.pop("username", None)
        return None, redirect(url_for("login_page"))

    return {"store": store, "record": record, "username": username}, None


def get_active_chat(history: dict) -> dict:
    chats = history.get("chats", [])
    active_id = history.get("active_chat_id")

    if not chats:
        new_chat = create_chat()
        history["chats"] = [new_chat]
        history["active_chat_id"] = new_chat["id"]
        return new_chat

    for chat in chats:
        if chat.get("id") == active_id:
            return chat

    history["active_chat_id"] = chats[0]["id"]
    return chats[0]


def find_chat_by_id(history: dict, chat_id: str):
    for chat in history.get("chats", []):
        if chat.get("id") == chat_id:
            return chat
    return None


def parse_sse_event(event_lines: list[str]):
    """
    Procesa un evento SSE completo.
    Un evento SSE termina con una línea vacía.
    """
    data_lines = []
    for raw_line in event_lines:
        if raw_line.startswith("data:"):
            data_lines.append(raw_line[5:].strip())

    if not data_lines:
        return None

    try:
        return json.loads("\n".join(data_lines))
    except Exception:
        return None


def get_previous_user_question(messages: list, assistant_index: int) -> str:
    for idx in range(assistant_index - 1, -1, -1):
        msg = messages[idx]
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


# ------------------------------
# Vistas de usuario
# ------------------------------
@app.route("/", methods=["GET"])
def login_page():
    store = load_store()
    return render_template("login.html", users=list_users(store), current_username=get_current_username())


@app.route("/select-user", methods=["POST"])
def select_user():
    payload = request.get_json(silent=True) or request.form or {}
    username = normalize_username(payload.get("username", ""))

    if not username:
        return jsonify({"ok": False, "error": "Debe ingresar un usuario."}), 400

    store = load_store()
    get_user_record(store, username, create_if_missing=True)
    save_store(store)
    session["username"] = username
    return jsonify({"ok": True, "username": username, "redirect": url_for("chat_home")})


@app.route("/logout", methods=["POST"])
def logout():
    session.pop("username", None)
    return jsonify({"ok": True, "redirect": url_for("login_page")})


@app.route("/chat", methods=["GET"])
def chat_home():
    ctx, redirect_response = require_user()
    if redirect_response:
        return redirect_response

    history = ctx["record"]["history"]
    requested_chat_id = request.args.get("chat_id", "").strip()
    if requested_chat_id:
        target = find_chat_by_id(history, requested_chat_id)
        if target is not None:
            history["active_chat_id"] = requested_chat_id
            save_store(ctx["store"])

    active_chat = get_active_chat(history)
    save_store(ctx["store"])

    assistant_pairs = {}
    for idx, msg in enumerate(active_chat.get("messages", [])):
        if msg.get("role") == "assistant":
            assistant_pairs[idx] = get_previous_user_question(active_chat["messages"], idx)

    return render_template(
        "index.html",
        chats=history.get("chats", []),
        active_chat=active_chat,
        username=ctx["username"],
        assistant_pairs=assistant_pairs,
    )


# ------------------------------
# API sesión y chats por usuario
# ------------------------------
@app.route("/api/session", methods=["GET"])
def get_session_info():
    username = get_current_username()
    return jsonify({"ok": True, "username": username})


@app.route("/api/users", methods=["GET"])
def get_users():
    store = load_store()
    return jsonify({"ok": True, "users": list_users(store)})


@app.route("/api/chats", methods=["GET"])
def list_chats_route():
    ctx, redirect_response = require_user()
    if redirect_response:
        return jsonify({"ok": False, "error": "Sesión no iniciada."}), 401

    history = ctx["record"]["history"]
    active_chat = get_active_chat(history)
    save_store(ctx["store"])
    return jsonify({
        "ok": True,
        "username": ctx["username"],
        "chats": history.get("chats", []),
        "active_chat_id": active_chat.get("id"),
    })


@app.route("/api/chats", methods=["POST"])
def new_chat():
    ctx, redirect_response = require_user()
    if redirect_response:
        return jsonify({"ok": False, "error": "Sesión no iniciada."}), 401

    history = ctx["record"]["history"]
    chat = create_chat()
    history.setdefault("chats", []).insert(0, chat)
    history["active_chat_id"] = chat["id"]
    save_store(ctx["store"])
    return jsonify({"ok": True, "chat": chat, "active_chat_id": chat["id"]})


@app.route("/api/chats/<chat_id>", methods=["GET"])
def get_chat(chat_id):
    ctx, redirect_response = require_user()
    if redirect_response:
        return jsonify({"ok": False, "error": "Sesión no iniciada."}), 401

    history = ctx["record"]["history"]
    chat = find_chat_by_id(history, chat_id)
    if chat is None:
        return jsonify({"ok": False, "error": "Chat no encontrado"}), 404

    history["active_chat_id"] = chat_id
    save_store(ctx["store"])
    return jsonify({"ok": True, "chat": chat})


@app.route("/api/chats/<chat_id>", methods=["PATCH"])
def update_chat(chat_id):
    ctx, redirect_response = require_user()
    if redirect_response:
        return jsonify({"ok": False, "error": "Sesión no iniciada."}), 401

    history = ctx["record"]["history"]
    chat = find_chat_by_id(history, chat_id)
    if chat is None:
        return jsonify({"ok": False, "error": "Chat no encontrado"}), 404

    payload = request.get_json(silent=True) or {}
    title = str(payload.get("title", "")).strip()

    if not title:
        return jsonify({"ok": False, "error": "El título no puede estar vacío"}), 400

    chat["title"] = title[:120]
    save_store(ctx["store"])
    return jsonify({"ok": True, "chat": chat})


@app.route("/api/chats/<chat_id>", methods=["DELETE"])
def delete_chat(chat_id):
    ctx, redirect_response = require_user()
    if redirect_response:
        return jsonify({"ok": False, "error": "Sesión no iniciada."}), 401

    history = ctx["record"]["history"]
    chats = history.get("chats", [])
    target = find_chat_by_id(history, chat_id)
    if target is None:
        return jsonify({"ok": False, "error": "Chat no encontrado"}), 404

    history["chats"] = [chat for chat in chats if chat.get("id") != chat_id]

    if not history["chats"]:
        new_chat = create_chat()
        history["chats"] = [new_chat]
        history["active_chat_id"] = new_chat["id"]
    elif history.get("active_chat_id") == chat_id:
        history["active_chat_id"] = history["chats"][0]["id"]

    save_store(ctx["store"])
    return jsonify({
        "ok": True,
        "deleted_chat_id": chat_id,
        "active_chat_id": history["active_chat_id"],
        "chats": history["chats"],
    })


@app.route("/api/chats/<chat_id>/message", methods=["POST"])
@app.route("/api/chats/<chat_id>/messages", methods=["POST"])
def send_message(chat_id):
    ctx, redirect_response = require_user()
    if redirect_response:
        return jsonify({"ok": False, "error": "Sesión no iniciada."}), 401

    history = ctx["record"]["history"]
    chat = find_chat_by_id(history, chat_id)
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

    user_message = {"role": "user", "content": prompt, "time": current_time_str()}
    chat.setdefault("messages", []).append(user_message)
    if len(chat["messages"]) == 1 and chat.get("title", "Nuevo chat") == "Nuevo chat":
        chat["title"] = prompt[:80]
    chat["preview"] = prompt[:120]
    history["active_chat_id"] = chat_id
    save_store(ctx["store"])

    def generate():
        backend_response = None
        final_answer = None
        event_lines = []

        try:
            backend_response = requests.post(
                BACKEND_URL,
                json={"prompt": prompt},
                timeout=(STREAM_CONNECT_TIMEOUT, REQUEST_TIMEOUT),
                stream=True,
                headers={"Accept": "text/event-stream"},
            )
            backend_response.raise_for_status()

            for line in backend_response.iter_lines(decode_unicode=True):
                if line is None:
                    continue

                # Reenviar exactamente el stream SSE hacia el navegador
                if line == "":
                    yield "\n"
                    if event_lines:
                        payload_event = parse_sse_event(event_lines)
                        if payload_event and payload_event.get("stage") == "fin":
                            final_answer = payload_event.get("resultado") or payload_event.get("message")
                        event_lines = []
                else:
                    event_lines.append(line)
                    yield line + "\n"

            # Procesar remanente por si el stream cerró sin línea vacía final
            if event_lines:
                payload_event = parse_sse_event(event_lines)
                if payload_event and payload_event.get("stage") == "fin":
                    final_answer = payload_event.get("resultado") or payload_event.get("message")

        except requests.exceptions.Timeout:
            final_answer = "Error: el backend tardó demasiado en responder."
            yield f"data: {json.dumps({'stage': 'error', 'message': final_answer}, ensure_ascii=False)}\n\n"
        except requests.exceptions.ConnectionError as e:
            final_answer = f"Error: no se pudo conectar con el backend en {BACKEND_URL}. Detalle: {e}"
            yield f"data: {json.dumps({'stage': 'error', 'message': final_answer}, ensure_ascii=False)}\n\n"
        except requests.exceptions.HTTPError as e:
            final_answer = f"Error HTTP del backend: {str(e)}"
            yield f"data: {json.dumps({'stage': 'error', 'message': final_answer}, ensure_ascii=False)}\n\n"
        except Exception as e:
            final_answer = f"Error inesperado: {str(e)}"
            yield f"data: {json.dumps({'stage': 'error', 'message': final_answer}, ensure_ascii=False)}\n\n"
        finally:
            if backend_response is not None:
                backend_response.close()

            if final_answer:
                refreshed_store = load_store()
                refreshed_record = get_user_record(refreshed_store, ctx["username"], create_if_missing=False)
                if refreshed_record is not None:
                    refreshed_chat = find_chat_by_id(refreshed_record["history"], chat_id)
                    if refreshed_chat is not None:
                        refreshed_chat["messages"].append(
                            {"role": "assistant", "content": final_answer, "time": current_time_str()}
                        )
                        save_store(refreshed_store)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ------------------------------
# API feedback
# ------------------------------
@app.route("/api/feedback", methods=["POST"])
def save_feedback():
    ctx, redirect_response = require_user()
    if redirect_response:
        return jsonify({"ok": False, "error": "Sesión no iniciada."}), 401

    payload = request.get_json(silent=True) or {}
    dominio = str(payload.get("domain", "")).strip()
    mejora = str(payload.get("issue", "")).strip()
    ejemplo = str(payload.get("example_response", "")).strip()
    question = str(payload.get("question", "")).strip()
    answer = str(payload.get("answer", "")).strip()
    chat_id = str(payload.get("chat_id", "")).strip()

    if not dominio:
        return jsonify({"ok": False, "error": "Debe seleccionar un dominio."}), 400
    if not mejora:
        return jsonify({"ok": False, "error": "Debe ingresar el error o mejora encontrada."}), 400
    if not question:
        return jsonify({"ok": False, "error": "No se encontró la pregunta asociada."}), 400
    if not answer:
        return jsonify({"ok": False, "error": "No se encontró la respuesta asociada."}), 400

    feedback_store = load_feedback_store()
    item = {
        "id": str(uuid.uuid4()),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "username": ctx["username"],
        "chat_id": chat_id,
        "question": question,
        "answer": answer,
        "domain": dominio,
        "issue_or_improvement": mejora,
        "example_response": ejemplo,
    }
    feedback_store.setdefault("items", []).append(item)
    save_feedback_store(feedback_store)
    return jsonify({"ok": True, "feedback": item})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "frontend", "backend_url": BACKEND_URL})


if __name__ == "__main__":
    app.run(host=FRONTEND_HOST, port=FRONTEND_PORT, debug=False)
