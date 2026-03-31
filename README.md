
# Proyecto dividido en frontend y backend

## Estructura

```bash
grupo_danec_split/
├── backend/
│   ├── app.py
│   └── requirements.txt
└── frontend/
    ├── app.py
    ├── requirements.txt
    ├── data/
    │   └── chat_history.json
    ├── static/
    │   ├── danecLogo.png
    │   └── style.css
    └── templates/
        └── index.html
```

## Descripción

- **Frontend**: Flask renderiza la interfaz, administra el historial de chats y consume el backend.
- **Backend**: Flask expone `/analizar` y se conecta con Ollama usando tu flujo principal.

## Ejecución

### 1. Backend
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### 2. Frontend
En otra terminal:
```bash
cd frontend
pip install -r requirements.txt
python app.py
```

## Puertos
- Frontend: `http://localhost:8000`
- Backend: `http://localhost:5000`

## Variables opcionales del frontend
```bash
export BACKEND_URL=http://localhost:5000
export BACKEND_TIMEOUT=180
```

## Observación
El backend fue tomado como base del archivo principal proporcionado por el usuario y se mantuvo la integración con Ollama. Se añadió CORS y una corrección en el ciclo de reintentos para evitar una condición lógica incorrecta.
