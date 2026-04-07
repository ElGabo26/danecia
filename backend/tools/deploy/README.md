# Refactor SQL Agent

## Archivos
- `tools/deploy/src/makeContext.py`: genera el contexto desde catálogo, joins, reglas, glosario y ejemplos.
- `tools/deploy/src/sql_rag_pipeline.py`: generación, revisión LLM y reparación.
- `sql_service.py`: wrapper compatible con el flujo anterior.

## Uso
```python
from sql_service import run_sql_generation_flow
r = run_sql_generation_flow("ventas por mes de la marca TONI")
print(r["sql"])
```
