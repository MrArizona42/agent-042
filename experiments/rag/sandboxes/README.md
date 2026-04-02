# RAG Sandboxes

This directory contains notebook-only experimental code.

Rules:
- Production services, Airflow DAGs, and production evals must never import from here.
- Each experiment lives in its own subpackage.
- Copy or fork only the production modules the experiment actually changes.
- Record the experiment identifier in collection `_meta` when building challengers from sandbox code.
- If an experiment changes query-time behavior, port it into `src/rag` before promoting to champion.
