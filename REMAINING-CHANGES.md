# Remaining Changes

Это короткий список пунктов из исторических audit / review документов, которые еще не реализованы
в текущей ветке.

## Незавершенные изменения

* Полный config cleanup не завершен: worker и отдельные runtime-path-ы все еще дублируют часть
  shared settings, а в compose и коде остаются отдельные hardcoded URLs / timeouts.
* UI и discovery API для KB еще не сведены к одному источнику: UI читает локальный registry, а
  `/v1/knowledge-bases` возвращает flat list, а не task-grouped discovery contract.
* Chat history частично server-side only: gateway сохраняет завершенные non-streaming exchanges,
  но prompt reconstruction все еще опирается на client-supplied history, а streaming responses не
  персистятся.
* Training orchestration пока заканчивается на `train -> inspect/promote`; автоматический шаг
  `train -> evaluate -> human decision` не подключен.
* Alembic migrations для `agent042` DB не заведены; schema bootstrap по-прежнему опирается на ORM
  table creation / SQL scripts.

## Future Work From Reviews

* RAG reranking и hybrid-search benchmarks.
* Token / cost tracking и более полная observability для LLM path.
* Hosted CI/CD workflows.
* Agent layer with dynamic tool selection.
