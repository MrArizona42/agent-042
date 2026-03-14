# RAG Aliases — Design Document

## 1. Цель

Добавить в RAG-систему alias-based lifecycle (`champion` / `challenger`) для Qdrant коллекций,
чтобы сравнивать retrieval-конфигурации без полного релиза в production.

## 2. Ключевые концепции

### Knowledge base vs. collection

* **Knowledge base (KB)** — логическое понятие, видимое пользователю в UI (например, `arxiv`,
  `pytorch_docs`). Соответствует понятию «dataset».
* **Collection** — физическая коллекция в Qdrant. Для каждой KB может существовать одна или
  более collections.
* **Alias** — именованный указатель на конкретную collection. Qdrant резолвит alias прозрачно.

### Alias roles

| Role | Назначение |
|---|---|
| `champion` | Текущая production-конфигурация. UI всегда использует `champion`. |
| `challenger` | Кандидат для A/B-тестирования и eval-экспериментов. |
| `*_staging` | Временный alias для вновь собираемой коллекции (только для replace strategy). |

### Promotion

Promotion — продвижение коллекции на более высокий уровень (из staging → alias, из
`challenger` → `champion`). Выполняется через CLI или автоматически в DAG.

### Retrieval parameters

Параметры `top_k`, `score_threshold`, `reranking` задаются глобально для всей RAG-конфигурации. Эксперименты с этими параметрами выполняются через деплой новой production-конфигурации.

## 3. Naming conventions

Aliases и collections в Qdrant должны быть глобально уникальны.

| Entity | Format | Example |
|---|---|---|
| Alias | `{kb}_{role}` | `arxiv_champion`, `pytorch_docs_challenger` |
| Staging alias | `{kb}_{role}_staging` | `pytorch_docs_champion_staging` |
| Collection | `{kb}_{timestamp}` | `pytorch_docs_20260314_120000` |

## 4. Конфигурация

В проекте используются две отдельные конфигурации с разной областью видимости.

### 4.1. Runtime config — `config/knowledge_bases.json`

Используется **gateway, eval runner и build scripts**. Определяет какие KB существуют,
какие aliases валидны, и какая стратегия обновления. Путь задаётся через env var
`GATEWAY_KNOWLEDGE_BASES_PATH` (default: `config/knowledge_bases.json`).
Загружается в Pydantic Settings.

```json
[
    {
        "knowledge_base": "arxiv",
        "aliases": ["champion", "challenger"],
        "update_strategy": "incremental"
    },
    {
        "knowledge_base": "pytorch_docs",
        "aliases": ["champion", "challenger"],
        "update_strategy": "replace"
    }
]
```

Поля:
* `knowledge_base` — имя KB, видимое пользователю и используемое в API.
* `aliases` — список ролей, которые существуют для данной KB. Определяет, какие aliases
  build scripts будут обновлять и какие aliases допустимы в API.
* `update_strategy` — `incremental` (данные добавляются к существующему индексу) или
  `replace` (индекс пересобирается через staging).

Если у KB нет alias `champion` в конфиге — эта KB **недоступна для production** (UI). Это не
ошибка конфигурации — KB может быть в стадии подготовки. API-вызовы с явным alias (например,
`challenger`) всё равно будут работать, если alias существует в Qdrant.

Gateway не знает и не должен знать о параметрах сборки (chunking, embedding model и т.д.) —
он только резолвит alias и делает запрос к Qdrant.

### 4.2. Build-time config — `experiments/conf/rag_build_profiles.yaml`

Используется **только build scripts и DAGs**. Определяет как собирать коллекции.
Gateway и eval runner этот файл **не читают**.

```yaml
profiles:
  baseline:
    chunking_strategy: fixed_token
    chunk_size: 512
    chunk_overlap: 64
    embedding_model: intfloat/e5-base-v2

  experimental_v2:
    chunking_strategy: section_aware
    chunk_size: 1024
    chunk_overlap: 128
    embedding_model: intfloat/e5-large-v2
```

Связь profile → alias:
* `champion` — всегда собирается с профилем `baseline` (hardcoded в DAG).
* `challenger` — собирается с профилем, указанным как параметр DAG (default: `experimental_v2`).

Build script принимает `--profile baseline`. Имя профиля сохраняется в metadata коллекции Qdrant
(payload на sentinel point с `id="_meta"`) для трассировки и аудита.

## 5. API контракт

### 5.1. Формат запроса

Текущий формат (`knowledge_base: str | None`) заменяется на список KB-запросов
с опциональным alias:

```json
{
  "rag_sources": [
    {"knowledge_base": "arxiv", "alias": "champion"},
    {"knowledge_base": "pytorch_docs", "alias": "challenger"}
  ]
}
```

* `alias` — optional, default: `"champion"`.
* UI всегда отправляет запросы без `alias` (используется `champion`).
* Eval runner и тестовые скрипты могут указывать произвольный alias.

### 5.2. Multi-KB retrieval

При запросах в несколько коллекций из каждой извлекается `top_k` чанков

### 5.3. Alias resolution

Aliases резолвятся **при каждом запросе** (без кеширования). Gateway является единственным
source of truth для резолва `(kb, alias)` → Qdrant collection name `{kb}_{alias}`.

### 5.4. Обработка ошибок

Если запрошенный alias не существует в Qdrant:
* **API** — HTTP 404 с сообщением, что KB недоступна для указанного alias.
* **UI** — отображает сообщение об ошибке: «Knowledge base недоступна».

Ошибка должна быть явной (fail-fast), без silent fallback на другой alias.

## 6. Build scripts

### 6.1. Incremental strategy — `build_chat_index`

Используется для KB с `update_strategy: "incremental"` (e.g., `arxiv` — статьи добавляются
к существующему индексу).

* Добавляет новые данные ко **всем** collections, на которые ссылаются aliases из runtime
  конфига (по умолчанию `champion` и `challenger`).
* Если одной из коллекций нет — обновить остальные и залогировать warning.
* Staging aliases **не используются** — данные добавляются инкрементально напрямую.

### 6.2. Replace strategy — `build_code_index`

Используется для KB с `update_strategy: "replace"` (e.g., `pytorch_docs` — документация
пересобирается полностью).

Пошаговый процесс (для каждого alias из runtime конфига):

1. Создать новую коллекцию `{kb}_{timestamp}` (e.g., `pytorch_docs_20260315_120000`).
2. Создать staging alias `{kb}_{role}_staging`
   (e.g., `pytorch_docs_champion_staging`) → указать на новую коллекцию.
3. Собрать индекс в новую коллекцию (через staging alias).
4. Запустить eval: **nDCG@10 на BEIR-SciFact** (единственная gating метрика).
5. Сравнить с текущим значением для `{kb}_{role}`:

   **Изменение в пределах -5% .. +20% → auto-promote.** Qdrant операции:
   - Re-point `{kb}_{role}` (e.g., `pytorch_docs_champion`) на новую коллекцию.
   - Staging alias `{kb}_{role}_staging` остаётся на той же коллекции
     (будет переключён на следующую коллекцию при следующем сборе).
   - Старая коллекция (на которую раньше указывал `{kb}_{role}`) теряет alias
     и будет удалена cleanup DAG-ом через 7 дней.

   **Изменение за пределами диапазона (drop > 5% или jump > 20%) → block.**
   Залогировать как аномалию (в дальнейшем — настроить уведомления). Все aliases остаются
   на месте. Staging alias по-прежнему указывает на новую коллекцию, но production alias
   не переключается. При следующем сборе staging будет переключён на ещё более новую
   коллекцию.

### 6.3. Manual promotion CLI

Для ручного promotion — отдельный скрипт по типу `manage_registry.py`:

```bash
python -m scripts.manage_rag promote --kb pytorch_docs --from challenger --to champion
```

## 7. Cleanup DAG

Отдельный Airflow DAG, запуск ежедневно. Логика:

1. Получить список всех Qdrant collections.
2. Получить список всех aliases → собрать set коллекций, на которые указывают aliases.
3. Для каждой коллекции **без alias**: распарсить timestamp из имени.
4. Если коллекция старше 7 дней → удалить.

Коллекции, не соответствующие формату `{kb}_{timestamp}` (legacy), пропускаются.

## 8. Eval pipeline integration

* **Gateway как single source of truth**: eval runner обращается к RAG **через gateway API**,
  передавая `knowledge_base` + `alias` в запросе. Это гарантирует единую точку резолва
  aliases и исключает дублирование логики. Требует запущенного gateway во время eval.
* **UI-запросы** — резолвятся только по KB (alias = `champion` по умолчанию).
* **Eval runner** — резолвит KB + explicit alias.
* **Таблица `eval_runs`** — добавить колонку `rag_alias` TEXT для фиксации alias,
  использованного в eval-run.

## 9. Admin endpoints

* `GET /v1/knowledge-bases` — список KB с их aliases и статусом доступности
  (alias существует в Qdrant или нет).
* Promotion **не выполняется** через API. Только CLI по SSH или автоматически в DAG.
