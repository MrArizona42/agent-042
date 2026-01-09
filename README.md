# Разработка и исследование интеллектуального ассистента для исследователей с использованием генерации на основе поиска и эффективного дообучения моделей.

## Инструкции

* `./infra/README.md` - настройка окружения и инфраструктуры
* `./experiments/README.md` - как проводить эксперименты

## Бизнес описание работы агентской системы

Цель проекта: Создать интеллектуальную ассистент‑систему для исследователей в области
ML/DL/AI/LLM, которая ускоряет поиск информации, суммаризацию научных материалов и генерацию кода,
повышая продуктивность и воспроизводимость исследований. Планируется создать агентскую систему,
которая будет работать локально, сохраняя конфиденциальность пользовательских данных. При этом
агентская система будет знать, с чем и с кем ведется работа, чтобы обеспечить максимально
возможную релевантность. Т.е. подобная агентская система будет представлять из себя
"коллегу-ассистента", который осведомлен, в какой системе он работает, "помнит" историю
обращений пользователей и историю выполняемых задач, может "загуглить" дополнительную информацию,
при этом не раскрывая конфиденциальные данные.

### Целевая аудитория

* Исследователи: быстро извлекает суть статей, находит релевантные цитаты и идеи для
  экспериментов.
* ML-инженеры: получает примеры кода, рефакторинг и помощь с интеграцией моделей в пайплайны.
* Студенты: получает объяснения концепций и примеры с минимальным барьером входа.
* Руководитель группы: получает обзор прогресса и агрегированные знания по проекту.

### Ключевые сценарии использования

* Чат с поддержкой поиска по внутренним и внешним источникам (RAG): ответы с указанием источников и
  цитат.
* Суммаризация статей и длинных документов (multi-level: от краткого «TL;DR» до подробной
  структуры).
* Генерация и доработка кода (шаблоны, тесты, советы по оптимизации).
* Поиск по корпоративным/локальным репозиториям знаний и коду.

### Основные ценности

* Экономия времени на обзор литературы и поиск решений.
* Быстрая генерация кода и примеров, релевантных имеющимся базам знаний, проектам и репозиториям.

### Ожидаемые возможности системы:

* Чат-бот. Ответы на вопросы про разные области и аспекты ML / DL / AI / LLM.
* Суммаризация документов / статей (в том же режиме чат-бота).
* Генерация кода (в том же режиме чат-бота).
* RAG система с базой знаний (статьи, документация, кодовые базы и т.д.).

### Возможные расширения функционала, возможностей RAG системы и решение проблемы cutoff-date:

* Tool-calling уровня Low-risk: web-search, code-search, knowledge-base lookups.
* Хранение и динамическое обновление информации о пользователе и истории переписки.

## Метрики успеха ответов в чате

### 1. Answer quality:

#### a) Relevance and Correctness.

* **What it measures:** Does the answer address the question? Are the statements factually correct?
* **How to measure:** LLM-as-judge with a fixed rubric: Relevance (1-5) and Correctness (1-5).
* **Datasets:** Natural Questions, HotpotQA, or your own curated ML/AI questions
* **Input:** Question + retrieved passages + generated answer
* **Judges:** Strong closed-source or open LLM (GPT-4 / Claude / Llama-70B)

#### b) ROUGE-L

#### c) BERTScore

### 2. Summarization Quality

#### a) Coverage and Faithfulness

* **What it measures:** Did the summarizer generate a summary that covers the answer? Are key
  ideas preserved?

* **How to measure:** LLM-as-judge with two questions:
    1. “Does the summary introduce unsupported
       claims?”
    2. “Does it cover the main contributions?”

* **Datasets:** ArXiv, OpenReview, PubMed

#### b) ROUGE-L

#### c) BERTScore

### 3. Code Generation Quality

#### a) Executable Correctness Rate

* **What it measures:** Did the code generator generate executable code that passes all tests?

* **How to measure:**

1. `Executable rate = runnable_outputs / total_code_outputs`
2. `Correctness rate = passed_tests / total_tests`

* **Dataset:** HumanEval (https://github.com/google-research/google-research/tree/master/human_eval)

## Метрики успеха RAG системы

### 4. RAG Grounding:

#### a) Grounded Answer Rate, LLM-as-judge based

* **What it measures:** Are claims supported by retrieved documents?

* **How to measure:** For each atomic claim in the answer: Is it supported by at least one
  retrieved chunk? `Groundedness = supported_claims / total_claims`

#### b) Answer–Context Similarity (BERTScore or cosine similarity)

### 5. Retrieval Quality

#### a) Recall@k

* **What it measures:** Did retrieval fetch documents that contain the answer?

* **How to measure:** For each query, Check whether at least one gold document is in top-k retrieved

* **Datasets:** MS MARCO, BEIR

#### b) nDCG@k

## Метрики успеха агентского сервиса

TBD

## Техническое описание и постановка задачи

Сервис будет иметь 2 основные платформы:

1. Платформа для экспериментирования и обучения моделей и адаптеров
2. Платформа с работающим LLM сервисом. Сам сервис будет строиться в 4 этапа:
    1. Базовая LLM.
    2. Базовая LLM с адаптерами под разные нужды. Фиксированный или rule-based выбор адаптеров
    3. Базовая LLM с адаптерами + RAG система. Фиксированный или rule-based выбор адаптеров и RAG
       пайплайна.
    4. Агентский сервис с динамическим выбором задействуемых инструментов

### Базовая LLM.

* Клиент делает запрос
* Запрос попадает в API Gateway (FastAPI)
* FastAPI использует Task Router, который на данный момент имеет одну функцию: chat
* FastAPI использует Prompt Builder, который собирает промпт
    * Промпт фиксированный
    * Собирается Prompt Config
* vLLM Inference Server всегда имеет загруженную базовую LLM
    * получает информацию, какой адаптер подгружать (на данный момент никакой)
    * получает промпт
    * vLLM генерирует ответ, который через FastAPI направляется клиенту

### Базовая LLM с адаптерами под разные нужды. Фиксированный или rule-based выбор адаптеров

* Клиент делает запрос
* Запрос попадает в API Gateway (FastAPI)
* **FastAPI использует Task Router, который определяет, что нужно сделать: chat / summarize /
  generate code**
    * таска определяется rule-based по кейвордам или вообще выбирается вручную в UI
    * под каждую таску существует свой LoRA
* FastAPI использует Prompt Builder, который собирает промпт
    * **Промпт фиксированный, но разный для каждого типа таски**
    * Собирается Prompt Config
* vLLM Inference Server всегда имеет загруженную базовую LLM
    * получает информацию, какой адаптер подгружать (или никакой)
    * получает промпт
    * vLLM генерирует ответ, который через FastAPI направляется клиенту

### Базовая LLM с адаптерами + RAG система. Фиксированный или rule-based выбор адаптеров и RAG пайплайна.

* Клиент делает запрос
* Запрос попадает в API Gateway (FastAPI)
* FastAPI использует Task Router, который определяет, что нужно сделать: chat / summarize /
  generate code
    * таска определяется rule-based по кейвордам или вообще выбирается вручную в UI
    * под каждую таску существует свой LoRA
* FastAPI использует Prompt Builder, который собирает промпт
    * **в UI можно выбрать, использовать ли RAG**
    * **Промпт теперь может собираться при помощи RAG**
    * Собирается Prompt Config
* vLLM Inference Server всегда имеет загруженную базовую LLM
    * получает информацию, какой адаптер подгружать (или никакой)
    * получает промпт
    * vLLM генерирует ответ, который через FastAPI направляется клиенту

### Агентский сервис с динамическим выбором задействуемых инструментов

* Клиент делает запрос
* Запрос попадает в API Gateway (FastAPI)
* **Между FastAPI и Task Router / Prompt Builder теперь есть еще один слой абстракции с отдельной
  LLM,
  которая автоматизирует выбор адаптеров и RAG, а также может задействовать другие инструменты.**
    * Подробности TBD
* FastAPI использует Task Router, который определяет, что нужно сделать: chat / summarize /
  generate code
    * в UI по-прежнему можно вручную выбрать, какую задачу нужно выполнять
    * под каждую таску существует свой LoRA
* FastAPI использует Prompt Builder, который собирает промпт
    * **в UI по-прежнему можно выбрать, использовать ли RAG**
    * Собирается Prompt Config
* vLLM Inference Server всегда имеет загруженную базовую LLM
    * получает информацию, какой адаптер подгружать (или никакой)
    * получает промпт
    * vLLM генерирует ответ, который через FastAPI направляется клиенту

### RAG пайплайны

TBD

### Платформа для экспериментов и обучение LoRA

TBD

## Данные

**Добавть описание и требования к данным**

* https://huggingface.co/datasets/ccdv/arxiv-summarization - датасет статей arxiv с разметкой
  “article - abstract”.
* https://huggingface.co/datasets/nvidia/OpenCodeInstruct - датасет с разметкой “input human
  instruction - code implementation”. Датасет имеет 5млн размеченных промптов.
* https://huggingface.co/datasets/openai/openai_humaneval - датасет с 164 задачами для тестирования
  результатов, оценка по Global leaderboard

## Workflow automation and CI/CD

**Нужно расширить этот раздел, сматчить с реальными датасетами, тестами и метриками.**

### Branch: experiments

Goal: speed, flexibility, low friction

#### Pre-commit (light)

* black
* isort
* trailing whitespace
* YAML/JSON validation

#### CI (minimal or optional)

* Syntax check
* Optional unit tests for shared utilities

Experiments should not feel like bureaucracy.

### Branch: develop (inference dev)

Goal: correctness & safety

#### Pre-commit

* formatting
* imports
* basic linting

#### CI

* unit tests (FastAPI, routing)
* config validation
* adapter registry checks
* inference smoke test (CPU or tiny model)

This is your engineering-quality bar.

### Branch: main

Goal: stability

#### CI

triggered when merge request is opened

* full inference test suite
* latency regression checks (light)
* startup test (service boots)

#### CD

triggered only by merge approval

* manual confirmation
* deploy inference
* rollback available
