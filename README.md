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

### Целевая аудитория (персоны)

* Исследователь / PhD: быстро извлекает суть статей, находит релевантные цитаты и идеи для
  экспериментов.
* ML-инженер: получает примеры кода, рефакторинг и помощь с интеграцией моделей в пайплайны.
* Студент / инженер-новичок: получает объяснения концепций и примеры с минимальным барьером входа.
* Руководитель группы / PI: получает обзор прогресса и агрегированные знания по проекту.

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

В рамках данного проекта будет реализована агентская система для помощи исследователям в
различных областях. Построение такой агентской системы production уровня требует наличия среды
для разработки этой системы и автоматизированных пайплайнов, сервинга самого агента и среды для 
экспериментирования различных вариантов и конфигураций. Другими словами, планируется разработки 
следующих платформ:

* Платформа для разработки сервиса
* Платформа для сервинга сервиса
* Платформа для экспериментирования
* Возможно, отдельная платформа для A/B тестирования

Для всего сервиса выставляются такие технические требования:

* Возможность работы всех платформ на одном сервере
* Возможность простого горизонтального масштабирования каждой платформы
* Модульность. Каждая платформа по отдельности и отдельные части одной платформы должны иметь
  модульную / микросервисную структуру
* Возможность разделения всех платформ на отдельные проекты в будущем (разделение на отдельные
  репозитории, пайплайны, CI/CD и т.д.)
* Простота разворачивания сервиса, IaC
* Наличие системы тестов при деплое и fail-safe режимы
* Прозрачный мониторинг всей системы

### Метрики

* Основные: BLEU, CodeBLEU - измеряет пресечение слов / биграмм в исходных и сгенерированных
  текстах.
* Для суммаризации: BERTScore - измеряет семантическую близость исходных и сгенерированных текстов.
* Для генерации кода: pass@k на бенчмарках (например HumanEval) - измеряет вероятность, что хотя бы
  один из топ-К ответов правильный

## Валидация и тест

...

## Датасеты

* https://huggingface.co/datasets/ccdv/arxiv-summarization - датасет статей arxiv с разметкой
  “article - abstract”. Уже имеет разбиение train / validation / test примерно 200000 / 6000 / 6000.
  Будет использоваться дефолтное разбиение.
* https://huggingface.co/datasets/nvidia/OpenCodeInstruct - датасет с разметкой “input human
  instruction - code implementation”. Датасет имеет 5млн размеченных промптов. В зависимости от
  ресурсов доступных при обучении, будет выбрано не менее 100000 / 5000 / 5000 семплов.
* https://huggingface.co/datasets/openai/openai_humaneval - датасет с 164 задачами для тестирования
  результатов, оценка по Global leaderboard

## Моделирование

### Бейзлайн

Ministral 3b

### Основная модель

Ministral 3b + LoRA адаптер для суммаризации
Ministral 3b + LoRA адаптер для задач кодинга

## Внедрение

TBD
