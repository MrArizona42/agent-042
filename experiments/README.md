# 🧪 Как проводить эксперименты

> Немного магии Hydra, немного дисциплины DVC и щепотка Lightning — быстрый и воспроизводимый цикл
> экспериментов.

## 🧭 Оглавление

- 🚀 Быстрый старт
- 🗂️ Структура директории
- 📦 DVC
- ⚙️ Hydra и конфигурирование
- ▶️ Примеры

## 🚀 Быстрый старт

1. Скачать датасет и модель — откройте ноутбук `experiments/misc_ops/prefetch_assets.ipynb`,
   задайте `PROJECT_ROOT` и выполните нужные ячейки.
2. Запустить обучение адаптера:

```bash
python -m experiments.training.train_adapter.start_train \
  +experiment=arxiv_summarization \
  paths.project_root="/home/user/agent-042"
```

`+experiment=arxiv_summarization` — обязательный аргумент: говорит Hydra, какой набор конфигов
использовать. Без него команда завершится с ошибкой (подробнее — в разделе ниже).

> ℹ️ На Windows в Git Bash удобнее использовать прямые слэши в путях: `C:/Users/user/MyGitRepos/agent-042`.

## 🗂️ Структура директории

* `./training/conf` - Hydra конфиги
* `./training` - Обучение адаптера
* `./training/lora_ops.ipynb` - Операции с LoRA (регистрация, промоушен, синхронизация)
* `./rag` - RAG operator notebooks и notebook wrappers поверх `src/rag/ops`
* `./rag/rag_ops.ipynb` - Операции с RAG (create / refresh / alias management / диагностика)
* `./rag/notebook_ops.py` - notebook façade над production entrypoints из `src/rag/ops`
* `./rag/sandboxes/` - notebook-only experimental forks, которые не импортируются production-кодом
* `./eval` - Оценка моделей
* `./eval/eval_results.ipynb` - Результаты оценки (сравнение, отчёты)
* `./eval/debug_eval.ipynb` - Отладка пайплайна оценки
* `./misc_ops` - Прочие операции (prefetch, MLflow, PostgreSQL диагностика)

## RAG operator path

- Production-safe lifecycle код для RAG живёт в `src/rag/ops/`.
- Создание новых коллекций выполняется через `src/rag/ops/create/` и notebook wrappers
  `create_arxiv(...)` / `create_pytorch_docs(...)` из `experiments/rag/notebook_ops.py`.
- Refresh существующих production-коллекций выполняется через `src/rag/ops/update/` и wrappers
  `refresh_arxiv(...)` / `refresh_pytorch_docs(...)`. Airflow DAG-и используют только эти
  production update entrypoints.
- Alias management и inspection идут через `src/rag/ops/aliases.py`, `src/rag/ops/inspect.py`
  и notebook wrappers `assign_alias(...)`, `promote(...)`, `detach(...)`, `inspect_kb_alias(...)`.
- `experiments/rag/rag_ops.ipynb` должен вызывать только `experiments.rag.notebook_ops`, чтобы
  notebook и Airflow использовали один и тот же production runtime.
- Если notebook или helper всё же импортирует registry-specific schema/loader напрямую, их источник
  должен быть `src/shared/operator_registry.py`, а не `shared.config`.
- `experiments/rag/sandboxes/` предназначен только для notebook-only experiments. Если sandbox
  эксперимент нужно продвигать в champion, код сначала переносится в `src/rag/` или
  `src/rag/ops/`, а уже потом пересобирается и промоутится коллекция.

## 📦 DVC

Remote хранилище называется ycloud

**Базовые команды DVC**

- `dvc init`: Инициализирует DVC в проекте (если ещё не сделано).
- `dvc add <file_or_dir>`: Добавляет файл или директорию под контроль DVC (например,
  `dvc add data/`).
- `dvc push`: Отправляет добавленные данные в удалённое хранилище (ycloud).
- `dvc pull`: Загружает данные из удалённого хранилища.
- `dvc repro`: Воспроизводит пайплайн эксперимента (на основе dvc.yaml).
- `dvc status`: Проверяет статус изменений в данных и пайплайнах.

Для настройки remote используется конфигурация в `.dvc/config`. Подробности в документации DVC.

> 💡 Рекомендация: держите крупные артефакты (данные, модели) под DVC — репозиторий станет легче и
> воспроизводимее.

Все используемые данные необходимо версионировать в DVC!

## ⚙️ Hydra и конфигурирование

### Как Hydra собирает конфиг (если видите её впервые)

Hydra не читает один большой `config.yaml` — она *компонует* итоговый конфиг из нескольких
файлов. Каждый файл — это **группа конфигов** (config group), отвечающая за одну «ось»:
модель, датасет, тренер, планировщик LR и т.д.

Итоговый конфиг строится в четыре уровня; каждый следующий перекрывает предыдущий:

```
1. Dataclass-defaults   ← config.py: типы полей и стабильные runtime defaults
         ↓ перекрывается
2. YAML-файлы групп     ← conf/<group>/*.yaml: конкретные значения (lr, пути, параметры LoRA…)
         ↓ перекрывается
3. Experiment preset    ← +experiment=arxiv_summarization: выбирает нужные YAML одним аргументом
         ↓ перекрывается
4. CLI-аргументы        ← trainer.max_epochs=3 training.lr=5e-5: правки для одного запуска
```

**Ключевое правило проекта:** настройки, специфичные для эксперимента (`task`, `dataset`, `model`,
`lora`, `data`, `training`, `scheduler`), в `config.yaml` помечены как обязательные (`???`).
Это гарантирует, что каждый запуск явно выбирает эксперимент — никаких «магических умолчаний».

Запустить обучение можно двумя способами:

- **Через пресет** (рекомендуется): `+experiment=arxiv_summarization` — задаёт все обязательные
  группы одним аргументом.
- **Вручную**: явно указать каждую группу: `task=summarization dataset=arxiv_summarization …`

Готовые пресеты (`conf/experiment/`):

| Аргумент | Задача | Датасет |
|----------|--------|---------|
| `+experiment=arxiv_summarization` | Суммаризация | arXiv |
| `+experiment=open_code_instruct_qwen` | Кодогенерация | Open Code Instruct |

---

### Файлы конфигов

- Конфиги лежат в `experiments/training/conf`; точка входа — `config.yaml`.
- Скрипт обучения читает конфиги через `@hydra.main(..., config_path="../conf", ...)` и
  принимает оверрайды из CLI.
- Скачивание датасетов и моделей выполняется интерактивно через ноутбук
  `experiments/misc_ops/prefetch_assets.ipynb` (без Hydra).

- Группы конфигов:
    - `conf/paths/paths_config.yaml` — ключ `paths.project_root`; по умолчанию Linux-путь, на
      своей машине переопределяйте через CLI
    - `conf/task/*.yaml` — имя задачи, шаблон промпта, MLflow-теги
    - `conf/dataset/*.yaml` — путь к датасету, имена сплитов, маппинг полей
    - `conf/model/*.yaml` — `local_path` и `name`; стабильные runtime defaults (квантизация и т.д.)
      берутся из dataclass и не дублируются в YAML
    - `conf/lora/*.yaml` — ранг адаптера `r`, alpha, dropout, целевые модули
    - `conf/data/*.yaml` — бюджеты токенов, batch size, num_workers
    - `conf/training/*.yaml` — seed, lr, weight_decay
    - `conf/scheduler/*.yaml` — тип планировщика LR и его параметры
    - `conf/trainer/*.yaml` — пресеты `pytorch_lightning.Trainer`
    - `conf/callbacks/*.yaml` — пресеты callbacks
    - `conf/logger/*.yaml` — пресеты MLflow logger
    - `conf/experiment/*.yaml` — тонкие оверлеи, выбирающие несколько групп одним аргументом

Про пути и рабочие директории в этом проекте

- Hydra меняет текущую рабочую директорию на `hydra.run.dir`. В скриптах пути приводятся к
  абсолютным через `paths.project_root`, поэтому:
    - относительные пути в конфигурациях трактуются относительно корня проекта (
      `paths.project_root`)
    - всегда передавайте корректный `paths.project_root` для своей машины (пример ниже)

### 🔎 Напоминалка про изучение конфигов из CLI (Hydra)

- Посмотреть доступные группы и опции (help выводит список override-групп):

```bash
python -m experiments.training.train_adapter.start_train --help
```

- Вывести финальный составленный конфиг, не запуская задачу (`--resolve` разворачивает интерполяции
  `${...}`):

```bash
python -m experiments.training.train_adapter.start_train \
  +experiment=arxiv_summarization --cfg job --resolve
```

- Показать только поддерево (например, секцию `model`):

```bash
python -m experiments.training.train_adapter.start_train \
  +experiment=arxiv_summarization --cfg job --resolve -p model
```

- Посмотреть конфиг самой Hydra (логгирование, каталоги и т.п.):

```bash
python -m experiments.training.train_adapter.start_train --cfg hydra --resolve
```

- Диагностическая информация (плагины, searchpath, версия):

```bash
python -m experiments.training.train_adapter.start_train --info
```

Примечания:

- Флаги `--cfg`/`--info` печатают информацию и завершают программу, сама тренировка не стартует.

> ⚠️ Windows: кавычки вокруг путей и прямые слэши (`C:/...`) избавляют от экранирования.

### Запуск скриптов

1) **Скачивание датасета и модели** — используйте ноутбук
   `experiments/misc_ops/prefetch_assets.ipynb`.
   Задайте `PROJECT_ROOT`, выберите конфигурацию датасета / модели и запустите ячейки.
   Ноутбук позволяет интерактивно изучить скачанные данные перед добавлением в DVC.

2) **Обучение адаптера (training/train_adapter/start_train.py)**

 - `config.yaml` задаёт инфраструктурные дефолты: `trainer=single_gpu`,
   `callbacks=checkpoint_only`, `logger=mlflow_train_adapter`. Experiment-specific
   группы обязательны — укажите `+experiment=<name>` или задайте каждую вручную.
- Основные секции, которые можно переопределять из CLI:
    - `task.*` (имя задачи, шаблон промпта, MLflow task tags)
    - `dataset.*` (local_path, split names, field mapping, validation_fraction)
    - `model.*` (обычно достаточно `local_path`; стабильные runtime defaults берутся из dataclass)
    - `lora.*` (r, lora_alpha, target_modules, ...)
    - `data.*` (лимиты токенов, batch size, num_workers, train_on_inputs)
  - `data_module.*` (например, `shuffle`)
    - `training.*` (seed, lr, weight_decay)
    - `scheduler.*` (enabled, warmup_steps, type, ...)
    - `trainer.*` (max_epochs, devices, accelerator, precision, ...)
    - `callbacks.checkpoint.*` (save_top_k, monitor, ...)
  - `logger.*` (параметры Lightning MLflow logger)
  - `tracking.*` (поведение MLflow tracking и env path)

  Отдельно:
  - `trainer=single_gpu`, `callbacks=checkpoint_only`, `logger=mlflow_train_adapter` выбираются в `config.yaml`
    через Hydra defaults и могут быть заменены на другие пресеты без вложенных package override-ов
  - `+experiment=<preset>` — задаёт все обязательные experiment-specific группы одним
    аргументом. Например: `+experiment=open_code_instruct_qwen`.
    Без пресета каждую группу нужно указать явно.

  Отдельно:
  - evaluation запускается через dedicated eval DAG, а не через training config
  - registration / alias promotion выполняются отдельным шагом после успешного train run

  Примеры:

```bash
# Стандартный запуск суммаризации
python -m experiments.training.train_adapter.start_train \
  +experiment=arxiv_summarization \
  paths.project_root="C:/Users/user/MyGitRepos/agent-042"

# Изменить число эпох и LR поверх experiment preset
python -m experiments.training.train_adapter.start_train \
  +experiment=arxiv_summarization \
  paths.project_root="C:/Users/user/MyGitRepos/agent-042" \
  trainer.max_epochs=3 \
  training.lr=5e-5

# Переопределить путь к модели (если скачали в другое место)
python -m experiments.training.train_adapter.start_train \
  +experiment=arxiv_summarization \
  paths.project_root="C:/Users/user/MyGitRepos/agent-042" \
  model.local_path="C:/data/models/Qwen/Qwen3-0.6B"

# Запустить coding SFT preset
python -m experiments.training.train_adapter.start_train \
  +experiment=open_code_instruct_qwen \
  paths.project_root="C:/Users/user/MyGitRepos/agent-042"
```

### Мультизапуски (sweeps) Hydra

- Hydra позволяет запускать набор экспериментов одной командой через `-m` и списки значений:

```bash
# Перебор LR и accumulate_grad_batches (2×2 = 4 запуска)
python -m experiments.training.train_adapter.start_train -m \
  +experiment=arxiv_summarization \
  paths.project_root="C:/Users/user/MyGitRepos/agent-042" \
  training.lr=1e-4,5e-5 \
  trainer.accumulate_grad_batches=4,8
```

- Каждый запуск получит собственную папку `hydra.run.dir` и запись логов.

> 💡 Совет: комбинируйте sweeps с небольшой `max_epochs` для быстрого перебора и sanity-check.

Где смотреть логи и артефакты

- Hydra конфиги и runtime-артефакты: `artifacts/training/hydra/...`
- Каждый запуск создаёт уникальную директорию: `artifacts/training/runs/<timestamp>-<uuid>/`
  - `checkpoints/` — чекпоинты PyTorch Lightning
  - `export/` — сохранённые веса и токенайзер лучшего чекпоинта
  - `metadata/` — resolved Hydra config и lineage metadata
  - `evaluation/` — post-train evaluation summary
- При включённом MLflow (настроить .env): артефакты и метрики в трекинге MLflow; Hydra-артефакты
  также загружаются в MLflow

Советы по конфигам

- Если `training/conf/paths/paths_config.yaml` содержит чужой путь — не редактируйте его в репозитории, а
  переопределяйте через CLI `paths.project_root=...`.
- Для Windows используйте прямые слэши (`C:/...`) или экранируйте обратные слэши в кавычках.
- Чтобы скачать новый датасет или модель — отредактируйте параметры в ноутбуке
  `experiments/misc_ops/prefetch_assets.ipynb` и запустите нужные ячейки.

## Куда что складывается (данные, логи, метрики, артефакты, параметры)

- **Датасеты**: сохраняются в директорию, указанную в ноутбуке `prefetch_assets.ipynb`
  (по умолчанию `assets/datasets/...`). Версионируются через DVC в Yandex Cloud.

- **Модели (prefetch)**: сохраняются в директорию, указанную в ноутбуке `prefetch_assets.ipynb`
  (по умолчанию `assets/models/...`).

- Каждый запуск обучения создаёт собственную директорию:
  `artifacts/training/runs/<timestamp>-<uuid>/`.
  Внутри неё:
  - `checkpoints/` — Lightning checkpoints
  - `export/` — экспортированный адаптер и токенайзер лучшего чекпоинта
  - `metadata/` — resolved config, git SHA, dataset DVC hash, hardware info
  - `evaluation/` — результаты автоматической post-train оценки

- Логи Hydra: `artifacts/training/hydra/...`. В каждой папке запуска (`hydra.run.dir`) Hydra
  сохраняет снимок конфигов в подпапке `.hydra`.

- Нативные логи Python: подхватываются Hydra и сохраняются в ту же папку (`hydra.run.dir`).

- Локальные артефакты Lightning: `artifacts/training/runs/<timestamp>-<uuid>/checkpoints/...`.
- Экспорт адаптера для регистрации и дальнейшего использования:
  `artifacts/training/runs/<timestamp>-<uuid>/export/...`.

- Логи MLflow: через MLFlow проходят:
  - Метрики и параметры - триггерятся через Lightning, можно отслеживать обучение в MLFlow UI
  - Post-train eval метрики логируются в тот же run как `eval.<metric>`
  - Вся папка Hydra - отправляется в Yandex Cloud (триггерится MLFlow, но НЕ проксируются через
    MLFlow server!)

- **Model Registry (MLflow)**: после обучения адаптер можно вручную зарегистрировать в MLflow Model
  Registry через ноутбук `experiments/training/lora_ops.ipynb`. Подробнее — ниже.

- Базовый корень для относительных путей: `paths.project_root` (обязательно указывайте корректный
  путь для своей машины).

## 🗂️ Model Registry — управление LoRA-адаптерами

Реестр адаптеров построен на **MLflow Model Registry** и решает ключевую задачу: обеспечить плавный
и воспроизводимый переход обученного адаптера из эксперимента в production inference.

### Концепция

Каждый обученный LoRA-адаптер регистрируется как **Model Version** внутри именованной группы
(**Registered Model**). Имена моделей следуют конвенции `lora-<task>`:

| Registered Model      | Задача                | LoRA для…           |
|-----------------------|-----------------------|---------------------|
| `lora-summarize`      | Суммаризация статей   | `summarize` task    |
| `lora-code`           | Генерация кода        | `code` task         |
| `lora-chat`           | Чат / QA              | `chat` task         |

Для управления жизненным циклом используются **aliases** (не deprecated stages):

- **`champion`** — текущий production-адаптер. Именно он загружается в vLLM.
- **`challenger`** — кандидат на замену champion (для A/B-тестирования или ревью).

### Регистрация адаптера в реестре

Обучение через `start_train.py` только логирует метрики и артефакты в MLflow Tracking.
Регистрация в Model Registry — отдельный осознанный шаг через ноутбук
`experiments/training/lora_ops.ipynb`:

```bash
# 1. Обучить адаптер (без регистрации в Registry)
python -m experiments.training.train_adapter.start_train \
  +experiment=arxiv_summarization \
  paths.project_root="C:/Users/user/MyGitRepos/agent-042"

# 2. Просмотреть результаты в MLflow UI, выбрать лучший run

# 3. Зарегистрировать выбранный run в Model Registry
#    (через lora_ops.ipynb или Python API: registry.register_adapter(...))
```

Такое разделение не засоряет реестр промежуточными экспериментами и обеспечивает
осознанный контроль над тем, какие адаптеры попадают в каталог развёртывания.

### CLI для управления реестром

> **Примечание:** CLI-скрипт `scripts/manage_registry.py` удалён. Операции с реестром
> (register, promote, demote, download, sync) теперь выполняются через ноутбук
> `experiments/training/lora_ops.ipynb` или напрямую через `shared.model_registry`
> Python API.

Скрипт `src/shared/model_registry.py` — программный интерфейс для управления
адаптерами в реестре. Для локальных запусков читает `MLFLOW_TRACKING_URI`
из корневого `.env` репозитория, созданного из корневого `.env.example`.

**Просмотр всех зарегистрированных адаптеров:**

```python
# Python API (из lora_ops.ipynb)
from shared.model_registry import AdapterRegistry
registry = AdapterRegistry()
registry.list_models()
```

**Промотирование версии в production (alias champion):**

```python
registry.promote(model_name="lora-summarize", version=3, alias="champion")
```

**Снять alias:**

```python
registry.demote(model_name="lora-summarize", alias="champion")
```

### Синхронизация адаптеров на inference-хосте

Для подготовки адаптеров и загрузки в работающий vLLM используется модуль `src/shared/model_registry.py`.
Он скачивает aliased-адаптеры (champion, challenger) из реестра и загружает их в vLLM
через hot-load REST API (`POST /v1/load_lora_adapter`) — без рестарта сервера.

```bash
# Из корня проекта (с настроенным .env)
python -m shared.model_registry sync --adapters-dir ./assets/adapters
```

По умолчанию команда читает endpoint vLLM из `VLLM_BASE_URL`
`--vllm-url` нужен только для явного override.

Результат на диске:

```
assets/adapters/
├── lora-summarize/
│   └── v3/
│       └── model/
│           ├── adapter_config.json
│           ├── adapter_model.safetensors
│           └── ...
└── lora-code/
    └── v2/
        └── model/
            └── ...
```

В vLLM адаптеры регистрируются с именами `{model}-{alias}`, например:
`lora-summarize-champion`, `lora-code-challenger`.

### Полный рабочий процесс: от обучения до inference

```bash
# 1. Обучить адаптер (метрики и артефакты логируются в MLflow)
python -m experiments.training.train_adapter.start_train \
  +experiment=arxiv_summarization \
  paths.project_root="C:/Users/user/MyGitRepos/agent-042"

# 2. Посмотреть версии, метрики в MLflow UI, выбрать лучший run

# 3. Зарегистрировать лучший run в Model Registry (через lora_ops.ipynb)
#    registry.register_adapter(run_id="<RUN_ID>", artifact_path="model", model_name="lora-summarize")

# 4. Промотировать зарегистрированную версию в production
#    registry.promote(model_name="lora-summarize", version=3, alias="champion")

# 5. Синхронизировать адаптеры на inference-хосте (hot-load в работающий vLLM)
python -m shared.model_registry sync
```

### Конфигурация vLLM для multi-LoRA

В `docker-compose.yaml` vLLM запускается с поддержкой multi-LoRA и hot-load:

```yaml
command: [
  "--model", "${VLLM_MODEL}",
  "--enable-lora",
  "--max-loras", "${VLLM_MAX_LORAS}",
  "--max-lora-rank", "${VLLM_MAX_LORA_RANK}",
  ...
]
environment:
  VLLM_ALLOW_RUNTIME_LORA_UPDATING: "true"
volumes:
  - ${PROJECT_ROOT}/assets/models:/models:rw
  - ${PROJECT_ROOT}/assets/adapters:/adapters:rw
```

Ключевые переменные окружения (`.env`):

```bash
VLLM_MAX_LORAS=4        # макс. число одновременно загруженных LoRA
VLLM_MAX_LORA_RANK=16   # >= lora.r из конфига обучения
```
