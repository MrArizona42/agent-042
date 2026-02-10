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

```bash
# 1) Скачать датасет (укажите путь к корню проекта на вашей машине)
python ./experiments/scripts/prefetch_data.py \
  paths.project_root="C:/Users/user/MyGitRepos/agent-042"

# 2) Скачать модель (по умолчанию Ministral-3b-instruct)
python ./experiments/scripts/prefetch_model.py \
  paths.project_root="C:/Users/user/MyGitRepos/agent-042"

# 3) Запустить обучение адаптера (дефолтные параметры из конфигов)
python ./experiments/scripts/train_hydra.py \
  paths.project_root="C:/Users/user/MyGitRepos/agent-042"
```

> ℹ️ На Windows в Git Bash удобнее использовать прямые слэши в путях: `C:/...`.

## 🗂️ Структура директории

* `./conf` - Hydra конфиги
* `./scripts` - Скрипты экспериментов

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

- Конфиги лежат в `experiments/conf` и собираются в два основных сценария:
    - Работа с активами (датасет/модель): `config-assets.yaml`
    - Обучение адаптера: `config.yaml`
- Скрипты читают конфиги через декоратор `@hydra.main(..., config_path="../conf", ...)` и принимают
  оверрайды из CLI.

- Группы конфигов:
    - `conf/paths/paths_config.yaml` — ключ `paths.project_root` (по умолчанию проставлен
      Linux-путь, на своей машине лучше переопределять через CLI)
    - `conf/assets/dataset/*.yaml` — описания датасетов (имя на HF, сплиты, целевая директория)
    - `conf/assets/model/*.yaml` — описания моделей (идентификатор на HF, целевая директория)
    - `conf/experiment/train_adapter.yaml` — все параметры обучения (
      model/lora/data/trainer/scheduler/output/mlflow)

Про пути и рабочие директории в этом проекте

- Hydra меняет текущую рабочую директорию на `hydra.run.dir`. В скриптах пути приводятся к
  абсолютным через `paths.project_root`, поэтому:
    - относительные пути в конфигурациях трактуются относительно корня проекта (
      `paths.project_root`)
    - всегда передавайте корректный `paths.project_root` для своей машины (пример ниже)

### 🔎 Напоминалка про изучение конфигов из CLI (Hydra)

- Посмотреть доступные группы и опции (help выводит список override-групп):

```bash
python ./experiments/scripts/train_hydra.py --help
python ./experiments/scripts/prefetch_data.py --help
python ./experiments/scripts/prefetch_model.py --help
```

- Вывести финальный составленный конфиг, не запуская задачу (`--resolve` разворачивает интерполяции
  `${...}`):

```bash
python ./experiments/scripts/train_hydra.py --cfg job --resolve
python ./experiments/scripts/prefetch_data.py --cfg job --resolve
python ./experiments/scripts/prefetch_model.py --cfg job --resolve
```

- Показать только поддерево (например, секцию `experiment` или `assets`):

```bash
# Только секция experiment
python ./experiments/scripts/train_hydra.py --cfg job --resolve -p experiment

# Только секция assets (актуально для prefetch_*)
python ./experiments/scripts/prefetch_data.py --cfg job --resolve -p assets
python ./experiments/scripts/prefetch_model.py --cfg job --resolve -p assets
```

- Посмотреть конфиг самой Hydra (логгирование, каталоги и т.п.):

```bash
python ./experiments/scripts/train_hydra.py --cfg hydra --resolve
```

- Диагностическая информация (плагины, searchpath, версия):

```bash
python ./experiments/scripts/train_hydra.py --info
```

Примечания:

- Флаги `--cfg`/`--info` печатают информацию и завершают программу, сама тренировка/загрузка не
  стартует.
- В `--help` в разделе Config groups отображаются группы, их имена и доступные варианты (например,
  `assets/model: mistral-7b, ministral-3b-instruct`).

> ⚠️ Windows: кавычки вокруг путей и прямые слэши (`C:/...`) избавляют от экранирования.

### Запуск скриптов с конфигами Hydra

Возможно, это лишнее, но скачивание датасета и моделей из Hugging Face было сделано тоже через
Hydra конфиги.

1) **Скачивание датасета (scripts/prefetch_data.py)**

- Использует `config-assets.yaml` и группу `assets/dataset`
- Важные ключи:
    - `assets.dataset.name`, `assets.dataset.config`, `assets.dataset.train_split`,
      `assets.dataset.val_split`
    - `assets.dataset.target_dir` — куда сохранить (относительно `paths.project_root`, если не
      абсолютный)
      Примеры:

```bash
# Windows (Git Bash). Рекомендуется использовать прямые слэши или экранировать обратные
python ./experiments/scripts/prefetch_data.py \
  paths.project_root="C:/Users/user/MyGitRepos/agent-042"

# Поменять датасет-конфиг (если добавите новый файл в conf/assets/dataset/)
python ./experiments/scripts/prefetch_data.py \
  paths.project_root="C:/Users/user/MyGitRepos/agent-042" \
  assets/dataset=arxiv-summarization-01
```

2) **Скачивание модели (scripts/prefetch_model.py)**

- Использует `config-assets.yaml` и группу `assets/model`
- Важные ключи:
    - `assets.model.id` — repo_id на Hugging Face
    - `assets.model.target_dir` — локальная папка (относительно `paths.project_root`, если не
      абсолютная)
      Примеры:

```bash
# Модель по умолчанию (ministral/Ministral-3b-instruct)
python ./experiments/scripts/prefetch_model.py \
  paths.project_root="C:/Users/user/MyGitRepos/agent-042"

# Переключиться на Мistral-7B
python ./experiments/scripts/prefetch_model.py \
  paths.project_root="C:/Users/user/MyGitRepos/agent-042" \
  assets/model=mistral-7b
```

3) **Обучение адаптера (scripts/train_hydra.py)**

- Использует `config.yaml` -> `experiment: train_adapter`
- Основные секции, которые можно переопределять из CLI:
    - `experiment.model.*` (dtype, 4-bit, gradient_checkpointing, local_path, и т.д.)
    - `experiment.lora.*` (r, lora_alpha, target_modules, ...)
    - `experiment.data.*` (max_seq_length, batch_size, local_path, prompt_template)
    - `experiment.training.*` (lr, weight_decay)
    - `experiment.scheduler.*` (enabled, warmup_steps, type, ...)
    - `experiment.trainer.*` (max_epochs, devices, accelerator, precision, ...)
    - `experiment.output.save_dir`
    - `experiment.mlflow.*` (при наличии настроенной среды)
      Примеры:

```bash
# Базовый запуск (использует значения по умолчанию из конфигов)
python ./experiments/scripts/train_hydra.py \
  paths.project_root="C:/Users/user/MyGitRepos/agent-042"

# Изменить число эпох и LR
python ./experiments/scripts/train_hydra.py \
  paths.project_root="C:/Users/user/MyGitRepos/agent-042" \
  experiment.trainer.max_epochs=3 \
  experiment.training.lr=5e-5

# Переопределить модель на локальный путь (если скачали в другое место)
python ./experiments/scripts/train_hydra.py \
  paths.project_root="C:/Users/user/MyGitRepos/agent-042" \
  experiment.model.local_path="C:/data/models/Ministral-3b-instruct"
```

### Мультизапуски (sweeps) Hydra

- Hydra позволяет запускать набор экспериментов одной командой через `-m` и списки значений:

```bash
# Перебор LR и accumulate_grad_batches (2×2 = 4 запуска)
python ./experiments/scripts/train_hydra.py -m \
  paths.project_root="C:/Users/user/MyGitRepos/agent-042" \
  experiment.training.lr=1e-4,5e-5 \
  experiment.trainer.accumulate_grad_batches=4,8
```

- Каждый запуск получит собственную папку `hydra.run.dir` и запись логов.

> 💡 Совет: комбинируйте sweeps с небольшой `max_epochs` для быстрого перебора и sanity-check.

Где смотреть логи и артефакты

- Hydra конфиги и runtime-артефакты: `experiments/logs/hydra-logs/...`
- Логи PyTorch Lightning: `experiments/logs/lightning_logs`
- Сохранённые веса и токенайзер: `experiment.output.save_dir` (по умолчанию вложено в
  `assets/newly_trained/<дата>/<время>`)
- При включённом MLflow (настроить .env): артефакты и метрики в трекинге MLflow; Hydra-артефакты
  также загружаются в MLflow

Советы по конфигам

- Если `conf/paths/paths_config.yaml` содержит чужой путь — не редактируйте его в репозитории, а
  переопределяйте через CLI `paths.project_root=...`.
- Для Windows используйте прямые слэши (`C:/...`) или экранируйте обратные слэши в кавычках.
- Чтобы добавить новый датасет или модель — создайте `.yaml` в соответствующей группе (
  `conf/assets/dataset/` или `conf/assets/model/`) и выбирайте через `assets/dataset=<name>` или
  `assets/model=<name>`.
- Для быстрой диагностики скрипты `prefetch_*` печатают итоговую конфигурацию (
  `OmegaConf.to_yaml(cfg)`). Сверяйте, что пути резолвятся правильно.

## Куда что складывается (данные, логи, метрики, артефакты, параметры)

- **Датасеты**: сохраняются в `assets.dataset.target_dir` (относительно `paths.project_root`, если
  путь не абсолютный). Версионируются через DVC в Yandex Cloud.

- **Модели (prefetch)**: сохраняются в `assets.model.target_dir` (относительно `paths.project_root`).

- Выходы обучения (веса адаптера/модели), токенайзер и
  конфиги: по умолчанию `assets/newly_trained/<дата>/<время>`.

- Логи Hydra: `experiments/logs/hydra-logs/...`. В каждой папке запуска (`hydra.run.dir`) Hydra
  сохраняет снимок конфигов в подпапке `.hydra`.

- Нативные логи Python: подхватываются Hydra и сохраняются в ту же папку (`hydra.run.dir`).

- Логи Lightning: `experiments/logs/lightning_logs/...`. Lightning логирует локально только
  чекпойнты по настройкам Trainer/Callbacks (если включены).

- Логи MLflow: через MLFlow проходят:
  - Метрики и параметры - триггерятся через Lightning, можно отслеживать обучение в MLFlow UI
  - Вся папка Hydra - отправляется в Yandex Cloud (триггерится MLFlow, но НЕ проксируются через
    MLFlow server!)

- **Model Registry (MLflow)**: после обучения адаптер автоматически регистрируется в MLflow Model
  Registry (если `experiment.mlflow.register_model=true`). Подробнее — ниже.

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
| `lora-summarization`  | Суммаризация статей   | `summarize` task    |
| `lora-code`           | Генерация кода        | `code` task         |
| `lora-chat`           | Чат / QA              | `chat` task         |

Для управления жизненным циклом используются **aliases** (не deprecated stages):

- **`champion`** — текущий production-адаптер. Именно он загружается в vLLM.
- **`challenger`** — кандидат на замену champion (для A/B-тестирования или ревью).

### Автоматическая регистрация при обучении

При обучении через `train_hydra.py` адаптер автоматически регистрируется в реестре, если в
конфиге указано:

```yaml
# conf/experiment/train_adapter.yaml
mlflow:
  register_model: true
  registered_model_name: "lora-summarization"
```

Чтобы обучить адаптер под другую задачу, переопределите имя через CLI:

```bash
# Обучить и зарегистрировать адаптер для генерации кода
python ./experiments/scripts/train_hydra.py \
  paths.project_root="C:/Users/user/MyGitRepos/agent-042" \
  experiment.mlflow.registered_model_name="lora-code"
```

Если `registered_model_name` не задано, имя формируется автоматически по шаблону
`lora-<experiment_name>`.

### CLI для управления реестром: `manage_registry.py`

Скрипт `experiments/scripts/manage_registry.py` — операционный инструмент (не Hydra) для просмотра
и управления адаптерами в реестре. Читает `MLFLOW_BACKEND_URI` из `experiments/.env`.

**Просмотр всех зарегистрированных адаптеров:**

```bash
python experiments/scripts/manage_registry.py list
```

**Все версии конкретного адаптера:**

```bash
python experiments/scripts/manage_registry.py versions lora-summarization
```

**Промотирование версии в production (alias champion):**

```bash
python experiments/scripts/manage_registry.py promote lora-summarization 3
```

**Промотировать в staging (alias challenger):**

```bash
python experiments/scripts/manage_registry.py promote lora-summarization 5 --alias challenger
```

**Снять alias:**

```bash
python experiments/scripts/manage_registry.py demote lora-summarization
```

**Посмотреть, какие адаптеры сейчас в production:**

```bash
python experiments/scripts/manage_registry.py production
```

**Скачать production-адаптер локально:**

```bash
python experiments/scripts/manage_registry.py download lora-summarization ./my_adapters
```

### Синхронизация адаптеров на inference-хосте

Для подготовки адаптеров к загрузке в vLLM используется модуль `src/shared/model_registry.py`.
Он скачивает все champion-адаптеры из реестра и генерирует `lora-modules.json` для vLLM.

```bash
# Из корня проекта (с настроенным .env)
python -m shared.model_registry sync --adapters-dir ./assets/adapters
```

Результат:

```
assets/adapters/
├── lora-summarization/
│   └── v3/
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── ...
├── lora-code/
│   └── v2/
│       └── ...
├── lora-modules.json          ← для vLLM --lora-modules
└── adapters-summary.json      ← человекочитаемый манифест
```

### Полный рабочий процесс: от обучения до inference

```bash
# 1. Обучить адаптер (автоматически регистрируется в Registry)
python ./experiments/scripts/train_hydra.py \
  paths.project_root="C:/Users/user/MyGitRepos/agent-042" \
  experiment.mlflow.registered_model_name="lora-summarization"

# 2. Посмотреть версии, метрики в MLflow UI, выбрать лучшую
python experiments/scripts/manage_registry.py versions lora-summarization

# 3. Промотировать лучшую версию в production
python experiments/scripts/manage_registry.py promote lora-summarization 3

# 4. Синхронизировать адаптеры на inference-хосте
python -m shared.model_registry sync --adapters-dir ./assets/adapters

# 5. Перезапустить vLLM (docker-compose) — он подхватит адаптеры из assets/adapters/
docker compose -f infra/compose/docker-compose.yaml restart vllm
```

### Конфигурация vLLM для multi-LoRA

В `docker-compose.yaml` vLLM запускается с поддержкой multi-LoRA:

```yaml
command: [
  "--model", "${VLLM_MODEL}",
  "--enable-lora",
  "--max-loras", "${VLLM_MAX_LORAS}",
  "--max-lora-rank", "${VLLM_MAX_LORA_RANK}",
  ...
]
volumes:
  - ${PROJECT_ROOT}/assets/models:/models:rw
  - ${PROJECT_ROOT}/assets/adapters:/adapters:ro
```

Ключевые переменные окружения (`.env`):

```bash
VLLM_MAX_LORAS=4        # макс. число одновременно загруженных LoRA
VLLM_MAX_LORA_RANK=16   # >= lora.r из конфига обучения
```
