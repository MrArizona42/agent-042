# 🧪 Как проводить эксперименты

> Немного магии Hydra, немного дисциплины DVC и щепотка Lightning — быстрый и воспроизводимый цикл экспериментов.

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
- `dvc add <file_or_dir>`: Добавляет файл или директорию под контроль DVC (например, `dvc add data/`).
- `dvc push`: Отправляет добавленные данные в удалённое хранилище (ycloud).
- `dvc pull`: Загружает данные из удалённого хранилища.
- `dvc repro`: Воспроизводит пайплайн эксперимента (на основе dvc.yaml).
- `dvc status`: Проверяет статус изменений в данных и пайплайнах.

Для настройки remote используется конфигурация в `.dvc/config`. Подробности в документации DVC.

> 💡 Рекомендация: держите крупные артефакты (данные, модели) под DVC — репозиторий станет легче и воспроизводимее.

Все используемые данные необходимо версионировать в DVC!

## ⚙️ Hydra и конфигурирование

Ниже — краткая и практичная инструкция по использованию Hydra в этом репозитории.

Что такое Hydra здесь
- Конфиги лежат в `experiments/conf` и собираются в два основных сценария:
  - Работа с активами (датасет/модель): `config-assets.yaml`
  - Обучение адаптера: `config.yaml`
- Скрипты читают конфиги через декоратор `@hydra.main(..., config_path="../conf", ...)` и принимают оверрайды из CLI.

Ключевые файлы конфигурации
- `conf/config-assets.yaml`
  - defaults:
    - `paths: paths_config` — базовый путь проекта
    - `assets/dataset: arxiv-summarization-01` — выбранный датасет
    - `assets/model: ministral-3b-instruct` — выбранная модель
  - hydra.run.dir: логи Hydra пишутся в `experiments/logs/hydra-logs/assets-hydra-logs/<дата>/<время>`
- `conf/config.yaml`
  - defaults:
    - `paths: paths_config`
    - `experiment: train_adapter`
  - hydra.run.dir: `experiments/logs/hydra-logs/train-hydra-logs/<дата>/<время>`
- Группы конфигов:
  - `conf/paths/paths_config.yaml` — ключ `paths.project_root` (по умолчанию проставлен Linux-путь, на своей машине лучше переопределять через CLI)
  - `conf/assets/dataset/*.yaml` — описания датасетов (имя на HF, сплиты, целевая директория)
  - `conf/assets/model/*.yaml` — описания моделей (идентификатор на HF, целевая директория)
  - `conf/experiment/train_adapter.yaml` — все параметры обучения (model/lora/data/trainer/scheduler/output/mlflow)

Важно про пути и рабочие директории
- Hydra меняет текущую рабочую директорию на `hydra.run.dir`. В скриптах пути приводятся к абсолютным через `paths.project_root`, поэтому:
  - относительные пути в конфигурациях трактуются относительно корня проекта (`paths.project_root`)
  - всегда передавайте корректный `paths.project_root` для своей машины (пример ниже)

### 🔎 Изучение конфигов из CLI (Hydra)
- Посмотреть доступные группы и опции (help выводит список override-групп):
```bash
python ./experiments/scripts/train_hydra.py --help
python ./experiments/scripts/prefetch_data.py --help
python ./experiments/scripts/prefetch_model.py --help
```
- Вывести финальный составленный конфиг, не запуская задачу (`--resolve` разворачивает интерполяции `${...}`):
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
- Флаги `--cfg`/`--info` печатают информацию и завершают программу, сама тренировка/загрузка не стартует.
- В `--help` в разделе Config groups отображаются группы, их имена и доступные варианты (например, `assets/model: mistral-7b, ministral-3b-instruct`).

> ⚠️ Windows: кавычки вокруг путей и прямые слэши (`C:/...`) избавляют от экранирования.

### Запуск скриптов с конфигами Hydra
1) **Скачивание датасета (scripts/prefetch_data.py)**
- Использует `config-assets.yaml` и группу `assets/dataset`
- Важные ключи:
  - `assets.dataset.name`, `assets.dataset.config`, `assets.dataset.train_split`, `assets.dataset.val_split`
  - `assets.dataset.target_dir` — куда сохранить (относительно `paths.project_root`, если не абсолютный)
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
  - `assets.model.target_dir` — локальная папка (относительно `paths.project_root`, если не абсолютная)
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
- Сохранённые веса и токенайзер: `experiment.output.save_dir` (по умолчанию вложено в `assets/newly_trained/<дата>/<время>`)
- При включённом MLflow (настроить .env): артефакты и метрики в трекинге MLflow; Hydra-артефакты также загружаются в MLflow

Советы по конфигам
- Если `conf/paths/paths_config.yaml` содержит чужой путь — не редактируйте его в репозитории, а переопределяйте через CLI `paths.project_root=...`.
- Для Windows используйте прямые слэши (`C:/...`) или экранируйте обратные слэши в кавычках.
- Чтобы добавить новый датасет или модель — создайте `.yaml` в соответствующей группе (`conf/assets/dataset/` или `conf/assets/model/`) и выбирайте через `assets/dataset=<name>` или `assets/model=<name>`.
- Для быстрой диагностики скрипты `prefetch_*` печатают итоговую конфигурацию (`OmegaConf.to_yaml(cfg)`). Сверяйте, что пути резолвятся правильно.
