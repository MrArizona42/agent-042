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

1. Скачать датасет и модель — откройте ноутбук `experiments/scripts/prefetch_assets.ipynb`,
   задайте `PROJECT_ROOT` и выполните нужные ячейки.
2. Запустить обучение адаптера:

```bash
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

- Конфиги лежат в `experiments/conf` и используются для обучения адаптера:
    - Обучение адаптера: `config.yaml`
- Скрипт обучения читает конфиги через декоратор `@hydra.main(..., config_path="../conf", ...)` и
  принимает оверрайды из CLI.
- Скачивание датасетов и моделей выполняется интерактивно через ноутбук
  `experiments/scripts/prefetch_assets.ipynb` (без Hydra).

- Группы конфигов:
    - `conf/paths/paths_config.yaml` — ключ `paths.project_root` (по умолчанию проставлен
      Linux-путь, на своей машине лучше переопределять через CLI)
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
```

- Вывести финальный составленный конфиг, не запуская задачу (`--resolve` разворачивает интерполяции
  `${...}`):

```bash
python ./experiments/scripts/train_hydra.py --cfg job --resolve
```

- Показать только поддерево (например, секцию `experiment`):

```bash
python ./experiments/scripts/train_hydra.py --cfg job --resolve -p experiment
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

- Флаги `--cfg`/`--info` печатают информацию и завершают программу, сама тренировка не стартует.

> ⚠️ Windows: кавычки вокруг путей и прямые слэши (`C:/...`) избавляют от экранирования.

### Запуск скриптов

1) **Скачивание датасета и модели** — используйте ноутбук
   `experiments/scripts/prefetch_assets.ipynb`.
   Задайте `PROJECT_ROOT`, выберите конфигурацию датасета / модели и запустите ячейки.
   Ноутбук позволяет интерактивно изучить скачанные данные перед добавлением в DVC.

2) **Обучение адаптера (scripts/train_hydra.py)**

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
- Чтобы скачать новый датасет или модель — отредактируйте параметры в ноутбуке
  `experiments/scripts/prefetch_assets.ipynb` и запустите нужные ячейки.

## Куда что складывается (данные, логи, метрики, артефакты, параметры)

- **Датасеты**: сохраняются в директорию, указанную в ноутбуке `prefetch_assets.ipynb`
  (по умолчанию `assets/datasets/...`). Версионируются через DVC в Yandex Cloud.

- **Модели (prefetch)**: сохраняются в директорию, указанную в ноутбуке `prefetch_assets.ipynb`
  (по умолчанию `assets/models/...`).

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

- Базовый корень для относительных путей: `paths.project_root` (обязательно указывайте корректный
  путь для своей машины).
