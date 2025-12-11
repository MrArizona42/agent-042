1. Установить [UV-менеджер](https://docs.astral.sh/uv/getting-started/installation/)
2. Запустить команду для синхронизации окружения:
```bash
uv sync
```
3. Добавить креды Yandex Cloud в файл `agent-042/.dvc/config.local`:
```text
['remote "ycloud"']
    access_key_id = YCA...
    secret_access_key = YCM...
```
4. Сделать dvc pull для скачивания данных:
```bash
dvc pull -r ycloud
```
Опционально, можно скачать датасет напрямую из Hugging Face, запустив скрипт 
`agent-042/experiments/scripits/prefetch_data.py`.
5. Скачать базовую модель Mistral 7B, запустив скрипт 
   `agent-042/experiments/scripts/prefetch_model.py`.