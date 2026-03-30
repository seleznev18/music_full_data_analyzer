# Music Full Data Analyzer

Сервис для массового анализа музыкальных треков: скачивание с S3, скрапинг текстов с Genius.com, генерация описаний через Gemini API (kie.ai прокси). Результат — `results.jsonl`.

## Быстрый старт

```bash
pip install -r requirements.txt
cp .env.example .env
# заполнить .env
./run_pipeline.sh
```

Требования: Python 3.11+, ffmpeg в PATH, AWS CLI (для S3), tmux.

## .env

```
GENIUS_API_TOKENS=key1,key2,key3
GEMINI_API_KEY=your_kie_ai_key
GEMINI_MODEL=gemini-3-flash
```

- `GENIUS_API_TOKENS` — несколько ключей через запятую. Пайплайн ротирует их автоматически, соблюдая интервал между запросами на каждый ключ. Чем больше ключей — тем быстрее обработка текстов.
- Если указан один ключ, можно использовать `GENIUS_API_TOKEN` (без S).
- `GEMINI_API_KEY` — ключ от [kie.ai](https://kie.ai) (OpenAI-совместимый прокси к Gemini).

## Формат manifest.csv

```csv
file_id,filename,song_name,artist,bpm,key,time_signature,has_vocals
abc123,track.flac,Song Name,Artist,120,C major,4/4,true
```

- `file_id` — уникальный ID (для резюма)
- `filename` — имя файла в S3
- `has_vocals` — `true`/`false`. Если `false`, текст с Genius не запрашивается

## Как работает pipeline.py

3-стадийный асинхронный конвейер на asyncio с очередями между стадиями:

```
manifest.csv
    ↓
[Stage 1] S3 Download → queue_a →
[Stage 2] Genius Lyrics → queue_b →
[Stage 3] Gemini Caption → queue_c →
[Writer] → results.jsonl
```

**Stage 1** — скачивает аудиофайлы с S3 через AWS CLI. Несколько воркеров параллельно.

**Stage 2** — ищет текст на Genius API (`/search` → скрапинг HTML страницы). Несколько ключей ротируются через `GeniusKeyPool` с настраиваемым интервалом. При 429 — экспоненциальный backoff.

**Stage 3** — конвертирует аудио в MP3 (mono, 96kbps, макс. 5 мин) через ffmpeg, отправляет base64 на kie.ai Gemini API, парсит SSE-стрим, очищает ответ от мусора.

**Writer** — записывает результат в JSONL построчно с flush после каждой записи.

**Резюм** — при перезапуске автоматически пропускает уже обработанные `file_id` из существующего `results.jsonl`.

**Graceful shutdown** — Ctrl+C завершает текущие задачи, сохраняет прогресс.

## Формат results.jsonl

Каждая строка — JSON-объект:

```json
{
  "file_id": "abc123",
  "file_name": "track.flac",
  "song_name": "Song Name",
  "artist": "Artist",
  "bpm": "120",
  "key": "C major",
  "time_signature": "4/4",
  "has_vocals": true,
  "lyrics": "[Verse 1]\nSome lyrics...",
  "found_on_genius": true,
  "has_markup": true,
  "caption": "Energetic pop track with bright synths..."
}
```

| Поле | Описание |
|------|----------|
| `found_on_genius` | `true` — песня найдена на Genius (даже если текст пустой/без разметки) |
| `has_markup` | `true` — текст содержит секционную разметку (`[Verse]`, `[Chorus]` и т.д.) |
| `lyrics` | Очищенный текст. Если `has_markup=false` но `found_on_genius=true` — содержит raw-текст без разметки |

## Запуск

`run_pipeline.sh` запускает пайплайн внутри tmux-сессии — он продолжит работу после отключения SSH.

```bash
./run_pipeline.sh [OPTIONS]
```

Все параметры прокидываются напрямую в `pipeline.py`.

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `--manifest` | `manifest.csv` | Путь к CSV |
| `--output` | `results.jsonl` | Выходной файл |
| `--temp-dir` | `/tmp/music_processing` | Папка для временных файлов |
| `--s3-bucket` | `n2gaz71k8d` | S3 бакет |
| `--s3-prefix` | `flac-dataset` | Префикс в S3 |
| `--s3-endpoint` | `https://s3api-eu-cz-1.runpod.io` | S3 endpoint |
| `--skip-lyrics` | `false` | Пропустить Genius — только скачивание + капшены |
| `--download-workers` | `10` | Воркеры скачивания |
| `--genius-workers` | `15` | Воркеры текстов (рекомендуется: кол-во ключей × 2) |
| `--genius-concurrency` | `5` | Макс. параллельных запросов к Genius (рекомендуется: = кол-во ключей) |
| `--genius-interval` | `2.0` | Секунд между запросами на 1 ключ (не менее 0.2 — лимит Genius 5 req/s на ключ) |
| `--gemini-workers` | `10` | Воркеры капшенов |
| `--gemini-concurrency` | `5` | Макс. параллельных запросов к Gemini |
| `--queue-a-size` | `150` | Размер очереди download → lyrics |
| `--queue-b-size` | `300` | Размер очереди lyrics → caption |

### Примеры

Стандартный запуск:
```bash
./run_pipeline.sh
```

Полный запуск (lyrics + captions):
```bash
./run_pipeline.sh \
  --gemini-workers 125 \
  --gemini-concurrency 125 \
  --genius-workers 12 \
  --genius-concurrency 6 \
  --genius-interval 2.0 \
  --download-workers 20 \
  --queue-a-size 500 \
  --queue-b-size 500
```

Только капшены (без Genius):
```bash
./run_pipeline.sh \
  --skip-lyrics \
  --gemini-workers 125 \
  --gemini-concurrency 125 \
  --download-workers 20 \
  --queue-a-size 500 \
  --queue-b-size 500
```

### Мониторинг

```bash
tmux attach -t pipeline       # подключиться к сессии (Ctrl+B, D — отключиться)
tail -f pipeline_output.log   # stdout в реальном времени
tail -f pipeline_errors.log   # ошибки
wc -l results.jsonl           # сколько обработано
```

Если сессия уже запущена, скрипт выдаст ошибку — используйте `tmux kill-session -t pipeline` чтобы завершить.

## Логи

- `pipeline_errors.log` — все ошибки с трейсбэками
- `lyrics_debug.log` — статус каждого запроса к Genius (OK, NOT_FOUND, NO_CONTAINER, CLEAN_EMPTY, EXCEPTION)

## Старый API (main.py)

Одиночный анализ треков через HTTP API:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000

curl -X POST http://localhost:8000/api/v1/analyze \
  -F "file=@track.flac" \
  -F "song_name=Song Name" \
  -F "artist=Artist"
```

Для пакетной обработки используйте `pipeline.py`.
