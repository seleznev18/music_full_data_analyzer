# Music Full Data Analyzer

Сервис для анализа музыкальных треков: извлечение BPM, тональности, размера, скрапинг текстов с Genius.com, генерация описаний через Gemini API. Включает экспорт в формат ACE-Step 1.5 для файн-тюнинга.

## Требования

- Python 3.11+
- ffmpeg (в PATH)
- API-ключи: Genius, Gemini (через kie.ai)

## Установка

```bash
pip install -r requirements.txt
cp .env.example .env
# заполнить .env своими ключами
```

## Структура файлов

```
music_for_preprocessing/
├── manifest.csv          # список треков для обработки
├── track1.flac           # аудиофайлы
└── track2.flac
```

### Формат manifest.csv

```csv
file_name,song_name,artist,has_vocals
track1.flac,Название,Артист,true
track2.flac,Инструментал,Артист,false
```

- `file_name` — имя файла в папке `music_for_preprocessing/`
- `has_vocals` — `true`/`false`. Если `false` — текст с Genius не скрапится

Форматы аудио: `.mp3`, `.wav`, `.flac`, `.ogg`, `.m4a`, `.aac`

## API

### `POST /api/v1/analyze`

Анализ одного аудиофайла. Принимает `multipart/form-data`.

**Параметры запроса:**

| Поле | Тип | Обязательное | Описание |
|------|-----|:---:|----------|
| `file` | file | да | Аудиофайл (.mp3, .wav, .flac, .ogg, .m4a, .aac) |
| `song_name` | string | да | Название трека (для поиска текста на Genius) |
| `artist` | string | да | Исполнитель |
| `has_vocals` | bool | нет | Есть ли вокал (default: `true`). Если `false` — Genius не запрашивается, `lyrics=""` |

**Пример запроса (curl):**

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "file=@track.flac" \
  -F "song_name=The Fate of Ophelia" \
  -F "artist=Taylor Swift" \
  -F "has_vocals=true"
```

**Ответ (`200 OK`):**

```json
{
  "file_name": "track.flac",
  "song_name": "The Fate of Ophelia",
  "artist": "Taylor Swift",
  "bpm": "84.72",
  "key": "C# minor",
  "time_signature": "4/4",
  "lyrics": "[Verse 1]\nSome lyrics here...",
  "caption": "Lush indie folk ballad with breathy female vocal..."
}
```

**Ошибки:** `400` — неподдерживаемый формат, `422` — ошибка анализа аудио.

## Использование

### 1. Анализ треков

Терминал 1 — запуск API-сервера:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Терминал 2 — пакетная обработка:
```bash
python process_songs.py
```

Флаги: `--host http://localhost:8000` `--output results.json`

Результат сохраняется в `results.json` — массив объектов с полями: `file_name`, `song_name`, `artist`, `bpm`, `key`, `time_signature`, `lyrics`, `caption`.

### 2. Экспорт в ACE-Step

После обработки — экспорт для файн-тюнинга ACE-Step 1.5:

```bash
python export_acestep.py
```

Флаги: `--results results.json` `--music-dir music_for_preprocessing` `--manifest music_for_preprocessing/manifest.csv` `--output output`

### Результат экспорта

```
output/
├── vae_training/          # только аудио (48kHz stereo FLAC)
│   ├── track_00001.flac
│   └── manifest.csv
│
└── dit_training/          # аудио + тексты + аннотации
    ├── track_00001.flac
    ├── track_00001.lyrics.txt
    ├── track_00001.json
    └── manifest.csv
```

- Аудио ресемплируется в **48kHz stereo FLAC**
- Тексты очищаются от артефактов Genius (Embed, You might also like, и т.д.)
- JSON-аннотации используют поля ACE-Step: `caption`, `bpm` (int), `keyscale`, `timesignature`, `language`
- Язык определяется автоматически из текста песни
- Инструменталы получают `[Instrumental]\n(instrumental)` в файле текста

### Валидация при экспорте

Пропускаются треки с:
- длительностью вне 30–600 сек
- BPM вне 20–300
- пустым caption после очистки
- повреждённым аудио

## .env

```
GENIUS_API_TOKEN=...
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-3-flash
```
