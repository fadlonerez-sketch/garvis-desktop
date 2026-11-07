# Configuration

All settings live in `.env` in the project folder. `.env.example` lists every
one of them with comments; `run.bat` creates `.env` from it on first run and
never overwrites an existing one.

## Your API key

Garvis reads the key from `ANTHROPIC_API_KEY` in `.env`. Set it in any one of
three ways:

- Answer **Y** when `run.bat` asks.
- Run this; it asks for the key with hidden input and writes only that line
  of `.env`:

  ```bat
  .venv\Scripts\python.exe scripts\check_anthropic.py --set-key
  ```

- Open `.env` in a text editor and set `ANTHROPIC_API_KEY=your-key`.

Restart Garvis after changing the key. To check it:

```bat
.venv\Scripts\python.exe scripts\check_anthropic.py
```

This makes one free check against the API and uses no tokens.

- **`.env` stays on your PC.** It is git-ignored, and the key is never
  written to logs or shown in Garvis's settings page.
- **Keep the key in `.env`**, not in a global Windows environment variable,
  so other tools don't pick it up.
- **Set a monthly spend limit in the Anthropic Console.** Garvis also has
  its own budget limits (`GARVIS_LLM_DAILY_BUDGET_USD`,
  `GARVIS_LLM_MONTHLY_BUDGET_USD`, `GARVIS_LLM_MAX_COST_PER_TASK_USD`),
  checked against its own estimates. They are **not** a billing cap.
- `.venv\Scripts\python.exe scripts\usage_report.py` shows Garvis's recorded
  spend.

## Model

Garvis uses one model: **Claude Haiku 5.5** (`claude-haiku-5-5`), through
the official Anthropic API, configured in `config/models.json`.

- `GARVIS_LLM_PROVIDER` must stay `tiered` (the default).
- Garvis refuses to start if the registry names any other model, and there
  is no fallback to other providers.
- Each request is classified locally as small, medium or large. That only
  sets Claude's effort level, which is the `reasoning_effort` per tier in
  `config/models.json`.
- Adapters for Gemini, Groq, OpenRouter and a local OmniRoute gateway exist
  in `app/llm/`, but this release does not use them for requests.

## Other settings

| Setting | Default | Effect |
|---|---|---|
| `GARVIS_TTS_VOICE` | `en_US-lessac-medium` | Piper voice; the `.onnx` and `.onnx.json` files go in `data\voices` |
| `GARVIS_STT_MODEL` / `GARVIS_STT_DEVICE` | `base` / `cpu` | faster-whisper model size and device (`cuda` if you have one) |
| `GARVIS_DESKTOP_CONTROL_ENABLED` | `true` | Mouse, keyboard, window and screen tools |
| `GARVIS_DESKTOP_CONFIRM_INPUT_ACTIONS` | `false` | `true` asks before **every** click and keystroke |
| `GARVIS_GLOBAL_HOTKEY` | `ctrl+alt+g` | Brings Garvis to the front (empty = off) |
| `GARVIS_BROWSER_HEADLESS` | `false` | Run the controlled browser hidden |
| `GARVIS_OCR_TESSERACT_PATH` | auto-detect | Path to `tesseract.exe` |
| `GARVIS_GMAIL_CLIENT_SECRET_PATH` | `config/gmail_client_secret.json` | Gmail OAuth client; see below |
| `GARVIS_LOG_CONTENT_EXCERPTS` | `false` | `true` adds the first ~120 characters of each model reply to the log (length and hash are always logged) |

## Gmail (optional)

1. In Google Cloud Console, enable the Gmail API.
2. Create an OAuth **Desktop app** client.
3. Save its JSON as `config/gmail_client_secret.json` (git-ignored).

The first Gmail request opens your browser for consent.

## Runtime data

Everything Garvis writes at runtime goes under `data/`, which is
git-ignored: logs, databases, tasks, screenshots and backups.

## Start-menu shortcut

A Start-menu shortcut with the global hotkey is optional:
`.venv\Scripts\python.exe scripts\install_shortcut.py` (undo with
`--remove`).
