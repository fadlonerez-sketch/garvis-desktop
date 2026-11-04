# Garvis Desktop (Beta)

Garvis is a personal AI assistant for Windows. You can type or talk to it.
It answers with a cloud language model and can act on your PC through
permission-checked tools:

- open apps and websites;
- research the web in a controlled browser;
- read and edit files and run PowerShell;
- operate windows;
- use Gmail.

Speech recognition (faster-whisper) and speech synthesis (Piper) run
locally. The language model always runs in the cloud, using **your own API
key**.

> Beta: a personal project, Windows only. Expect rough edges.

## Requirements

- **Windows 10 or 11, 64-bit.**
- **Python 3.14, 64-bit.** Get it from
  [python.org](https://www.python.org/downloads/); "Install for current
  user" is enough, no admin rights. This is the version Garvis is tested
  with.
- **Internet access** for the first setup: about 1 GB of Python packages
  plus a 60 MB voice.
- **An Anthropic API key.** Create one in the
  [Anthropic Console](https://console.anthropic.com/). API use is billed by
  Anthropic to **your** account.
- **Optional:**
  - microphone and speakers for voice;
  - Google Chrome for the browser tools (without it, run
    `.venv\Scripts\python.exe -m playwright install chromium` once);
  - [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for
    reading text off the screen.

## Install and start

1. Download the project and unzip it, or `git clone` it, into a **short
   path** such as `C:\garvis` or `C:\Users\you\garvis`. The folder path
   must stay within about 80 characters unless Windows long paths are
   enabled; `run.bat` checks this.
2. Double-click **`run.bat`**, or run it from a terminal in the project
   folder.

On first run, `run.bat`:

1. checks for Python 3.14 (64-bit);
2. creates a private `.venv` in the project folder;
3. installs the pinned dependencies (`requirements.txt` +
   `constraints.txt`);
4. creates your `.env` from `.env.example` (an existing `.env` is never
   overwritten) and offers to save your API key;
5. downloads the default Piper voice into `data\voices`;
6. starts Garvis.

To start Garvis later, run `run.bat` again; it skips the steps that are
already done. Keep the console window open while Garvis runs, because it
shows errors. `run.bat --setup-only` sets everything up without starting
Garvis. Nothing outside the project folder is changed, and no admin rights
are needed.

## Your API key

Garvis reads the key from `ANTHROPIC_API_KEY` in the `.env` file in the
project folder. Set it in any one of three ways:

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
- `.venv\Scripts\python.exe scripts\usage_report.py` shows Garvis's recorded spend.

### Model

Garvis uses one model: **Claude Haiku 5.5** (`claude-haiku-5-5`), through
the official Anthropic API, configured in `config/models.json`.

- `GARVIS_LLM_PROVIDER` must stay `tiered` (the default).
- Garvis refuses to start if the registry names any other model, and there
  is no fallback to other providers.
- Each request is classified locally as small, medium or large. That only
  sets Claude's effort level, which is the `reasoning_effort` per tier in
  `config/models.json`.
- Adapters for Gemini, Groq, OpenRouter and a local OmniRoute gateway exist
  in `app/llm/`, but this beta does not use them for requests.

## Other settings

All settings live in `.env`, and `.env.example` lists them with comments.
The most useful ones:

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

**Gmail (optional):**

1. In Google Cloud Console, enable the Gmail API.
2. Create an OAuth **Desktop app** client.
3. Save its JSON as `config/gmail_client_secret.json` (git-ignored).

The first Gmail request opens your browser for consent.

Everything Garvis writes at runtime goes under `data/`, which is
git-ignored: logs, databases, tasks, screenshots and backups.

## Using it

- Type a message and press Enter, or click **Start listening** to talk.
- Risky actions open a confirmation dialog first. Garvis's own code decides
  this, not the model.
- A Start-menu shortcut with the global hotkey is optional:
  `.venv\Scripts\python.exe scripts\install_shortcut.py` (undo with
  `--remove`).

## Tests

```bat
.venv\Scripts\python.exe -m pytest
```

The suite is offline:

- It never uses your API key or your `data/` folder.
- It fakes the browser, the audio devices and the model.

A few tests use real Windows facilities and only read system state. The
exception is one that briefly registers an unused hotkey
(`Ctrl+Alt+Shift+F23`). `scripts\smoke_test_*.py` are manual checks that use
real hardware or a real API key. Read each one before running it.

## Security

- **Approval rules are code, not prompts.** The model can only call
  registered tools with validated arguments. A permission checker decides
  what needs approval, and prompt injection can't change that.
- **These always ask first:**
  - sending email and submitting forms;
  - committing clicks (Send, Delete, Pay, …);
  - Enter in terminals, chat and mail apps and IDEs;
  - PowerShell commands that aren't provably read-only, and every test
    runner;
  - writes of executable or startup files and of Garvis's own code;
  - reading or writing credential files.
- **Untrusted input stays contained:**
  - web content and tool output are treated as untrusted;
  - secrets are redacted from tool output;
  - web fetches can't reach private or local network addresses.
- **No listening port by default.** The optional phone interface in
  `app/mobile/` is off unless you set `GARVIS_MOBILE_ENABLED=true`, and it
  isn't supported in this beta.

Report vulnerabilities privately; see [SECURITY.md](SECURITY.md).
`.venv\Scripts\python.exe scripts\secret_scan.py` checks a working copy for committed secrets
before you publish a fork.

## Known limitations

- Windows only. One model (Claude Haiku 5.5), with no fallback.
- The separate "Garvis workspace" virtual desktop relies on an undocumented
  Windows API (`pyvda`). If a Windows update breaks it, background research
  reports an error instead.
- Garvis can't control elevated (admin) windows. Real mouse and keyboard
  input works only on the desktop you're looking at.
- Wake phrases, OCR and reminder wording are tuned for English and Hebrew.

## Licences

No licence has been chosen for Garvis yet, so all rights are reserved. The
bundled fonts keep their own licences, next to the font files in
`app/ui/fonts/`:

- Inter and Manrope: SIL Open Font License 1.1;
- Material Symbols: Apache 2.0.

Python packages, Piper voices and Whisper models are downloaded under their
own licences.
