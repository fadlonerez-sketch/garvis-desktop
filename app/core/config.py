"""Central configuration for Garvis.

All tunables (model names, hosts, paths, secrets) must be read from here.
Nothing else in the codebase should hardcode a model name, API key, or
machine-specific path. Values are sourced from environment variables / a
.env file, with sensible local-first defaults.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        env_prefix="GARVIS_",
        extra="ignore",
        # Lets fields with an unprefixed env alias (ANTHROPIC_API_KEY) still be
        # set by field name in code/tests.
        populate_by_name=True,
    )

    # --- Paths -----------------------------------------------------------
    data_dir: Path = Field(default=_PROJECT_ROOT / "data")
    log_dir: Path = Field(default=_PROJECT_ROOT / "data" / "logs")

    # --- LLM ---------------------------------------------------------------
    llm_provider: str = Field(
        default="tiered",
        description="Must be 'tiered' (the default): every request goes to Claude Haiku 5.5 through "
        "the Anthropic API (config/models.json, one model per tier, no fallback -- see "
        "app/llm/router.py:TieredLLMRouter). The former 'gemini', 'groq', 'openrouter', "
        "'ollama' and 'auto' modes are refused at startup.",
    )
    llm_model_registry_path: Path = Field(default=_PROJECT_ROOT / "config" / "models.json")
    max_context_tokens: int | None = Field(
        default=None,
        description="Conversation-history token budget (excluding the system prompt). Unset = "
        "1200 for the legacy providers (sized for Groq's 8000-TPM limit, unchanged) and 12000 "
        "for 'tiered'/'gemini', whose models have far larger limits -- 1200 would trim away "
        "the user's question as soon as one web_research result arrives.",
    )

    gemini_api_key: str = Field(
        default="",
        description="Secret. Google AI Studio API key (https://aistudio.google.com/apikey). "
        "Never log or print this value.",
    )
    gemini_base_url: str = Field(default="https://generativelanguage.googleapis.com/v1beta/openai")
    gemini_model: str = Field(
        default="gemini-3.8-flash", description="Model for GARVIS_LLM_PROVIDER=gemini (single-model mode)."
    )
    gemini_request_timeout_seconds: float = Field(default=60.0)

    groq_api_key: str = Field(default="", description="Secret. Never log or print this value.")
    groq_model: str = Field(default="openai/gpt-oss-120b")
    groq_request_timeout_seconds: float = Field(default=60.0)

    openrouter_api_key: str = Field(
        default="", description="Secret. Never log or print this value."
    )
    openrouter_model: str = Field(
        default="",
        description="OpenRouter model id, e.g. 'anthropic/claude-haiku-4.5'. Required when "
        "GARVIS_LLM_PROVIDER=openrouter -- OpenRouterProvider raises ConfigurationError if "
        "unset, since OpenRouter serves hundreds of models and there is no safe default.",
    )
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")
    openrouter_request_timeout_seconds: float = Field(default=60.0)

    omniroute_api_key: str = Field(
        default="",
        description="Secret. OmniRoute's local client API key (created in its dashboard). Used by "
        "'omniroute' entries in config/models.json. Never log or print this value.",
    )
    omniroute_base_url: str = Field(
        default="http://127.0.0.1:20128/v1",
        description="Local OmniRoute gateway. Must be a loopback address -- OmniRouteProvider "
        "refuses anything else so the key never leaves this machine.",
    )
    omniroute_request_timeout_seconds: float = Field(default=60.0)

    # --- Direct Anthropic API (primary provider, app/llm/anthropic.py) ------
    anthropic_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "GARVIS_ANTHROPIC_API_KEY"),
        description="Secret. Claude API key from platform.claude.com (Settings > API keys), read from "
        "ANTHROPIC_API_KEY in Garvis's .env (or the environment). Unset = Garvis starts, but every model "
        "request fails with an error asking for the key (no other model is used). Never log or print it.",
    )
    anthropic_base_url: str = Field(
        default="https://api.anthropic.com",
        description="Pinned explicitly so a stray ANTHROPIC_BASE_URL in the environment can't redirect the key.",
    )
    anthropic_request_timeout_seconds: float = Field(
        default=45.0, gt=0,
        description="Per-request timeout. A timed-out request is not resent (it may have been processed "
        "and billed); the error is reported instead.",
    )

    # --- LLM usage accounting and local spending limits (app/llm/usage.py) --
    # These are Garvis's own limits on its own cost *estimates* (provider-
    # reported tokens x config/models.json prices). They stop Garvis from
    # sending requests; they are not a billing cap -- set a hard spend limit
    # in the provider's console (Anthropic Console > Limits) for that.
    llm_usage_db_path: Path = Field(
        default=_PROJECT_ROOT / "data" / "usage.db",
        description="Local ledger: one row per model request (tokens, estimated cost, model). No prompt text.",
    )
    llm_budget_enabled: bool = Field(default=True, description="Check the limits below before every request.")
    llm_max_calls_per_task: int = Field(
        default=15, ge=1, description="Model requests allowed for one user request (retries included)."
    )
    llm_max_cost_per_task_usd: float = Field(default=0.05, ge=0)
    llm_daily_budget_usd: float = Field(default=0.50, ge=0)
    llm_monthly_budget_usd: float = Field(default=5.00, ge=0)
    llm_max_prompt_tokens: int = Field(
        default=60_000, ge=1000,
        description="Refuse a request whose estimated prompt is larger. Claude Haiku 5.5 charges 5x for "
        "prompts over 100K tokens; this keeps a wide margin below that.",
    )
    llm_output_reserve_tokens: int = Field(
        default=2000, ge=0,
        description="Reply size assumed when projecting the next request's cost for the budget check.",
    )
    llm_max_retries: int = Field(
        default=1, ge=0, le=5,
        description="Extra attempts on the SAME model after a rate-limit/unavailable error (exponential "
        "backoff). OmniRoute already retries a 429 twice and a 5xx once on its side.",
    )
    llm_retry_backoff_seconds: float = Field(default=2.0, ge=0)

    # --- Speech-to-text ------------------------------------------------
    stt_provider: str = Field(default="faster_whisper")
    stt_model: str = Field(default="base")
    stt_device: str = Field(default="cpu")
    stt_compute_type: str = Field(default="int8")
    stt_sample_rate: int = Field(default=16000)
    stt_language: str | None = Field(
        default="en",
        description="Language code passed to faster-whisper (e.g. 'en'). Pinning this "
        "avoids per-utterance language auto-detection, which is unreliable on short or "
        "noisy clips and was observed producing garbled non-English hallucinated "
        "transcriptions. Set to None/empty to fall back to auto-detect.",
    )
    stt_min_duration_seconds: float = Field(
        default=0.3,
        description="Utterances shorter than this (after VAD segmentation) are treated "
        "as noise and skipped rather than sent to Whisper -- very short clips are a "
        "classic hallucination trigger for faster-whisper.",
    )

    # --- Text-to-speech ------------------------------------------------
    tts_provider: str = Field(default="piper")
    tts_voice: str = Field(default="en_US-lessac-medium")
    tts_voice_dir: Path = Field(default=_PROJECT_ROOT / "data" / "voices")
    tts_executable: str | None = Field(
        default=None,
        description="Path to a standalone piper executable. If unset, Garvis invokes "
        "the 'piper' pip package (piper-tts) via 'python -m piper' instead.",
    )

    # --- Hands-free conversation session (VAD tuning) ---------------------
    # Energy-threshold VAD (app/voice/vad.py) has no ground truth without a
    # real microphone/room, so every one of these is exposed for the user to
    # tune rather than hardcoded. Defaults are reasonable starting points
    # for a quiet-ish desktop room with a normal microphone, not a
    # scientifically calibrated value.
    session_poll_interval_seconds: float = Field(
        default=0.03, description="How often the session loop drains the microphone queue."
    )
    session_energy_threshold: float = Field(
        default=500.0,
        description="RMS amplitude (0-32767) above which audio counts as speech while "
        "listening for the user's utterance.",
    )
    session_min_speech_seconds: float = Field(
        default=0.15,
        description="Minimum continuous loud audio before an utterance is considered "
        "started -- higher avoids triggering on brief accidental noises.",
    )
    session_min_silence_seconds: float = Field(
        default=0.7,
        description="Minimum continuous quiet audio after speech before an utterance is "
        "considered finished -- higher avoids cutting the user off during a short pause; "
        "lower reduces the wait before Garvis responds.",
    )
    session_barge_in_energy_threshold: float = Field(
        default=900.0,
        description="RMS threshold used only while Garvis is speaking (higher than "
        "session_energy_threshold) to reduce false barge-in triggers from Garvis's own "
        "TTS output bleeding back into the microphone from speakers. No acoustic echo "
        "cancellation is performed -- a headset avoids this failure mode entirely.",
    )
    session_max_utterance_seconds: float = Field(
        default=30.0, description="Safety cap: an utterance is force-ended after this long."
    )
    session_echo_guard_enabled: bool = Field(
        default=True,
        description="Tell Garvis's own voice (echo from the speakers) apart from new speech by comparing "
        "the microphone with the exact audio being played (app/voice/echo.py). Off = the older "
        "loudness-only heuristic.",
    )
    session_echo_margin: float = Field(
        default=0.2,
        description="Share of a frame's energy that may stay unexplained by the playback before it counts "
        "as new sound (absorbs echo-model error when the speakers are loud). Higher = fewer false "
        "interruptions, but quieter speech over Garvis is missed. 0.2 measured: zero false barge-ins on "
        "real speaker echo, real double-talk caught in ~0.19 s.",
    )
    tts_playback_backend: str = Field(
        default="sounddevice",
        description="'sounddevice' (exact playback timing, needed by the echo guard; falls back to "
        "winsound if no output stream opens) or 'winsound' (the original player).",
    )

    # --- Wake-word standby (app/voice/wake.py) ----------------------------
    # Local only: standby uses the same mic/VAD/faster-whisper as sessions
    # and never calls the LLM. Reuses session_poll_interval_seconds,
    # session_energy_threshold and session_min_speech_seconds.
    wake_word_enabled: bool = Field(
        default=False,
        description="Kept for compatibility; no longer arms anything at launch. With voice ownership "
        "(app/core/voice_ownership.py) the Desktop never claims the microphone just because it started: "
        "the user ticks 'Wake word standby' on the Voice page, and it listens only while no other Garvis "
        "device (phone, Eye Station) holds the voice.",
    )
    wake_min_segment_seconds: float = Field(
        default=0.4, description="Speech shorter than this is never transcribed in standby."
    )
    wake_max_segment_seconds: float = Field(
        default=3.0,
        description="At most this much of one speech segment is transcribed in standby -- "
        "bounds Whisper's cost per segment.",
    )
    wake_min_silence_seconds: float = Field(
        default=0.6, description="Silence that ends a standby speech segment."
    )
    wake_match_threshold: float = Field(
        default=0.75,
        description="Per-word fuzzy similarity (0-1) required to accept a wake phrase.",
    )
    wake_cooldown_seconds: float = Field(
        default=2.0, description="After a wake, further wake candidates are ignored this long."
    )
    wake_playback_tail_seconds: float = Field(
        default=0.5,
        description="Standby ignores the mic this long after Garvis stops speaking, so its "
        "own voice is never evaluated as a wake phrase.",
    )

    # --- Memory ----------------------------------------------------------
    memory_db_path: Path = Field(default=_PROJECT_ROOT / "data" / "garvis.db")
    max_context_messages: int = Field(default=30)

    # --- Confirmations / permissions -----------------------------------
    confirmation_timeout_seconds: float = Field(default=120.0)

    # --- Desktop app --------------------------------------------------------
    global_hotkey: str = Field(
        default="ctrl+alt+g",
        description="System-wide shortcut that brings the running Garvis window to the front (empty = "
        "off). To also START Garvis with it, create the Start-menu shortcut: "
        "python scripts/install_shortcut.py.",
    )

    # --- Agent ------------------------------------------------------------
    agent_max_tool_iterations: int = Field(
        default=15,
        description="Max LLM <-> tool round-trips in one turn. Multi-step computer tasks (inspect, act, "
        "verify) need more than a handful; each step is one LLM request.",
    )

    # --- Gmail (OAuth) ---------------------------------------------------
    gmail_client_secret_path: Path = Field(
        default=_PROJECT_ROOT / "config" / "gmail_client_secret.json"
    )
    gmail_token_path: Path = Field(default=_PROJECT_ROOT / "data" / "gmail_token.json")

    # --- Web search -------------------------------------------------------
    web_search_max_results: int = Field(default=5)

    # --- Browser automation (Playwright) ---------------------------------
    browser_headless: bool = Field(
        default=False,
        description="Run the controlled browser headless. Defaults to False (visible) "
        "so browser actions are easy to observe/debug; set True for unattended use.",
    )
    browser_nav_timeout_ms: int = Field(default=15000)
    browser_action_timeout_ms: int = Field(default=10000)
    browser_channel: str = Field(
        default="chrome",
        description="Playwright browser channel. 'chrome' drives the installed Google Chrome "
        "(no extra download); if it can't launch, Garvis logs it and falls back to the bundled "
        "Chromium ('playwright install chromium'). Empty = always bundled Chromium.",
    )

    # --- Garvis workspace (dedicated Windows virtual desktop) -------------
    workspace_enabled: bool = Field(
        default=True,
        description="Keep Garvis's controlled browser on its own virtual desktop so research "
        "never takes over the user's screen. Needs pyvda; degrades (logged) to the current "
        "desktop if the virtual-desktop API is unavailable on this Windows build.",
    )
    workspace_name: str = Field(default="Garvis")

    # --- Browser research (web_research tool) ----------------------------
    research_search_engine: str = Field(
        default="duckduckgo", description="'duckduckgo' or 'bing' -- see app/browser/research.py."
    )
    research_max_sources: int = Field(default=3, description="Result pages opened and read per research task.")
    research_source_chars: int = Field(
        default=1500, description="Max characters of relevant text returned per source."
    )

    # --- Desktop control (mouse / keyboard / windows / screen) ------------
    desktop_control_enabled: bool = Field(
        default=True,
        description="Register the desktop-control tools (mouse_*, keyboard_*, scroll, "
        "focus_window, switch_window, list_windows, screen_info, screen_capture). Windows only; "
        "on other platforms the tools are registered but fail with a clear message.",
    )
    desktop_confirm_input_actions: bool = Field(
        default=False,
        description="True = confirm EVERY click / typing / key press / hotkey. Default False: routine "
        "input runs directly and only committing actions are confirmed -- clicks on controls named "
        "like Send/Delete/Pay, Enter or newlines in terminal/chat/mail apps, send and "
        "permanent-delete hotkeys (see app/tools/desktop.py:CommitGuard).",
    )
    desktop_commit_guard_apps: str = Field(
        default="",
        description="Comma-separated process names where Enter counts as committing (sends a message "
        "or runs a command). Empty = the built-in list of terminals, chat and mail apps.",
    )
    desktop_context_in_prompt: bool = Field(
        default=True,
        description="Tell the model at the start of each turn which window is active and which apps are "
        "open (the active window's title and the other apps' names only -- no screen content).",
    )
    ocr_tesseract_path: str = Field(
        default="",
        description="tesseract.exe (or its folder) for read_screen_text. Empty = auto-detect: PATH "
        "and the standard install folders (Program Files, LocalAppData).",
    )
    ocr_languages: str = Field(default="eng+heb", description="Tesseract languages, '+'-separated.")
    ocr_model: str = Field(default="fast", description="'fast' or 'best' tessdata subfolder, if present.")
    file_backup_dir: Path = Field(
        default=_PROJECT_ROOT / "data" / "file_backups",
        description="Copies of files taken before write_text_file/edit_text_file change them.",
    )
    desktop_max_type_chars: int = Field(default=2000)
    desktop_screenshot_dir: Path = Field(default=_PROJECT_ROOT / "data" / "screenshots")
    desktop_screenshot_keep: int = Field(
        default=20, description="Only the newest N screenshots are kept on disk."
    )

    # --- Mobile interface (iPhone / Android over the local network, app/mobile) -----
    mobile_enabled: bool = Field(
        default=False,
        description="Serve the mobile web interface over HTTPS on the local network. Off by "
        "default: turning it on opens a port on this PC (LAN clients only, paired devices only).",
    )
    mobile_host: str = Field(
        default="0.0.0.0",
        description="Interface to listen on. 0.0.0.0 = all IPv4 interfaces; set this PC's LAN "
        "address to listen on that network only. Clients outside mobile_allowed_networks are "
        "refused either way.",
    )
    mobile_port: int = Field(default=8443)
    mobile_allowed_networks: str = Field(
        default="",
        description="Comma-separated CIDRs allowed to connect. Empty = private/link-local/"
        "loopback ranges only (10/8, 172.16/12, 192.168/16, 169.254/16, 127/8, fc00::/7, "
        "fe80::/10, ::1).",
    )
    mobile_extra_hosts: str = Field(
        default="",
        description="Comma-separated extra hostnames/IPs to put in the HTTPS certificate, on top "
        "of the auto-detected private LAN addresses and <computer-name>.local.",
    )
    mobile_data_dir: Path = Field(
        default=_PROJECT_ROOT / "data" / "mobile",
        description="Local CA, server certificate and paired-device records. The CA private key "
        "never leaves this directory.",
    )
    mobile_ca_name_constraints: bool = Field(
        default=True,
        description="Limit the local CA to private IPs and *.local names, so even a stolen CA key "
        "can't impersonate real websites to the phone. Changing it only affects a newly "
        "generated CA.",
    )
    mobile_pairing_code_ttl_seconds: int = Field(default=300)
    mobile_device_token_days: int = Field(
        default=30, description="A paired phone must pair again after this many days."
    )
    mobile_max_message_chars: int = Field(default=4000)
    mobile_max_audio_bytes: int = Field(
        default=8 * 1024 * 1024, description="Upload limit for one voice recording."
    )
    mobile_max_audio_seconds: float = Field(default=60.0)
    mobile_stt_timeout_seconds: float = Field(default=120.0)
    mobile_tts_timeout_seconds: float = Field(default=120.0)
    mobile_max_tts_chars: int = Field(
        default=3000, description="Longer replies are shown in full but only this much is spoken."
    )
    mobile_remote_access: str = Field(
        default="off",
        description="'off' (default): LAN only. 'tailscale': also accept paired phones that connect over "
        "this PC's Tailscale address (a private, encrypted network between your own devices -- no router "
        "port is opened). Clients must still be paired; TLS and the device token are unchanged.",
    )
    mobile_remote_hosts: str = Field(
        default="",
        description="Comma-separated extra host names the phone may use to reach this PC remotely, e.g. "
        "the Tailscale MagicDNS name (desktop-xxxx.tailnet-name.ts.net). Only accepted with "
        "mobile_remote_access=tailscale.",
    )
    mobile_phone_action_timeout_seconds: float = Field(
        default=90.0, description="How long Garvis waits for the phone to run (or the user to approve) a phone action."
    )

    # --- Reminders and scheduled calls (app/reminders; rung by the Android app) ---
    reminders_timezone: str = Field(
        default="",
        description="IANA time zone for reminder times (e.g. Asia/Jerusalem). Empty = the zone the paired "
        "phone reports, else this PC's current UTC offset.",
    )
    user_name: str = Field(default="", description="Your name as Garvis says it in English reminder calls (e.g. Alex).")
    user_name_he: str = Field(default="", description="Your name in Hebrew, for Hebrew reminder calls (e.g. אלכס). "
                              "Empty = user_name.")
    user_gender: Literal["male", "female"] = Field(
        default="male", description="Grammatical gender Garvis uses when it speaks Hebrew to you (תזכור / תזכרי).")
    reminders_model_phrasing: bool = Field(
        default=True,
        description="Let the assistant's model (the same Claude Haiku, no other model) word each scheduled call "
        "to fit it; its sentence is checked against the reminder and the built-in wording is kept otherwise. "
        "False = built-in wording only (no model call).",
    )

    # --- Image generation (app/images; separate from the LLM, off by default) ---
    image_provider: str = Field(
        default="",
        description="'' (off) or 'cloudflare' (Cloudflare Workers AI). Never enabled implicitly.",
    )
    cloudflare_account_id: str = Field(default="")
    cloudflare_api_token: str = Field(
        default="", description="Secret. Cloudflare API token with the Workers AI permission. Never log it."
    )
    image_model: str = Field(default="@cf/black-forest-labs/flux-1-schnell")
    image_steps: int = Field(default=4, description="Diffusion steps (1-8 for flux-1-schnell).")
    image_timeout_seconds: float = Field(default=120.0)

    # --- Logging ----------------------------------------------------------
    log_level: str = Field(default="INFO")
    log_content_excerpts: bool = Field(
        default=False,
        description="Opt-in: include the first ~120 characters of each LLM reply in the log "
        "(content_excerpt). Off by default -- replies can contain personal data; length and hash "
        "are always logged.",
    )

    def effective_max_context_tokens(self) -> int:
        if self.max_context_tokens is not None:
            return self.max_context_tokens
        return 12000 if self.llm_provider in ("tiered", "gemini") else 1200

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.tts_voice_dir.mkdir(parents=True, exist_ok=True)
        self.gmail_client_secret_path.parent.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the process-wide Settings singleton, creating it on first use."""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_dirs()
    return _settings
