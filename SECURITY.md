# Security policy

## Reporting a vulnerability

Please **do not** report security problems in public issues, discussions or
pull requests.

Report them privately through GitHub's **private vulnerability reporting**:
open the repository's **Security** tab and choose **Report a
vulnerability**. If that button is missing, open a public issue that only
asks the maintainer for a private way to send a report. Leave out any
details of the problem.

A useful report includes:

- the affected file(s) and the version or commit;
- what an attacker can do and what they need first. For example: content
  on a web page Garvis reads, a device on the same network, or local
  access to the PC;
- steps to reproduce, ideally without real credentials or personal data;
- any suggested fix.

This is a personal project maintained on a best-effort basis. There is no
bug bounty and no guaranteed response time. Reports are acknowledged when
they are read and fixed as time allows. Please give a reasonable amount of
time before any public disclosure.

## Supported versions

Only the latest commit on the default branch is supported.

## Scope

In scope:

- **Ways around the permission and confirmation system:**
  - a tool call that runs without the confirmation its policy requires;
  - prompt injection that leads to a tool action without confirmation;
  - PowerShell commands misclassified as read-only.
- **Secret leaks:** API keys, OAuth tokens or pairing tokens reaching logs,
  tool output, the model or the phone interface.
- **Network-facing code:** the optional phone web interface (`app/mobile/`),
  including pairing, authentication, the network allowlist, TLS, and
  request handling.
- **Unsafe fetching:** SSRF or unsafe URL handling in the web-page and
  browser tools.
- **File-system misuse:** path traversal or unsafe file handling in the
  file tools.

Out of scope:

- Problems that need an attacker who already runs code as your Windows
  user.
- The behaviour of third-party services (the model provider, Google,
  Cloudflare) and of software you install yourself (Chrome, Tesseract,
  OmniRoute).
- Findings that only apply after you deliberately weaken a default, such
  as enabling the phone interface on an untrusted network or widening
  `GARVIS_MOBILE_ALLOWED_NETWORKS`.

## If you run Garvis

- Keep your `.env`, `config/gmail_client_secret.json` and the `data/`
  folder private. They hold your API key, OAuth tokens, paired-device
  records, the phone interface's local certificate authority, logs and
  databases.
- If a key may have been exposed, revoke it at the provider and create a new
  one. Deleting a file or a commit doesn't make an exposed key safe again.
- Leave `GARVIS_MOBILE_ENABLED=false` unless you need the phone interface,
  and allow it through Windows Firewall for **Private** networks only.
