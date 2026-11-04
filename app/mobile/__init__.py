"""Mobile interface: an HTTPS web app the phone opens (Safari on iPhone, Chrome on Android) on the
local network. A transport layer only -- it drives the one Agent/EventBus/
VoicePipeline that AgentBridge already owns (see build.py), never its own."""
