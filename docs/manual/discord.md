# Discord

![Discord inspector](assets/ui-placeholder.svg)

## Purpose

Send **outbound webhooks** to Discord (for example captions, announcements, or bot channels), configured from the host. This is unrelated to **VRChat** (which uses OSC from the desktop app)—separate settings and transport.

## Open the inspector

Sidebar → **Discord**.

## Setup (high level)

1. Create a **Webhook URL** in your Discord server channel settings (Integrations → Webhooks). Use only `https://discord.com/api/webhooks/...` URLs.
2. Paste it into the Sigil inspector (treat it like a password).
3. Use **Send test message** to verify the webhook before relying on STT.
4. For **forum** or **media** channels, set either an existing **thread ID** (query) or a **new thread name** so Discord accepts the message—see Discord’s Execute Webhook docs.

## Tips

- Anyone with the webhook URL can post to that channel—rotate it if leaked.
- Discord limits message `content` length (2000 characters); Sigil splits longer finals into multiple webhook posts with a short delay.
- **Post only when live** uses Twitch or Kick live status (either platform counts as live).
- Rate limits apply per Discord rules; the app retries briefly on HTTP 429.

## Related

- [Twitch](twitch.md)
- [Quick actions](ui-overview.md) — dock may link here
