from __future__ import annotations

import os
import requests


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing")
    if not chat_id:
        raise RuntimeError("TELEGRAM_CHAT_ID is missing")

    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={
            "chat_id": chat_id,
            "text": "✅ NHL Goal Model connecté. Les futurs picks buteur qualifiés pourront être envoyés ici.",
            "disable_web_page_preview": True,
        },
        timeout=30,
    )
    try:
        payload = r.json()
    except Exception:
        payload = {}
    if not r.ok or not payload.get("ok"):
        description = str(payload.get("description") or f"HTTP {r.status_code}")
        raise RuntimeError(f"Telegram send failed: {description}")
    print("Telegram smoke test sent successfully")


if __name__ == "__main__":
    main()
