"""One-time interactive OAuth setup for the Gmail REST API.

Run once to authorize the app and write a reusable token (with a
refresh_token) to GMAIL_TOKEN_FILE. The runtime client then refreshes that
token silently — it never opens a browser.

    PYTHONPATH=. python scripts/gmail_oauth_setup.py

Prereq: download an OAuth 2.0 "Desktop app" client_secret JSON from the
Google Cloud Console and save it to GMAIL_CREDENTIALS_FILE.
"""
from __future__ import annotations

import os
import sys

from google_auth_oauthlib.flow import InstalledAppFlow

from config.settings import GMAIL_CREDENTIALS_FILE, GMAIL_TOKEN_FILE

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]


def main() -> int:
    if not os.path.exists(GMAIL_CREDENTIALS_FILE):
        print(
            f"ERROR: client secret not found at {GMAIL_CREDENTIALS_FILE}\n"
            "Download an OAuth 2.0 'Desktop app' client_secret JSON from the "
            "Google Cloud Console and save it there (or set GMAIL_CREDENTIALS_FILE).",
            file=sys.stderr,
        )
        return 1

    flow = InstalledAppFlow.from_client_secrets_file(GMAIL_CREDENTIALS_FILE, SCOPES)
    creds = flow.run_local_server(port=0)

    os.makedirs(os.path.dirname(GMAIL_TOKEN_FILE) or ".", exist_ok=True)
    with open(GMAIL_TOKEN_FILE, "w") as f:
        f.write(creds.to_json())

    print(f"Authorized. Token written to {GMAIL_TOKEN_FILE}")
    print("The Inbox tab and scripts/fetch_inbox.py will now use it automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
