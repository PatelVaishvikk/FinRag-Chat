# app/auth.py

from typing import Optional

# static maps
USERS = {
    "alice":   "finance",
    "bob":     "marketing",
    "charlie": "hr",
    "dave":    "engineering",
    "ceo":     "c_level",
    "eve":     "employee"
}

PASSWORDS = {
    "alice": "alice123",
    "bob":   "bob123",
    "charlie": "charlie123",
    "dave":  "dave123",
    "ceo":   "c3oP@ss",
    "eve":   "eve123"
}

def authenticate(username: str, password: str) -> Optional[str]:
    u = username.strip().lower()
    # check both exist
    if u in USERS and PASSWORDS.get(u) == password:
        return USERS[u]
    return None
