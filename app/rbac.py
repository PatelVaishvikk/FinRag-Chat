# app/rbac.py

USERS = {
    "alice": "finance",
    "bob": "marketing",
    "charlie": "hr",
    "dave": "engineering",
    "ceo": "c_level",
    "eve": "employee"
}

ROLE_COLLECTIONS = {
    "finance": ["finance_docs"],
    "marketing": ["marketing_docs"],
    "hr": ["hr_docs"],
    "engineering": ["engineering_docs"],
    "c_level": ["finance_docs", "marketing_docs", "hr_docs", "engineering_docs", "general_docs"],
    "employee": ["general_docs"]
}

def authenticate(username):
    return USERS.get(username)

def get_collections_for_role(role):
    return ROLE_COLLECTIONS.get(role, [])
