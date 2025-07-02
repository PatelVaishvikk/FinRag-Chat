from app.auth import authenticate
from app.rbac import get_collections_for_role
from app.vector_store import search_collections

def main():
    print("\n==== FinSolve RAG + RBAC Terminal Assistant ====\n")
    username = input("Enter your username: ").strip()
    role = authenticate(username)
    if not role:
        print("Invalid user. Exiting.")
        return
    print(f"Authenticated as '{username}' (Role: {role})")

    allowed_collections = get_collections_for_role(role)
    if not allowed_collections:
        print("You do not have access to any data collections. Exiting.")
        return

    print(f"You can search in: {', '.join(allowed_collections)}")
    print("\nType your questions (or 'exit' to quit):\n")

    while True:
        q = input("Your question: ").strip()
        if q.lower() in ["exit", "quit"]:
            break
        results = search_collections(allowed_collections, q)
        if not results:
            print("No relevant results found.\n")
            continue
        for idx, r in enumerate(results, 1):
            print(f"\nResult {idx}:")
            print(f"Document: {r['document'][:400]}{'...' if len(r['document']) > 400 else ''}")
            print(f"Metadata: {r['metadata']}")
            print(f"Source Collection: {r['collection']}")
            print(f"Similarity Score: {r['similarity']:.2f}")
            print("-" * 50)
        print()

if __name__ == "__main__":
    main()
