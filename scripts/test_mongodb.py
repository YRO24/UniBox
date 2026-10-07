import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.mongo_client import get_db_client


if __name__ == "__main__":

    print("Testing MongoDB connection...")

    try:
        client = get_db_client()
        client.admin.command("ping")

        print("MongoDB connection successful!")

    except Exception as e:
        print("\nMongoDB connection failed!")
        print("\nActual error:")
        print(e)