from modules.database import Database
import config
import os

print(f"Log Path: {config.ACCESS_LOG_PATH}")
if os.path.exists(config.ACCESS_LOG_PATH):
    print("File exists.")
    with open(config.ACCESS_LOG_PATH, "r") as f:
        print(f"Content preview (first 100 chars): {f.read(100)}")
else:
    print("File does not exist.")

db = Database()
logs = db.read_access_logs(limit=5)
print(f"Logs read: {len(logs)}")
for log in logs:
    print(log)
