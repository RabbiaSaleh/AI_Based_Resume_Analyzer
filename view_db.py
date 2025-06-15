import mysql.connector

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="1234",
    database="ai_resume"
)
cursor = conn.cursor()

# Check if tables exist and print row counts for debugging
for table in ["users", "user_info", "resume_data"]:
    cursor.execute(f"SHOW TABLES LIKE '{table}'")
    if not cursor.fetchone():
        print(f"Table '{table}' does NOT exist in database 'ai_resume'.")
    else:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"Table '{table}' row count: {count}")

# Print all data if any rows exist
print("\nUsers:")
cursor.execute("SELECT * FROM users")
users = cursor.fetchall()
if users:
    for row in users:
        print(row)
else:
    print("No users found.")

print("\nUser Info:")
cursor.execute("SELECT * FROM user_info")
user_info = cursor.fetchall()
if user_info:
    for row in user_info:
        print(row)
else:
    print("No user info found.")

print("\nResume Data:")
cursor.execute("SELECT * FROM resume_data")
resume_data = cursor.fetchall()
if resume_data:
    for row in resume_data:
        print(row)
else:
    print("No resume data found.")

cursor.close()
conn.close()
