import mysql.connector
import bcrypt

def connect_to_db():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="1234",
            database="ai_resume"
        )
        return conn
    except mysql.connector.Error as e:
        print(f"Database connection error: {e}")
        return None

def save_user_credentials(email, password):
    conn = connect_to_db()
    if not conn:
        return None
    cursor = conn.cursor()
    try:
        # Hash the password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        # Store as utf-8 string, not bytes
        hashed_password_str = hashed_password.decode('utf-8')
        query = """
            INSERT INTO users (email, password)
            VALUES (%s, %s)
        """
        cursor.execute(query, (email, hashed_password_str))
        conn.commit()
        user_id = cursor.lastrowid
        return user_id
    except mysql.connector.Error as e:
        print(f"Error saving user credentials: {e}")
        return None
    finally:
        cursor.close()
        conn.close()

def verify_user_credentials(email, password):
    conn = connect_to_db()
    if not conn:
        return None
    cursor = conn.cursor()
    try:
        query = "SELECT user_id, password FROM users WHERE email = %s"
        cursor.execute(query, (email,))
        result = cursor.fetchone()
        if result and bcrypt.checkpw(password.encode('utf-8'), result[1].encode('utf-8')):
            return result[0]  # Return user_id
        return None
    except mysql.connector.Error as e:
        print(f"Error verifying user: {e}")
        return None
    finally:
        cursor.close()
        conn.close()

def save_user_info(user_id, name, email, phone, file_path):
    conn = connect_to_db()
    if not conn:
        return None
    cursor = conn.cursor()
    try:
        # Fetch current values if any are None
        if None in (name, email, phone, file_path):
            cursor.execute("SELECT name, email, phone, file_path FROM user_info WHERE user_id = %s", (user_id,))
            row = cursor.fetchone()
            if row:
                name = name if name is not None else row[0]
                email = email if email is not None else row[1]
                phone = phone if phone is not None else row[2]
                file_path = file_path if file_path is not None else row[3]
        print(f"[DEBUG] Saving user_info: user_id={user_id}, name={name}, email={email}, phone={phone}, file_path={file_path}")
        query = """
            INSERT INTO user_info (user_id, name, email, phone, file_path)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                name=VALUES(name), email=VALUES(email), phone=VALUES(phone), file_path=VALUES(file_path)
        """
        values = (user_id, name, email, phone, file_path)
        cursor.execute(query, values)
        conn.commit()
        return user_id
    except mysql.connector.Error as e:
        print(f"Error saving user info: {e}")
        return None
    finally:
        cursor.close()
        conn.close()

def save_resume_data(user_id, skills, education, experience, category, confidence):
    conn = connect_to_db()
    if not conn:
        return None
    cursor = conn.cursor()
    try:
        query = """
            INSERT INTO resume_data (user_id, skills, education, experience, category, confidence)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        values = (user_id, skills, education, experience, category, confidence)
        cursor.execute(query, values)
        conn.commit()
        return cursor.lastrowid
    except mysql.connector.Error as e:
        print(f"Error saving resume data: {e}")
        return None
    finally:
        cursor.close()
        conn.close()

def save_resume_score(job_id, user_id, match_score):
    conn = connect_to_db()
    if not conn:
        return None
    cursor = conn.cursor()
    try:
        query = """
            INSERT INTO resume_score (job_id, user_id, match_score)
            VALUES (%s, %s, %s)
        """
        values = (job_id, user_id, match_score)
        cursor.execute(query, values)
        conn.commit()
        return True
    except mysql.connector.Error as e:
        print(f"Error saving resume score: {e}")
        return None
    finally:
        cursor.close()
        conn.close()