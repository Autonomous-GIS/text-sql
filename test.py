import psycopg2
import json
import os

# --- CONFIGURATION ---
DB_NAME = "UMS"
DB_USER = "bigdata_local"
DB_PASS = "1201"
DB_HOST = "127.0.0.1"
DB_PORT = "5432"

# --- CONFIG FOR DYNAMIC TABLES ---
SERVICE_FEEDBACK_CONFIG = {
    "GIS Copilot": {
        "table_name": "feedback_giscopilot",
        "fields": ["question", "feedback", "feedback_message",
                   "error_msg", "error_traceback", "generated_code",
                   "data_overview", "task_breakdown", "selected_tools", "workflow"]
    },
    "Spatial Data Retrieval Agent": {
        "table_name": "feedback_spatial_retrieval_agent",
        "fields": ["question", "feedback", "feedback_message",
                   "error_msg", "error_traceback", "generated_code","selected_datasource"]
    },
    "Spatial Query Agent": {
        "table_name": "feedback_spatial_query_agent",
        "fields": ["question", "comment", "error"]
    }
}

def get_pg_connection():
    return psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
    )

def init_postgres_schema(cursor):
    print("Creating Tables...")

    # 1. USERS Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            email VARCHAR(255) UNIQUE NOT NULL,
            password VARCHAR(255) NOT NULL,
            first_name VARCHAR(255) NOT NULL,
            last_name VARCHAR(255) NOT NULL,
            country VARCHAR(255),
            city VARCHAR(255),
            phone_number VARCHAR(255),
            occupation VARCHAR(255),
            role VARCHAR(255) DEFAULT 'user',
            is_admin INTEGER DEFAULT 0,
            is_verified INTEGER DEFAULT 0,
            verification_token VARCHAR(255),
            password_reset_token VARCHAR(255),
            password_reset_expires TIMESTAMP,
            remaining_credit REAL DEFAULT 50.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            daily_requests INTEGER DEFAULT 100,
            last_reset_date TIMESTAMP
        );
    """)

    # 2. API_KEYS Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id),
            api_key VARCHAR(255) UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # 3. SUPPORT_MESSAGES Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS support_messages (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id),
            message TEXT NOT NULL,
            reply TEXT,
            status VARCHAR(255) DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            replied_at TIMESTAMP
        );
    """)

    # 4. QUESTIONS Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS questions (
            id SERIAL PRIMARY KEY,
            question_id VARCHAR(255) UNIQUE NOT NULL,
            user_id INTEGER NOT NULL REFERENCES users(id),
            service_name VARCHAR(255) NOT NULL,
            llm_call_count INTEGER DEFAULT 0,
            llm_call_limit INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_questions_qid ON questions(question_id);")

    # 5. LLM_USAGE_LOG Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS llm_usage_log (
            id SERIAL PRIMARY KEY,
            question_id VARCHAR(255) NOT NULL,
            user_id INTEGER NOT NULL REFERENCES users(id),
            service_name VARCHAR(255),
            llm_model VARCHAR(255),
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            cost REAL DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_usage_qid ON llm_usage_log(question_id);")

    # 6. FEEDBACK TABLES (Dynamic Creation)
    for service, config in SERVICE_FEEDBACK_CONFIG.items():
        table_name = config["table_name"]
        
        # --- FIX 1: Drop the table first to remove the bad schema ---
        cursor.execute(f'DROP TABLE IF EXISTS "{table_name}" CASCADE')
        
        # --- FIX 2: Use Quoted "requestID" ---
        field_defs = ", ".join([f'"{field}" TEXT' for field in config["fields"]])
        
        query = f"""
            CREATE TABLE "{table_name}" (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id),
                date_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_email VARCHAR(255),
                user_name VARCHAR(255),
                "requestID" VARCHAR(255), 
                question_id VARCHAR(255) UNIQUE,
                {field_defs}
            );
        """
        cursor.execute(query)

def load_data(cursor):
    print("Loading Data from JSON...")
    
    if not os.path.exists('migration_data.json'):
        print("❌ Error: migration_data.json not found!")
        return

    with open('migration_data.json', 'r') as f:
        data_dump = json.load(f)

    # Order matters because of Foreign Keys!
    table_order = ['users', 'api_keys', 'questions', 'support_messages', 'llm_usage_log']
    
    # Add any extra tables found in the dump (like feedback tables)
    for t in data_dump.keys():
        if t not in table_order:
            table_order.append(t)

    for table in table_order:
        if table not in data_dump:
            continue
            
        print(f"Importing {table}...")
        columns = data_dump[table]['columns']
        rows = data_dump[table]['data']
        
        if not rows:
            continue

        # Quote columns to handle reserved words or mixed case ("requestID")
        col_str = ", ".join([f'"{c}"' for c in columns])
        placeholders = ", ".join(["%s"] * len(columns))
        sql = f'INSERT INTO "{table}" ({col_str}) VALUES ({placeholders})'

        # Execute
        try:
            cursor.executemany(sql, rows)
            
            # Reset the ID sequence so new inserts don't crash
            if 'id' in columns:
                try:
                    cursor.execute(f"SELECT setval(pg_get_serial_sequence('\"{table}\"', 'id'), max(id)) FROM \"{table}\"")
                except:
                    pass 
        except Exception as e:
            print(f"⚠️ Error importing {table}: {e}")
            raise e # Raise to trigger rollback

if __name__ == "__main__":
    conn = None
    try:
        conn = get_pg_connection()
        conn.autocommit = False 
        cursor = conn.cursor()
        
        init_postgres_schema(cursor)
        load_data(cursor)
        
        conn.commit()
        print("✅ Migration Complete! Data successfully saved.")
        
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"\n❌ Critical Error: Transaction Aborted. No data was saved.")
        print(f"Details: {e}")
    finally:
        if conn:
            conn.close()