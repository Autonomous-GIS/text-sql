#### SERVER FINAL
########

import os
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

from sqlalchemy.exc import SQLAlchemyError
from pathlib import Path
import json
import pandas as pd
import time
import pickle
from flask import Flask, request, jsonify, Response, stream_with_context, render_template
from flask_cors import CORS
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError
import re
from difflib import SequenceMatcher
import os
import requests
import html
import xml.etree.ElementTree as ET
from typing import Dict
# import geopandas as gpd  # This import was in your notebook but not used. Uncomment if needed.
from io import StringIO

# --- Load Pickled Data ---
# Ensure these files are in the same directory as Fserver.py
try:
    with open(f"prompts", "rb") as promptPath:
        prompts = pickle.load(promptPath)

    with open(f"KNOWLEDGE_GRAPH", "rb") as KNOWLEDGE_GRAPH_FILE:
        KNOWLEDGE_GRAPH = pickle.load(KNOWLEDGE_GRAPH_FILE)
except FileNotFoundError as e:
    print(f"CRITICAL ERROR: Could not load data file. Make sure 'prompts' and 'KNOWLEDGE_GRAPH' are present.")
    print(f"Error: {e}")
    # Define dummy data to allow server to start, but it will fail.
    prompts = {"EEA": {}, "QLA": {}, "SQL": {}, "RA": {}, "has goem": {}}
    KNOWLEDGE_GRAPH = {}


# ===============================================
# === HELPER & AGENT FUNCTIONS (Copied from Notebook) ===
# ===============================================

def check_geometry(sql: str) -> Dict[str, str]:
    # This function was defined inside /execute-query.
    # I've moved it here to be a general helper, but it's not used by the generator.
    # The version inside /execute-query is the one that will be called.
    pass 

def to_markdown(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    separator = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = []
    for _, row in df.iterrows():
        cells = ["" if pd.isna(x) else str(x) for x in row]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator] + rows)

def run_query(sql: str, database: str) -> dict:
    """
    Executes a standard SQL query and returns tabular results.
    Uses .mappings().all() and explicit dict conversion for direct JSON serialization.
    """
    db_url = os.getenv('DATABASE_URL', f'postgresql+psycopg2://abk6234:1201@localhost:5432/{database}')
    
    try:
        engine = create_engine(db_url)
        with engine.connect() as connection:
            rows = connection.execute(text(sql)).mappings().all()
            result = [dict(row) for row in rows]
            
            if not result:
                return {"status": "Success", "Output": []}
            
            return {"status": "Success", "Output": result} 

    except SQLAlchemyError as e:
        print(f"SQLAlchemy Error in run_query: {e.__cause__ or e}")
        return {"status": "Error", "Output": str(e.__cause__ or e)}
    except Exception as e:
        print(f"General Error in run_query: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "Error", "Output": str(e)}

def validate_query(query: str, DB):
    # WARNING: Using user's credentials here. Be sure this is secure.
    engine = create_engine(f'postgresql+psycopg2://abk6234:1201@localhost:5432/{DB}')
    try:
        with engine.connect() as connection:
            connection.execute(text("SET statement_timeout = 3000"))
            connection.execute(text(f"EXPLAIN {query}"))
            return {"status": "Valid", "Output": "Query is syntactically correct and executable"}
    except SQLAlchemyError as e:
        return {"status": "Error", "Output": str(e.__cause__ or e)}

def EEA_Func(Question, max_retries=3):
        def EEA_task(Question):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompts["EEA"]["system"]},
                    {"role": "user", "content": prompts["EEA"]["user"].format(Question=Question)}
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content) # Use json.loads instead of eval
        def EEA_check(EEA_result):
            for key in ['intent', 'entities', 'operations']:
                if key not in EEA_result:
                    return False
            return True

        for attempt in range(1, max_retries + 1):
            try:
                EEA_result = EEA_task(Question)
                if EEA_check(EEA_result):
                    return EEA_result
                else:
                    print(f"Attempt {attempt}: EEA check failed, retrying...")
            except Exception as e:
                print(f"Attempt {attempt}: Exception occurred: {e}, retrying...")
        return "EEA failed" # Should be a JSON object or dict for consistency
        
def MRA_Func(question, keywords, Database_description, tables_json_abrvs, columns_json_abrvs, column_values_json, AI_Descriptions):

    def text_score(text, keywords):
        score = 0
        text_low = text.lower()
        for k in keywords:
            k_low = k.lower()
            if k_low in text_low:
                score += 3
            elif SequenceMatcher(None, text_low, k_low).ratio() > 0.7:
                score += 1
        return score

    def prefilter_metadata(keywords, db_desc, tables_abbr, cols_abbr, col_values, ai_desc, max_items=150):
        """Identify candidate tables/columns based on keyword matches before passing to the LLM."""
        hits = []

        if any(k.lower() in db_desc.lower() for k in keywords):
            hits.append({"type": "db", "entity": "database", "desc": db_desc, "score": 3})

        for t in tables_abbr:
            if isinstance(t, str):
                try:
                    t = json.loads(t)
                except Exception:
                    t = {"name": t, "desc": ""}
            name = t.get("name", "")
            desc = t.get("desc", "")
            s = text_score(name + " " + desc, keywords)
            if s > 0:
                hits.append({"type": "table", "entity": name, "desc": desc, "score": s})

        for c in cols_abbr:
            if isinstance(c, str):
                try:
                    c = json.loads(c)
                except Exception:
                    c = {"name": c, "desc": "", "table": None}
            name = c.get("name", "")
            desc = c.get("desc", "")
            s = text_score(name + " " + desc, keywords)
            if s > 0:
                hits.append({
                    "type": "column",
                    "table": c.get("table"),
                    "entity": name,
                    "desc": desc,
                    "score": s
                })

        for col, samples in col_values.items():
            for k in keywords:
                if any(k.lower() in str(v).lower() for v in samples):
                    hits.append({
                        "type": "value",
                        "entity": col,
                        "samples": samples,
                        "score": 3
                    })
                    break

        hits_sorted = sorted(hits, key=lambda x: x["score"], reverse=True)[:max_items]
        return hits_sorted

    def run_metadata_reasoning(client, question, keywords, db_desc, tables_abbr, cols_abbr, col_values, ai_desc):
        """Send prefiltered metadata and user question to GPT for reasoning."""
        hits = prefilter_metadata(keywords, db_desc, tables_abbr, cols_abbr, col_values, ai_desc)

        metadata_summary = {
            "Database_description": db_desc,
            "Candidate_metadata": hits,
            "AI_Descriptions": ai_desc[:30] if len(ai_desc) > 30 else ai_desc
        }

        system_prompt = """
        You are a **metadata reasoning agent**.
        Your task is to analyze a user question and its keywords, along with structured database metadata,
        and output a **JSON object** representing which tables and columns are potentially useful
        for answering the question.

        **Output Rules:**
        - Output only valid JSON (no text outside JSON).
        - Use this structure:
        {
            "table_name": {
            "column_name": {
                "confidence": "float 0-1",
                "rationale": "why it's relevant"
            },
            ...
            },
            ...
        }
        - Confidence should reflect your certainty that the column is useful.
        - Provide a short, explicit rationale for each column.
        - Maximize recall (include anything potentially relevant).
        """

        user_prompt = f"""
        QUESTION: {question}
        KEYWORDS: {keywords}

        METADATA:
        {json.dumps(metadata_summary, indent=2)}
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "developer", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()}
            ],
            response_format={"type": "json_object"}
        )

        result = response.choices[0].message.content
        return json.loads(result)

    def normalize_output(raw_json):
        """Ensure consistent JSON structure: {table: {column: {...}}}."""
        structured = {}
        for table, columns in raw_json.items():
            if not isinstance(columns, dict): continue # Add safety check
            structured[table] = {}
            for col, info in columns.items():
                if not isinstance(info, dict): continue # Add safety check
                structured[table][col] = {
                    "confidence": float(info.get("confidence", 0.0)),
                    "rationale": info.get("rationale", "").strip()
                }
        return structured

    raw_output = run_metadata_reasoning(
        client=client,
        question=question,
        keywords=keywords,
        db_desc=Database_description,
        tables_abbr=tables_json_abrvs,
        cols_abbr=columns_json_abrvs,
        col_values=column_values_json,
        ai_desc=AI_Descriptions
    )

    final_output = normalize_output(raw_output)
    tab_list = {}
    sample_values = ""
    
    # Check if column_values_json is nested as expected
    if not (column_values_json and isinstance(next(iter(column_values_json.values()), {}), dict)):
        print("Warning: column_values_json is not in the expected format {table: {col: [values]}}")
        # Create a dummy structure to prevent crashes
        column_values_json = {table: {col: [] for col in cols} for table, cols in final_output.items()}
        
    if not (columns_json_abrvs and isinstance(next(iter(columns_json_abrvs.values()), {}), dict)):
        print("Warning: columns_json_abrvs is not in the expected format {table: {col: desc}}")
        # Create a dummy structure to prevent crashes
        columns_json_abrvs = {table: {col: "" for col in cols} for table, cols in final_output.items()}
        
    if not (AI_Descriptions and isinstance(next(iter(AI_Descriptions.values()), {}), dict)):
        print("Warning: AI_Descriptions is not in the expected format {table: {col: desc}}")
        # Create a dummy structure to prevent crashes
        AI_Descriptions = {table: {col: "" for col in cols} for table, cols in final_output.items()}


    for table, cols in final_output.items():
        sample_values += f"- Table **{table}**\n"
        tab_list[table] = {}
        for col in cols:
            # Add safety checks
            tab_list[table][col] = column_values_json.get(table, {}).get(col, [])
            if len(tab_list[table][col]) < 10:
                tab_list[table][col] = tab_list[table][col] + [None] * (10 - len(tab_list[table][col]))
            
            col_desc = columns_json_abrvs.get(table, {}).get(col, "N/A")
            sample_values +=f"**{col}** : {col_desc}\n"
        
        if tab_list[table]: # Only build dataframe if there are columns
            sample_values +="\n"+to_markdown(pd.DataFrame(tab_list[table])) +"\\n\\n"
    
    tab_list = {}
    sample_values_rich = ""
    for table, cols in final_output.items():
        sample_values_rich += f"- Table **{table}**\n"
        tab_list[table] = {}
        for col in cols:
            tab_list[table][col] = column_values_json.get(table, {}).get(col, [])
            if len(tab_list[table][col]) < 10:
                tab_list[table][col] = tab_list[table][col] + [None] * (10 - len(tab_list[table][col]))
            
            ai_desc = AI_Descriptions.get(table, {}).get(col, "N/A")
            sample_values_rich +=f"**{col}** : {ai_desc}\n"
        sample_values_rich += "--------------------------------\\n\\n"
    
    return sample_values, sample_values_rich


def QLA_Func(Question, sample_values):
    # This function in your notebook uses a deprecated 'client.responses.create'
    # Re-writing it using the standard 'client.chat.completions.create'
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a query logic planner. Your role is to generate a high-level, step-by-step plan for answering a user's question, given a set of relevant tables and columns. Do not write SQL."},
            {"role": "user", "content": prompts["QLA"]["input"].format(Question=Question, sample_values=sample_values)}
        ]
    )
    return response.choices[0].message.content

def SQLA_Func(Question, QL, potential_table_columns, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "developer", "content": prompts["SQL"]["system"]},
                    {"role": "user", "content": prompts["SQL"]["user"].format(Question=Question, QLA=QL, col_desc=potential_table_columns)}
                ],
                response_format={"type": "json_object"}
            )
            SQL = json.loads(response.choices[0].message.content)['query'] # Use json.loads
            return SQL
        except Exception as e:
            print(f"Attempt {attempt}: Exception occurred: {e}, retrying...")

def double_check(Question, SQL, MRA2):
    for i in range(3):
        try:
            response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "user", "content": f"""For this question: \n'{Question}'\n -The SQL query is generated: \n {SQL}\n -If the postgresql query requires revison based on the schema information do that. \n  - Here is the schema description of the database: \n{MRA2} \n Your response must be a SQL query in a json file in this format {{"query":"SQL"}}"""}
                        ],
                        response_format={"type": "json_object"}
                    )
            result = response.choices[0].message.content
            return json.loads(result)["query"] # Use json.loads
        except:
            pass
    return "double ckeck faild"

def validate_spatial_query(SQL, Question, spatial_guide, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompts["RA"]["validate"]["system"]},
                    {"role": "user", "content": prompts["RA"]["validate"]["user"].format(SQL=SQL, Question=Question, spatial_guide=spatial_guide)}
                ],
                response_format={"type": "json_object"}
            )
            SQL = json.loads(response.choices[0].message.content)['query'] # Use json.loads
            return SQL
        except Exception as e:
            print(f"Attempt {attempt}: Exception occurred: {e}, retrying...")
    return "Validation failed"

def format_check(question, SQL, columns_json_abrvs):
    for i in range(3):
        try:
            all_table_columns = ""
            for table in columns_json_abrvs:
                all_table_columns+= f"Table {table} :\n"
                for col in columns_json_abrvs[table]:
                    all_table_columns+= f"{col} : {columns_json_abrvs[table][col]}\n"
                all_table_columns+= f"---------------------\n"
            response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "user", "content": f"""If the question wants the locations as output and the generated query doesn't return a geometry column as output, check the schema informatio and if the geometry column can be considered add that. \n
                                -The question: \n'{question}'\n - The SQL query: {SQL}\n \n
                                - The schema information:\n {all_table_columns} \n\n
                                If the question wants a List as output remove the geometry column in outputs (if exists in the output of query)\n\n
                                If the question wants Locations as output consider the geometry column in outputs (if the geometry column can be considered in the output considering the schema)\n\n
                                - Do not make any change to other parts of the query, Just consider the output. \n\n
                                Your response must be a SQL query in a json file in this format {{"query":"SQL"}}"""}
                            ],
                            response_format={"type": "json_object"}
                        )
            result = response.choices[0].message.content
            return json.loads(result)["query"] # Use json.loads
        except:
             pass
    print ("format check faild")
    return SQL

def double_check_conditions(Question, SQL, DB):
    for i in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompts["RA"]["Double Check"]["system"]},
                    {"role": "user", "content": prompts["RA"]["Double Check"]["user"].format(SQL=SQL)}
                ],
                response_format={"type": "json_object"}
            )
            result_json = json.loads(response.choices[0].message.content) # Use json.loads

            for filter in result_json["filters"]:
                if filter["operator"].strip()=="=":
                    test_qry = f"""SELECT
                        {filter["column"]},
                        similarity({filter["column"]}, '{filter["value"].replace("'", "''")}') AS similarity_score
                    FROM
                        {filter["table"]}
                    ORDER BY
                        similarity_score DESC
                    LIMIT 10;""" # Fixed SQL injection risk

                    rst = run_query(test_qry, DB)
                    if rst["status"]=='Success':
                        if (pd.DataFrame(rst["Output"])[filter["column"]] == (filter["value"])).any() == False:
                            response = client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[
                                    {"role": "user", "content": f"For this question: \n {Question} \n I have this SQL query: \n {SQL} \n But in the column {filter['column']} of table {filter['table']} there is not exactly value for '{filter['value']}'. Similar values in the column are {', '.join(pd.DataFrame(rst['Output'])[filter['column']].tolist())}. \n If the approprate value for the condition is obvious use that otherwise change the condition to ILIKE and use %%. Also, if the approprate value for the condition is NOT obvious and the conditon is like Teharan City keep the part that might return the most relevant result for example:  ILIKE '%tehran%' \n Your response MUST JUST be a SQL query in json format like:  Required JSON Format: {{'query': 'The complete PostgreSQL SQL query string'}}\""}
                                ],
                                response_format={"type": "json_object"}
                            )
                            result = response.choices[0].message.content
                            SQL = json.loads(result)['query'] # Use json.loads
            return SQL
        except Exception as e:
            print(f"double_check_conditions failed on attempt {i+1}: {e}")
    return SQL


def RA_Func(SQL, question, DB, MRA2):
    print("RA:")
    for i in range(4):
        validation_result = validate_query(SQL, DB) # Changed from run_query to validate_query
        if validation_result["status"] == 'Valid':
            print(f"The SQL query was valid after {i} run of RA")
            return {"status":"valid", "sql":SQL}
        else:
            print("run:", i)
            print("query:", SQL)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": f"""For this question: '{question}'\n This postgresql quary is generated : {SQL}\n BUT the query faced this error: {validation_result["Output"]} please fix the error and if required suggest an new working query.\n - Here is the schema description of the database: \n{MRA2} \n Your response must be a SQL query in a json file in this format {{"query":"SQL"}}"""}
                ],
                response_format={"type": "json_object"}
            )
            result = response.choices[0].message.content
            SQL = json.loads(result)["query"] # Use json.loads
            print("error:", validation_result["Output"])
            print("refined:", SQL)
    
    return {"status":"failed", "sql":""}

# ===============================================
# === SQL QUERY GENERATOR (The core logic) ===
# ===============================================

def generate_sql_query(Question, DB):
    # WARNING: Using user's credentials here. Be sure this is secure.
    engine = create_engine(f'postgresql+psycopg2://abk6234:1201@localhost:5432/{DB}')
    
    metadata_file = f"metadata/{DB}.meta"
    if not os.path.exists(metadata_file):
        print(f"Error: Metadata file not found: {metadata_file}")
        yield {"node": "ERROR", "result": f"Metadata file not found for database: {DB}"}
        return

    with open(metadata_file, "rb") as metaPath:
        meta = pickle.load(metaPath)
    
    columns_json_abrvs = meta["columns_json_abrvs"]
    tables_json_abrvs = meta["tables_json_abrvs"]
    column_values_json = meta["column_values_json"]
    AI_Descriptions = meta["AI_Descriptions"]
    Database_description = meta["Database_description"]
    
    try:
        EEA = EEA_Func(Question)
        yield {"node": "EEA", "result": EEA}
        
        entities = EEA["entities"]
        MRA1, MRA2 = MRA_Func(Question, entities, Database_description, tables_json_abrvs, columns_json_abrvs, column_values_json, AI_Descriptions)
        yield {"node": "MRA", "result": MRA1}
        
        QL = QLA_Func(Question, MRA1)
        yield {"node": "QLA", "result": QL}
        
        SQL = SQLA_Func(Question, QL, MRA1)
        yield {"node": "SQLA", "result": SQL}
        
        SQL = double_check(Question, SQL, MRA2)
        SQL = validate_spatial_query(SQL, Question, prompts["Spatial guide"])
        SQL = double_check_conditions(Question, SQL, DB)
        SQL = format_check(Question, SQL, columns_json_abrvs)
        
        RA = RA_Func(SQL,Question, DB, MRA2)
        yield {"node": "RA", "result": RA}

        final_sql_result = "faild"
        if RA["status"]=='valid':
            final_sql_result = RA.get("sql", SQL) # Use the refined SQL from RA
        
        # Yield for the "SQL" node in the workflow
        yield {"node": "SQL", "result": final_sql_result}
        
        # NEW: Yield a final, structured message for the main page
        # We now call Colorize_SQL here and include it in the payload
        formatted_sql = Colorize_SQL(final_sql_result) if final_sql_result != "faild" else None
        yield {"node": "FINAL", "sql_query": final_sql_result, "formatted_sql": formatted_sql, "status": RA["status"]}
        
    except Exception as e:
        print(f"Error during query generation: {e}")
        import traceback
        traceback.print_exc()
        yield {"node": "ERROR", "result": str(e)}
    
    return None

# ===============================================
# === FLASK APP AND ROUTES ===
# ===============================================

app = Flask(__name__)
CORS(app) # Allow all origins for simplicity in development

conversations = {}
LongMemory = []
UserFeedbacks = {}

def is_confirmation(message: str) -> bool:
    """Check if user message is a confirmation."""
    confirmations = {"yes", "y", "ok", "okay", "go ahead", "please do", "confirm", 
                    "sure", "absolutely", "of course", "yeah", "yep", "alright",
                    "execute", "run it", "show me", "let's see", "proceed"}
    return message.lower().strip() in confirmations

def Colorize_SQL(sql):
    """
    Colorizes and formats SQL query using external API.
    Returns HTML-formatted SQL with syntax highlighting.
    """
    try:
        xml_request = f"""<sqlpp_request>
<clientid>dpriver-9094-8133-2031</clientid>
<dbvendor>mssql</dbvendor>
<outputfmt>html2</outputfmt>
<inputsql>{html.escape(sql)}</inputsql>
<formatoptions>
<keywordcs>Uppercase</keywordcs>
<tablenamecs>Lowercase</tablenamecs>
<columnnamecs>Lowercase</columnnamecs>
<functioncs>InitCap</functioncs>
<datatypecs>Uppercase</datatypecs>
<variablecs>Unchanged</variablecs>
<aliascs>Unchanged</aliascs>
<quotedidentifiercs>Unchanged</quotedidentifiercs>
<identifiercs>Lowercase</identifiercs>
<lnbrwithcomma>after</lnbrwithcomma>
<liststyle>stack</liststyle>
<salign>sleft</salign>
<quotechar>"</quotechar>
<maxlenincm>80</maxlenincm>
</formatoptions>
</sqlpp_request>"""
        
        response = requests.post("https://www.dpriver.com/cgi-bin/ppserver_service", data=xml_request)
        xml_string = response.text
        
        root = ET.fromstring(xml_string)
        encoded_sql = root.find("formattedsql").text
        decoded_sql = html.unescape(encoded_sql)
        
        return decoded_sql
    except Exception as e:
        print(f"Error colorizing SQL: {e}")
        return f"<pre>{html.escape(sql)}</pre>"

def process_message(user_id: str, user_message: str, knowledge_graph: dict) -> dict:
    """
    Processes a user's message using an agentic AI model with state management.
    """
    if user_id not in conversations:
        conversations[user_id] = {
            "history": [],
            "current_state": "NO_PLAN",
            "current_plan": None,
            "confirmed_parameters": {}
        }
    
    conversation_state = conversations[user_id]
    
    conversation_state["history"].append({"role": "user", "content": user_message})
    
    kg_string = json.dumps(knowledge_graph, indent=2)

    # (System prompt from your notebook is copied here)
# ... (inside process_message)
    
    system_prompt = f"""
You are a friendly and precise AI assistant. Your primary purpose is to help a user build a complete and unambiguous query plan by asking clarifying questions.

Your ONLY source of truth for available data is this Knowledge Graph:
KNOWLEDGE GRAPH:
{kg_string}

--- YOUR MODES OF OPERATION ---
... (Keep this section as-is) ...

--- STATE AWARENESS RULES ---

CRITICAL: You MUST be aware of the conversation state by reading the message history.

1.  **If the state is NO_PLAN (new chat):**
    * If user asks for data, start building a plan. Status: `clarification_needed`.
    * If user is conversational, answer. Status: `answered_directly`.

2.  **If the state is CLARIFICATION_NEEDED (you just asked a question):**
    * If the user's new message answers your question and the plan is *now complete*:
        * Fill all plan details.
        * Status: `plan_ready`.
    * If the user's message answers, but *more* info is needed:
        * Ask the *next* question.
        * Status: `clarification_needed`.

3.  **If the state is PLAN_READY (you just proposed a plan in the last turn):**
    * **This is the most critical state.**
    * **A. If the user *confirms*** (e.g., "yes", "ok", "go ahead", "run it"):
        * Your status MUST be `execute_plan`.
        * Your `response_to_user` should be: "I am now executing your request: [Restate the clear_question here]."
        * You MUST return the *exact same* `query_plan` from the previous turn.
    * **B. If the user asks a *conversational question*** about the plan (e.g., "describe that", "what does that mean?", "why that table?"):
        * Your status MUST be `answered_directly`.
        * Your `response_to_user` must be the *answer* to their question.
        * DO NOT return a `query_plan`.
    * **C. If the user wants to *modify* the plan** (e.g., "actually, for New York instead"):
        * Update the plan. If it's now complete, return `plan_ready`. If it needs more info, return `clarification_needed`.

--- REQUIREMENTS FOR A COMPLETE QUERY PLAN (`plan_ready`) ---
... (Keep this section as-is) ...

--- YOUR FINAL JSON OUTPUT FORMAT ---

1.  If the plan is ready (Rule 2):
    {{
        "status": "plan_ready",
        "response_to_user": "I am now processing your request to: [Restate the clear_question here].",
        "query_plan": {{ ... }}
    }}

2.  If you need to ask ONE clarifying question (Rule 2):
    {{
        "status": "clarification_needed",
        "response_to_user": "Your friendly and specific question for the user",
        "query_plan": null
    }}

3.  If you are in Conversational Mode (Rule 1 or 3.B):
    {{
        "status": "answered_directly",
        "response_to_user": "Your friendly, conversational response.",
        "query_plan": null
    }}

4.  If the user confirms execution of an existing plan (Rule 3.A):
    {{
        "status": "execute_plan",
        "response_to_user": "I am now executing your request: [Restate the clear_question here].",
        "query_plan": {{
            "clear_question": "The final, unambiguous, self-contained question",
            "target_database": "The single database name from the Knowledge Graph",
            "representation_type": "table" | "map"
        }}
    }}
"""
    
    messages = [{"role": "system", "content": system_prompt}]
    history = conversation_state["history"][-10:]
    messages.extend(history)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        status = result.get("status")

        if status == "plan_ready":
            # The plan is PROPOSED. Stay in this state.
            conversation_state["current_state"] = "PLAN_READY"
            conversation_state["current_plan"] = result.get("query_plan")
        
        elif status == "execute_plan":
            # The plan is CONFIRMED. Reset state for the next request.
            conversation_state["current_state"] = "NO_PLAN"
            conversation_state["current_plan"] = None
        
        elif status == "clarification_needed":
            conversation_state["current_state"] = "CLARIFICATION_NEEDED"
            
        elif status == "answered_directly":
            # This is the fix. If the user just asked a question,
            # we answer it, but we DON'T change the underlying state.
            # If we were in PLAN_READY, we remain in PLAN_READY.
            pass 

        if result.get("response_to_user"):
            conversation_state["history"].append({"role": "assistant", "content": result["response_to_user"]})
        
        return result
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return {
            "status": "error",
            "response_to_user": "I'm sorry, I encountered an error. Please try your request again.",
            "query_plan": None
        }

@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint to handle chat messages from the frontend."""
    try:
        data = request.json
        user_id = data.get('user_id')
        message = data.get('message')
        
        if not user_id or not message:
            return jsonify({"error": "Missing user_id or message"}), 400
        
        result = process_message(user_id, message, KNOWLEDGE_GRAPH)
        
        return jsonify({
            "response": result.get("response_to_user", ""),
            "status": result.get("status"),
            "query_plan": result.get("query_plan")
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/knowledge-graph', methods=['GET'])
def get_knowledge_graph():
    """Endpoint to serve the knowledge graph to the frontend."""
    return jsonify({"knowledge_graph": KNOWLEDGE_GRAPH})

@app.route('/history/<user_id>', methods=['GET'])
def get_history(user_id):
    """Endpoint to retrieve conversation history for a user."""
    if user_id in conversations:
        history = conversations[user_id].get("history", [])
    else:
        history = []
    return jsonify({"history": history})

@app.route('/clear/<user_id>', methods=['POST'])
def clear_history(user_id):
    """Endpoint to clear conversation history for a user."""
    if user_id in conversations:
        del conversations[user_id]
    return jsonify({"message": "Conversation cleared"})

# ===============================================
# === NEW STREAMING SQL GENERATION ENDPOINT ===
# ===============================================
@app.route('/generate-sql-stream', methods=['POST'])
def generate_sql_stream():
    """
    This is the new streaming endpoint.
    It returns a stream of Server-Sent Events (SSE).
    """
    try:
        data = request.json
        user_id = data.get('user_id')
        question = data.get('question')
        database = data.get('database')
        
        if not question or not database:
            return jsonify({"error": "Missing question or database"}), 400

        query_generator = generate_sql_query(question, database)
        
        def stream():
            final_sql_from_stream = None
            try:
                # Add the initial "Question" node data
                question_payload = {"node": "question", "result": question}
                yield f"data: {json.dumps(question_payload)}\n\n"

                for progress_update in query_generator:
                    # Send this update to the client, formatted as an SSE message
                    yield f"data: {json.dumps(progress_update)}\n\n"
                    
                    if progress_update.get("node") == "FINAL":
                        final_sql_from_stream = progress_update.get("sql_query")

                # After the stream is done, save to memory
                if final_sql_from_stream:
                    print(f"Saving to LongMemory for user {user_id}")
                    memory_entry = {
                        "user_id": user_id,
                        "question": question,
                        "Query": final_sql_from_stream,
                        "feedback": None,
                        "comment": ""
                    }
                    LongMemory.append(memory_entry)
                    
                    if user_id in conversations:
                        conversations[user_id]["current_state"] = "NO_PLAN"
                        conversations[user_id]["current_plan"] = None
                        
            except Exception as e:
                print(f"Error during stream generation: {e}")
                error_data = {"node": "ERROR", "result": str(e)}
                yield f"data: {json.dumps(error_data)}\n\n"
            
            print("Stream complete.")
        
        # Return the streaming response
        return Response(stream_with_context(stream()), mimetype='text/event-stream')

    except Exception as e:
        print(f"Error in generate-sql-stream setup: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500

# ===============================================
# === ORIGINAL (NON-STREAMING) ENDPOINT ===
# ===============================================
@app.route('/generate-sql', methods=['POST'])
def generate_sql():
    """Endpoint to generate SQL query when plan is ready."""
    try:
        data = request.json
        user_id = data.get('user_id')
        question = data.get('question')
        database = data.get('database')
        
        if not question or not database:
            return jsonify({"error": "Missing question or database"}), 400
        
        query_generator = generate_sql_query(question, database)
        
        final_sql = None
        progress_updates = []
        
        for progress_update in query_generator:
            # Changed to append the dict, not str()
            progress_updates.append(progress_update) 
            if isinstance(progress_update, dict) and progress_update.get("node") == "FINAL":
                final_sql = progress_update.get("sql_query")
        
        if not final_sql:
            final_sql = "faild" # Use the value from the stream
        
        formatted_sql = Colorize_SQL(final_sql)
        
        # Memory logic is now handled in the streaming endpoint, 
        # but we'll keep it here for this legacy route.
        memory_entry = {
            "user_id": user_id,
            "question": question,
            "Query": final_sql,
            "feedback": None,
            "comment": ""
        }
        LongMemory.append(memory_entry)
        
        if user_id in conversations:
            conversations[user_id]["current_state"] = "NO_PLAN"
            conversations[user_id]["current_plan"] = None
        
        response_data = {
            "sql_query": final_sql,
            "question": question,
            "database": database,
            "progress_updates": progress_updates # Now a list of dicts
        }
        
        if formatted_sql:
            response_data["formatted_sql"] = formatted_sql
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in generate-sql endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500

@app.route('/feedback', methods=['POST'])
def handle_feedback():
    """Endpoint to handle user feedback on generated SQL queries."""
    try:
        data = request.json
        user_id = data.get('user_id')
        question = data.get('question')
        query = data.get('query')
        feedback = data.get('feedback')
        comment = data.get('comment', '')
        
        for entry in reversed(LongMemory):
            if entry.get('user_id') == user_id and entry.get('Query') == query:
                if feedback in ['like', 'dislike']:
                    entry['feedback'] = feedback
                if comment:
                    entry['comment'] = comment
                break
        
        if user_id not in UserFeedbacks:
            UserFeedbacks[user_id] = []
        
        feedback_text = f"Query: {query[:100]}... | Feedback: {feedback}"
        if comment:
            feedback_text += f" | Comment: {comment}"
        
        UserFeedbacks[user_id].append(feedback_text)
        
        print(f"Feedback received: {feedback} for query: {query[:50]}...")
        print(f"Current LongMemory entries: {len(LongMemory)}")
        
        return jsonify({"message": "Feedback recorded successfully"})
        
    except Exception as e:
        print(f"Error in feedback endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/long-memory', methods=['GET'])
def get_long_memory():
    """Endpoint to retrieve all stored queries and feedback."""
    return jsonify({"memory": LongMemory, "total_entries": len(LongMemory)})

@app.route('/execute-query', methods=['OPTIONS', 'POST'])
def execute_sql():
    """Endpoint to execute SQL queries and return results."""
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    
    data = request.json
    sql = data.get('sql')
    database = data.get('database', 'default')
    outType = data.get('outType')
    if not sql:
        return jsonify({"error": "Missing SQL query"}), 400
    
    disallowed_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'TRUNCATE', 'GRANT', 'ALTER', 'CREATE']
    if any(keyword in sql.upper() for keyword in disallowed_keywords):
        return jsonify({"error": "Only SELECT queries are allowed."}), 403

    def check_geometry(sql: str) -> Dict[str, str]:
        """Uses LLM to check if query returns geometry and ensures proper formatting."""
        prompt = f"""
                    You are a SQL assistant. Analyze the SQL query below.

                    Task:
                    1. Identify if the query will return any geometry column (Please consider JUST geomety data type is intended for example if the query retun latitude or/and longitude they are not geometry column JUST geom). 
                    2. If yes, ensure the geometry column(s) have the alias exactly geom
                    (replace existing aliases or add one if missing).
                    3. Also, if has geometry column, ensure that output gemetry is NOT transformed format to other formats like using ST_AsText or ST_AsGeoJSON etc. Output of geometry column must be like geom.
                    4. If no geometry column, keep query unchanged.
                    5. Return ONLY a valid JSON object in this format:

                    {{
                    "has_geometry": "yes" or "no",
                    "query": "<the SQL query>"
                    }}

                    SQL query:
                    {sql}
                    """

        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            text_out = resp.choices[0].message.content.strip()
            return json.loads(text_out)
        except Exception as e:
            print(f"Error in check_geometry: {e}")
            return {"has_geometry": "no", "query": sql}
    
    if outType.lower() == "table":
        result = run_query(re.sub(r'\s+', ' ', sql.strip()) , database)
        if result["status"] == "Error":
            return jsonify({"error": result["Output"]}), 500
        output_list = [{k: v for k, v in dict(t).items() if k != "geom"} for t in {tuple(row.items()) for row in result["Output"]}]
        return jsonify({"table": output_list, "type": "table"})

    try:
        query_info = check_geometry(sql)
        
        if query_info['has_geometry'].lower() == 'yes':
            geojson_query = f"""WITH Results AS (
    {query_info["query"].replace(";", "")}
)
SELECT jsonb_build_object(
    'type',     'FeatureCollection',
    'features', jsonb_agg(
        jsonb_build_object(
            'type',       'Feature',
            'geometry',   ST_AsGeoJSON(geom)::jsonb,
            'properties', to_jsonb(Results) - 'geom'
        )
    )
) AS geojson
FROM Results;
"""
            
            result = run_query(geojson_query, database)
            
            if result["status"] == "Error":
                return jsonify({"error": result["Output"]}), 500
            
            if result["Output"] and len(result["Output"]) > 0:
                geojson_data = result["Output"][0].get("geojson", {})
                return jsonify({"geojson": geojson_data, "type": "map"})
            else:
                return jsonify({"error": "No results returned"}), 404
        else:
            result = run_query(query_info["query"], database)
            
            if result["status"] == "Error":
                return jsonify({"error": result["Output"]}), 500
            
            output_list = [{k: v for k, v in dict(t).items() if k != "geom"} for t in {tuple(row.items()) for row in result["Output"]}]
            return jsonify({"table": output_list, "type": "table"})
            
    except Exception as e:
        print(f"Error in execute_sql endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500

@app.route('/', methods=['GET'])
def index():
    return render_template("main.html")

@app.route('/workflow.html', methods=['GET'])
def workflowf():
    return render_template("workflow.html")

if __name__ == '__main__':
    # Running on port 6060 as specified in your main.html
    app.run(host='0.0.0.0', port=3000)
