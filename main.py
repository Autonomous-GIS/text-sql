import os
from sqlalchemy import create_engine, inspect, text
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError
from pathlib import Path
import json
import pandas as pd
from typing import Dict
import time
import pickle
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError

with open(f"prompts", "rb") as promptPath:
    prompts = pickle.load(promptPath)

def check_geometry(sql: str) -> Dict[str, str]:
    resp = client.responses.create(
        model="gpt-5",
        input=prompts["has goem"]["input"].format(sql=sql),
        text={"verbosity": "low"},
        reasoning={"effort": "minimal"}
    )

    # Extract model output text
    text_out = resp.output_text.strip()
    try:
        return json.loads(text_out)
    except json.JSONDecodeError:
        return {"has_geometry": "no", "query": sql}

def to_markdown(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    separator = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = []
    for _, row in df.iterrows():
        cells = ["" if pd.isna(x) else str(x) for x in row]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator] + rows)

def run_query(query: str, DB):
    engine = create_engine(f'postgresql+psycopg2://abk6234:1201@localhost:5432/{DB}')
    try:
        with engine.connect() as connection:
            result = connection.execute(text(query)).mappings().all()
            return {"status": "Executed", "Output": result}
    except SQLAlchemyError as e:
        return {"status": "Error", "Output": str(e.__cause__ or e)}

def validate_query(query: str, engine):
    try:
        with engine.connect() as connection:
            connection.execute(text("SET statement_timeout = 3000"))
            connection.execute(text(f"EXPLAIN {query}"))
            return {"status": "Valid", "Output": "Query is syntactically correct and executable"}
    except SQLAlchemyError as e:
        return {"status": "Error", "Output": str(e.__cause__ or e)}

def EEA_Func(Question, max_retries=3):
        def EEA_task(Question):
            prompt = prompt.format(Question=Question)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompts["EEA"]["system"]},
                    {"role": "user", "content": prompts["EEA"]["user"].format(Question=Question)}
                ],
                response_format={"type": "json_object"}
            )
            return eval(response.choices[0].message.content)
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
        return "EEA failed"

def MRA_Func(Question, entities, Database_description, tables_json_abrvs, columns_json_abrvs, column_values_json, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            Potentials = []
            for entity in entities:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": prompts["MRA"]["S-1"]["system"]},
                        {"role": "user", "content": prompts["MRA"]["S-1"]["user"].format(Database_description=Database_description, Question=Question, entity=entity)}
                    ],
                    response_format={"type": "json_object"}
                )
                is_db = eval(response.choices[0].message.content)
                if is_db['entire'].lower() == "yes":
                    is_db_flag = True
                else:
                    is_db_flag = False

                # Step 2: Identify potential tables
                potential_tables = []
                if not is_db_flag:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": prompts["MRA"]["S-2"]["system"]},
                            {
                            "role": "user",
                            "content": prompts["MRA"]["S-2"]["user"].format(Table_description="\n".join([f"- {table}: {tables_json_abrvs[table]}" for table in tables_json_abrvs]), Question=Question, entity=entity)
                        }
                        ],
                        response_format={"type": "json_object"}
                    )
                    potential_tables = eval(response.choices[0].message.content)["tables"]

                if len(potential_tables) > 0:
                    for table in potential_tables:
                        if table not in Potentials:
                            Potentials.append(table)

            # Step 3: Prepare columns info
            pot_tab_des = {pot_tab: columns_json_abrvs[pot_tab] for pot_tab in Potentials}
            table_columns = ""
            for table, cols in pot_tab_des.items():
                table_columns += f"\n*{table}*:\n"
                for col, desc in cols.items():
                    table_columns += f"- '{col}': {desc}\n"

            # Step 4: Ask which columns might be useful
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompts["MRA"]["S-4"]["system"]},
                    {"role": "user", "content": prompts["MRA"]["S-4"]["user"].format(Question=Question, table_columns=table_columns)}
                ],
                response_format={"type": "json_object"}
            )
            potential_table_columns = eval(response.choices[0].message.content)

            # Step 5: Prepare sample values
            sample_values = ""
            tab_list = {}
            for table, cols in potential_table_columns.items():
                sample_values += f"- Table '*{table}*'\n"
                for v in cols:
                    tab_list[v] = column_values_json[table][v]
                sample_values += to_markdown(pd.DataFrame(tab_list)) + "\n\n"

            return {"potential_table_columns":potential_table_columns,"sample_values":sample_values}

        except Exception as e:
            print(f"Attempt {attempt}: Exception occurred: {e}, retrying...")

    # If all retries fail
    return "MRA failed"

def QLA_Func(Question, sample_values):
    response = client.responses.create(
        model="gpt-5",
        input=prompts["QLA"]["input"].format(Question=Question, sample_values=sample_values),
    reasoning={"effort": "low"},
    text={"verbosity": "low"}
    )
    return response.output_text

def SQLA_Func(Question, QL, AI_Descriptions, potential_table_columns, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            col_desc = ""
            for table in potential_table_columns:
                col_desc += f"\n*{table}*:\n"
                for col in potential_table_columns[table]:
                    col_desc += f"{col}: {AI_Descriptions[table][col]}\n"

            # Call the LLM to generate the SQL query
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "developer", "content": prompts["SQL"]["system"]},
                    {"role": "user", "content": prompts["SQL"]["user"].format(Question=Question, QLA=QL, col_desc=col_desc)}
                ],
                response_format={"type": "json_object"}
            )
            SQL = eval(response.choices[0].message.content)['query']
            return SQL
        except Exception as e:
            print(f"Attempt {attempt}: Exception occurred: {e}, retrying...")

    # If all retries fail
    return "SQL failed"

def double_check_filters(SQL, engine, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            # Step 1: Extract textual WHERE filters
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": prompts["RA"]["Double Check"]["system"],
                    },
                    {
                        "role": "user",
                        "content": prompts["RA"]["Double Check"]["user"].format(SQL=SQL)
                    },
                ],
                response_format={"type": "json_object"},
            )

            filters = eval(response.choices[0].message.content.replace("null", "''"))

            # Step 2: Check availability of each textual filter
            if len(filters["filters"]) > 0:
                availability = ""
                for condition in filters["filters"]:
                    table = condition["table"] or ''
                    check_query = f"SELECT {condition['column']} FROM {table} WHERE {condition['column']} ILIKE {condition['value']} LIMIT 1"
                    availability += f"Check if column {condition['column']} has value {condition['value']}:\n"
                    availability += check_query + "\n"
                    availability += "Output: " + str(run_query(check_query, engine=engine)) + "\n__________\n"

                # Step 3: Refine SQL based on availability
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert in PostGIS extension of PostgreSQL database"},
                        {"role": "user", "content": prompts["RA"]["Refine"]["user"].format(SQL=SQL, availability=availability)
                        },
                    ],
                    response_format={"type": "json_object"},
                )

                SQL = eval(response.choices[0].message.content)['query']

            return SQL  # Success

        except Exception as e:
            print(f"Attempt {attempt}: Exception occurred: {e}, retrying...")

    # All retries failed
    return "Double_Check failed"
    
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

            SQL = eval(response.choices[0].message.content)['query']
            return SQL  # Success

        except Exception as e:
            print(f"Attempt {attempt}: Exception occurred: {e}, retrying...")

    return "Validation failed"

def error_handler(SQL, Error):
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": prompts["error"]["system"]},
        {"role": "user", "content": prompts["error"]["user"].format(SQL=SQL, Error=Error)}
        ],
        response_format={"type": "json_object"}
    )
    try :
        return eval(response.choices[0].message.content)
    except:
        return "Error_Handler faild"

def generate_sql_query(Question, DB):
    engine = create_engine(f'postgresql+psycopg2://abk6234:1201@localhost:5432/{DB}')
    with open(f"metadata/{DB}.meta", "rb") as metaPath:
        meta = pickle.load(metaPath)
    columns_json_abrvs = meta["columns_json_abrvs"]
    tables_json_abrvs = meta["tables_json_abrvs"]
    column_values_json = meta["column_values_json"]
    AI_Descriptions = meta["AI_Descriptions"]
    Database_description = meta["Database_description"]
    yield "Entity Extraction Agent"
    EEA = EEA_Func(Question)
    yield f"User's intent: {EEA['intent']}"
    yield f"Important entities: {', '.join(EEA['entities'])}"
    entities = EEA["entities"]
    MRA = MRA_Func(Question, entities, Database_description, tables_json_abrvs, columns_json_abrvs, column_values_json)
    yield "Metadata Retrival Agent:"
    pt_tb = ""
    for table in MRA["potential_table_columns"]:
        pt_tb+=f"From table {table} we might need these columns: "
        pt_tb+=", ".join(MRA["potential_table_columns"][table])
        yield pt_tb
    # Part 3
    sample_values = MRA["sample_values"]
    potential_table_columns = MRA["potential_table_columns"]
    QL = QLA_Func(Question, sample_values)
    yield "Query Logic Agent:"
    yield QL
    SQL = SQLA_Func(Question, QL, AI_Descriptions, potential_table_columns)
    yield "SQL Generation Agent:"
    yield "Unreviewed Query:"
    yield SQL
    yield "Reviwer Agent is working"
    SQL = double_check_filters(SQL, engine)
    yield "Filter Validation Complete"
    spatial_guide = prompts["Spatial guide"]
    SQL = validate_spatial_query(SQL, Question, spatial_guide, max_retries=3)
    yield "Spatial Validation Complete"
    is_executable = validate_query(SQL, engine)
    if is_executable["status"] == 'Error':
        Error = is_executable["Output"]
        SQL = error_handler(SQL, Error)
    is_executable = validate_query(SQL, engine)
    if is_executable["status"] == 'Error':
        yield "I couldn't generate a SQL query in this attempt, would you like me make another effort?"
    else:
        yield f"FINAL_SQL:{SQL}"

    return None



app = Flask(__name__)
CORS(app)
query_results = {}

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*' 
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    return response

@app.route('/generate-sql', methods=['OPTIONS', 'POST'])
def generate_sql():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    data = request.json
    question = data.get('question')
    db = data.get('database')
    if not question or not db:
        return jsonify({'error': 'Question and database are required'}), 400
    
    def generate():
        try:
            request_id = str(int(time.time() * 1000))
            query_results[request_id] = {'status': 'processing', 'progress': []}
            yield f"data: {json.dumps({'request_id': request_id, 'step': 'started'})}\n\n"
            generator = generate_sql_query(question)
            final_sql = None
            for progress_update in generator:
                if progress_update.startswith("FINAL_SQL:"):
                    final_sql = progress_update.replace("FINAL_SQL:", "").strip()
                    query_results[request_id]['sql'] = final_sql
                    yield f"data: {json.dumps({'request_id': request_id, 'type': 'complete', 'sql': final_sql})}\n\n"
                else:
                    query_results[request_id]['progress'].append(progress_update)
                    yield f"data: {json.dumps({'request_id': request_id, 'type': 'progress', 'message': progress_update})}\n\n"
            query_results[request_id]['status'] = 'completed'
        except Exception as e:
            yield f"data: {json.dumps({'request_id': request_id, 'type': 'error', 'error': str(e)})}\n\n"
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/query-status/<request_id>')
def query_status(request_id):
    result = query_results.get(request_id, {'error': 'Request not found'})
    return jsonify(result)

@app.route('/execute-query', methods=['OPTIONS', 'POST'])
def execute_sql():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    data = request.json
    sql = data.get('sql')
    query = check_geometry(sql)
    if query['has_geometry'].lower()== 'yes':
        try:
            R = run_query(f"""WITH Results AS (
            {query["query"].replace(";","")}
            )
            SELECT jsonb_build_object(
                'type',     'FeatureCollection',
                'features', jsonb_agg(
                    jsonb_build_object(
                        'type',       'Feature',
                        'geometry',   ST_AsGeoJSON(geometry)::jsonb,
                        'properties', to_jsonb(Results) - 'geometry'
                    )
                )
            ) AS geojson
            FROM Results;
            """)
            return dict(R["Output"][0])
        except:
            R = run_query(query["query"])
            return {"table":pd.DataFrame(R["Output"]).to_dict(orient='records')}
    else:
        R = run_query(query["query"])
        return {"table":pd.DataFrame(R["Output"]).to_dict(orient='records')}

@app.route('/', methods=['GET'])
def index():
    return "Hi"

if __name__ == '__main__':
    app.run(threaded=True, port=3000, host='0.0.0.0')