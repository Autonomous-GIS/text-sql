import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import unified_postgis_agent as m


def test_standalone_select_screening():
    assert m._read_only_select('SELECT * FROM t')
    assert m._read_only_select('WITH c AS (SELECT 1) SELECT * FROM c')
    assert not m._read_only_select('DROP TABLE t')
    assert not m._read_only_select('SELECT 1; DELETE FROM t')
    assert not m._read_only_select('SELECT * INTO new_table FROM old_table')


def test_database_url_preserves_secret_and_db_selection(monkeypatch):
    monkeypatch.setattr(m, 'allowed_databases', lambda: ['GeoDB'])
    monkeypatch.delenv('POSTGIS_DATABASE_URL_TEMPLATE', raising=False)
    monkeypatch.setenv('DATABASE_URL', 'postgresql+psycopg2://u:secret@localhost:5432/startdb')
    assert m.database_url('GeoDB') == 'postgresql+psycopg2://u:secret@localhost:5432/GeoDB'
    with pytest.raises(ValueError):
        m.database_url('another')


def test_progress_order_even_with_out_of_order_optional_reporting():
    a = m.ProgressAdapter('Find hospitals')
    assert a.record('QLA', 'Using spatial joins') == []
    assert [x['node'] for x in a.record('EEA', {'intent': 'Find hospitals'})] == ['EEA']
    assert [x['node'] for x in a.record('MRA', 'Found geometry')] == ['MRA','QLA']
    assert [x['node'] for x in a.complete('SELECT 1', True, 'EXPLAIN passed')] == ['SQLA', 'RA']


def test_one_react_agent_observes_python_before_answer(monkeypatch):
    agent = m.StreamingPostGISAgent(skill_markdown='PostGIS', model='test-model')
    call = SimpleNamespace(type='function_call',name='execute_python',arguments=json.dumps({'code':'print(1)'}),call_id='x')
    tool_response = SimpleNamespace(output=[call], output_text='')
    answer_response = SimpleNamespace(output=[],output_text=json.dumps({'sql_query':'SELECT 1'}))
    received = iter([tool_response,answer_response])
    monkeypatch.setattr(agent, '_run_model', lambda: next(received))
    monkeypatch.setattr(m, 'execute_python', lambda code, env_overrides=None:{'ok': True,'stdout':'1\n', 'stderr':'', 'exit_code':0, 'file_changes':[]})
    events = list(agent.iter_events('mode=sql'))
    assert [x['type'] for x in events] == ['tool_started', 'tool_completed', 'answer']
    assert agent.input_items[-1]['type'] == 'function_call_output'
    assert events[-1]['text'] == '{"sql_query": "SELECT 1"}'


def test_stream_contract_uses_one_agent(monkeypatch):
    service = m.UnifiedPostGISService()
    monkeypatch.setattr(m, 'allowed_databases', lambda: ['GeoDB'])
    monkeypatch.setattr(m, 'database_url', lambda db: 'postgresql+psycopg2://localhost/GeoDB')
    monkeypatch.setattr(m, 'validate_final_sql', lambda sql,db:(True,'Read-only EXPLAIN succeeded'))
    session = service.session('alice')
    def fake_events(*args,**kwargs):
        yield {'type':'tool_started','step':1,'code':'import os'}
        yield {'type':'tool_completed','step':1,'code':'from sqlalchemy import create_engine',
               'observation': {'ok':True, 'stdout': 'UI_PROGRESS: {"kind":"metadata","summary":"Checked geometry columns"}\n'}}
        yield {'type':'answer','text':json.dumps({'sql_query':'SELECT geom FROM hospitals','summary':'Found the features'})}
    monkeypatch.setattr(session.agent, 'iter_events', fake_events)
    stream=list(service.stream_sql('alice','Map hospitals','GeoDB'))
    assert [x['node'] for x in stream] == ['EEA','MRA','QLA','SQLA','RA','SQL','FINAL']
    assert stream[-1]['status'] == 'valid'
    assert stream[-1]['sql_query'] == 'SELECT geom FROM hospitals'
    assert session.plan is None and session.state=='NO_PLAN'


def test_chat_plan_keeps_the_existing_response_contract(monkeypatch):
    service = m.UnifiedPostGISService()
    monkeypatch.setattr(m, 'allowed_databases', lambda:['GeoDB'])
    session = service.session('alice')
    def fake_events(*args,**kwargs):
        yield {'type':'answer', 'text':json.dumps({'status':'plan_ready','response_to_user':'Ready',
             'query_plan':{'clear_question':'Map hospitals','target_database':'GeoDB','representation_type':'map'}})}
    monkeypatch.setattr(session.agent, 'iter_events',fake_events)
    response=service.chat('alice','Map hospitals',{})
    assert response['status']=='plan_ready'
    assert response['query_plan']['target_database']=='GeoDB'
    assert session.state=='PLAN_READY'


def test_gpt6_luna_defaults_and_main_responses_parameters(monkeypatch):
    """Check the actual outgoing Responses request without requiring an API key."""
    from GeneralAgent import general_python_agent as general_module
    from GeneralAgent import experience_manager as memory_module

    monkeypatch.delenv('POSTGIS_AGENT_MODEL', raising=False)
    service = m.UnifiedPostGISService()
    assert service.model == 'gpt-6-luna'
    assert general_module.MODEL == 'gpt-6-luna'
    assert memory_module.DEFAULT_REFLECTION_MODEL == 'gpt-6-luna'

    agent = m.StreamingPostGISAgent(skill_markdown='PostGIS', model=service.model)
    sent = []
    agent.client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: sent.append(kwargs)))
    agent._run_model()
    assert len(sent) == 1
    request = sent[0]
    assert request['model'] == 'gpt-6-luna'
    assert request['reasoning'] == {'effort': 'medium'}
    assert request['tool_choice'] == 'auto'
    assert request['parallel_tool_calls'] is False
    assert request['tools'][0]['strict'] is True
    assert request['tools'][0]['name'] == 'execute_python'
    assert {'temperature', 'top_p', 'top_logprobs', 'logprobs'}.isdisjoint(request)


def test_gpt6_luna_experience_reflection_request(tmp_path):
    """Keep the optional experience request valid on the Responses API."""
    from GeneralAgent.experience_manager import ExperienceManager

    sent = []

    def create(**kwargs):
        sent.append(kwargs)
        return SimpleNamespace(output_text='{"changes": []}')

    manager = ExperienceManager(
        memory_path=tmp_path / 'experiences.json',
        model='gpt-6-luna',
        client=SimpleNamespace(responses=SimpleNamespace(create=create)),
        show_activity=False,
    )
    report = manager.reflect({'turns': [[
        {'role': 'user', 'content': 'Find nearby facilities.'},
        {'role': 'assistant', 'content': 'A query was returned.'},
    ]]}, active_skill_key='postgis_sql_skill')
    assert report.attempted
    assert len(sent) == 1
    request = sent[0]
    assert request['model'] == 'gpt-6-luna'
    assert request['reasoning'] == {'effort': 'low'}
    assert request['store'] is False
    assert request['max_output_tokens'] > 0
    assert request['text']['format']['type'] == 'json_schema'
    assert request['text']['format']['strict'] is True
    assert {'temperature', 'top_p', 'top_logprobs', 'logprobs'}.isdisjoint(request)


def test_all_responses_create_sites_omit_unsupported_sampling_parameters():
    """Verify both reflection and consolidation, even if reflection is disabled."""
    import ast
    from pathlib import Path

    root = Path(m.__file__).resolve().parent
    paths = [
        root / 'GeneralAgent' / 'general_python_agent.py',
        root / 'GeneralAgent' / 'experience_manager.py',
    ]
    calls = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == 'create'
                    and isinstance(func.value, ast.Attribute) and func.value.attr == 'responses'):
                continue
            keywords = {keyword.arg for keyword in node.keywords}
            assert {'temperature', 'top_p', 'top_logprobs', 'logprobs'}.isdisjoint(keywords)
            assert {'model', 'reasoning'}.issubset(keywords)
            calls.append((path.name, keywords))
    assert len(calls) == 3  # main ReAct, experience reflection, experience consolidation


def test_osm_pa_uses_spatialbenchmark_database(monkeypatch):
    monkeypatch.setattr(m, "allowed_databases", lambda: ["osm_pa", "GeoDB"])
    monkeypatch.delenv("POSTGIS_DATABASE_URL_TEMPLATE", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg2://u:secret@localhost:5432/startdb")
    assert m.database_url("osm_pa") == "postgresql+psycopg2://u:secret@localhost:5432/SpatialBenchmark"
    assert m.database_url("GeoDB") == "postgresql+psycopg2://u:secret@localhost:5432/GeoDB"
    monkeypatch.setenv("POSTGIS_DATABASE_URL_TEMPLATE", "postgresql+psycopg2://u:secret@localhost:5432/{database}")
    assert m.database_url("osm_pa") == "postgresql+psycopg2://u:secret@localhost:5432/SpatialBenchmark"
