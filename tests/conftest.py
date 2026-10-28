import sys
import types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The development notebook lacks sqlparse; deployed requirements install it.
try:
    import sqlparse  # noqa: F401
except ImportError:
    stub = types.ModuleType('sqlparse')
    class Statement:
        def __init__(self, sql): self.sql = sql
        def get_type(self):
            return 'SELECT' if self.sql.strip().upper().startswith(('SELECT ', 'WITH ')) else 'UNKNOWN'
    stub.parse = lambda s: [Statement(x) for x in s.split(';') if x.strip()]
    stub.format = lambda s, **kw: s
    sys.modules['sqlparse'] = stub
try:
    from openai import OpenAI  # noqa
except ImportError:
    fake = types.ModuleType('openai')
    class OpenAI:
        def __init__(self, *a, **kw): pass
    fake.OpenAI = OpenAI
    sys.modules['openai'] = fake