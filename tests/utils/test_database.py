import psycopg2

from utils import database


class _DummyCursor:
    def __init__(self, results):
        # Copy list to avoid cross-test mutation
        self._results = list(results)

    def execute(self, query, params=None):
        # No-op: queries only influence fetchone sequencing
        return None

    def fetchone(self):
        if not self._results:
            raise AssertionError("fetchone called more times than expected")
        return self._results.pop(0)

    def close(self):
        return None


class _DummyConnection:
    def __init__(self, results):
        self._cursor = _DummyCursor(results)

    def cursor(self):
        return self._cursor

    def close(self):
        return None


def _patch_connect(monkeypatch, results):
    def _fake_connect(connection_string):
        return _DummyConnection(results)

    monkeypatch.setattr(database.psycopg2, "connect", _fake_connect)


def test_test_database_connection_success(monkeypatch):
    results = [
        ("PostgreSQL 16.0",),
        ("vector",),
    ]
    _patch_connect(monkeypatch, results)

    success, message = database.test_database_connection("postgresql://user:pass@host/db")

    assert success is True
    assert "pgvector" in message


def test_test_database_connection_missing_extension(monkeypatch):
    results = [
        ("PostgreSQL 16.0",),
        None,
    ]
    _patch_connect(monkeypatch, results)

    success, message = database.test_database_connection("postgresql://user:pass@host/db")

    assert success is False
    assert "NÃO está instalada" in message


def test_test_database_connection_operational_error(monkeypatch):
    def _raise_error(connection_string):
        raise psycopg2.OperationalError("connection refused")

    monkeypatch.setattr(database.psycopg2, "connect", _raise_error)

    success, message = database.test_database_connection("postgresql://user:pass@host/db")

    assert success is False
    assert "Erro de conexão" in message


def test_get_vector_store_stats_not_found(monkeypatch):
    results = [
        (False,),
    ]
    _patch_connect(monkeypatch, results)

    stats = database.get_vector_store_stats("postgresql://user:pass@host/db", "documents")

    assert stats == {
        "exists": False,
        "total_documents": 0,
        "message": "Collection 'documents' não existe ainda",
    }


def test_get_vector_store_stats_success(monkeypatch):
    results = [
        (True,),
        (5,),
    ]
    _patch_connect(monkeypatch, results)

    stats = database.get_vector_store_stats("postgresql://user:pass@host/db", "documents")

    assert stats == {
        "exists": True,
        "total_documents": 5,
        "collection_name": "documents",
    }


def test_get_vector_store_stats_exception(monkeypatch):
    def _raise_error(connection_string):
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr(database.psycopg2, "connect", _raise_error)

    stats = database.get_vector_store_stats("postgresql://user:pass@host/db", "documents")

    assert stats == {
        "exists": False,
        "error": "unexpected failure",
    }
