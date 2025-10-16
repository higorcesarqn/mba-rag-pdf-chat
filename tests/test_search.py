from types import SimpleNamespace

import pytest

import search


class DummyDoc:
    def __init__(self, content):
        self.page_content = content


class DummyVectorStore:
    def __init__(self):
        self.docs_with_scores = []
        self.calls = []

    def similarity_search_with_score(self, query, k=None):
        self.calls.append((query, k))
        return [(DummyDoc(content), score) for content, score in self.docs_with_scores]


class DummyLLM:
    def __init__(self):
        self.prompts = []
        self.next_content = ""

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.next_content)


@pytest.fixture
def patched_search_service(monkeypatch):
    vector_store = DummyVectorStore()
    llm = DummyLLM()

    monkeypatch.setattr(search, "PGVector", lambda *args, **kwargs: vector_store)
    monkeypatch.setattr(search.LLMFactory, "create_all", lambda *args, **kwargs: ("embeddings", llm))

    service = search.SearchService()

    return service, vector_store, llm


def test_search_similar_documents_returns_documents(patched_search_service):
    service, vector_store, _ = patched_search_service
    vector_store.docs_with_scores = [("Doc 1", 0.1), ("Doc 2", 0.2)]

    docs = service.search_similar_documents("consulta", k=2)

    assert [doc.page_content for doc in docs] == ["Doc 1", "Doc 2"]
    assert vector_store.calls == [("consulta", 2)]


@pytest.mark.asyncio
async def test_generate_answer_returns_llm_response(patched_search_service):
    service, vector_store, llm = patched_search_service
    vector_store.docs_with_scores = [("Contexto relevante", 0.05)]
    llm.next_content = "Resposta gerada"

    result = await service.generate_answer("Qual o assunto?", k=1)

    assert result == "Resposta gerada"
    assert "Contexto relevante" in llm.prompts[0]
    assert "Qual o assunto?" in llm.prompts[0]


@pytest.mark.asyncio
async def test_generate_answer_returns_default_when_no_docs(patched_search_service):
    service, vector_store, llm = patched_search_service
    vector_store.docs_with_scores = []
    llm.next_content = "Should not be used"

    result = await service.generate_answer("Pergunta sem dados")

    assert result == "Não tenho informações necessárias para responder sua pergunta."
    assert llm.prompts == []
