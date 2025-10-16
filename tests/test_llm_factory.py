import pytest

import llm_factory


class DummyEmbeddings:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class DummyChatModel:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


@pytest.fixture(autouse=True)
def restore_config():
    original_provider = llm_factory.Config.LLM_PROVIDER
    original_openai_key = llm_factory.Config.OPENAI_API_KEY
    original_google_key = llm_factory.Config.GOOGLE_API_KEY
    yield
    llm_factory.Config.LLM_PROVIDER = original_provider
    llm_factory.Config.OPENAI_API_KEY = original_openai_key
    llm_factory.Config.GOOGLE_API_KEY = original_google_key


def test_create_embeddings_openai(monkeypatch):
    llm_factory.Config.LLM_PROVIDER = "openai"
    llm_factory.Config.OPENAI_API_KEY = "sk-test"

    monkeypatch.setattr(llm_factory, "OpenAIEmbeddings", DummyEmbeddings)

    embeddings = llm_factory.LLMFactory.create_embeddings()

    assert isinstance(embeddings, DummyEmbeddings)
    assert embeddings.kwargs["model"] == llm_factory.Config.OPENAI_EMBEDDING_MODEL


def test_create_chat_model_google(monkeypatch):
    llm_factory.Config.LLM_PROVIDER = "google"
    llm_factory.Config.GOOGLE_API_KEY = "google-test-key"

    monkeypatch.setattr(llm_factory, "ChatGoogleGenerativeAI", DummyChatModel)

    chat_model = llm_factory.LLMFactory.create_chat_model(temperature=0.7)

    assert isinstance(chat_model, DummyChatModel)
    assert chat_model.kwargs["temperature"] == 0.7


def test_create_chat_model_unknown_provider(monkeypatch):
    llm_factory.Config.LLM_PROVIDER = "unknown"

    with pytest.raises(ValueError) as excinfo:
        llm_factory.LLMFactory.create_chat_model()

    assert "Provider não suportado" in str(excinfo.value)


def test_create_all_reuses_factory(monkeypatch):
    llm_factory.Config.LLM_PROVIDER = "openai"
    llm_factory.Config.OPENAI_API_KEY = "sk-test"

    def _fake_create_embeddings():
        return "embeddings"

    def _fake_create_chat_model(temp=0.0):
        return f"chat-{temp}"

    monkeypatch.setattr(llm_factory.LLMFactory, "create_embeddings", staticmethod(_fake_create_embeddings))
    monkeypatch.setattr(llm_factory.LLMFactory, "create_chat_model", staticmethod(_fake_create_chat_model))

    embeddings, chat_model = llm_factory.LLMFactory.create_all(temperature=0.2)

    assert embeddings == "embeddings"
    assert chat_model == "chat-0.2"


def test_get_provider_info_flags_missing_keys():
    llm_factory.Config.LLM_PROVIDER = "google"
    llm_factory.Config.GOOGLE_API_KEY = ""

    info = llm_factory.LLMFactory.get_provider_info()

    assert info["provider"] == "Google Gemini"
    assert info["api_key_set"] is False
