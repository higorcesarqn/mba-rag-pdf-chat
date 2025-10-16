"""
Factory Pattern para criação de instâncias de LLM e Embeddings.

centraliza a criação de objetos baseado em configuração, 
permitindo trocar facilmente entre providers
(OpenAI ou Google Gemini) sem modificar código cliente.

Exemplo de uso:
    ```python
    from llm_factory import LLMFactory
    
    # Criar apenas embeddings
    embeddings = LLMFactory.create_embeddings()
    
    # Criar apenas chat model
    chat_model = LLMFactory.create_chat_model()
    
    # Criar ambos de uma vez
    embeddings, chat_model = LLMFactory.create_all()
    ```
"""

from typing import Tuple
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.embeddings.base import Embeddings
from langchain.chat_models.base import BaseChatModel

from config import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)


class LLMFactory:
    """
    Factory para criar instâncias de LLM e Embeddings.
    
    Attributes:
        Nenhum (apenas métodos estáticos)
    """
    
    @staticmethod
    def create_embeddings() -> Embeddings:
        """
        Cria instância de embeddings baseado no provider configurado.
        
        Returns:
            Instância de Embeddings (OpenAI ou Google)
            
        Raises:
            ValueError: Se provider não for suportado
            
        Exemplo:
            ```python
            embeddings = LLMFactory.create_embeddings()
            vectors = embeddings.embed_documents(["texto exemplo"])
            ```
        """
        provider = Config.LLM_PROVIDER
        
        if provider == "openai":
            logger.info(f"🤖 Inicializando OpenAI Embeddings: {Config.OPENAI_EMBEDDING_MODEL}")
            
            return OpenAIEmbeddings(
                model=Config.OPENAI_EMBEDDING_MODEL,
                api_key=Config.OPENAI_API_KEY
            )
        
        elif provider == "google":
            logger.info(f"🤖 Inicializando Google Embeddings: {Config.GOOGLE_EMBEDDING_MODEL}")
            
            return GoogleGenerativeAIEmbeddings(
                model=Config.GOOGLE_EMBEDDING_MODEL,
                google_api_key=Config.GOOGLE_API_KEY
            )
        
        else:
            error_msg = f"Provider não suportado: {provider}. Use 'openai' ou 'google'"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)
    
    @staticmethod
    def create_chat_model(temperature: float = 0.0) -> BaseChatModel:
        """
        Cria instância de chat model baseado no provider configurado.
        
        Args:
            temperature: Temperatura para geração (0.0 = determinístico, 1.0 = criativo)
                        Default: 0.0 para respostas consistentes
        
        Returns:
            Instância de ChatModel (OpenAI ou Google)
            
        Raises:
            ValueError: Se provider não for suportado
            
        Exemplo:
            ```python
            chat = LLMFactory.create_chat_model()
            response = chat.invoke("Olá, como você está?")
            ```
        """
        provider = Config.LLM_PROVIDER
        
        if provider == "openai":
            logger.info(f"🤖 Inicializando OpenAI Chat: {Config.OPENAI_CHAT_MODEL}")
            
            return ChatOpenAI(
                model=Config.OPENAI_CHAT_MODEL,
                api_key=Config.OPENAI_API_KEY,
                temperature=temperature
            )
        
        elif provider == "google":
            logger.info(f"🤖 Inicializando Google Chat: {Config.GOOGLE_CHAT_MODEL}")
            
            return ChatGoogleGenerativeAI(
                model=Config.GOOGLE_CHAT_MODEL,
                google_api_key=Config.GOOGLE_API_KEY,
                temperature=temperature
            )
        
        else:
            error_msg = f"Provider não suportado: {provider}. Use 'openai' ou 'google'"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)
    
    @staticmethod
    def create_all(temperature: float = 0.0) -> Tuple[Embeddings, BaseChatModel]:
        """
        Cria embeddings e chat model de uma vez.
        
        Útil quando você precisa de ambos (como no SearchService).
        
        Args:
            temperature: Temperatura para o chat model
        
        Returns:
            Tupla (embeddings, chat_model)
            
        Exemplo:
            ```python
            embeddings, chat = LLMFactory.create_all()
            
            # Usar embeddings para busca
            vectors = embeddings.embed_query("minha pergunta")
            
            # Usar chat para gerar resposta
            response = chat.invoke("responda isso")
            ```
        """
        logger.info(f"🏭 Factory: Criando embeddings e chat model ({Config.LLM_PROVIDER})")
        
        embeddings = LLMFactory.create_embeddings()
        chat_model = LLMFactory.create_chat_model(temperature)
        
        logger.info("✅ Factory: Instâncias criadas com sucesso")
        
        return embeddings, chat_model
    
    @staticmethod
    def get_provider_info() -> dict:
        """
        Retorna informações sobre o provider atual.
        
        Returns:
            Dicionário com informações do provider
        """
        provider = Config.LLM_PROVIDER
        
        if provider == "openai":
            return {
                "provider": "OpenAI",
                "embedding_model": Config.OPENAI_EMBEDDING_MODEL,
                "chat_model": Config.OPENAI_CHAT_MODEL,
                "api_key_set": bool(Config.OPENAI_API_KEY)
            }
        elif provider == "google":
            return {
                "provider": "Google Gemini",
                "embedding_model": Config.GOOGLE_EMBEDDING_MODEL,
                "chat_model": Config.GOOGLE_CHAT_MODEL,
                "api_key_set": bool(Config.GOOGLE_API_KEY)
            }
        else:
            return {
                "provider": "Unknown",
                "error": f"Provider inválido: {provider}"
            }


if __name__ == "__main__":
    """Teste standalone do factory"""
    
    print("\n" + "=" * 60)
    print("🧪 TESTANDO LLM FACTORY")
    print("=" * 60 + "\n")
    
    # Mostrar configuração
    Config.display_config()
    
    print("\n📊 Informações do Provider:")
    info = LLMFactory.get_provider_info()
    for key, value in info.items():
        print(f"   - {key}: {value}")
    
    print("\n🏗️  Testando criação de instâncias...\n")
    
    try:
        # Testar embeddings
        print("1️⃣  Criando Embeddings...")
        embeddings = LLMFactory.create_embeddings()
        print(f"   ✅ Tipo: {type(embeddings).__name__}\n")
        
        # Testar chat model
        print("2️⃣  Criando Chat Model...")
        chat = LLMFactory.create_chat_model()
        print(f"   ✅ Tipo: {type(chat).__name__}\n")
        
        # Testar create_all
        print("3️⃣  Criando ambos (create_all)...")
        emb, ch = LLMFactory.create_all()
        print(f"   ✅ Embeddings: {type(emb).__name__}")
        print(f"   ✅ Chat: {type(ch).__name__}\n")
        
        print("=" * 60)
        print("✅ TODOS OS TESTES PASSARAM!")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERRO: {str(e)}\n")
        logger.error("Erro no teste do factory", exc_info=True)
