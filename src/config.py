import os
from dotenv import load_dotenv
from typing import Literal

load_dotenv()


class Config:
    """
    Configuração centralizada da aplicação com suporte multi-LLM.
    
    Suporta OpenAI e Google Gemini através da variável LLM_PROVIDER.

    """
    
    # ========== LLM Provider Configuration ==========
    LLM_PROVIDER: Literal["openai", "google"] = os.getenv("LLM_PROVIDER", "openai").lower()
    
    # ========== OpenAI Configuration ==========
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    
    # ========== Google Gemini Configuration ==========
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")
    GOOGLE_CHAT_MODEL = os.getenv("GOOGLE_CHAT_MODEL", "gemini-2.0-flash-exp")
    
    # ========== Database Configuration ==========
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rag")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB = os.getenv("POSTGRES_DB", "rag")
    POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
    
    # ========== Vector Store Configuration ==========
    COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "pdf_documents")
    
    # ========== Document Processing Configuration ==========
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
    PDF_PATH = os.getenv("PDF_PATH", "document.pdf")
    
    # ========== Application Configuration ==========
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    SEARCH_K = int(os.getenv("SEARCH_K", "10"))  # Número de documentos similares a retornar
    
    @classmethod
    def validate(cls):
        """
        Valida configurações baseado no provider escolhido.
 
        Raises:
            ValueError: Se configuração obrigatória estiver ausente ou inválida
        """
        # Validar provider
        if cls.LLM_PROVIDER not in ["openai", "google"]:
            raise ValueError(
                f"❌ LLM_PROVIDER inválido: '{cls.LLM_PROVIDER}'. "
                f"Valores aceitos: 'openai' ou 'google'"
            )
        
        # Validar API Key baseado no provider
        if cls.LLM_PROVIDER == "openai":
            if not cls.OPENAI_API_KEY:
                raise ValueError(
                    "❌ OPENAI_API_KEY não configurada. "
                    "Configure a variável de ambiente ou arquivo .env"
                )
            if not cls.OPENAI_API_KEY.startswith("sk-"):
                raise ValueError(
                    "❌ OPENAI_API_KEY parece inválida. "
                    "Chaves OpenAI começam com 'sk-'"
                )
        
        elif cls.LLM_PROVIDER == "google":
            if not cls.GOOGLE_API_KEY:
                raise ValueError(
                    "❌ GOOGLE_API_KEY não configurada. "
                    "Configure a variável de ambiente ou arquivo .env"
                )
            if len(cls.GOOGLE_API_KEY) < 30:
                raise ValueError(
                    "❌ GOOGLE_API_KEY parece inválida. "
                    "Chaves Google AI normalmente têm 39+ caracteres"
                )
        
        # Validar configuração de banco de dados
        if not cls.DATABASE_URL:
            raise ValueError(
                "❌ DATABASE_URL não configurada. "
                "Configure a variável de ambiente ou arquivo .env"
            )
        
        if not cls.DATABASE_URL.startswith("postgresql://"):
            raise ValueError(
                "❌ DATABASE_URL deve começar com 'postgresql://'"
            )
        
        # Validar parâmetros numéricos
        if cls.CHUNK_SIZE < 100:
            raise ValueError(
                f"❌ CHUNK_SIZE muito pequeno: {cls.CHUNK_SIZE}. "
                f"Mínimo recomendado: 100"
            )
        
        if cls.CHUNK_OVERLAP >= cls.CHUNK_SIZE:
            raise ValueError(
                f"❌ CHUNK_OVERLAP ({cls.CHUNK_OVERLAP}) deve ser menor que "
                f"CHUNK_SIZE ({cls.CHUNK_SIZE})"
            )
        
        if cls.SEARCH_K < 1:
            raise ValueError(
                f"❌ SEARCH_K deve ser >= 1, valor atual: {cls.SEARCH_K}"
            )
    
    @classmethod
    def display_config(cls):
        """
        Exibe configuração atual (útil para debug).
        Oculta chaves de API por segurança.
        """
        def mask_key(key: str) -> str:
            """Mascara chave de API mostrando apenas primeiros/últimos caracteres"""
            if not key or len(key) < 8:
                return "***"
            return f"{key[:4]}...{key[-4:]}"
        
        print("=" * 60)
        print("⚙️  CONFIGURAÇÃO ATUAL")
        print("=" * 60)
        print(f"🤖 LLM Provider: {cls.LLM_PROVIDER.upper()}")
        print()
        
        if cls.LLM_PROVIDER == "openai":
            print("📡 OpenAI:")
            print(f"   - API Key: {mask_key(cls.OPENAI_API_KEY)}")
            print(f"   - Embedding Model: {cls.OPENAI_EMBEDDING_MODEL}")
            print(f"   - Chat Model: {cls.OPENAI_CHAT_MODEL}")
        else:
            print("📡 Google Gemini:")
            print(f"   - API Key: {mask_key(cls.GOOGLE_API_KEY)}")
            print(f"   - Embedding Model: {cls.GOOGLE_EMBEDDING_MODEL}")
            print(f"   - Chat Model: {cls.GOOGLE_CHAT_MODEL}")
        
        print()
        print("🗄️  Database:")
        print(f"   - Host: {cls.POSTGRES_HOST}:{cls.POSTGRES_PORT}")
        print(f"   - Database: {cls.POSTGRES_DB}")
        print(f"   - Collection: {cls.COLLECTION_NAME}")
        print()
        print("📄 Document Processing:")
        print(f"   - Chunk Size: {cls.CHUNK_SIZE}")
        print(f"   - Chunk Overlap: {cls.CHUNK_OVERLAP}")
        print(f"   - Search K: {cls.SEARCH_K}")
        print()
        print("🔧 Application:")
        print(f"   - Log Level: {cls.LOG_LEVEL}")
        print("=" * 60)


# Validar configurações ao importar o módulo
try:
    Config.validate()
except ValueError as e:
    print(f"\n⚠️  ERRO DE CONFIGURAÇÃO:\n{str(e)}\n")
    print("💡 Dica: Verifique seu arquivo .env ou variáveis de ambiente")
    print("💡 Exemplo: cp .env.example .env (e preencha as chaves de API)\n")
    raise
