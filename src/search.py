"""
Serviço de busca semântica com suporte multi-LLM.

Realiza busca vetorial no PostgreSQL e gera respostas
usando LLM (OpenAI ou Google Gemini).

Exemplo de uso:
    from search import SearchService
    
    service = SearchService()
    answer = await service.generate_answer("Qual o faturamento?")
    print(answer)
"""

import sys
import os
from typing import List

from langchain_postgres import PGVector
from langchain.schema import Document

# Adicionar src ao path para imports funcionarem
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from llm_factory import LLMFactory
from utils.logger import setup_logger

logger = setup_logger(__name__)


# Template do prompt (conforme especificação EXATA)
PROMPT_TEMPLATE = """CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


class SearchService:
    """
    Serviço de busca semântica com suporte multi-LLM.
    
    Attributes:
        embeddings: Instância de embeddings (OpenAI ou Google)
        llm: Instância de chat model (OpenAI ou Google)
        vector_store: Store vetorial conectado ao PostgreSQL
    """
    
    def __init__(self):
        """
        Inicializa o serviço de busca.
        
        Cria instâncias de embeddings, chat model e conecta
        ao vector store no PostgreSQL.
        """
        logger.info("=" * 60)
        logger.info("🔍 Inicializando SearchService")
        logger.info(f"📡 Provider: {Config.LLM_PROVIDER.upper()}")
        logger.info("=" * 60)
        
        # Usar factory para criar embeddings e chat model
        self.embeddings, self.llm = LLMFactory.create_all()
        
        # Conectar ao vector store
        logger.info(f"🔗 Conectando ao vector store...")
        logger.info(f"   - Database: {Config.POSTGRES_HOST}:{Config.POSTGRES_PORT}/{Config.POSTGRES_DB}")
        logger.info(f"   - Collection: {Config.COLLECTION_NAME}")
        
        try:
            self.vector_store = PGVector(
                embeddings=self.embeddings,
                collection_name=Config.COLLECTION_NAME,
                connection=Config.DATABASE_URL,
            )
            logger.info("✅ Vector store conectado com sucesso")
        except Exception as e:
            logger.error(f"❌ Erro ao conectar vector store: {str(e)}")
            logger.error("💡 Verifique se:")
            logger.error("   1. O banco de dados está rodando (docker-compose up -d)")
            logger.error("   2. A ingestão foi executada (python src/ingest.py)")
            raise
        
        logger.info("=" * 60)
        logger.info("✅ SearchService inicializado com sucesso")
        logger.info("=" * 60 + "\n")
    
    def search_similar_documents(self, query: str, k: int = None) -> List[Document]:
        """
        Busca documentos similares usando embeddings.
        
        Args:
            query: Pergunta do usuário
            k: Número de documentos a retornar (default: Config.SEARCH_K)
            
        Returns:
            Lista de documentos relevantes ordenados por similaridade
            
        Raises:
            Exception: Se houver erro na busca
        """
        if k is None:
            k = Config.SEARCH_K
        
        try:
            query_preview = query[:50] + "..." if len(query) > 50 else query
            logger.info(f"🔎 Buscando documentos similares para: '{query_preview}'")
            logger.info(f"   - Retornar top {k} documentos")
            
            # Busca com scores
            docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
            docs = [doc for doc, score in docs_with_scores]
            
            logger.info(f"✅ Encontrados {len(docs)} documento(s) similar(es)")
            
            # Log dos top 3 scores (útil para debug)
            if docs_with_scores:
                logger.debug("📊 Top 3 documentos mais similares:")
                for i, (doc, score) in enumerate(docs_with_scores[:3], 1):
                    preview = doc.page_content[:100].replace('\n', ' ')
                    logger.debug(f"   {i}. Score: {score:.4f} | Preview: {preview}...")
            
            return docs
            
        except Exception as e:
            logger.error(f"❌ Erro na busca: {str(e)}")
            raise
    
    async def generate_answer(self, query: str, k: int = None) -> str:
        """
        Gera resposta baseada no contexto encontrado.
        
        Args:
            query: Pergunta do usuário
            k: Número de documentos para buscar (default: Config.SEARCH_K)
            
        Returns:
            Resposta gerada pelo LLM baseada no contexto
            
        Raises:
            Exception: Se houver erro na geração
        """
        try:
            logger.info("\n" + "💬 GERANDO RESPOSTA " + "=" * 42)
            query_preview = query[:80] + "..." if len(query) > 80 else query
            logger.info(f"❓ Pergunta: {query_preview}")
            
            # 1. Buscar documentos relevantes
            logger.info("\n⏳ Etapa 1/4: Buscando documentos relevantes...")
            relevant_docs = self.search_similar_documents(query, k=k)
            
            if not relevant_docs:
                logger.warning("⚠️  Nenhum documento relevante encontrado")
                return "Não tenho informações necessárias para responder sua pergunta."
            
            logger.info(f"✅ {len(relevant_docs)} documento(s) recuperado(s)")
            
            # 2. Construir contexto
            logger.info("\n⏳ Etapa 2/4: Construindo contexto...")
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            context_length = len(context)
            
            logger.info(f"✅ Contexto construído: {context_length} caracteres")
            logger.debug(f"   Preview do contexto: {context[:200].replace(chr(10), ' ')}...")
            
            # 3. Construir prompt
            logger.info("\n⏳ Etapa 3/4: Montando prompt...")
            prompt = PROMPT_TEMPLATE.format(
                contexto=context,
                pergunta=query
            )
            
            prompt_length = len(prompt)
            logger.info(f"✅ Prompt montado: {prompt_length} caracteres")
            
            # 4. Gerar resposta
            logger.info(f"\n⏳ Etapa 4/4: Gerando resposta com {Config.LLM_PROVIDER.upper()}...")
            response = await self.llm.ainvoke(prompt)
            
            answer = response.content.strip()
            answer_preview = answer[:100] + "..." if len(answer) > 100 else answer
            
            logger.info(f"✅ Resposta gerada: {len(answer)} caracteres")
            logger.info(f"   Preview: {answer_preview}")
            
            logger.info("\n" + "=" * 60)
            logger.info("✅ RESPOSTA CONCLUÍDA")
            logger.info("=" * 60 + "\n")
            
            return answer
            
        except Exception as e:
            logger.error(f"❌ Erro ao gerar resposta: {str(e)}")
            logger.error("💡 Retornando mensagem de erro genérica")
            return "Erro interno ao processar sua pergunta. Tente novamente."


if __name__ == "__main__":
    """Teste standalone do SearchService"""
    import asyncio
    
    async def test_search():
        """Função de teste"""
        print("\n" + "=" * 60)
        print("🧪 TESTANDO SEARCH SERVICE")
        print("=" * 60 + "\n")
        
        try:
            # Criar serviço
            service = SearchService()
            
            # Teste 1: Pergunta de exemplo
            test_query = "Qual é o principal assunto do documento?"
            print(f"\n📝 Teste 1: Busca de documentos")
            print(f"Pergunta: {test_query}\n")
            
            docs = service.search_similar_documents(test_query, k=3)
            print(f"✅ Encontrados {len(docs)} documentos\n")
            
            # Teste 2: Geração de resposta
            print(f"\n📝 Teste 2: Geração de resposta")
            print(f"Pergunta: {test_query}\n")
            
            answer = await service.generate_answer(test_query)
            print(f"\n💬 Resposta:")
            print(f"{answer}\n")
            
            print("=" * 60)
            print("✅ TESTES CONCLUÍDOS!")
            print("=" * 60 + "\n")
            
        except Exception as e:
            print(f"\n❌ Erro no teste: {str(e)}\n")
            logger.error("Erro no teste:", exc_info=True)
    
    asyncio.run(test_search())