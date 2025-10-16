"""
Serviço de ingestão de PDF com suporte multi-LLM.

Processa arquivos PDF, divide em chunks e armazena embeddings
no PostgreSQL usando pgVector.

Exemplo de uso:
    # Ingerir PDF específico
    python src/ingest.py caminho/para/arquivo.pdf
    
    # Usar PDF_PATH do .env
    python src/ingest.py
"""

import asyncio
import sys
import os
from typing import List
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from langchain.schema import Document

# Adicionar src ao path para imports funcionarem
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from llm_factory import LLMFactory
from utils.logger import setup_logger

logger = setup_logger(__name__)


class PDFIngestionService:
    """
    Serviço de ingestão de PDF com suporte multi-LLM.
    
    Attributes:
        embeddings: Instância de embeddings (OpenAI ou Google)
        text_splitter: Divisor de texto em chunks
    """
    
    def __init__(self):
        """
        Inicializa o serviço de ingestão.
        
        Cria instâncias de embeddings e text splitter baseado
        nas configurações do Config.
        """
        logger.info("=" * 60)
        logger.info(f"🚀 Inicializando PDFIngestionService")
        logger.info(f"📡 Provider: {Config.LLM_PROVIDER.upper()}")
        logger.info("=" * 60)
        
        # Usar factory para criar embeddings
        self.embeddings = LLMFactory.create_embeddings()
        
        # Configurar text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]  # Prioridade de separação
        )
        
        logger.info(f"⚙️  Configuração de chunking:")
        logger.info(f"   - Chunk Size: {Config.CHUNK_SIZE} caracteres")
        logger.info(f"   - Chunk Overlap: {Config.CHUNK_OVERLAP} caracteres")
        logger.info(f"   - Collection: {Config.COLLECTION_NAME}")
        logger.info("=" * 60)
    
    def _validate_pdf_path(self, pdf_path: str) -> Path:
        """
        Valida se o arquivo PDF existe e é acessível.
        
        Args:
            pdf_path: Caminho para o arquivo PDF
            
        Returns:
            Path object do arquivo validado
            
        Raises:
            FileNotFoundError: Se arquivo não existir
            ValueError: Se arquivo não for PDF
        """
        path = Path(pdf_path)
        
        if not path.exists():
            error_msg = f"Arquivo não encontrado: {pdf_path}"
            logger.error(f"❌ {error_msg}")
            raise FileNotFoundError(error_msg)
        
        if not path.is_file():
            error_msg = f"Caminho não é um arquivo: {pdf_path}"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)
        
        if path.suffix.lower() != '.pdf':
            error_msg = f"Arquivo não é PDF: {pdf_path} (extensão: {path.suffix})"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)
        
        logger.info(f"✅ Arquivo validado: {path.name} ({path.stat().st_size / 1024:.2f} KB)")
        return path
    
    async def ingest_pdf(self, pdf_path: str, clear_existing: bool = False) -> dict:
        """
        Processa PDF e salva embeddings no banco.
        
        Args:
            pdf_path: Caminho para o arquivo PDF
            clear_existing: Se True, limpa collection antes de inserir
            
        Returns:
            Dicionário com estatísticas da ingestão:
            {
                'success': bool,
                'pdf_path': str,
                'total_pages': int,
                'total_chunks': int,
                'collection_name': str,
                'provider': str
            }
            
        Raises:
            FileNotFoundError: Se PDF não existir
            Exception: Outros erros de processamento
        """
        try:
            logger.info("\n" + "🔄 INICIANDO INGESTÃO DE PDF " + "=" * 35)
            
            # 1. Validar arquivo
            validated_path = self._validate_pdf_path(pdf_path)
            logger.info(f"📄 Arquivo: {validated_path.absolute()}")
            
            # 2. Carregar PDF
            logger.info("\n⏳ Carregando PDF...")
            loader = PyPDFLoader(str(validated_path))
            documents: List[Document] = loader.load()
            
            total_pages = len(documents)
            logger.info(f"✅ PDF carregado: {total_pages} página(s)")
            
            # Log de amostra do conteúdo
            if documents:
                first_page_preview = documents[0].page_content[:200].replace('\n', ' ')
                logger.debug(f"   Preview página 1: {first_page_preview}...")
            
            # 3. Dividir em chunks
            logger.info("\n⏳ Dividindo em chunks...")
            texts: List[Document] = self.text_splitter.split_documents(documents)
            
            total_chunks = len(texts)
            logger.info(f"✅ Documento dividido em {total_chunks} chunk(s)")
            
            # Estatísticas de chunks
            if texts:
                chunk_sizes = [len(chunk.page_content) for chunk in texts]
                avg_size = sum(chunk_sizes) / len(chunk_sizes)
                logger.info(f"   - Tamanho médio: {avg_size:.0f} caracteres")
                logger.info(f"   - Maior chunk: {max(chunk_sizes)} caracteres")
                logger.info(f"   - Menor chunk: {min(chunk_sizes)} caracteres")
            
            # 4 & 5. Gerar embeddings e salvar no banco
            logger.info("\n⏳ Gerando embeddings e salvando no banco...")
            logger.info(f"   - Provider: {Config.LLM_PROVIDER.upper()}")
            logger.info(f"   - Database: {Config.POSTGRES_HOST}:{Config.POSTGRES_PORT}/{Config.POSTGRES_DB}")
            logger.info(f"   - Collection: {Config.COLLECTION_NAME}")
            
            # Criar/conectar ao vector store
            vector_store = PGVector.from_documents(
                documents=texts,
                embedding=self.embeddings,
                collection_name=Config.COLLECTION_NAME,
                connection=Config.DATABASE_URL,
                pre_delete_collection=clear_existing  # Limpar collection se solicitado
            )
            
            logger.info("✅ Embeddings gerados e salvos com sucesso!")
            
            # Resultado
            result = {
                'success': True,
                'pdf_path': str(validated_path.absolute()),
                'pdf_name': validated_path.name,
                'total_pages': total_pages,
                'total_chunks': total_chunks,
                'collection_name': Config.COLLECTION_NAME,
                'provider': Config.LLM_PROVIDER,
                'chunk_size': Config.CHUNK_SIZE,
                'chunk_overlap': Config.CHUNK_OVERLAP
            }
            
            logger.info("\n" + "=" * 60)
            logger.info("✅ INGESTÃO CONCLUÍDA COM SUCESSO!")
            logger.info("=" * 60)
            logger.info(f"📊 Resumo:")
            logger.info(f"   - Arquivo: {result['pdf_name']}")
            logger.info(f"   - Páginas: {result['total_pages']}")
            logger.info(f"   - Chunks armazenados: {result['total_chunks']}")
            logger.info(f"   - Collection: {result['collection_name']}")
            logger.info(f"   - Provider: {result['provider'].upper()}")
            logger.info("=" * 60 + "\n")
            
            return result
            
        except FileNotFoundError:
            logger.error("❌ Arquivo PDF não encontrado!")
            raise
        
        except Exception as e:
            logger.error(f"❌ Erro durante ingestão: {str(e)}")
            logger.error("💡 Verifique:")
            logger.error("   1. Se o banco de dados está rodando (docker-compose up -d)")
            logger.error("   2. Se a extensão pgvector está instalada")
            logger.error("   3. Se a API key está configurada corretamente")
            logger.error("   4. Se o arquivo PDF não está corrompido")
            raise


async def main():
    """
    Função principal para execução standalone via CLI.
    
    Aceita caminho do PDF como argumento ou usa PDF_PATH do .env
    """
    import argparse
    
    # Parser de argumentos
    parser = argparse.ArgumentParser(
        description="Ingestão de PDF para vector database (PostgreSQL + pgVector)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Exemplos:
  # Usar arquivo específico
  python src/ingest.py document.pdf
  
  # Usar caminho absoluto
  python src/ingest.py C:\\docs\\relatorio.pdf
  
  # Usar PDF_PATH do .env
  python src/ingest.py
  
  # Limpar collection antes de inserir
  python src/ingest.py document.pdf --clear
  
  # Modo debug
  python src/ingest.py document.pdf --debug

Provider atual: {Config.LLM_PROVIDER.upper()}
Collection: {Config.COLLECTION_NAME}
        """
    )
    
    parser.add_argument(
        'pdf_path',
        nargs='?',  # Opcional
        default=Config.PDF_PATH,
        help=f"Caminho para o arquivo PDF (default: {Config.PDF_PATH})"
    )
    
    parser.add_argument(
        '--clear',
        action='store_true',
        help="Limpar collection existente antes de inserir"
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help="Ativar modo debug (logs detalhados)"
    )
    
    args = parser.parse_args()
    
    # Configurar debug
    if args.debug:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.info("🐛 Modo DEBUG ativado")
    
    # Verificar se caminho foi fornecido
    if not args.pdf_path:
        print("\n❌ Erro: Nenhum arquivo PDF especificado!")
        print("💡 Use: python src/ingest.py <caminho_do_pdf>")
        print(f"💡 Ou configure PDF_PATH no arquivo .env\n")
        parser.print_help()
        sys.exit(1)
    
    # Executar ingestão
    try:
        service = PDFIngestionService()
        result = await service.ingest_pdf(
            pdf_path=args.pdf_path,
            clear_existing=args.clear
        )
        
        # Sucesso!
        sys.exit(0)
        
    except FileNotFoundError as e:
        print(f"\n❌ Erro: {str(e)}\n")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Erro fatal: {str(e)}\n")
        logger.error("Detalhes do erro:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())