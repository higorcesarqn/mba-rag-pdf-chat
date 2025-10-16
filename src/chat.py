"""
Interface CLI para chat interativo com PDF usando RAG.

Permite fazer perguntas sobre o conteúdo do PDF através
de busca semântica e geração de respostas com LLM.

Exemplo de uso:
    # Chat interativo normal
    python src/chat.py
    
    # Com modo debug
    python src/chat.py --debug
    
    # Testar uma pergunta única
    python src/chat.py --query "Qual o assunto do documento?"
"""

import asyncio
import argparse
import sys
import os
from typing import Optional

# Adicionar src ao path para imports funcionarem
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search import SearchService
from config import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ChatInterface:
    """
    Interface de linha de comando para chat interativo.
    
    Attributes:
        search_service: Instância do SearchService
    """
    
    def __init__(self):
        """Inicializa interface de chat"""
        logger.info("🎯 Inicializando ChatInterface")
        
        try:
            self.search_service = SearchService()
            logger.info("✅ ChatInterface pronto para uso\n")
        except Exception as e:
            logger.error(f"❌ Erro ao inicializar chat: {str(e)}")
            raise
    
    def _print_header(self):
        """Imprime cabeçalho do chat"""
        print("\n" + "=" * 60)
        print("💬 CHAT COM PDF - RAG SYSTEM")
        print("=" * 60)
        print(f"🤖 Provider: {Config.LLM_PROVIDER.upper()}")
        print(f"📚 Collection: {Config.COLLECTION_NAME}")
        print(f"🔍 Search K: {Config.SEARCH_K}")
        print("=" * 60)
        print("\n💡 Dicas:")
        print("   - Digite sua pergunta e pressione Enter")
        print("   - Digite 'sair', 'exit', 'quit' ou 'q' para encerrar")
        print("   - Digite 'help' para ver comandos especiais")
        print("=" * 60 + "\n")
    
    def _print_help(self):
        """Imprime comandos disponíveis"""
        print("\n" + "=" * 60)
        print("📖 COMANDOS DISPONÍVEIS")
        print("=" * 60)
        print("  help, ajuda, ?     - Mostra esta mensagem")
        print("  sair, exit, quit   - Encerra o chat")
        print("  clear, cls         - Limpa a tela")
        print("  info               - Mostra informações do sistema")
        print("=" * 60 + "\n")
    
    def _print_info(self):
        """Imprime informações do sistema"""
        print("\n" + "=" * 60)
        print("ℹ️  INFORMAÇÕES DO SISTEMA")
        print("=" * 60)
        print(f"Provider: {Config.LLM_PROVIDER.upper()}")
        
        if Config.LLM_PROVIDER == "openai":
            print(f"Embedding Model: {Config.OPENAI_EMBEDDING_MODEL}")
            print(f"Chat Model: {Config.OPENAI_CHAT_MODEL}")
        else:
            print(f"Embedding Model: {Config.GOOGLE_EMBEDDING_MODEL}")
            print(f"Chat Model: {Config.GOOGLE_CHAT_MODEL}")
        
        print(f"\nDatabase: {Config.POSTGRES_HOST}:{Config.POSTGRES_PORT}/{Config.POSTGRES_DB}")
        print(f"Collection: {Config.COLLECTION_NAME}")
        print(f"\nChunk Size: {Config.CHUNK_SIZE}")
        print(f"Chunk Overlap: {Config.CHUNK_OVERLAP}")
        print(f"Search K: {Config.SEARCH_K}")
        print("=" * 60 + "\n")
    
    def _clear_screen(self):
        """Limpa a tela (Windows e Unix)"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    async def ask_question(self, query: str) -> str:
        """
        Processa uma pergunta e retorna resposta.
        
        Args:
            query: Pergunta do usuário
            
        Returns:
            Resposta gerada
        """
        try:
            answer = await self.search_service.generate_answer(query)
            return answer
        except Exception as e:
            logger.error(f"Erro ao processar pergunta: {str(e)}", exc_info=True)
            return "❌ Erro ao processar sua pergunta. Tente novamente."
    
    async def run_single_query(self, query: str):
        """
        Executa uma única pergunta (modo não-interativo).
        
        Args:
            query: Pergunta a processar
        """
        print(f"\n❓ Pergunta: {query}\n")
        print("⏳ Processando...\n")
        
        answer = await self.ask_question(query)
        
        print("=" * 60)
        print("💬 RESPOSTA:")
        print("=" * 60)
        print(answer)
        print("=" * 60 + "\n")
    
    async def run_interactive_chat(self):
        """
        Loop principal do chat interativo.
        
        Gerencia entrada do usuário, comandos especiais
        e processamento de perguntas.
        """
        self._print_header()
        
        while True:
            try:
                # Input do usuário
                query = input("💬 Sua pergunta: ").strip()
                
                # Verificar se input está vazio
                if not query:
                    continue
                
                # Processar comandos especiais
                query_lower = query.lower()
                
                # Comando: sair
                if query_lower in ['sair', 'exit', 'quit', 'q']:
                    print("\n👋 Encerrando chat. Até logo!")
                    print("=" * 60 + "\n")
                    break
                
                # Comando: help
                if query_lower in ['help', 'ajuda', '?']:
                    self._print_help()
                    continue
                
                # Comando: clear
                if query_lower in ['clear', 'cls']:
                    self._clear_screen()
                    self._print_header()
                    continue
                
                # Comando: info
                if query_lower == 'info':
                    self._print_info()
                    continue
                
                # Processar pergunta
                print("\n⏳ Processando...\n")
                answer = await self.ask_question(query)
                
                # Mostrar resposta
                print("=" * 60)
                print("💬 RESPOSTA:")
                print("=" * 60)
                print(answer)
                print("=" * 60 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrompido pelo usuário (Ctrl+C).")
                print("=" * 60 + "\n")
                break
            
            except Exception as e:
                print(f"\n❌ Erro inesperado: {str(e)}")
                logger.error(f"Erro no loop do chat: {str(e)}", exc_info=True)
                print("💡 Tente novamente ou digite 'sair' para encerrar.\n")


def main():
    """
    Função principal com argumentos de linha de comando.
    
    Suporta:
    - Modo interativo (padrão)
    - Modo single query (--query)
    - Modo debug (--debug)
    """
    parser = argparse.ArgumentParser(
        description="Chat interativo com PDF usando RAG (OpenAI ou Google Gemini)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Exemplos:
  # Chat interativo (padrão)
  python src/chat.py
  
  # Pergunta única
  python src/chat.py --query "Qual o assunto do documento?"
  
  # Modo debug
  python src/chat.py --debug
  
  # Pergunta única com debug
  python src/chat.py --query "Resumo" --debug

Provider atual: {Config.LLM_PROVIDER.upper()}
Collection: {Config.COLLECTION_NAME}
        """
    )
    
    parser.add_argument(
        '--query', '-q',
        type=str,
        help="Fazer uma pergunta única (modo não-interativo)"
    )
    
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help="Ativar modo debug (logs detalhados)"
    )
    
    args = parser.parse_args()
    
    # Configurar debug
    if args.debug:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        
        # Aplicar debug a todos os loggers do projeto
        for logger_name in ['config', 'llm_factory', 'search', 'ingest', 'utils.database']:
            logging.getLogger(logger_name).setLevel(logging.DEBUG)
        
        logger.info("🐛 Modo DEBUG ativado globalmente")
    
    # Executar chat
    try:
        chat = ChatInterface()
        
        if args.query:
            # Modo single query
            asyncio.run(chat.run_single_query(args.query))
        else:
            # Modo interativo
            asyncio.run(chat.run_interactive_chat())
    
    except KeyboardInterrupt:
        print("\n\n👋 Programa interrompido.")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"Erro fatal: {str(e)}", exc_info=True)
        print(f"\n❌ Erro fatal: {str(e)}")
        print("💡 Verifique:")
        print("   1. Se o banco de dados está rodando (docker-compose up -d)")
        print("   2. Se a ingestão foi executada (python src/ingest.py)")
        print("   3. Se as variáveis de ambiente estão configuradas (.env)")
        print("   4. Use --debug para mais detalhes\n")
        sys.exit(1)


if __name__ == "__main__":
    main()