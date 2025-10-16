"""
Utilitários para conexão e validação do PostgreSQL + pgVector.
"""

import sys
import os
import psycopg2
from typing import Tuple

# Adicionar src ao path para imports funcionarem
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_logger

logger = setup_logger(__name__)


def test_database_connection(connection_string: str) -> Tuple[bool, str]:
    """
    Testa conexão com PostgreSQL e valida extensão pgvector.
    
    Args:
        connection_string: String de conexão PostgreSQL
                          Ex: "postgresql://user:pass@localhost:5432/dbname"
    
    Returns:
        Tupla (sucesso: bool, mensagem: str)
    
    Exemplo:
        ```python
        from config import Config
        from utils.database import test_database_connection
        
        success, message = test_database_connection(Config.DATABASE_URL)
        if success:
            print(f"✅ {message}")
        else:
            print(f"❌ {message}")
        ```
    """
    try:
        logger.info("Testando conexão com PostgreSQL...")
        
        # Tentar conectar
        conn = psycopg2.connect(connection_string)
        cursor = conn.cursor()
        
        # Verificar versão do PostgreSQL
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        logger.debug(f"PostgreSQL version: {version}")
        
        # Verificar extensão pgvector
        cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
        result = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if result:
            logger.info("✅ Conexão com banco OK - pgvector instalado")
            return True, "Conexão bem-sucedida! Extensão pgvector está instalada."
        else:
            logger.warning("⚠️  Conexão OK mas pgvector não encontrado")
            return False, "Conexão OK, mas extensão pgvector NÃO está instalada. Execute: CREATE EXTENSION vector;"
            
    except psycopg2.OperationalError as e:
        error_msg = f"Erro de conexão: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return False, error_msg
    
    except Exception as e:
        error_msg = f"Erro ao testar banco: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return False, error_msg


def get_vector_store_stats(connection_string: str, collection_name: str) -> dict:
    """
    Obtém estatísticas da collection de vetores.
    
    Args:
        connection_string: String de conexão PostgreSQL
        collection_name: Nome da collection (tabela)
    
    Returns:
        Dicionário com estatísticas (total de documentos, etc.)
    """
    try:
        conn = psycopg2.connect(connection_string)
        cursor = conn.cursor()
        
        # Verificar se a tabela existe
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = %s
            );
        """, (collection_name,))
        
        table_exists = cursor.fetchone()[0]
        
        if not table_exists:
            cursor.close()
            conn.close()
            return {
                "exists": False,
                "total_documents": 0,
                "message": f"Collection '{collection_name}' não existe ainda"
            }
        
        # Contar documentos
        cursor.execute(f"SELECT COUNT(*) FROM {collection_name};")
        count = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        logger.info(f"✅ Collection '{collection_name}': {count} documentos")
        
        return {
            "exists": True,
            "total_documents": count,
            "collection_name": collection_name
        }
        
    except Exception as e:
        logger.error(f"❌ Erro ao obter estatísticas: {str(e)}")
        return {
            "exists": False,
            "error": str(e)
        }


if __name__ == "__main__":
    """Teste standalone"""
    from config import Config
    
    print("\n🧪 Testando conexão com banco de dados...\n")
    
    success, message = test_database_connection(Config.DATABASE_URL)
    print(f"\n{'✅' if success else '❌'} {message}\n")
    
    if success:
        stats = get_vector_store_stats(Config.DATABASE_URL, Config.COLLECTION_NAME)
        print(f"📊 Estatísticas da collection:")
        print(f"   - Existe: {stats.get('exists', False)}")
        print(f"   - Total documentos: {stats.get('total_documents', 0)}\n")
