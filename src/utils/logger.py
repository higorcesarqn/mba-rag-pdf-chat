"""
Sistema de logging configurável para a aplicação.
"""

import logging
import sys
from typing import Optional


def setup_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Configura e retorna um logger.
    
    Args:
        name: Nome do logger (geralmente __name__ do módulo)
        level: Nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
               Se None, usa Config.LOG_LEVEL
    
    Returns:
        Logger configurado
    
    Exemplo:
        ```python
        from utils.logger import setup_logger
        
        logger = setup_logger(__name__)
        logger.info("✅ Operação concluída")
        logger.error("❌ Erro ao processar")
        ```
    """
    logger = logging.getLogger(name)
    
    # Evitar duplicação de handlers se logger já foi configurado
    if logger.handlers:
        return logger
    
    # Determinar nível de log
    if level is None:
        # Importação tardia para evitar circular dependency
        try:
            from config import Config
            level = Config.LOG_LEVEL
        except (ImportError, AttributeError):
            level = "INFO"
    
    # Converter string para constante do logging
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    # Criar handler para console (stdout)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)
    
    # Formato detalhado com timestamp, nome do módulo e nível
    # Exemplo: 2025-10-15 14:30:45 - src.ingest - INFO - ✅ PDF carregado
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    # Não propagar para o logger raiz (evita duplicação)
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Alias para setup_logger (compatibilidade).
    
    Args:
        name: Nome do logger
        
    Returns:
        Logger configurado
    """
    return setup_logger(name)


# Logger para este módulo (exemplo)
logger = setup_logger(__name__)
