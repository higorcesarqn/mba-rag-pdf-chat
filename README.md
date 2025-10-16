# Sistema RAG - Ingestão e Busca Semântica com LangChain 🚀

Sistema completo de **RAG (Retrieval-Augmented Generation)** que permite fazer perguntas sobre documentos PDF usando busca semântica e LLM (Large Language Models).

## 🎯 Funcionalidades

- ✅ **Ingestão de PDF**: Processa arquivos PDF, divide em chunks e armazena embeddings
- ✅ **Busca Semântica**: Busca vetorial usando PostgreSQL + pgVector
- ✅ **Multi-LLM**: Suporta OpenAI e Google Gemini (configurável)
- ✅ **Chat Interativo**: Interface CLI amigável com comandos especiais
- ✅ **Respostas Baseadas em Contexto**: Responde apenas com informações do PDF

## 🛠️ Tecnologias

| Categoria          | Tecnologia                  |
| ------------------ | --------------------------- |
| **Linguagem**      | Python 3.13+                |
| **Framework**      | LangChain                   |
| **Banco de Dados** | PostgreSQL 17 + pgVector    |
| **Gerenciador**    | uv (Python package manager) |
| **Container**      | Docker & Docker Compose     |
| **LLM Providers**  | OpenAI ou Google Gemini     |

## 📋 Pré-requisitos

Antes de começar, certifique-se de ter instalado:

- ✅ **Python 3.10+** ([Download](https://www.python.org/downloads/))
- ✅ **Docker Desktop** ([Download](https://www.docker.com/products/docker-desktop/))
- ✅ **uv** - Gerenciador de pacotes Python ([Instalação](https://docs.astral.sh/uv/))
- ✅ **API Key** da OpenAI OU Google Gemini

### Como obter API Keys:

- **OpenAI**: https://platform.openai.com/api-keys
- **Google Gemini**: https://aistudio.google.com/app/apikey

## 🚀 Instalação

### 1. Clone o repositório

```bash
git clone <url-do-repositorio>
cd projeto1
```

### 2. Instale as dependências

**Usando UV (recomendado):**

```bash
uv pip install -r requirements.txt
```

**Ou usando Python/pip padrão:**

```bash
python -m pip install -r requirements.txt
```

### 3. Configure as variáveis de ambiente

```bash
# Copiar template
cp .env.example .env

# Editar .env e adicionar suas API keys
# No Windows: notepad .env
# No Linux/Mac: nano .env
```

**Configuração mínima no `.env`:**

```env
# Escolha o provider: "openai" ou "google"
LLM_PROVIDER=google

# Se usar Google Gemini
GOOGLE_API_KEY=sua_chave_aqui

# Se usar OpenAI
# OPENAI_API_KEY=sk-sua_chave_aqui

# Database (já configurado)
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/rag
```

### 4. Suba o banco de dados

```bash
docker-compose up -d
```

Aguarde alguns segundos e verifique se está rodando:

```bash
docker ps
```

### 5. Instale a extensão pgVector (se necessário)

```bash
docker exec postgres_rag psql -U postgres -d rag -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

## 📖 Uso

### Passo 1: Ingerir um PDF

Processe seu PDF e armazene no banco de dados:

**Usando UV:**

```bash
uv run python src/ingest.py seu_arquivo.pdf
uv run python src/ingest.py
uv run python src/ingest.py document.pdf --clear
uv run python src/ingest.py document.pdf --debug
```

**Ou usando Python diretamente:**

```bash
python src/ingest.py seu_arquivo.pdf
python src/ingest.py
python src/ingest.py document.pdf --clear
python src/ingest.py document.pdf --debug
```

**Saída esperada:**

```
🚀 Inicializando PDFIngestionService
📡 Provider: GOOGLE
✅ PDF carregado: 15 página(s)
✅ Documento dividido em 42 chunk(s)
✅ Embeddings gerados e salvos com sucesso!
✅ INGESTÃO CONCLUÍDA COM SUCESSO!
```

### Passo 2: Fazer perguntas (Chat Interativo)

**Usando UV:**

```bash
uv run python src/chat.py
uv run python src/chat.py --query "Qual o assunto principal?"
uv run python src/chat.py --debug
```

**Ou usando Python diretamente:**

```bash
python src/chat.py
python src/chat.py --query "Qual o assunto principal?"
python src/chat.py --debug
```

**Exemplo de interação:**

```
💬 CHAT COM PDF - RAG SYSTEM
🤖 Provider: GOOGLE
💡 Dicas: Digite 'help' para comandos especiais
============================================================

💬 Sua pergunta: Qual o faturamento da empresa?

⏳ Processando...

============================================================
💬 RESPOSTA:
============================================================
O faturamento foi de 10 milhões de reais.
============================================================

💬 Sua pergunta: Quantos clientes temos em 2024?

⏳ Processando...

============================================================
💬 RESPOSTA:
============================================================
Não tenho informações necessárias para responder sua pergunta.
============================================================

💬 Sua pergunta: sair
👋 Encerrando chat. Até logo!
```

### Comandos do Chat

| Comando | Descrição              |
| ------- | ---------------------- |
| `help`  | Mostra ajuda           |
| `info`  | Informações do sistema |
| `clear` | Limpa a tela           |
| `sair`  | Encerra o chat         |

## ⚙️ Configuração Multi-LLM

O sistema suporta dois provedores de LLM. Edite o `.env` para trocar:

### Opção 1: OpenAI (GPT)

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-sua_chave_aqui
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o-mini
```

### Opção 2: Google Gemini

```env
LLM_PROVIDER=google
GOOGLE_API_KEY=AIza_sua_chave_aqui
GOOGLE_EMBEDDING_MODEL=models/embedding-001
GOOGLE_CHAT_MODEL=gemini-2.0-flash-exp
```

## 📁 Estrutura do Projeto

```
projeto1/
├── src/
│   ├── config.py              # Configurações centralizadas
│   ├── llm_factory.py         # Factory para criar LLMs
│   ├── ingest.py              # Script de ingestão de PDF
│   ├── search.py              # Serviço de busca semântica
│   ├── chat.py                # Interface CLI interativa
│   └── utils/
│       ├── logger.py          # Sistema de logging
│       └── database.py        # Utilitários de banco
├── tests/
│   ├── test_llm_factory.py    # Testes do factory de LLM
│   └── utils/
│       └── test_database.py   # Testes de utilitários de banco
├── pytest.ini                 # Configuração do pytest (pythonpath/tests)
├── docker-compose.yml         # PostgreSQL + pgVector
├── requirements.txt           # Dependências Python
├── .env.example               # Template de configuração
├── .env                       # Configuração (não versionar)
└── README.md                  # Este arquivo
```

## 🔧 Parâmetros Configuráveis

Todas as configurações estão no arquivo `.env`:

| Variável        | Descrição                              | Padrão   |
| --------------- | -------------------------------------- | -------- |
| `LLM_PROVIDER`  | Provider de LLM (`openai` ou `google`) | `openai` |
| `CHUNK_SIZE`    | Tamanho dos chunks em caracteres       | `1000`   |
| `CHUNK_OVERLAP` | Sobreposição entre chunks              | `150`    |
| `SEARCH_K`      | Número de documentos similares         | `10`     |
| `LOG_LEVEL`     | Nível de log (DEBUG, INFO, etc.)       | `ERROR`  |

## 🧪 Testes

Os testes automatizados vivem em `tests/` e utilizam `pytest`.

### Instalar dependências de desenvolvimento

**Usando UV (recomendado):**

```bash
uv sync
```

**Ou sincronizando via requirements.txt:**

```bash
uv pip install -r requirements.txt
```

### Executar a suíte completa

```bash
uv run pytest
```

### Executar com relatório de cobertura (opcional)

```bash
uv run pytest --cov=src --cov-report=term-missing
```

### Ajuste manual de PYTHONPATH (apenas se ignorar o pytest.ini)

```bash
# Linux/macOS
PYTHONPATH=src uv run pytest -q

# Windows (PowerShell)
$env:PYTHONPATH="src"; uv run pytest -q
```

## 🐛 Troubleshooting

### Erro: "No module named 'dotenv'"

**Usando UV:**

```bash
uv pip install -r requirements.txt
uv sync
```

**Ou usando Python/pip padrão:**

```bash
python -m pip install -r requirements.txt
```

### Erro: "pgvector extension not found"

```bash
docker exec postgres_rag psql -U postgres -d rag -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Erro: "API key not configured"

Verifique se o arquivo `.env` tem a chave correta:

```bash
cat .env  # Linux/Mac
type .env # Windows
```

### Banco de dados não conecta

```bash
# Reiniciar containers
docker-compose down
docker-compose up -d

# Ver logs
docker-compose logs postgres
```

### Collection vazia (sem documentos)

Execute a ingestão primeiro:

**Usando UV:**

```bash
uv run python src/ingest.py seu_arquivo.pdf
```

**Ou usando Python diretamente:**

```bash
python src/ingest.py seu_arquivo.pdf
```

## 📚 Pacotes Principais

| Pacote                   | Versão | Uso                     |
| ------------------------ | ------ | ----------------------- |
| `langchain`              | 0.3.27 | Framework principal     |
| `langchain-openai`       | 0.3.30 | Integração OpenAI       |
| `langchain-google-genai` | 2.1.9  | Integração Google       |
| `langchain-postgres`     | 0.0.15 | Vector store PostgreSQL |
| `pypdf`                  | 6.0.0  | Leitura de PDF          |
| `psycopg2-binary`        | 2.9.10 | Driver PostgreSQL       |
| `pgvector`               | 0.3.6  | Extensão de vetores     |

## 🎓 Conceitos Implementados

- **RAG (Retrieval-Augmented Generation)**: Combina busca semântica com LLM
- **Vector Database**: Armazenamento eficiente de embeddings
- **Semantic Search**: Busca por similaridade vetorial
- **Factory Pattern**: Criação de objetos LLM de forma flexível
- **Dependency Injection**: Configuração centralizada
- **Async/Await**: Operações assíncronas para melhor performance

## 📄 Licença

Este projeto foi desenvolvido para fins educacionais para o Desafio MBA Engenharia de Software com IA - Full Cycle.

## 📞 Suporte

Se encontrar problemas:

1. Verifique a seção [Troubleshooting](#-troubleshooting)
2. Use `--debug` para ver logs detalhados
3. Verifique os logs do Docker: `docker-compose logs`

---

**Desenvolvido com ❤️ usando Python, LangChain, PostgreSQL e muita IA**

## Requisitos

### 1. Ingestão do PDF

- O PDF deve ser dividido em chunks de 1000 caracteres com overlap de 150.
- Cada chunk deve ser convertido em embedding.
- Os vetores devem ser armazenados no banco de dados PostgreSQL com pgVector.

### 2. Consulta via CLI

- Criar um script Python para simular um chat no terminal.
- Passos ao receber uma pergunta:
  1. Vetorizar a pergunta.
  2. Buscar os 10 resultados mais relevantes (k=10) no banco vetorial.
  3. Montar o prompt e chamar a LLM.
  4. Retornar a resposta ao usuário.

Prompt a ser utilizado:

```
CONTEXTO:
{resultados concatenados do banco de dados}

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
{pergunta do usuário}

RESPONDA A "PERGUNTA DO USUÁRIO"
```

## Estrutura obrigatória do projeto

```
├── docker-compose.yml
├── requirements.txt      # Dependências
├── .env.example          # Template da variável OPENAI_API_KEY
├── src/
│   ├── ingest.py         # Script de ingestão do PDF
│   ├── search.py         # Script de busca
│   ├── chat.py           # CLI para interação com usuário
├── document.pdf          # PDF para ingestão
└── README.md             # Instruções de execução
```
