# TermEpidemi

**TermEpidemi** é uma aplicação desenvolvida para análise e previsão de ocupação hospitalar com base em dados históricos de internações.  
O sistema foi criado no contexto de **hackathon**, com foco em execução rápida, testes práticos e visualização clara dos resultados.

A aplicação permite importar dados hospitalares, gerar análises históricas e executar previsões de admissões hospitalares por unidade de saúde, auxiliando na identificação de padrões e antecipação de picos de demanda.

---

# 🚀 O que o sistema faz

A aplicação fornece um fluxo completo para análise de dados hospitalares:

- Importação de registros hospitalares via CSV  
- Armazenamento estruturado em PostgreSQL  
- Consolidação de histórico de internações por semana  
- Geração de previsões de admissões hospitalares  
- Consulta de hospitais com previsão disponível  
- Consulta de previsões por hospital  
- Dashboard de visualização de previsão  
- Dashboard histórico com filtros analíticos  
- Área administrativa para execução de testes e simulações  
- Processamento assíncrono usando Celery e Redis  

O objetivo do projeto é transformar dados brutos de internação em **informação útil para planejamento hospitalar**.

---

# 🛠 Tecnologias utilizadas

- Python 3.12  
- Django 6.0.3  
- Django REST Framework 3.16.1  
- PostgreSQL 16  
- Celery 5.5.3  
- Redis 7  
- Pandas 2.3.2  
- NumPy 1.26.4  
- scikit-learn 1.5.1  
- joblib 1.4.2  
- Docker  
- Docker Compose  

---

# 📦 Pré-requisitos

Antes de rodar o projeto, é necessário ter instalado:

- Docker  
- Docker Compose  

Portas utilizadas pelo sistema:

- **8000** — API Django  
- **5432** — PostgreSQL  
- **6379** — Redis  

---

# 🧰 Variáveis de ambiente

Para facilitar a execução durante o hackathon, o arquivo `.env` já foi incluído no projeto.

Isso permite executar o sistema rapidamente sem configuração adicional.

Exemplo:

```env
DJANGO_SECRET_KEY=django-insecure-6q3n9r2f0k8m1v5p7s4t9x2y6z1a3b8c5d0e7f2h9j4k1m
DJANGO_DEBUG=1
DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1

DJANGO_PORT=8000

POSTGRES_HOST=hackadashs_db
POSTGRES_PORT=5432
POSTGRES_DB=hackadashs
POSTGRES_USER=hackadashs
POSTGRES_PASSWORD=hackadashs

DATABASE_URL=postgresql://hackadashs:hackadashs@db:5432/hackadashs

CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/1
```

---

# 🐳 Como rodar o projeto

### 1. Clonar o repositório

```bash
git clone git@github.com:RianAndrade/TermEpidemi.git
```

### 2. Entrar na pasta do projeto

```bash
cd TermEpidemi
```

### 3. Subir os containers

```bash
docker compose up --build
```

Na  primeira vez vai demorar um pouco, pois o Docker irá baixar as imagens, construir o container da API e instalar as dependências popular as tabelas e rodar o algortimo de predição. Quando terminar aparecerá no seu terminal a mensagem: 

```
| March 06, 2026 - 14:26:35```
| Django version 6.0.3, using settings 'config.settings'
| Starting development server at http://0.0.0.0:8000/
| Quit the server with CONTROL-C.
```

O Docker irá automaticamente:

- iniciar o PostgreSQL  
- iniciar o Redis  
- aplicar as migrations do Django  
- iniciar o servidor da API  
- iniciar o worker do Celery  

---

# 🌐 Como acessar o projeto

Depois que os containers estiverem rodando:


### Dashboard de previsão
```
http://localhost:8000/hospital-occupancy-dashboard/
```

### Dashboard histórico
```
http://localhost:8000/hospital-admissions/historical/dashboard/
```

### Área administrativa
```
http://localhost:8000/hospital-occupancy-admin/
```


---



# 📌 Observações

O projeto foi desenvolvido com foco em **demonstração e testes rápidos** durante hackathon.

Por isso foram priorizados:

- ambiente pronto com Docker  
- importação simples via CSV automazida na inicialização
- execução rápida de previsões  
- dashboards para visualização dos resultados  
- Paineis com facil acesso e navegação

Essas escolhas permitem que qualquer avaliador consiga rodar e testar o sistema rapidamente.
<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->
