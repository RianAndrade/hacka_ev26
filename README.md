# hackadashs
**hackadashs** é uma aplicação criada para enfrentar um problema recorrente na gestão de saúde pública: **a dificuldade de antecipar picos de internações hospitalares**.

O sistema utiliza **dados históricos de internações** para gerar **análises e previsões de ocupação hospitalar**. Além das previsões, a plataforma também oferece **dashboards com dados de anos anteriores**, permitindo que gestores identifiquem padrões históricos e tomem decisões mais informadas sobre planejamento de capacidade e distribuição de recursos.

---

# 🚀 Funcionalidades do sistema

A aplicação oferece um fluxo completo de análise de dados hospitalares, incluindo:

- Importação de registros hospitalares via arquivos CSV  
- Armazenamento estruturado dos dados em PostgreSQL  
- Consolidação do histórico de internações por semana epidemiológica  
- Geração de previsões de admissões hospitalares por unidade de saúde  
- Consulta de hospitais com previsões disponíveis  
- Consulta detalhada de previsões por hospital  
- Dashboard de visualização de previsões  
- Dashboard histórico com filtros analíticos  
- Área administrativa para execução de testes e simulações  
- Processamento assíncrono de tarefas utilizando Celery e Redis  

O objetivo principal do projeto é transformar **dados brutos de internação em informações estratégicas**, capazes de apoiar decisões relacionadas ao planejamento hospitalar e à gestão da capacidade da rede de saúde.

---

# 🛠 Tecnologias utilizadas

O sistema foi desenvolvido utilizando as seguintes tecnologias:

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

Antes de executar o projeto, é necessário ter instalado no sistema:

- Docker  
- Docker Compose  

Portas utilizadas pela aplicação:

- **8000** — API Django  
- **5432** — Banco de dados PostgreSQL  
- **6379** — Redis  

---

# 🧰 Variáveis de ambiente

Para facilitar a execução durante o hackathon, o arquivo `.env` já está incluído no projeto.

Isso permite iniciar o sistema rapidamente sem necessidade de configuração adicional.

Exemplo de configuração:

DJANGO_SECRET_KEY=django-insecure-6q3n9r2f0k8m1v5p7s4t9x2y6z1a3b8c5d0e7f2h9j4k1m  
DJANGO_DEBUG=1  
DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1  

DJANGO_PORT=8000  

POSTGRES_HOST=term_epidemic_db  
POSTGRES_PORT=5432  
POSTGRES_DB=term_epidemic  
POSTGRES_USER=term_epidemic  
POSTGRES_PASSWORD=term_epidemic  

DATABASE_URL=postgresql://term_epidemic:term_epidemic@db:5432/term_epidemic  

CELERY_BROKER_URL=redis://redis:6379/0  
CELERY_RESULT_BACKEND=redis://redis:6379/1  

---

# 🐳 Como rodar o projeto

### 1. Clonar o repositório

git clone git@github.com:RianAndrade/hackadashs.git

### 2. Entrar na pasta do projeto

cd hackadashs

### 3. Subir os containers

docker compose up --build

Na primeira execução, o processo pode demorar alguns minutos, pois o Docker irá:

- baixar as imagens necessárias  
- construir o container da API  
- instalar as dependências do projeto  
- aplicar as migrations do banco  
- importar os dados iniciais  
- executar o algoritmo de previsão  

Quando a inicialização terminar, a seguinte mensagem aparecerá no terminal:

| March 06, 2026 - 14:26:35  
| Django version 6.0.3, using settings 'config.settings'  
| Starting development server at http://0.0.0.0:8000/  
| Quit the server with CONTROL-C.

Durante a inicialização, o Docker automaticamente irá:

- iniciar o PostgreSQL  
- iniciar o Redis  
- aplicar as migrations do Django  
- iniciar o servidor da API  
- iniciar o worker do Celery  

---

# 🌐 Como acessar o sistema

Após os containers estarem em execução, o sistema poderá ser acessado pelos seguintes endereços:

### Dashboard de previsão
http://localhost:8000/hospital-occupancy-dashboard/

### Dashboard histórico
http://localhost:8000/hospital-admissions/historical/dashboard/

### Área administrativa
http://localhost:8000/hospital-occupancy-admin/

---

# 📌 Observações

O projeto foi desenvolvido com foco em **demonstração e testes rápidos durante um hackathon**.  
Por esse motivo, algumas decisões de arquitetura priorizaram simplicidade e facilidade de execução.

Entre as principais escolhas do projeto estão:

- ambiente completamente configurado com Docker  
- importação automática de dados via CSV durante a inicialização  
- execução rápida de previsões  
- dashboards para visualização imediata dos resultados  
- navegação simples entre os painéis do sistema  

Essas características permitem que qualquer avaliador consiga **executar, testar e compreender o funcionamento do sistema rapidamente**, sem necessidade de configurações complexas.