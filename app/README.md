# 🎯 Sistema de Deteção de Objetos Industriais - Aplicação de Demonstração

Este diretório contém a aplicação web de demonstração desenvolvida em **Streamlit** para o Trabalho Prático da Unidade Curricular de **Inteligência Artificial** do curso de **Licenciatura em Engenharia Informática** (Escola Superior de Tecnologia e Gestão - P.PORTO).

A interface responde na íntegra a todos os critérios e funcionalidades obrigatórias (e bónus) estipulados no enunciado , permitindo carregar imagens , testar amostras aleatórias do dataset local e capturar imagens em tempo real através da câmara do telemóvel para realizar inferências com os modelos YOLOv8 e YOLO11 treinados.

## 🛠️ Instalação

### 1. Instalar as dependências

Abrir o terminal na pasta `app/` e executar:

```bash
pip install -r requirements.txt
```

Ou manualmente:

```bash
pip install streamlit ultralytics opencv-python numpy pandas pillow
```

---

## 🚀 Como Executar

### Execução Local

```bash
python -m streamlit run app6.py
```

A aplicação abre automaticamente no navegador em `http://localhost:8501`.

> Recomenda-se utilizar `python -m streamlit` em vez de `streamlit` diretamente, para evitar erros de bloqueio de segurança no Windows (política Device Guard).

---

### Acesso pelo Telemóvel via ngrok

Para utilizar a câmara do telemóvel, é necessário criar um túnel HTTPS com o ngrok, uma vez que os navegadores modernos bloqueiam o acesso à câmara em ligações HTTP normais.

**Configuração inicial (apenas uma vez):**

**1.** Aceder a [ngrok.com](https://ngrok.com) no navegador e clicar em **Sign up** para criar uma conta gratuita

**2.** Após iniciar sessão, aceder ao dashboard e copiar o **authtoken** pessoal

**3.** Instalar o ngrok através do terminal:
```bash
winget install ngrok.ngrok
```
> Fechar e reabrir o terminal após a instalação

**4.** Configurar o token de autenticação:
```bash
ngrok config add-authtoken O_TEU_TOKEN_AQUI
```

**Utilização (cada sessão):**

**5.** Abrir dois terminais em simultâneo:

```bash
# Terminal 1 — iniciar a aplicação
python -m streamlit run app6.py

# Terminal 2 — criar o túnel seguro
ngrok http 8501
```

**6.** No Terminal 2, identificar a linha **Forwarding** e copiar o endereço público `https://xxxx.ngrok-free.app`

**7.** Abrir esse endereço no navegador do telemóvel → aceder ao separador **Câmara** → aceitar a permissão de acesso à câmara

---

## 📊 Funcionalidades Implementadas

| Separador | Descrição |
|---|---|
| 📷 **Imagem** | Carregamento de imagem com deteção de objetos, bounding boxes, etiquetas e nível de confiança |
| 🎥 **Câmara** | Captura de frames em tempo real — utiliza a câmara do telemóvel via ngrok |
| 📁 **Dataset** | Carregamento de imagens aleatórias da pasta de teste do dataset |
| ⚖️ **Comparar Modelos** ⭐ | Comparação de dois modelos em simultâneo sobre a mesma imagem |
| 📜 **Histórico** ⭐ | Registo das últimas N inferências com exportação para JSON |

**Painel Lateral (Sidebar):**
- Seleção do modelo — quatro modelos treinados disponíveis
- Controlo deslizante (slider) para o limiar de confiança mínima

**Saída Estruturada:** tabela Pandas e bloco JSON detalhado para cada inferência

---

## 🤖 Modelos Treinados

| Ficheiro | Arquitetura | Velocidade |
|---|---|---|
| `yolo8s_best.pt` | YOLOv8 Small | Rápido |
| `yolo8m_best.pt` | YOLOv8 Medium | Equilibrado |
| `yolo11s_best.pt` | YOLO11 Small | Rápido |
| `yolo11m_best.pt` | YOLO11 Medium | Equilibrado |

> Os ficheiros `.pt` já contêm a arquitetura e os pesos treinados — não é necessário instalar nenhuma dependência adicional relativa aos modelos.

> ⚠️ **Importante:** Os quatro ficheiros `.pt` devem estar obrigatoriamente na mesma pasta que o `app6.py`, caso contrário a aplicação não consegue carregar os modelos.

---

## 📦 Conteúdo do requirements.txt

```
streamlit
ultralytics
opencv-python
numpy
pandas
pillow
```
