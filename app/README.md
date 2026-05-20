# 🎯 Sistema de Deteção de Objetos — Aplicação de Demonstração

Aplicação web desenvolvida em **Streamlit** para o Trabalho Prático da Unidade Curricular de **Inteligência Artificial** — Licenciatura em Engenharia Informática (ESTG - P.PORTO).

---

## 📁 Estrutura da Pasta

```
app/
├── app6.py              # Ficheiro principal da aplicação
├── README.md            # Este ficheiro
├── requirements.txt     # Dependências
├── yolo8s_best.pt       # Modelo YOLOv8 Small treinado
├── yolo8m_best.pt       # Modelo YOLOv8 Medium treinado
├── yolo11s_best.pt      # Modelo YOLO11 Small treinado
└── yolo11m_best.pt      # Modelo YOLO11 Medium treinado
```

---

## 🛠️ Instalação

### 1. Instalar as dependências

```bash
pip install -r requirements.txt
```

Ou manualmente:

```bash
pip install streamlit ultralytics opencv-python numpy pandas pillow
```

---

## 🚀 Como Correr

### Execução local

```bash
python -m streamlit run app6.py
```

A aplicação abre automaticamente no browser em `http://localhost:8501`.

> Usar `python -m streamlit` em vez de `streamlit` diretamente evita erros de bloqueio de segurança no Windows (Device Guard).

---

### Acesso pelo telemóvel via ngrok

Para usar a câmara do telemóvel é necessário criar um túnel HTTPS com o ngrok (os browsers bloqueiam acesso à câmara em ligações HTTP normais).

**Passo a passo:**

**1.** Cria uma conta gratuita no browser em [ngrok.com](https://ngrok.com) → clica em **Sign up**

**2.** Após login, vai ao dashboard e copia o teu **authtoken**

**3.** Instala o ngrok pelo terminal:
```bash
winget install ngrok.ngrok
```
> Fecha e reabre o terminal após instalar

**4.** Configura o token (só uma vez):
```bash
ngrok config add-authtoken O_TEU_TOKEN_AQUI
```

**3.** Abre dois terminais:

```bash
# Terminal 1 — corre a app
python -m streamlit run app6.py

# Terminal 2 — cria o túnel
ngrok http 8501
```

**4.** Copia o link `https://xxxx.ngrok-free.app` que aparece no Terminal 2

**5.** Abre esse link no browser do telemóvel → vai ao separador **Câmara** → aceita a permissão da câmara

---

## 📊 Funcionalidades

| Separador | Descrição |
|---|---|
| 📷 **Imagem** | Carrega uma imagem e deteta objetos com bounding boxes, etiquetas e confiança |
| 🎥 **Câmara** | Captura frames em tempo real — usa câmara do telemóvel via ngrok |
| 📁 **Dataset** | Carrega imagens aleatórias da pasta de teste do dataset |
| ⚖️ **Comparar Modelos** ⭐ | Compara dois modelos lado a lado sobre a mesma imagem |
| 📜 **Histórico** ⭐ | Registo das últimas N inferências com exportação JSON |

**Sidebar:**
- Seleção do modelo (4 modelos treinados)
- Slider de confiança mínima ajustável

**Saída estruturada:** tabela Pandas + bloco JSON para cada inferência

---

## 🤖 Modelos Treinados

| Ficheiro | Arquitetura | Tamanho |
|---|---|---|
| `yolo8s_best.pt` | YOLOv8 Small | Rápido |
| `yolo8m_best.pt` | YOLOv8 Medium | Equilibrado |
| `yolo11s_best.pt` | YOLO11 Small | Rápido |
| `yolo11m_best.pt` | YOLO11 Medium | Equilibrado |

> Os ficheiros `.pt` já contêm a arquitetura e os pesos treinados — não é necessário instalar nada adicional.

---
