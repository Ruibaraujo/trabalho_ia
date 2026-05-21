# Sistema de Deteção de Objetos Industriais — Aplicação de Demonstração

Aplicação web em **Streamlit** desenvolvida no âmbito do Trabalho Prático da Unidade Curricular de **Inteligência Artificial** do curso de **Licenciatura em Engenharia Informática** (Escola Superior de Tecnologia e Gestão — P.PORTO).

A interface responde a todos os requisitos obrigatórios e bónus do enunciado (D4): upload de imagem, captura via câmara (foto e em tempo real), seleção de modelo, inferência com bounding boxes, vistas estruturadas (JSON + tabela), histórico, exportação, comparação de modelos e slider de confiança.

---

## Pré-requisitos

- **Python 3.10** ou superior
- Pelo menos um modelo treinado (ficheiro `.pt`) numa subpasta de `../modelos/`
- *(Opcional, só para acesso pelo telemóvel)* `cloudflared`

## Estrutura esperada

A app descobre os modelos automaticamente a partir da pasta `../modelos/` (relativa ao `app.py`). Cada modelo deve estar numa subpasta contendo, no mínimo, um ficheiro `.pt` (idealmente chamado `best.pt`):

```
submissao/
├── app/
│   ├── app.py
│   ├── README.md
│   └── requirements.txt
├── dataset/
│   └── ... (test/valid/train + data.yaml)
└── modelos/
    ├── yolov11m/
    │   ├── best.pt                       # necessário para a app
    │   ├── model_card_yolov11m.md        # entregável D3 — não lido pela app
    │   └── model_manifest_yolov11m.json  # entregável D3 — opcionalmente lido pela app (ver nota abaixo)
    ├── yolov11s/
    │   └── ...
    └── ... (uma subpasta por modelo)
```

Adicionar mais um modelo no futuro é tão simples como criar mais uma subpasta com um `.pt` lá dentro e dar refresh à app — o dropdown atualiza-se automaticamente. O nome da pasta é, por defeito, a label que aparece no dropdown. Para sobrescrever essa label sem renomear a pasta, basta adicionar um campo `"display_name": "..."` ao respetivo `model_manifest_*.json`.

---

## Instalação

### 1. Dependências Python (todas as plataformas)

A partir da pasta `app/`:

```bash
pip install -r requirements.txt
```

> Em macOS e Linux, se tiveres ambas as versões do Python instaladas, podes precisar de `pip3` em vez de `pip`.

### 2. `cloudflared` (opcional)

Só é necessário para aceder à app pela câmara do **telemóvel**. Para uso só no PC, ignora este passo.

**Linux (Fedora / RHEL / Ubuntu / Debian — qualquer distro):**

Caminho universal por binário direto, evita as diferenças entre dnf/dnf5/apt:

```bash
sudo curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared
sudo chmod +x /usr/local/bin/cloudflared
cloudflared --version
```

**macOS:**

Via Homebrew (recomendado):

```bash
brew install cloudflared
cloudflared --version
```

Caso não tenhas Homebrew, instala-o primeiro em [brew.sh](https://brew.sh). Para Apple Silicon (M1/M2/M3/M4) o Homebrew deteta a arquitetura automaticamente — não é preciso configurar nada à mão.

**Windows:**

Via winget (vem com Windows 10/11):

```powershell
winget install Cloudflare.cloudflared
```

Fechar e reabrir o terminal após a instalação para o PATH ser recarregado.

---

## Execução

### Local (PC)

A partir da pasta `app/`:

```bash
python -m streamlit run app.py
```

A app abre automaticamente em `http://localhost:8501`.

> Recomenda-se `python -m streamlit ...` em vez de `streamlit ...` diretamente, para evitar problemas no Windows com a política Device Guard.

A câmara do **PC** (`tab Câmara → Em tempo real`) funciona em `localhost` sem mais configuração — o browser considera `localhost` um contexto seguro.

### Acesso pelo telemóvel

A câmara do **telemóvel** requer HTTPS (limitação dos browsers modernos). O caminho mais simples é um túnel HTTPS gratuito do Cloudflare:

1. No PC, num **segundo terminal** (deixa o primeiro a correr a app):

   ```bash
   cloudflared tunnel --url http://localhost:8501
   ```

2. Na saída do comando, procura a linha com a URL pública:

   ```
   Your quick Tunnel has been created! Visit it at (it may take some time to be reachable):
   https://palavras-aleatorias-1234.trycloudflare.com
   ```

3. Na app (browser do PC): tab **Câmara** → sub-tab **Em tempo real** → abrir o expander **Aceder do telemóvel (código QR)** → colar a URL no campo de texto.

4. A app desenha um QR code com essa URL. Apontar a câmara do telemóvel ao QR — abre a app no browser do telemóvel.

5. O telemóvel pede autorização de câmara da primeira vez. Aceitar.

> O túnel só vive enquanto o terminal estiver aberto. Se o fechares, a URL deixa de funcionar e tens de gerar uma nova (que será diferente).

---

## Como funciona o modo telemóvel

Os modelos correm **no PC**, não no telemóvel. O telemóvel funciona apenas como sensor (câmara) e ecrã. O fluxo de cada frame é:

> câmara do telemóvel → frame enviado por WebRTC para o PC → YOLO corre no PC e desenha as bounding boxes → frame anotado é enviado de volta para o telemóvel → telemóvel mostra o resultado em direto.

Implicações práticas:

- O **telemóvel não precisa de ser potente**. Qualquer Android/iPhone dos últimos anos chega.
- O **PC faz o trabalho pesado**. Os FPS dependem da CPU/GPU do PC.
- Quando ambos os dispositivos estão na **mesma rede WiFi**, a ligação WebRTC tende a estabelecer-se peer-to-peer através da rede local — o túnel cloudflared só é usado para o aperto de mão inicial (signaling), não para o vídeo em si.
- A app pede preferencialmente a **câmara traseira** do telemóvel (mais útil para a tarefa de deteção), mas se só houver a frontal disponível usa essa.

---

## Funcionalidades

| Tab | Sub-tab | Descrição |
|---|---|---|
| **Imagem** | — | Upload de imagem com inferência, bounding boxes, etiquetas e confiança |
| **Câmara** | Em tempo real | Stream WebRTC com inferência frame a frame, em direto. Botão de snapshot para gerar vista estruturada. Suporta câmara do PC e câmara do telemóvel (via cloudflared + QR) |
| **Câmara** | Tirar foto | Captura única via câmara e inferência sobre essa foto |
| **Dataset** | — | Seleção aleatória de imagens do split de teste do dataset |
| **Comparar Modelos** | Upload de imagem | Dois modelos em paralelo sobre a mesma imagem carregada |
| **Comparar Modelos** | Tirar foto | Dois modelos em paralelo sobre uma foto tirada na hora |
| **Histórico** | — | Últimas N inferências guardadas; exportação JSON completa |

**Barra lateral:**

- **Modelo** — dropdown populado automaticamente a partir das subpastas de `../modelos/`
- **Limiar de confiança** — slider de `0.00` a `1.00`, aplicado a todas as inferências
- **Máximo de inferências guardadas** — controlo do tamanho do histórico

**Saída estruturada (por inferência):** estatísticas (totais e por classe), tabela de deteções (classe / confiança / bounding box), e bloco JSON exportável.

---

## Notas por plataforma

### macOS

- Para o browser ter acesso à webcam ou câmara, pode ser preciso autorizar manualmente em **Definições do Sistema → Privacidade e Segurança → Câmara** (ativar Chrome / Safari / o browser que estás a usar).
- Em Apple Silicon (M1/M2/M3/M4), todas as dependências têm wheels nativos arm64 — o `pip install` é direto, sem compilação.
- Se a instalação do `av` (PyAV) falhar com erro de compilação (improvável em versões recentes do Python), instalar previamente as dependências de sistema com `brew install ffmpeg pkg-config` resolve.
- Se a primeira execução de `cloudflared` for bloqueada pelo Gatekeeper (só acontece com binário descarregado à mão, não com `brew`), correr `xattr -d com.apple.quarantine $(which cloudflared)`.

### Linux

- Pode ser necessário ter `python3-venv` ou equivalente para criar ambientes isolados (recomendado: `python3 -m venv .venv && source .venv/bin/activate`).
- Em distros com SELinux ativo (Fedora, RHEL), não há nada de especial a fazer — o Streamlit corre em modo utilizador normal.

### Windows

- Usar PowerShell ou cmd. O Windows Terminal é o mais confortável.
- Se `streamlit` não for reconhecido após `pip install`, usar `python -m streamlit ...` (já é a forma recomendada acima).
- Para o `cloudflared`, fechar e reabrir o terminal depois do `winget install` para o PATH ser recarregado.

---

## Troubleshooting

**`UnpicklingError: Weights only load failed` ao selecionar um modelo:**
Indica que a `ultralytics` instalada é demasiado antiga para o `torch` instalado (problema do `weights_only=True` introduzido no PyTorch 2.6). Atualizar:

```bash
pip install --upgrade ultralytics
```

E reiniciar o Streamlit (Ctrl+C no terminal e arrancar novamente — o cache de modelos do Streamlit fica com o resultado falhado).

**Nenhum modelo aparece no dropdown:**
A app procura subpastas de `../modelos/` que contenham um ficheiro `.pt`. Verificar a estrutura de pastas descrita acima.

**Stream do telemóvel não conecta ou fica negro:**
- Confirmar que o terminal do `cloudflared` continua aberto e a URL ainda é válida.
- Confirmar que aceitaste a permissão de câmara no browser do telemóvel.
- Se o telemóvel está em rede móvel (4G/5G) e o PC numa WiFi diferente, a ligação WebRTC pode demorar mais ou falhar em alguns operadores com NAT agressivo. Solução: pôr os dois na mesma WiFi.

**A app no telemóvel mostra a câmara frontal em vez da traseira:**
A app pede preferencialmente a câmara traseira (`facingMode: environment`), mas algumas combinações de browser e telemóvel ignoram essa preferência. Nesses casos, dentro do widget de vídeo do Streamlit, há um seletor de câmara que permite trocar manualmente.
