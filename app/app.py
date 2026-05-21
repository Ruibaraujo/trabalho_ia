import streamlit as st
import cv2
import os
import json
import random
import threading
import numpy as np
import pandas as pd
import av
import qrcode
from io import BytesIO
from ultralytics import YOLO
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase


# Configuração da página
st.set_page_config(
    page_title="Deteção de Objetos - Projeto IA",
    layout="wide"
)

st.title("Sistema de Deteção de Objetos")
st.markdown("**Projeto de Inteligência Artificial**")
st.markdown("---")

# ============================================
# DESCOBERTA DINÂMICA DE MODELOS
# ============================================
# Pasta-raiz onde estão os modelos. Cada modelo é uma subpasta contendo:
#   - um ficheiro .pt (preferencialmente 'best.pt')
#   - opcionalmente um model_manifest_*.json (para nome amigável)
#   - opcionalmente um model_card_*.md
# Resolvido a partir do próprio app6.py, portanto funciona independentemente
# do CWD em que o Streamlit é arrancado.
MODELS_DIR = Path(__file__).resolve().parent.parent / "modelos"


def _ler_manifesto(pasta_modelo: Path) -> dict:
    """Lê o primeiro model_manifest_*.json encontrado na pasta. {} se não houver ou falhar."""
    for candidato in pasta_modelo.glob("model_manifest_*.json"):
        try:
            with open(candidato, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _label_dropdown(pasta_modelo: Path, manifesto: dict) -> str:
    """
    Label do dropdown. Por defeito usa o nome da pasta (curto e distintivo).
    Para sobrescrever sem renomear a pasta, adicionar um campo 'display_name'
    ao model_manifest_*.json. O campo 'name' do manifesto é deliberadamente
    ignorado por costumar conter o identificador longo do projeto.
    """
    return manifesto.get("display_name") or pasta_modelo.name


def descobrir_modelos(raiz: Path) -> dict:
    """
    Varre as subpastas de 'raiz' à procura de modelos. Devolve um dict
    label_dropdown -> caminho_absoluto_do_.pt.
    """
    raiz = Path(raiz)
    if not raiz.exists() or not raiz.is_dir():
        return {}

    modelos = {}
    for pasta in sorted(raiz.iterdir()):
        if not pasta.is_dir():
            continue
        pt_files = sorted(pasta.glob("*.pt"))
        if not pt_files:
            continue
        # Prefere 'best.pt'; caso contrário usa o primeiro .pt por ordem alfabética
        peso = next((p for p in pt_files if p.name == "best.pt"), pt_files[0])

        manifesto = _ler_manifesto(pasta)
        label = _label_dropdown(pasta, manifesto)

        # Anti-colisão: improvável, mas seguro
        label_final, n = label, 2
        while label_final in modelos:
            label_final = f"{label} ({n})"
            n += 1

        modelos[label_final] = str(peso)

    return modelos


modelos_disponiveis = descobrir_modelos(MODELS_DIR)

if not modelos_disponiveis:
    st.error(
        f"Nenhum modelo encontrado em `{MODELS_DIR}`.\n\n"
        "Cada modelo deve estar numa subpasta contendo pelo menos um ficheiro `.pt` "
        "(idealmente `best.pt`)."
    )
    st.stop()

# ============================================
# CONFIGURAÇÕES NA SIDEBAR
# ============================================
st.sidebar.header("Configurações")

# Seleção do modelo
st.sidebar.subheader("Modelo")

modelo_selecionado = st.sidebar.selectbox(
    "Escolhe o modelo:",
    list(modelos_disponiveis.keys())
)

# Slider de confiança
limiar_confianca = st.sidebar.slider(
    "Limiar de confiança mínimo:",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)

# Histórico
st.sidebar.subheader("Histórico")
max_historico = st.sidebar.number_input(
    "Máximo de inferências guardadas:",
    min_value=1,
    max_value=50,
    value=10
)

# ============================================
# INICIALIZAÇÃO
# ============================================

@st.cache_resource
def carregar_modelo(caminho):
    if not os.path.exists(caminho):
        st.error(f"Ficheiro não existe: `{caminho}`")
        return None
    try:
        return YOLO(caminho)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        st.error(
            f"**Erro ao carregar modelo**\n\n"
            f"**Caminho:** `{caminho}`\n\n"
            f"**Tipo:** `{type(e).__name__}`\n\n"
            f"**Mensagem:** {e}"
        )
        with st.expander("Traceback completo"):
            st.code(tb)
        return None

modelo = carregar_modelo(modelos_disponiveis[modelo_selecionado])

if "historico" not in st.session_state:
    st.session_state.historico = []

# ============================================
# FUNÇÕES
# ============================================

def fazer_inferencia(imagem, modelo, limiar):
    results = modelo(imagem, conf=limiar, verbose=False)
    detecoes = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        class_name = modelo.names[class_id]
        conf = float(box.conf[0])
        detecoes.append({
            "classe": class_name,
            "confianca": round(conf, 4),
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        })
    return results, detecoes

def desenhar_detecoes(imagem, detecoes):
    img_copy = imagem.copy()
    cores = {}
    for det in detecoes:
        classe = det["classe"]
        if classe not in cores:
            np.random.seed(hash(classe) % 2**32)
            cores[classe] = tuple(np.random.randint(50, 255, 3).tolist())
        cor = cores[classe]
        bbox = det["bbox"]
        cv2.rectangle(img_copy, (bbox["x1"], bbox["y1"]), (bbox["x2"], bbox["y2"]), cor, 2)
        label = f"{classe}: {det['confianca']:.1%}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_copy, (bbox["x1"], bbox["y1"]-25), (bbox["x1"]+w, bbox["y1"]), cor, -1)
        cv2.putText(img_copy, label, (bbox["x1"], bbox["y1"]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img_copy

def calcular_estatisticas(detecoes):
    if not detecoes:
        return {}
    contagem = defaultdict(int)
    confianca_lista = defaultdict(list)
    for det in detecoes:
        contagem[det["classe"]] += 1
        confianca_lista[det["classe"]].append(det["confianca"])
    stats = {}
    for classe in contagem:
        stats[classe] = {
            "quantidade": contagem[classe],
            "confianca_media": round(np.mean(confianca_lista[classe]), 4)
        }
    return stats

def guardar_historico(fonte, detecoes, stats):
    entrada = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "fonte": fonte,
        "modelo": modelo_selecionado,
        "limiar": limiar_confianca,
        "num_detecoes": len(detecoes),
        "detecoes": detecoes,
        "estatisticas": stats
    }
    st.session_state.historico.insert(0, entrada)
    if len(st.session_state.historico) > max_historico:
        st.session_state.historico = st.session_state.historico[:max_historico]

def exportar_json(dados):
    return json.dumps(dados, indent=2, ensure_ascii=False)

def mostrar_resultados(detecoes, stats, fonte, mostrar_guardar=True):
    st.markdown("---")
    st.subheader("Estatísticas")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Objetos", len(detecoes))
    with col2:
        st.metric("Classes Diferentes", len(stats))
    with col3:
        if detecoes:
            media = np.mean([d["confianca"] for d in detecoes])
            st.metric("Confiança Média", f"{media:.1%}")

    if stats:
        st.subheader("Contagem por Classe")
        for classe, info in stats.items():
            st.write(f"**{classe}**: {info['quantidade']} detetado(s) | Confiança média: {info['confianca_media']:.1%}")

    st.subheader("Tabela de Deteções")
    if detecoes:
        df = pd.DataFrame([{
            "Classe": d["classe"],
            "Confiança": f"{d['confianca']:.1%}",
            "Bounding Box": f"({d['bbox']['x1']}, {d['bbox']['y1']}) - ({d['bbox']['x2']}, {d['bbox']['y2']})"
        } for d in detecoes])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Nenhum objeto detetado com o limiar atual. Tenta baixar o slider de confiança.")

    st.subheader("Saída JSON")
    json_output = {
        "modelo": modelo_selecionado,
        "limiar_confianca": limiar_confianca,
        "total_detecoes": len(detecoes),
        "estatisticas": stats,
        "detecoes": detecoes
    }
    st.json(json_output)

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.download_button(
            label="Exportar JSON",
            data=exportar_json(json_output),
            file_name=f"detecao_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            key=f"export_{fonte}_{datetime.now().timestamp()}"
        )
    if mostrar_guardar:
        with col_e2:
            if st.button("Guardar no Histórico", key=f"guardar_{fonte}"):
                guardar_historico(fonte, detecoes, stats)
                st.success("Guardado no histórico.")

def _comparar_dois_modelos(img_comp, modelo_a_nome, modelo_b_nome, limiar):
    """
    Corre os dois modelos selecionados sobre a mesma imagem e mostra os
    resultados lado a lado (caixa anotada, métricas e contagem por classe).
    """
    modelo_a = carregar_modelo(modelos_disponiveis[modelo_a_nome])
    modelo_b = carregar_modelo(modelos_disponiveis[modelo_b_nome])

    col_a, col_b = st.columns(2)

    if modelo_a:
        _, detecoes_a = fazer_inferencia(img_comp, modelo_a, limiar)
        stats_a = calcular_estatisticas(detecoes_a)
        img_a = desenhar_detecoes(img_comp, detecoes_a)
        with col_a:
            st.subheader(f"Modelo A: {modelo_a_nome}")
            st.image(img_a, channels="BGR")
            st.metric("Objetos detetados", len(detecoes_a))
            if detecoes_a:
                media_a = np.mean([d["confianca"] for d in detecoes_a])
                st.metric("Confiança média", f"{media_a:.1%}")
            for classe, info in stats_a.items():
                st.write(f"**{classe}**: {info['quantidade']}x ({info['confianca_media']:.1%})")

    if modelo_b:
        _, detecoes_b = fazer_inferencia(img_comp, modelo_b, limiar)
        stats_b = calcular_estatisticas(detecoes_b)
        img_b = desenhar_detecoes(img_comp, detecoes_b)
        with col_b:
            st.subheader(f"Modelo B: {modelo_b_nome}")
            st.image(img_b, channels="BGR")
            st.metric("Objetos detetados", len(detecoes_b))
            if detecoes_b:
                media_b = np.mean([d["confianca"] for d in detecoes_b])
                st.metric("Confiança média", f"{media_b:.1%}")
            for classe, info in stats_b.items():
                st.write(f"**{classe}**: {info['quantidade']}x ({info['confianca_media']:.1%})")

def _gerar_qr_code(url: str) -> bytes:
    """Gera um PNG em bytes com o QR code para a URL dada."""
    img = qrcode.make(url)
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()

# ============================================
# PROCESSADOR DE VÍDEO EM TEMPO REAL
# ============================================
class YOLOVideoProcessor(VideoProcessorBase):
    """
    Processa cada frame da webcam: corre o YOLO, desenha as bounding boxes
    e devolve o frame anotado. O modelo e o limiar são injetados em direto
    a partir da barra lateral.
    """
    def __init__(self):
        self.modelo = None
        self.limiar = 0.5
        self.lock = threading.Lock()
        self.detecoes_atuais = []
        self.ultima_imagem_anotada = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if self.modelo is None:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        results = self.modelo(img, conf=self.limiar, verbose=False)
        detecoes = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            class_name = self.modelo.names[class_id]
            conf = float(box.conf[0])
            detecoes.append({
                "classe": class_name,
                "confianca": round(conf, 4),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            })

        img_anotada = desenhar_detecoes(img, detecoes)

        with self.lock:
            self.detecoes_atuais = detecoes
            self.ultima_imagem_anotada = img_anotada

        return av.VideoFrame.from_ndarray(img_anotada, format="bgr24")

# ============================================
# TABS
# ============================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Imagem",
    "Câmara",
    "Dataset",
    "Comparar Modelos",
    "Histórico"
])

# ============================================
# TAB 1: IMAGEM
# ============================================
with tab1:
    st.header("Carregar Imagem")

    ficheiro = st.file_uploader(
        "Escolhe uma imagem:",
        type=["jpg", "jpeg", "png", "bmp"],
        key="upload_imagem"
    )

    if ficheiro and modelo:
        conteudo = ficheiro.read()
        arr = np.asarray(bytearray(conteudo), dtype=np.uint8)
        imagem = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Imagem Original")
            st.image(imagem, channels="BGR")

        results, detecoes = fazer_inferencia(imagem, modelo, limiar_confianca)
        stats = calcular_estatisticas(detecoes)

        with col2:
            st.subheader("Resultado da Deteção")
            img_resultado = desenhar_detecoes(imagem, detecoes)
            st.image(img_resultado, channels="BGR")

        mostrar_resultados(detecoes, stats, ficheiro.name)

# ============================================
# TAB 2: CÂMARA (tempo real + foto)
# ============================================
with tab2:
    st.header("Câmara")

    sub_tempo_real, sub_foto = st.tabs(["Em tempo real", "Tirar foto"])

    # --- Sub-tab: tempo real (WebRTC) ---
    with sub_tempo_real:
        st.caption(
            "Inferência YOLO frame a frame, em direto. "
            "Ajusta o modelo e o limiar na barra lateral — a alteração propaga-se ao stream em curso."
        )

        with st.expander("Aceder do telemóvel (código QR)", expanded=False):
            st.markdown(
                "1. Num segundo terminal, arranca o túnel: "
                "`cloudflared tunnel --url http://localhost:8501`  \n"
                "2. Cola aqui a URL HTTPS que aparecer (algo como "
                "`https://xxx.trycloudflare.com`)  \n"
                "3. Aponta a câmara do telemóvel para o QR — abre a app no browser do telemóvel."
            )
            tunnel_url = st.text_input(
                "URL do túnel:",
                key="tunnel_url",
                placeholder="https://xxx.trycloudflare.com",
            )
            if tunnel_url:
                try:
                    qr_bytes = _gerar_qr_code(tunnel_url.strip())
                    st.image(qr_bytes, width=280, caption=tunnel_url.strip())
                except Exception as e:
                    st.error(f"Não foi possível gerar o QR code: {e}")

        ctx = webrtc_streamer(
            key="webcam-tempo-real",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=YOLOVideoProcessor,
            # facingMode 'environment' = câmara traseira no telemóvel (no PC é ignorado)
            media_stream_constraints={
                "video": {"facingMode": {"ideal": "environment"}},
                "audio": False,
            },
            async_processing=True,
        )

        if ctx.video_processor is not None:
            ctx.video_processor.modelo = modelo
            ctx.video_processor.limiar = limiar_confianca

        st.markdown("---")

        if ctx.state.playing and ctx.video_processor is not None:
            st.info(
                "Carrega no botão para congelar o frame atual e gerar a "
                "vista estruturada (tabela + JSON + guardar no histórico)."
            )
            if st.button("Capturar frame atual", key="snapshot_webcam"):
                with ctx.video_processor.lock:
                    snap_detecoes = list(ctx.video_processor.detecoes_atuais)
                    snap_imagem = (
                        ctx.video_processor.ultima_imagem_anotada.copy()
                        if ctx.video_processor.ultima_imagem_anotada is not None
                        else None
                    )

                if snap_imagem is not None:
                    st.session_state.cam_snapshot_detecoes = snap_detecoes
                    st.session_state.cam_snapshot_imagem = snap_imagem
                else:
                    st.warning("Ainda não foi processado nenhum frame. Aguarda um instante.")
        else:
            st.info("Clica em **Start** para iniciar o stream.")

        if "cam_snapshot_imagem" in st.session_state:
            st.markdown("### Frame capturado")
            st.image(
                st.session_state.cam_snapshot_imagem,
                channels="BGR",
                use_container_width=True,
            )
            snap_stats = calcular_estatisticas(st.session_state.cam_snapshot_detecoes)
            mostrar_resultados(
                st.session_state.cam_snapshot_detecoes,
                snap_stats,
                "camara_tempo_real",
            )

    # --- Sub-tab: foto única (captura simples) ---
    with sub_foto:
        st.caption(
            "Tira uma foto via câmara e corre a inferência sobre essa imagem. "
            "Útil quando não interessa stream contínuo ou se o WebRTC der problemas."
        )

        foto = st.camera_input("Tirar foto", key="camera_input_foto")

        if foto and modelo:
            img_pil = Image.open(foto)
            img_array = np.array(img_pil)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            results, detecoes = fazer_inferencia(img_bgr, modelo, limiar_confianca)
            stats = calcular_estatisticas(detecoes)
            img_resultado = desenhar_detecoes(img_bgr, detecoes)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original")
                st.image(foto)
            with col2:
                st.subheader("Deteção")
                st.image(img_resultado, channels="BGR")

            mostrar_resultados(detecoes, stats, "camara_foto")

# ============================================
# TAB 3: DATASET
# ============================================
with tab3:
    st.header("Testar com Imagem do Dataset")

    pasta_teste = st.text_input(
        "Pasta de imagens de teste:",
        value="dataset/test/images"
    )

    col1, col2 = st.columns([1, 3])

    with col1:
        if st.button("Imagem Aleatória"):
            if os.path.exists(pasta_teste):
                imagens = [f for f in os.listdir(pasta_teste)
                           if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                if imagens:
                    st.session_state.imagem_dataset = random.choice(imagens)
                    st.session_state.imagem_dataset_path = os.path.join(pasta_teste, st.session_state.imagem_dataset)
                else:
                    st.error("Nenhuma imagem encontrada na pasta.")
            else:
                st.error(f"Pasta não existe: {pasta_teste}")

    if "imagem_dataset" in st.session_state and modelo:
        caminho = st.session_state.imagem_dataset_path
        with col2:
            st.write(f"Imagem selecionada: `{st.session_state.imagem_dataset}`")

        img = cv2.imread(caminho)
        if img is not None:
            results, detecoes = fazer_inferencia(img, modelo, limiar_confianca)
            stats = calcular_estatisticas(detecoes)
            img_resultado = desenhar_detecoes(img, detecoes)

            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Original")
                st.image(img, channels="BGR")
            with col_b:
                st.subheader("Deteção")
                st.image(img_resultado, channels="BGR")

            mostrar_resultados(detecoes, stats, st.session_state.imagem_dataset)
        else:
            st.error("Erro ao carregar a imagem.")

# ============================================
# TAB 4: COMPARAR MODELOS
# ============================================
with tab4:
    st.header("Comparar Dois Modelos")

    # Modelos partilhados entre as duas sub-tabs (upload e foto)
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        modelo_a_nome = st.selectbox("Modelo A:", list(modelos_disponiveis.keys()), key="modelo_a")
    with col_m2:
        # Se houver apenas 1 modelo descoberto, o índice 1 rebenta — defensive default
        default_b = 1 if len(modelos_disponiveis) > 1 else 0
        modelo_b_nome = st.selectbox(
            "Modelo B:",
            list(modelos_disponiveis.keys()),
            index=default_b,
            key="modelo_b",
        )

    sub_upload, sub_foto = st.tabs(["Upload de imagem", "Tirar foto"])

    with sub_upload:
        ficheiro_comp = st.file_uploader(
            "Carrega uma imagem para comparar:",
            type=["jpg", "jpeg", "png", "bmp"],
            key="upload_comparacao"
        )
        if ficheiro_comp:
            conteudo = ficheiro_comp.read()
            arr = np.asarray(bytearray(conteudo), dtype=np.uint8)
            img_comp = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            _comparar_dois_modelos(img_comp, modelo_a_nome, modelo_b_nome, limiar_confianca)

    with sub_foto:
        foto_comp = st.camera_input("Tirar foto", key="camera_comparacao")
        if foto_comp:
            img_pil = Image.open(foto_comp)
            img_array = np.array(img_pil)
            img_comp = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            _comparar_dois_modelos(img_comp, modelo_a_nome, modelo_b_nome, limiar_confianca)

# ============================================
# TAB 5: HISTÓRICO
# ============================================
with tab5:
    st.header("Histórico de Inferências")

    if not st.session_state.historico:
        st.info("Ainda não há inferências guardadas. Usa as outras tabs e carrega em 'Guardar no Histórico'.")
    else:
        col_h1, col_h2 = st.columns([3, 1])
        with col_h2:
            if st.button("Limpar Histórico"):
                st.session_state.historico = []
                st.rerun()

        resumo = []
        for i, entrada in enumerate(st.session_state.historico):
            resumo.append({
                "#": i + 1,
                "Timestamp": entrada["timestamp"],
                "Fonte": entrada["fonte"],
                "Modelo": entrada["modelo"],
                "Nº Deteções": entrada["num_detecoes"],
                "Limiar": f"{entrada['limiar']:.0%}"
            })
        st.dataframe(pd.DataFrame(resumo), use_container_width=True)

        st.subheader("Detalhe por Inferência")
        for i, entrada in enumerate(st.session_state.historico):
            with st.expander(f"#{i+1} — {entrada['timestamp']} | {entrada['fonte']} | {entrada['num_detecoes']} objeto(s)"):
                st.json(entrada)

        st.download_button(
            label="Exportar Histórico Completo (JSON)",
            data=exportar_json(st.session_state.historico),
            file_name=f"historico_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
