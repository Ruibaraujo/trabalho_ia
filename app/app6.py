import streamlit as st
import cv2
import os
import json
import random
import numpy as np
import pandas as pd
from ultralytics import YOLO
from datetime import datetime
from collections import defaultdict

# Configuração da página
st.set_page_config(
    page_title="Deteção de Objetos - Projeto IA",
    page_icon="🎯",
    layout="wide"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .stats-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .detection-label {
        font-size: 1.2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Título
st.markdown('<p class="main-header">🎯 Sistema de Deteção de Objetos</p>', unsafe_allow_html=True)
st.markdown("**Projeto de Inteligência Artificial**")
st.markdown("---")

# ============================================
# CONFIGURAÇÕES NA SIDEBAR
# ============================================
st.sidebar.header("⚙️ Configurações")

# Seleção do modelo
st.sidebar.subheader("📦 Modelo")
modelos_disponiveis = {
    "YOLOv8 Small - Treinado (yolo8s_best.pt)":   "yolo8s_best.pt",
    "YOLOv8 Medium - Treinado (yolo8m_best.pt)":  "yolo8m_best.pt",
    "YOLO11 Small - Treinado (yolo11s_best.pt)":  "yolo11s_best.pt",
    "YOLO11 Medium - Treinado (yolo11m_best.pt)": "yolo11m_best.pt",
}

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
st.sidebar.subheader("📜 Histórico")
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
    try:
        return YOLO(caminho)
    except:
        st.error(f"Erro ao carregar modelo: {caminho}")
        return None

modelo = carregar_modelo(modelos_disponiveis[modelo_selecionado])

if "historico" not in st.session_state:
    st.session_state.historico = []

if "webcam_ativa" not in st.session_state:
    st.session_state.webcam_ativa = False

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
    st.subheader("📊 Estatísticas")
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
        st.subheader("📦 Contagem por Classe")
        for classe, info in stats.items():
            st.write(f"**{classe}**: {info['quantidade']} detetado(s) | Confiança média: {info['confianca_media']:.1%}")

    st.subheader("📋 Tabela de Deteções")
    if detecoes:
        df = pd.DataFrame([{
            "Classe": d["classe"],
            "Confiança": f"{d['confianca']:.1%}",
            "Bounding Box": f"({d['bbox']['x1']}, {d['bbox']['y1']}) - ({d['bbox']['x2']}, {d['bbox']['y2']})"
        } for d in detecoes])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Nenhum objeto detetado com o limiar atual. Tenta baixar o slider de confiança.")

    st.subheader("🔧 Saída JSON")
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
            label="📥 Exportar JSON",
            data=exportar_json(json_output),
            file_name=f"detecao_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            key=f"export_{fonte}_{datetime.now().timestamp()}"
        )
    if mostrar_guardar:
        with col_e2:
            if st.button("💾 Guardar no Histórico", key=f"guardar_{fonte}"):
                guardar_historico(fonte, detecoes, stats)
                st.success("Guardado no histórico!")

# ============================================
# TABS
# ============================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📷 Imagem",
    "🎥 Webcam",
    "📁 Dataset",
    "⚖️ Comparar Modelos",
    "📜 Histórico"
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
# TAB 2: CÂMARA EM TEMPO REAL
# ============================================
with tab2:
    st.header("🎥 Câmara em Tempo Real")

    # Usa st.camera_input — funciona na webcam do PC e na câmara do telemóvel via ngrok
    foto = st.camera_input("📸 Capturar")

    if foto and modelo:
        from PIL import Image

        img_pil = Image.open(foto)
        img_array = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        results, detecoes = fazer_inferencia(img_bgr, modelo, limiar_confianca)
        stats = calcular_estatisticas(detecoes)
        img_resultado = desenhar_detecoes(img_bgr, detecoes)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📷 Original")
            st.image(foto)
        with col2:
            st.subheader("🎯 Deteção")
            st.image(img_resultado, channels="BGR")

        mostrar_resultados(detecoes, stats, "camara")

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
        if st.button("🎲 Imagem Aleatória"):
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
            st.write(f"📄 Imagem selecionada: `{st.session_state.imagem_dataset}`")

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
    st.header("⚖️ Comparar Dois Modelos")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        modelo_a_nome = st.selectbox("Modelo A:", list(modelos_disponiveis.keys()), key="modelo_a")
    with col_m2:
        modelo_b_nome = st.selectbox("Modelo B:", list(modelos_disponiveis.keys()), index=1, key="modelo_b")

    ficheiro_comp = st.file_uploader(
        "Carrega uma imagem para comparar:",
        type=["jpg", "jpeg", "png", "bmp"],
        key="upload_comparacao"
    )

    if ficheiro_comp:
        conteudo = ficheiro_comp.read()
        arr = np.asarray(bytearray(conteudo), dtype=np.uint8)
        img_comp = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        modelo_a = carregar_modelo(modelos_disponiveis[modelo_a_nome])
        modelo_b = carregar_modelo(modelos_disponiveis[modelo_b_nome])

        col_a, col_b = st.columns(2)

        if modelo_a:
            results_a, detecoes_a = fazer_inferencia(img_comp, modelo_a, limiar_confianca)
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
            results_b, detecoes_b = fazer_inferencia(img_comp, modelo_b, limiar_confianca)
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

# ============================================
# TAB 5: HISTÓRICO
# ============================================
with tab5:
    st.header("📜 Histórico de Inferências")

    if not st.session_state.historico:
        st.info("Ainda não há inferências guardadas. Usa as outras tabs e clica em 'Guardar no Histórico'.")
    else:
        col_h1, col_h2 = st.columns([3, 1])
        with col_h2:
            if st.button("🗑️ Limpar Histórico"):
                st.session_state.historico = []
                st.rerun()

        # Tabela resumo
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

        # Detalhe de cada entrada
        st.subheader("Detalhe por Inferência")
        for i, entrada in enumerate(st.session_state.historico):
            with st.expander(f"#{i+1} — {entrada['timestamp']} | {entrada['fonte']} | {entrada['num_detecoes']} objeto(s)"):
                st.json(entrada)

        # Exportar histórico completo
        st.download_button(
            label="📥 Exportar Histórico Completo (JSON)",
            data=exportar_json(st.session_state.historico),
            file_name=f"historico_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )