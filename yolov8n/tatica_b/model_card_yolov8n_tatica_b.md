# Model Card — YOLOv8n Tática B (v1.0.0)

## 1. Overview
- **Goal:** Testar hiperparametros, o uso do otimizador `AdamW` com learning rate inicial `lr0=0.001`, para verificar qual das taticas é a melhor para o treino dos modelos. Com o fim de ,onitorizar e auditar autonomamente o estado e a conformidade das luvas dos operadores em bancadas de trabalho farmacêuticas e laboratórios, prevenindo contaminações cruzadas de substâncias e salvaguardando a integridade física dos trabalhadores contra agentes químicos nocivos.
- **Task:** Object Detection (Deteção de Objetos)
- **Classes:**
  1. `glove_on` (Luvas corretamente calçadas e ajustadas)
  2. `glove_partial` (Luvas parcialmente calçadas, sem proteção total da mão)
  3. `glove_not_fitted` (Luvas presentes sobre a mão, mas não devidamente ajustadas)
  4. `glove_off` (Ausência total de luvas nas mãos do operador — Risco Crítico)
  5. `gloves_contaminated` (Luvas com sinais de contaminação por agentes externos, líquidos ou descoloração química)
  6. `gloves_damaged` (Luvas que apresentam danos físicos visíveis, como rasgos ou perfurações)
- **Intended users:** Auditores de Garantia de Qualidade (QA), Engenheiros de MLOps, Supervisores de Produção Farmacêutica e Responsáveis por Higiene e Segurança no Trabalho (SST).

## 2. Intended Use and Scope
- **Intended environment:** Bancadas de manipulação de compostos biológicos/químicos, cleanrooms (salas limpas estéreis), linhas de enchimento primário e eclusas de vestição (*gowning areas*).
- **Assumptions:** Captura de imagem e feeds de vídeo de alta resolução orientados numa perspetiva estritamente zenital (*top-down view*), sob iluminação artificial industrial estável.
- **Out-of-scope uses:** Este modelo foca-se exclusivamente em luvas e no estado das mãos. Está explicitamente **fora do âmbito** a deteção de qualquer outro componente do traje laboratorial (como batas, máscaras ou óculos) e luvas de cores que diferem da tonalidade azul padrão utilizada no ciclo de treino.

## 3. Training Data
- **Data source:** Conjunto de dados próprio, capturado e anotado manualmente de forma distribuída e equitativa pelos três membros do grupo de investigação, onde cada colaborador ficou encarregue de cobrir variações morfológicas de duas classes específicas.
- **Dataset size:** A partição original seguiu uma proporção estrita de 70% para treino, 20% para validação e 10% para teste.
  - **Partição de Treino Original (Pré-Augmentation):** 6.128 imagens.
  - **Exemplares Sintéticos Gerados:** 10.460 novas imagens.
  - **Partição de Treino Final (Pós-Augmentation):** Aproximadamente 14.748 imagens.
- **Class distribution:** Estratégia de balanceamento aplicada na expansão para mitigar o desbalanceamento natural da amostragem. A distribuição final de imagens no conjunto de treino pós-augmentation ficou fixada em:
  - `glove_on`: 2 778 imagens
  - `glove_partial`: 2 455 imagens
  - `glove_not_fitted`: 2 040 imagens
  - `glove_off`: 2 440 imagens
  - `gloves_contaminated`: 2 982 imagens
  - `gloves_damaged`: 2 055 imagens
- **Labeling guidelines:** As caixas delimitadoras cobrem a totalidade da extensão visível do objeto. Foi definido um limiar de visibilidade mínima de 0.3 (30%) no script de transformação; bounding boxes cuja área útil ficasse abaixo desse limite após as alterações geométricas foram automaticamente descartadas para mitigar ruído de otimização. Em partições de treino, priorizou-se imagens com separação clara de mãos.
- **Augmentations:** O pipeline de ampliação de dados foi executado programaticamente **pós-Roboflow** via script no Google Colab, recorrendo à biblioteca profissional **`Albumentations`** para aplicar transformações compostas:
  - *Geométricas (Invariância Espacial):* `HorizontalFlip`, `VerticalFlip` e `RandomRotate90` para imunizar o modelo à lateralidade da mão (esquerda/direita) ou posição do operador em torno da bancada.
  - *Iluminação (Resiliência Cromática):* Ajustes aleatórios de Brilho e Contraste (`RandomBrightnessContrast`) num intervalo de ±30% para prevenir a dependência de exposição perfeita e combater reflexos severos em bancadas de aço inox.
  - *Ruído Ótico e Dinâmica:* Desfocagem Gaussiana (`GaussianBlur`) atuando como regularizador para mimetizar o efeito de arrasto induzido pelo movimento rápido das mãos (*motion blur*) ou desalinhamento focal da lente.

## 4. Evaluation
- **Test set description:** Separado na proporção original de 10%, correspondendo a um subconjunto estritamente isolado de imagens (613 imagens, 771 instâncias) que não sofreram transformações sintéticas de treino e permaneceram totalmente ocultas à rede. Avaliação executada com `conf=0.001`, `iou=0.6`, `imgsz=640`.

- **Métricas Globais (All Classes):**

| Métrica        | Valor  |
|----------------|--------|
| `mAP@0.5`      | 0.9881 |
| `mAP@0.5:0.95` | 0.9364 |
| `Precision`    | 0.9993 |
| `Recall`       | 0.9860 |
| `F1`           | 0.9926 |

- **Métricas por Classe:**

| Classe                | P      | R      | F1     | AP@0.5 | AP@0.5:0.95 |
|-----------------------|--------|--------|--------|--------|-------------|
| `glove_not_fitted`    | 0.998  | 0.917  | 0.955  | 0.953  | 0.857       |
| `glove_off`           | 1.000  | 1.000  | 1.000  | 0.995  | 0.904       |
| `glove_on`            | 0.999  | 1.000  | 1.000  | 0.995  | 0.977       |
| `glove_partial`       | 0.999  | 1.000  | 0.999  | 0.995  | 0.947       |
| `gloves_contaminated` | 1.000  | 0.999  | 1.000  | 0.995  | 0.978       |
| `gloves_damaged`      | 1.000  | 1.000  | 1.000  | 0.995  | 0.955       |

- **Qualitative analysis:** A matriz de confusão normalizada revela que cinco das seis classes atingem 1.00 na diagonal, sem qualquer confusão entre classes — todos os erros residem exclusivamente na linha e coluna de background. A classe `glove_not_fitted` apresenta Recall de 0.917, confirmando que passos de atualização mais conservadores beneficiam a deteção desta classe. Na coluna background, 75% dos falsos positivos são classificados como `gloves_contaminated` e 25% como `glove_not_fitted`.
- **Recommended confidence threshold(s):** O limiar ótimo segundo a curva F1 é `conf=0.773`, onde o modelo atinge F1=0.99. Para deployment operacional, o threshold `conf=0.40` pode ser preferível em contexto de segurança industrial para maximizar recall.

## 5. Limitations and Failure Modes
- **Condições de Falha:** Efeitos severos de reflexo lumínico (*glare*) provocados por iluminação direta sobre luvas molhadas ou superfícies adjacentes de aço inoxidável podem segmentar ou distorcer as caixas delimitadoras. O modelo foi treinado exclusivamente com luvas de tonalidade azul padrão, pelo que luvas de outras cores ou materiais estão explicitamente fora do âmbito de deteção fiável. Condições de iluminação muito distintas das de treino ou ângulos de câmara não-zenitais podem degradar significativamente o desempenho.
- **Padrões de Erro:** A classe `glove_not_fitted` apresenta 8% de falsos negativos (Recall 0.917). Na coluna background, 75% dos falsos positivos são classificados como `gloves_contaminated` e 25% como `glove_not_fitted`.

## 6. Deployment Notes
- **Input requirements:** Resolução estrita de entrada de $640 \times 640$ píxeis, mapeamento em formato de três canais RGB.
- **Output format:** Ficheiro estruturado em lote contendo matrizes com o formato de coordenadas $[x_1, y_1, x_2, y_2]$, identificadores inteiros de classe (`category_id`), rótulos descritivos e scores probabilísticos de confiança $[0.0, 1.0]$. Recomendado o limiar operacional de `conf=0.773`, correspondente ao pico da curva macro F1. A latência de inferência não foi medida experimentalmente neste ciclo de desenvolvimento, sendo recomendada a realização de um benchmark no hardware alvo antes de qualquer deployment em produção.

## 7. Ethical / Safety / Privacy Considerations
- **Riscos:** Captura acidental e armazenamento de rostos de colaboradores na linha de montagem, gerando problemas de conformidade com o Regulamento Geral sobre a Proteção de Dados (RGPD). O dataset foi recolhido num contexto laboratorial específico com diversidade limitada de tipos de pele, ângulos de câmara e ambientes de produção, podendo comprometer a generalização do modelo. Em contexto farmacêutico, um falso negativo — nomeadamente a não deteção de `gloves_contaminated` ou `gloves_damaged` — pode ter consequências graves para a integridade do produto e para a segurança do operador.
- **Mitigação:** Como o modelo opera em perspetiva zenital (*top-down*), o risco de captura de rostos é estruturalmente reduzido. Contudo, recomenda-se a aplicação de uma máscara estática opaca (*blurring*) nas áreas limítrofes superiores do frame onde o rosto do operador possa surgir, garantindo total anonimização antes do fluxo preditivo. O modelo não deve ser utilizado como único mecanismo de auditoria, devendo ser complementado por inspeção humana em situações de risco crítico, em particular para a classe `glove_not_fitted` onde se observa a maior taxa de falsos negativos (8%).

## 8. Versioning and Contact
- **Version:** v1.0.0
- **Date:** 2026-05
- **Authors:** Ana Sofia Gonçalves Vaz (8230095), Maria Eduarda Meireles Sameiro (8230227), Rui Miguel Teixeira Borges de Araújo (8230112).
- **Institution:** Instituto Politécnico do Porto (IPP) — Escola Superior de Tecnologia e Gestão (ESTG). Licenciatura em Engenharia Informática, Unidade Curricular de Inteligência Artificial, 2025/2026.