# Dataset — Deteção de Luvas de Proteção em Ambiente Farmacêutico

## 1. Visão Geral

Este dataset foi recolhido e anotado manualmente pelos três membros do grupo de investigação, com o objetivo de treinar um modelo de deteção de objetos capaz de monitorizar o estado e a conformidade das luvas de proteção em contexto laboratorial e farmacêutico.

- **Formato:** YOLOv8 (bounding boxes normalizadas)
- **Número de classes:** 6
- **Total de imagens (pré-augmentation):** 6.128
- **Total de anotações:** 7.604
- **Resolução:** 640 × 640 píxeis (redimensionado pelo Roboflow)
- **Licença:** CC BY 4.0
- **Fonte:** [Roboflow Universe — trabalho_ia-opgfv v2](https://universe.roboflow.com/ruis-workspace-qvcoy/trabalho_ia-opgfv/dataset/2)

---

## 2. Divisão do Dataset

| Split      | Imagens | % Total |
|------------|--------:|--------:|
| Treino     |   4.288 |   70,0% |
| Validação  |   1.227 |   20,0% |
| Teste      |     613 |   10,0% |
| **Total**  | **6.128** | **100%** |

A divisão foi aplicada de forma estratificada pelo Roboflow, mantendo a proporção de classes em cada split. O conjunto de teste foi mantido estritamente isolado — as suas imagens não foram vistas pelo modelo durante qualquer fase de treino ou validação, nem sofreram qualquer transformação sintética de augmentation.

---

## 3. Classes e Nomenclatura

| ID | Nome                  | Descrição resumida                                                    |
|----|-----------------------|-----------------------------------------------------------------------|
| 0  | `glove_not_fitted`    | Luva presente na mão mas não devidamente ajustada                    |
| 1  | `glove_off`           | Mão sem qualquer luva — ausência total de proteção (Risco Crítico)   |
| 2  | `glove_on`            | Luva corretamente calçada e ajustada                                 |
| 3  | `glove_partial`       | Luva parcialmente calçada, sem cobertura total da mão                |
| 4  | `gloves_contaminated` | Luva com sinais visíveis de contaminação (líquidos, descoloração)    |
| 5  | `gloves_damaged`      | Luva com danos físicos visíveis (rasgos, perfurações)                |

Todas as classes referem-se exclusivamente a **luvas azuis de nitrilo**, que é o equipamento de proteção individual (EPI) padrão utilizado nos cenários capturados. Luvas de outras cores ou materiais estão explicitamente fora do âmbito deste dataset.

---

## 4. Distribuição de Anotações por Classe

| Classe                | ID | Treino | Validação | Teste | Total | % Total |
|-----------------------|----|-------:|----------:|------:|------:|--------:|
| `glove_not_fitted`    |  0 |    181 |        56 |    24 |   261 |    3,4% |
| `glove_off`           |  1 |    610 |       174 |    87 |   871 |   11,5% |
| `glove_on`            |  2 |  1.392 |       397 |   198 | 1.987 |   26,1% |
| `glove_partial`       |  3 |    701 |       196 |   105 | 1.002 |   13,2% |
| `gloves_contaminated` |  4 |  1.739 |       506 |   259 | 2.504 |   32,9% |
| `gloves_damaged`      |  5 |    685 |       196 |    98 |   979 |   12,9% |
| **Total**             |    |**5.308**|  **1.525**| **771**|**7.604**|**100%**|

> **Nota de desbalanceamento:** A classe `glove_not_fitted` é a menos representada (3,4% das anotações). Este desequilíbrio reflete a dificuldade prática em capturar este estado de forma sistemática e consistente — existem inúmeras variações morfológicas possíveis de uma luva mal ajustada, e a fronteira visual com `glove_partial` pode ser ténue. A decisão do grupo foi manter um número reduzido de exemplos desta classe em vez de introduzir exemplos ambíguos que pudessem prejudicar a aprendizagem.

---

## 5. Processo de Recolha

A recolha foi executada de forma colaborativa, com cada um dos três membros responsável por capturar as variações morfológicas de duas classes específicas. As imagens foram capturadas com **smartphones de vários membros do grupo** (câmaras traseiras de modelos variados), sempre numa **perspetiva zenital** (*top-down*), simulando a visão de uma câmara fixa sobre uma bancada laboratorial.

### Cenários simulados

Cada membro replicou dois ambientes de bancada:
- **Bancada branca** — superfície lisa de laboratório
- **Bancada metálica** — simulada com papel de alumínio para imitar superfícies de aço inoxidável

### Variações de captura

O protocolo de recolha incluiu as seguintes variações para aumentar a robustez do modelo:

- **Poses da mão:** captura de diferentes posições (mão aberta, fechada, em rotação e em posições típicas de manuseamento laboratorial)
- **Interação com objetos:** introdução de objetos como canetas a simular pipetas e instrumentos, para testar oclusão parcial e preensão
- **Complexidade de fundo:** inclusão de luvas dispersas no cenário como elementos de fundo, forçando o modelo a distinguir entre luvas em uso e objetos estáticos

---

## 6. Semântica e Interpretação das Etiquetas

Cada linha de anotação segue o formato YOLO:

```
<class_id> <cx> <cy> <width> <height>
```

Onde `cx`, `cy`, `width` e `height` são valores normalizados entre 0.0 e 1.0 relativamente às dimensões da imagem. A bounding box cobre a **totalidade da extensão visível da luva** (ou da mão, no caso de `glove_off`).

### Critérios de anotação por classe

- **`glove_on` (ID 2):** Luva azul de nitrilo completamente calçada, cobrindo a mão até ao pulso, sem dobras visíveis nem dedos expostos. É a classe visualmente mais inequívoca.

- **`glove_off` (ID 1):** Mão totalmente descoberta, sem qualquer luva. A bounding box cobre a região da mão visível. Juntamente com `glove_on`, é das classes mais fáceis de distinguir visualmente.

- **`glove_partial` (ID 3):** Luva em processo de ser calçada ou retirada — parte da mão está coberta e outra exposta. A luva está claramente presente mas incompleta.

- **`glove_not_fitted` (ID 0):** Luva colocada na mão mas sem estar devidamente ajustada — pode estar enrugada, com folgas excessivas, dedos mal alinhados ou pulso não vedado. A mão está coberta mas a proteção é inadequada.

- **`gloves_contaminated` (ID 4):** Luva com marcas visíveis de contaminação — manchas, descoloração química ou resíduos líquidos ou sólidos na superfície. Foram anotados **apenas casos em que a contaminação era claramente visível a olho nu** na imagem, para evitar falsos positivos derivados de ruído do sensor ou sombras.

- **`gloves_damaged` (ID 5):** Luva com danos estruturais visíveis — rasgos, perfurações, cortes ou deformações. Tal como na classe anterior, foram anotados **apenas danos claramente visíveis a olho nu**, evitando ambiguidade com grão de imagem ou artefactos de compressão JPEG.

### Regra de visibilidade mínima

Bounding boxes com menos de **30% da área útil visível** (por oclusão ou saída de frame) foram descartadas durante o processo de anotação e augmentation, para evitar ruído nos dados de treino.

### Imagens com múltiplas anotações

A maioria das imagens contém **uma única anotação**. Existem 1.465 imagens com múltiplas bounding boxes (duas ou três luvas visíveis em simultâneo). Apenas 3 imagens contêm anotações de **classes diferentes em simultâneo** (combinação `glove_on + gloves_contaminated`), confirmando que o dataset é predominantemente single-label por imagem.

Em cenários de sobreposição de mãos, a política de etiquetagem ditou a identificação individual de cada instância sempre que possível. Para o conjunto de treino, priorizaram-se imagens com separação clara de mãos para garantir uma aprendizagem inicial mais estável.

---

## 7. Problemas Conhecidos e Casos Ambíguos

### 7.1 `gloves_contaminated` vs `glove_on`

O principal caso ambíguo identificado durante a anotação é a distinção entre luvas contaminadas e luvas limpas em condições de iluminação desfavorável. Sombras projetadas sobre a superfície azul da luva de nitrilo — em especial nas superfícies de papel de alumínio — podem mimetizar manchas ou descoloração química, dificultando a classificação visual. Sempre que existia contaminação real confirmada visualmente pelo anotador, a imagem foi etiquetada como `gloves_contaminated`, mesmo que a contaminação fosse subtil. Em casos de dúvida genuína sem contaminação confirmada, a imagem foi etiquetada como `glove_on`.

### 7.2 Desbalanceamento de `glove_not_fitted`

A classe `glove_not_fitted` apresenta um número de exemplos significativamente inferior às restantes (261 anotações, 3,4% do total). Este desequilíbrio não foi um erro de processo — foi uma decisão deliberada face à dificuldade em capturar este estado de forma sistemática e sem ambiguidade com `glove_partial`.

### 7.3 Photographer bias

A recolha de dados foi dividida por classe entre os membros do grupo, com cada membro responsável por duas classes específicas. Este protocolo introduziu um efeito conhecido em visão computacional como *photographer bias*: o modelo aprende não apenas as características visuais da classe, mas também características específicas do fotógrafo — forma e dimensão da mão, tom de pele e condições de iluminação do ambiente recriado.

Este efeito tem duas consequências importantes:

1. **Nos testes de inferência informal** com imagens externas ao dataset, os modelos apresentaram falhas consistentes nas classes capturadas por membros diferentes do utilizador das imagens de teste, mas identificaram corretamente as classes do próprio membro.
2. **Nas métricas formais**, como o split treino/validação/teste foi realizado aleatoriamente sobre o dataset completo, o conjunto de teste contém imagens dos mesmos fotógrafos presentes no treino para cada classe. As métricas obtidas (mAP@0.5 ≈ 0.99) podem por isso ser **otimistas** relativamente ao desempenho esperado em condições reais de produção.

A solução para mitigar este efeito passaria por cada membro recolher imagens de **todas as classes**, isolando a variável "fotógrafo" da variável "classe".

### 7.4 Variabilidade de dispositivos e iluminação

As imagens foram capturadas com diferentes smartphones, introduzindo variabilidade em termos de qualidade de imagem, temperatura de cor e compressão JPEG. Esta variabilidade é considerada benéfica para a generalização do modelo, mas pode ter introduzido inconsistências subtis nas anotações entre membros do grupo.

---

## 8. Estrutura de Ficheiros

```
datset/
├── data.yaml
├──dataset.md
├── train/
│   ├── images/     # 4.288 imagens
│   └── labels/     # 4.288 ficheiros .txt (formato YOLO)
├── valid/
│   ├── images/     # 1.227 imagens
│   └── labels/     # 1.227 ficheiros .txt
└── test/
    ├── images/     # 613 imagens
    └── labels/     # 613 ficheiros .txt
```

O ficheiro `data.yaml` define os caminhos relativos dos splits, o número de classes (`nc: 6`) e os nomes das classes na ordem dos IDs.

