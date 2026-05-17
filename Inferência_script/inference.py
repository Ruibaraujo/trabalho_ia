"""
infer.py — Script de inferência autónomo para modelos YOLO de deteção de luvas.

Uso:
    python inference.py --model path/to/best.pt --image path/to/image.jpg
    python inference.py --model path/to/best.pt --input path/to/folder/ --output path/to/results/
    python inference.py --model path/to/best.pt --image path/to/image.jpg --conf 0.828 --iou 0.70
"""

import argparse
import json
import os
from pathlib import Path

import cv2
from ultralytics import YOLO

# ── Extensões de imagem suportadas ────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


# ── Inferência sobre uma única imagem ─────────────────────────────────────────

def run_inference(model, image_path: Path, output_dir: Path, conf: float, iou: float, imgsz: int, device: str):
    """Executa inferência numa imagem, guarda JSON e imagem anotada."""

    results = model.predict(
        source=str(image_path),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        max_det=300,
        device=device,
        verbose=False,
    )

    result = results[0]

    # Desconstrução dos tensores de saída
    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "class_id":   int(box.cls),
            "class_name": model.names[int(box.cls)],
            "confidence": round(float(box.conf), 4),
            "bbox": {
                "x1":     round(x1, 2),
                "y1":     round(y1, 2),
                "x2":     round(x2, 2),
                "y2":     round(y2, 2),
                "width":  round(x2 - x1, 2),
                "height": round(y2 - y1, 2),
            }
        })

    # Guardar JSON
    json_path = output_dir / f"{image_path.stem}.json"
    with open(json_path, "w") as f:
        json.dump(detections, f, indent=2)

    # Guardar imagem anotada
    annotated = result.plot(conf=True, labels=True, boxes=True, line_width=2)
    output_image_path = output_dir / image_path.name
    cv2.imwrite(str(output_image_path), annotated)

    print(f"  → {len(detections)} deteção(ões) | imagem → '{output_image_path}' | JSON → '{json_path}'")
    return detections


# ── Ponto de entrada ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Inferência YOLO para deteção de luvas de proteção."
    )

    # Argumentos obrigatórios
    parser.add_argument(
        "--model", required=True,
        help="Caminho para o ficheiro de pesos do modelo (ex: runs/yolov8s/weights/best.pt)"
    )

    # Fonte de entrada — imagem única ou pasta
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--image",
        help="Caminho para uma imagem única (ex: path/to/image.jpg)"
    )
    group.add_argument(
        "--input",
        help="Caminho para uma pasta com imagens (processa todas as imagens suportadas)"
    )

    # Argumentos opcionais
    parser.add_argument(
        "--output", default="output",
        help="Pasta de destino para imagens anotadas e ficheiros JSON (default: ./output)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.40,
        help="Limiar mínimo de confiança para deteções (default: 0.40)"
    )
    parser.add_argument(
        "--iou", type=float, default=0.70,
        help="Limiar IoU para Non-Maximum Suppression (default: 0.70)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Resolução de entrada do modelo em píxeis (default: 640)"
    )
    parser.add_argument(
        "--device", default="0",
        help="Dispositivo de inferência: '0' para GPU, 'cpu' para CPU (default: 0)"
    )

    args = parser.parse_args()

    # Criar pasta de output
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Carregar modelo
    print(f"[Inference] A carregar modelo: {args.model}")
    model = YOLO(args.model)

    # Recolher imagens a processar
    if args.image:
        image_paths = [Path(args.image)]
    else:
        input_dir = Path(args.input)
        if not input_dir.exists():
            raise FileNotFoundError(f"Pasta de input não encontrada: {input_dir}")
        image_paths = [
            p for p in input_dir.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if not image_paths:
            print(f"[AVISO] Nenhuma imagem encontrada em '{input_dir}'.")
            return

    print(f"[Inference] {len(image_paths)} imagem(ns) a processar...\n")

    # Processar imagens
    for image_path in image_paths:
        print(f"A processar: {image_path.name}")
        run_inference(model, image_path, output_dir, args.conf, args.iou, args.imgsz, args.device)

    print(f"\n[Inference] Concluído. Resultados guardados em '{output_dir}'.")


if __name__ == "__main__":
    main()
