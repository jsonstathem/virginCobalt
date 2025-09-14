"""
Скрипт для интеграции модели от вашего друга.
Замените этот код на реальную загрузку модели.
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import json

# Пример интеграции кода от друга:

def load_and_test_model(model_path: str, test_image_path: str):
    """
    Функция для загрузки и тестирования модели.
    Основана на коде от вашего друга.
    """

    # 1. Загружаем модель (раскомментируйте когда будет файл модели)
    # model = torch.load(model_path, map_location='cpu')
    # model.eval()

    # 2. Подготавливаем transforms
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # 3. Загружаем тестовое изображение
    image = Image.open(test_image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)  # [1, C, H, W]

    # 4. Прогон через модель (раскомментируйте)
    # with torch.no_grad():
    #     outputs = model(img_tensor)

    # Пока используем mock данные как в примере от друга:
    mock_outputs = [{
        "boxes": torch.tensor([
            [1132.896, 541.730, 1476.456, 1010.929],
            [1139.789, 451.677, 1586.240, 1190.254],
            [1411.924, 851.393, 1626.153, 1511.069]
        ]),
        "labels": torch.tensor([3, 3, 3]),
        "scores": torch.tensor([0.5620, 0.5087, 0.5031])
    }]

    outputs = mock_outputs  # Заменить на реальные outputs

    threshold = 0.5
    filtered_boxes = []
    filtered_labels = []
    filtered_scores = []

    CLASS_MAP = {
        1: "car",
        2: "dent",      # исправлено с "dunt"
        3: "rust",
        4: "scratch"
    }

    for box, label, score in zip(outputs[0]["boxes"], outputs[0]["labels"], outputs[0]["scores"]):
        if score >= threshold:
            filtered_boxes.append(box.cpu().numpy().tolist())
            filtered_labels.append(label.item())
            filtered_scores.append(score.item())

    # 5. Формируем JSON как в коде от друга
    result = {
        "boxes": filtered_boxes,
        "labels": filtered_labels,
        "scores": filtered_scores
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Интерпретация результатов:
    print("\nИнтерпретация результатов:")
    for i, (box, label, score) in enumerate(zip(result["boxes"], result["labels"], result["scores"])):
        class_name = CLASS_MAP.get(label, "unknown")
        print(f"Объект {i+1}: {class_name} (уверенность: {score:.2%})")
        print(f"  Координаты: {box}")

    return result

if __name__ == "__main__":
    # Тестирование (замените пути на реальные)
    # model_path = "model.pth"
    test_image_path = "test_image.jpg"  # Поместите тестовое изображение

    print("Тестирование модели...")
    # result = load_and_test_model(model_path, test_image_path)
    print("Готово!")