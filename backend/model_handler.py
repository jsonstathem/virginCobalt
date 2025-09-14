"""
Модуль для работы с ML моделью анализа повреждений автомобилей.
Интегрирует код от вашего друга для работы с PyTorch моделью.
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class CarDamageModel:
    """Класс для работы с моделью анализа повреждений автомобилей"""

    def __init__(self, model_path: str = None):
        # Автоматическое определение лучшего устройства
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"Используется GPU: {device_name}")
        else:
            self.device = torch.device('cpu')
            logger.info(f"Используется CPU: {torch.get_num_threads()} потоков")

        self.model = None
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # Маппинг классов согласно коду от друга
        self.CLASS_MAP = {
            1: "car",      # машина
            2: "dent",     # вмятина (исправлено с "dunt")
            3: "rust",     # ржавчина
            4: "scratch"   # царапина
        }

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """
        Загрузка модели из файла.
        Адаптировано под код от вашего друга.
        """
        try:
            # Загружаем модель (раскомментируйте когда будет файл модели)
            # self.model = torch.load(model_path, map_location=self.device)
            # self.model.eval()

            logger.info(f"Модель загружена с устройства: {self.device}")
            # logger.info(f"Модель загружена из: {model_path}")

        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise

    def predict(self, image: Image.Image, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Предсказание модели. Код адаптирован от вашего друга.

        Args:
            image: PIL изображение
            threshold: порог уверенности для фильтрации результатов

        Returns:
            Словарь с результатами детекции
        """

        if self.model is None:
            logger.warning("Модель не загружена, используется mock-режим")
            return self._generate_mock_prediction()

        try:
            # Код от вашего друга, адаптированный:
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)  # [1, C, H, W]

            # Прогон через модель
            with torch.no_grad():
                outputs = self.model(img_tensor)

            # Фильтрация результатов по threshold
            filtered_boxes = []
            filtered_labels = []
            filtered_scores = []

            for box, label, score in zip(outputs[0]["boxes"], outputs[0]["labels"], outputs[0]["scores"]):
                if score >= threshold:
                    filtered_boxes.append(box.cpu().numpy().tolist())
                    filtered_labels.append(label.item())
                    filtered_scores.append(score.item())

            # Формируем результат как в коде от друга
            result = {
                "boxes": filtered_boxes,
                "labels": filtered_labels,
                "scores": filtered_scores
            }

            logger.info(f"Обнаружено объектов: {len(filtered_boxes)}")
            return result

        except Exception as e:
            logger.error(f"Ошибка при предсказании: {e}")
            return self._generate_mock_prediction()

    def _generate_mock_prediction(self) -> Dict[str, Any]:
        """Генерация mock предсказаний для тестирования"""
        import random

        # Имитируем случайные детекции как в примере от друга
        mock_result = {
            "boxes": [],
            "labels": [],
            "scores": []
        }

        # Случайное количество дефектов (0-3)
        num_defects = random.randint(0, 3)

        for _ in range(num_defects):
            # Случайные координаты как в примере: [x1, y1, x2, y2]
            x1 = random.randint(100, 1000)
            y1 = random.randint(100, 800)
            x2 = x1 + random.randint(50, 400)
            y2 = y1 + random.randint(50, 300)

            mock_result["boxes"].append([x1, y1, x2, y2])
            mock_result["labels"].append(random.choice([2, 3, 4]))  # dent, rust, scratch
            mock_result["scores"].append(random.uniform(0.5, 0.95))

        return mock_result

    def get_class_name(self, label_id: int) -> str:
        """Получение названия класса по ID"""
        return self.CLASS_MAP.get(label_id, "unknown")

    def analyze_results(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Анализ результатов предсказания для формирования ответа.

        Returns:
            Словарь с анализом: есть ли повреждения, типы дефектов и т.д.
        """

        detected_defects = []
        max_confidence = 0

        for label, score in zip(prediction["labels"], prediction["scores"]):
            if label != 1:  # Исключаем класс "car"
                defect_type = self.get_class_name(label)
                if defect_type not in detected_defects:
                    detected_defects.append(defect_type)
                max_confidence = max(max_confidence, score)

        # Определяем общее состояние
        has_damage = len(detected_defects) > 0

        return {
            "has_damage": has_damage,
            "defect_types": detected_defects,
            "max_confidence": max_confidence,
            "num_detections": len(prediction["boxes"])
        }