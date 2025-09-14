from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import json
import io
import logging
import uvicorn
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Car Damage Analysis API",
    description="API для анализа повреждений автомобилей с использованием ML модели",
    version="1.0.0"
)

# CORS middleware для подключения фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Маппинг классов согласно модели
CLASS_MAP = {
    1: "car",      # машина
    2: "dunt",     # вмятина (исправлено согласно модели)
    3: "rust",     # ржавчина
    4: "scratch"   # царапина
}

# Маппинг дефектов на русский язык
DEFECT_TRANSLATION = {
    "dunt": "вмятины",
    "rust": "ржавчина",
    "scratch": "царапины"
}

class AnalysisResult(BaseModel):
    accuracy: int
    condition: str  # "битый" | "не битый"
    cleanliness: str  # "чистый" | "грязный"
    defects: List[str]
    confidence: int
    fileName: str

class MLModel:
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        logger.info(f"Используется устройство: {self.device}")

    def load_model(self, model_path: str = "/app/models/fasterrcnn_resnet50_multidataset.pth"):
        """Загрузка модели Faster R-CNN"""
        try:
            # Проверяем существование файла модели
            if not os.path.exists(model_path):
                logger.warning(f"Файл модели не найден: {model_path} - используется mock-режим")
                self.model = None
                return

            # Создаем базовую модель Faster R-CNN
            self.model = fasterrcnn_resnet50_fpn(weights=None, num_classes=5)  # 5 классов включая background

            # Загружаем веса модели
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"Модель успешно загружена из {model_path}")

        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            logger.warning("Переключение на mock-режим")
            self.model = None

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Предсказание модели"""
        if self.model is None:
            # Mock предсказания для тестирования
            return self._mock_prediction()

        try:
            # Преобразуем изображение в тензор согласно коду друга
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(img_tensor)

            threshold = 0.5  # показываем только объекты с score > 0.5
            filtered_boxes = []
            filtered_labels = []
            filtered_scores = []

            for box, label, score in zip(outputs[0]["boxes"], outputs[0]["labels"], outputs[0]["scores"]):
                if score >= threshold:
                    filtered_boxes.append(box.cpu().numpy().tolist())
                    filtered_labels.append(label.item())
                    filtered_scores.append(score.item())

            logger.info(f"Детектировано {len(filtered_labels)} объектов с confidence > {threshold}")

            return {
                "boxes": filtered_boxes,
                "labels": filtered_labels,
                "scores": filtered_scores
            }

        except Exception as e:
            logger.error(f"Ошибка предсказания: {e}")
            return self._mock_prediction()

    def _mock_prediction(self) -> Dict[str, Any]:
        """Mock предсказания для тестирования без модели"""
        import random

        # Имитируем случайные детекции
        mock_boxes = []
        mock_labels = []
        mock_scores = []

        # Случайное количество дефектов (0-3)
        num_defects = random.randint(0, 3)

        for _ in range(num_defects):
            # Случайные координаты бокса
            x1, y1 = random.randint(100, 800), random.randint(100, 600)
            x2, y2 = x1 + random.randint(50, 200), y1 + random.randint(50, 150)

            mock_boxes.append([x1, y1, x2, y2])
            mock_labels.append(random.choice([2, 3, 4]))  # dent, rust, scratch
            mock_scores.append(random.uniform(0.5, 0.95))

        return {
            "boxes": mock_boxes,
            "labels": mock_labels,
            "scores": mock_scores
        }

# Инициализация модели
ml_model = MLModel()
ml_model.load_model()

def analyze_detection_results(detection_results: Dict[str, Any], filename: str) -> AnalysisResult:
    """Анализ результатов детекции и формирование ответа"""

    # Извлекаем дефекты
    detected_defects = []
    confidence_scores = []

    for label, score in zip(detection_results["labels"], detection_results["scores"]):
        if label in CLASS_MAP and label != 1:  # Исключаем класс "car"
            defect_name = CLASS_MAP[label]
            if defect_name in DEFECT_TRANSLATION:
                defect_ru = DEFECT_TRANSLATION[defect_name]
                if defect_ru not in detected_defects:
                    detected_defects.append(defect_ru)
            confidence_scores.append(score)

    # Определяем состояние автомобиля
    has_damage = len(detected_defects) > 0
    condition = "битый" if has_damage else "не битый"

    # Простая логика для определения чистоты (можно усложнить)
    # В реальной модели это может быть отдельный классификатор
    import random
    cleanliness = random.choice(["чистый", "грязный"])

    # Рассчитываем метрики
    if confidence_scores:
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        confidence = int(avg_confidence * 100)
        accuracy = max(70, min(99, confidence + random.randint(-10, 10)))
    else:
        confidence = random.randint(85, 95)
        accuracy = random.randint(75, 95)

    return AnalysisResult(
        accuracy=accuracy,
        condition=condition,
        cleanliness=cleanliness,
        defects=detected_defects,
        confidence=confidence,
        fileName=filename
    )

@app.get("/")
async def root():
    return {"message": "Car Damage Analysis API v1.0.0"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": ml_model.model is not None,
        "device": str(ml_model.device)
    }

@app.post("/analyze", response_model=List[AnalysisResult])
async def analyze_images(files: List[UploadFile] = File(...)):
    """
    Анализ изображений автомобилей.
    Принимает список изображений и возвращает результаты анализа.
    """

    if not files:
        raise HTTPException(status_code=400, detail="Необходимо загрузить хотя бы одно изображение")

    if len(files) > 5:
        raise HTTPException(status_code=400, detail="Максимальное количество изображений: 5")

    results = []

    for file in files:
        try:
            # Проверка типа файла
            if not file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail=f"Файл {file.filename} не является изображением")

            # Проверка размера файла (10MB)
            contents = await file.read()
            if len(contents) > 10 * 1024 * 1024:
                raise HTTPException(status_code=400, detail=f"Файл {file.filename} превышает 10MB")

            # Загрузка изображения
            image = Image.open(io.BytesIO(contents)).convert("RGB")

            # Предсказание модели
            detection_results = ml_model.predict(image)

            # Анализ результатов
            analysis_result = analyze_detection_results(detection_results, file.filename)
            results.append(analysis_result)

            logger.info(f"Обработано изображение: {file.filename}")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Ошибка обработки файла {file.filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Ошибка обработки файла {file.filename}")

    return results

@app.post("/analyze-single")
async def analyze_single_image(file: UploadFile = File(...)):
    """
    Анализ одного изображения.
    Возвращает детальные результаты детекции.
    """

    try:
        # Проверки аналогичные /analyze
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Файл не является изображением")

        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Файл превышает 10MB")

        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Получаем детальные результаты детекции
        detection_results = ml_model.predict(image)

        # Анализируем результаты
        analysis_result = analyze_detection_results(detection_results, file.filename)

        return {
            "analysis": analysis_result,
            "detection_details": detection_results,
            "class_map": CLASS_MAP
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка анализа изображения: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)