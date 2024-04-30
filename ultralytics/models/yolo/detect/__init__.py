# Ultralytics YOLO 🚀, AGPL-3.0 license

from .predict import DetectionPredictor
from .train import DetectionTrainer, DistilationDetectionTrainer
from .val import DetectionValidator, DetectionDistillValidator

__all__ = "DetectionPredictor", "DetectionTrainer", "DetectionValidator"
