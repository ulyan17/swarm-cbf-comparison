"""Абстрактный базовый класс для контроллеров."""
from abc import ABC, abstractmethod
import numpy as np
from src.environment import Environment

class BaseController(ABC):
    @abstractmethod
    def compute(self, env: Environment) -> list[np.ndarray]:
        """Возвращает список управлений [u1, u2, ...] для каждого агента."""
        pass