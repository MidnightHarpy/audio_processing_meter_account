import re
import logging
from rus2num import Rus2Num

logger = logging.getLogger(__name__)


class NumberConverter:
    def __init__(self):
        self.converter = Rus2Num()
        self.digit_merge_pattern = re.compile(r"\b(\d)\s+(?=\d\b)")

    def convert(self, text: str) -> str:
        # конвертирует текстовые числа и объединяет разделённые цифры
        try:
            converted = self.converter(text)

            merged = self._merge_separated_digits(converted)

            return merged
        except Exception as e:
            logger.error(f"Ошибка конвертации: {str(e)}")
            return text

    def _merge_separated_digits(self, text: str) -> str:
        # объединение отдельных цифр
        return self.digit_merge_pattern.sub(r'\1', text)