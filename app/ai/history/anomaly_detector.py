import numpy as np
import joblib
import os
import logging
from tensorflow.keras.models import load_model
import tensorflow as tf


class AnomalyDetector:
    def __init__(self,
                 model_path='model/anomaly_detector.keras',
                 threshold_path='model/threshold.txt',
                 scaler_path='model/global_scaler.pkl'):

        current_dir = os.path.dirname(os.path.abspath(__file__))

        self.model_path = os.path.join(current_dir, "model/anomaly_detector")
        self.threshold_path = os.path.join(current_dir, threshold_path)
        self.scaler_path = os.path.join(current_dir, scaler_path)

        self.logger = logging.getLogger(f"{__name__}.AnomalyDetector")
        self.logger.info("Initializing AnomalyDetector...")

        # Проверяем существование файлов
        if not os.path.exists(self.model_path):
            error_msg = f"Model file not found: {self.model_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        if not os.path.exists(self.threshold_path):
            error_msg = f"Threshold file not found: {self.threshold_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        if not os.path.exists(self.scaler_path):
            error_msg = f"Scaler file not found: {self.scaler_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # загружаем модель, порог и скейлер
        self.model = tf.keras.models.load_model(
            self.model_path,
            compile=False,
            custom_objects={
                'LSTM': tf.keras.layers.LSTM,
                'RepeatVector': tf.keras.layers.RepeatVector,
                'TimeDistributed': tf.keras.layers.TimeDistributed
            }
        )
        self.threshold = self._load_threshold(self.threshold_path)
        self.scaler = joblib.load(self.scaler_path)
        self.window_size = 12

        self.logger.info("Anomaly detector initialized")
        self.logger.info(f"Anomaly threshold: {self.threshold:.6f}")

    def _load_threshold(self, path):
        try:
            with open(path, 'r') as f:
                return float(f.read().strip())
        except Exception as e:
            self.logger.error(f"Error loading threshold: {e}")
            return 0.01

    def detect(self, account_id, history, new_value):
        try:
            all_values = np.array(history + [new_value]).reshape(-1, 1)

            #нормализация с использованием глобального скейлера
            scaled_values = self.scaler.transform(all_values).flatten()

            window = scaled_values[-self.window_size:]
            window = window.reshape(1, self.window_size, 1)

            #реконструкция
            reconstruction = self.model.predict(window, verbose=0)

            #вычисление MSE
            mse = np.mean(np.square(window - reconstruction))

            #денормализация реконструированного значения
            denorm_reconstruction = self.scaler.inverse_transform(
                reconstruction.reshape(-1, 1)
            )
            denorm_value = denorm_reconstruction[-1][0]

            #проверка на аномалию
            is_anomaly = mse > self.threshold

            self.logger.debug(f"Account: {account_id}, Value: {new_value}, "
                              f"Reconstructed: {denorm_value:.2f}, MSE: {mse:.6f}, "
                              f"Anomaly: {is_anomaly}")

            return is_anomaly, mse, denorm_value

        except Exception as e:
            self.logger.error(f"Error detecting anomaly: {e}")
            # В случае ошибки возвращаем "не аномалия"
            return False, 0.0, new_value