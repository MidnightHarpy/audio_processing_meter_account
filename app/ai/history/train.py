import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
import joblib


current_dir = os.path.dirname(os.path.abspath(__file__))


def prepare_training_data(df, window_size=12):
    df = df.sort_values(['account_id', 'date'])
    groups = df.groupby('account_id')
    windows = []

    #глобальный скейлер для всех клиентов
    global_scaler = MinMaxScaler()
    all_values = df['value'].values.reshape(-1, 1)
    global_scaler.fit(all_values)

    for name, group in groups:
        values = group['value'].values

        #глобальная нормализация
        scaled = global_scaler.transform(values.reshape(-1, 1)).flatten()

        num_windows = len(scaled) // window_size
        for i in range(num_windows):
            start_idx = i * window_size
            end_idx = start_idx + window_size
            window = scaled[start_idx:end_idx]
            windows.append(window)

    return np.array(windows).reshape(-1, window_size, 1), global_scaler


def create_autoencoder(input_shape):
    #создание LSTM автоэнкодера
    model = Sequential([
        LSTM(128, activation='relu', input_shape=input_shape, return_sequences=True),
        LSTM(64, activation='relu', return_sequences=True),
        LSTM(32, activation='relu', return_sequences=False),
        Dense(16, activation='relu'),
        RepeatVector(input_shape[0]),
        LSTM(32, activation='relu', return_sequences=True),
        LSTM(64, activation='relu', return_sequences=True),
        LSTM(128, activation='relu', return_sequences=True),
        TimeDistributed(Dense(1))
    ])

    model.compile(optimizer='adam', loss='mse')
    return model


def train_model():
    data_path = os.path.join(current_dir, "historical_dataset.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # загрузка данных
    df = pd.read_csv(data_path)
    print(f"Загружено записей: {len(df)}")

    # подготовка данных
    WINDOW_SIZE = 12
    train_data, global_scaler = prepare_training_data(df, WINDOW_SIZE)
    print(f"Форма обучающих данных: {train_data.shape}")

    # создание и обучение модели
    autoencoder = create_autoencoder((WINDOW_SIZE, 1))

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        min_delta=0.0001
    )

    history = autoencoder.fit(
        train_data, train_data,
        epochs=15,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stopping],
        shuffle=True
    )


    # создаем директорию для сохранения
    model_dir = os.path.join(current_dir, "model")
    os.makedirs(model_dir, exist_ok=True)


    # сохранение модели
    model_path = os.path.join(model_dir, "anomaly_detector.keras")
    autoencoder.save(model_path, save_format="tf")
    print(f"Модель сохранена: {model_path}")

    # сохраняем глобальный скейлер
    scaler_path = os.path.join(model_dir, "global_scaler.pkl")
    joblib.dump(global_scaler, scaler_path)
    print(f"Глобальный скейлер сохранен: {scaler_path}")

    return autoencoder, history, global_scaler


def calculate_threshold(model, data):
    reconstructions = model.predict(data)
    mse = np.mean(np.square(data - reconstructions), axis=(1, 2))

    return np.percentile(mse, 99)


if __name__ == "__main__":
    try:
        model, history, global_scaler = train_model()

        # формируем путь к данным для расчета порога
        data_path = os.path.join(current_dir, "historical_dataset.csv")
        df = pd.read_csv(data_path)

        train_data, _ = prepare_training_data(df)
        threshold = calculate_threshold(model, train_data)
        print(f"Порог аномалий: {threshold:.4f}")

        # сохраняем порог
        threshold_path = os.path.join(current_dir, "model", "threshold.txt")
        with open(threshold_path, "w") as f:
            f.write(str(threshold))
        print(f"Порог сохранен: {threshold_path}")

        # сохраняем историю обучения
        history_df = pd.DataFrame(history.history)
        history_path = os.path.join(current_dir, "model", "training_history.csv")
        history_df.to_csv(history_path, index=False)
        print(f"История обучения сохранена: {history_path}")

    except Exception as e:
        print(f"Ошибка: {str(e)}")
        raise