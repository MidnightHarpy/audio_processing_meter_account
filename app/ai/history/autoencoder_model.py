import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def prepare_training_data(df, window_size=6):
    #сортировка и группировка
    df = df.sort_values(['account_id', 'date'])
    groups = df.groupby('account_id')

    windows = []
    scaler = MinMaxScaler()

    for name, group in groups:
        values = group['value'].values

        #нормализация для каждого клиента отдельно
        scaled = scaler.fit_transform(values.reshape(-1, 1)).flatten()

        for i in range(window_size, len(scaled)):
            window = scaled[i - window_size:i]
            windows.append(window)

    return np.array(windows).reshape(-1, window_size, 1)


#загрузка данных
df = pd.read_csv("training_dataset.csv")

train_data = prepare_training_data(df)

print(f"Форма обучающих данных: {train_data.shape}")