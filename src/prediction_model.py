import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


def main():
    # usunąłem dane do 1993 roku ze względu na ich brak, był tylko rok, miesiąc i wysokość inflacji
    data = pd.read_csv('../data/Inflacja-dane.csv', delimiter=',', dtype=float)
    data = data.fillna(0.0)

    # w danych wejściowych mogę uwzględnić inflację, ponieważ wszystkie dane dotyczą danych z poprzedniego miesiąca
    input_data = data[['inflacja r/r', 'stopa bezrobocia', 'PKB r/r', 'PKB kwartalnie', 'podaż pieniądza M1 (mld zł)',
                    'Wynagrodzenia przeciętne', 'Inflacja HICP r/r (Strefa Euro)', 'Kurs dolar amerykański NBP',
                    'Produkcja przemysłowa r/r']]
    inflation = data['inflacja r/r']

    # przesunięcie danych - jako wynik wynik funkcji wzracany wskaźnik inflacji w następnym miesiącu
    inflation = inflation[1:]
    last_row = input_data.tail(1)
    # usunięcie ostatniego wiersza - zgodność danych
    input_data = input_data.drop(input_data.index[-1])

    # normalizacja danych wejściowych
    scaler = MinMaxScaler()
    normalizated_data = scaler.fit_transform(input_data)

    # Budowa modelu 
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(9,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Uczenie modelu na całym zbiorze danych
    model.fit(normalizated_data, inflation, epochs=400)

    # Podział na zbiór treningowy i testowy
    split_index = int(0.8 * len(normalizated_data))
    train_data, train_labels = normalizated_data[:split_index], inflation[:split_index]
    test_data, test_labels = normalizated_data[split_index:], inflation[split_index:]

    model_splitted = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(9,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model_splitted.compile(optimizer='adam', loss='mean_squared_error')
    model_splitted.fit(train_data, train_labels, epochs=400)

    loss = model.evaluate(normalizated_data, inflation)
    print('All data Loss:', loss)

    loss = model_splitted.evaluate(test_data, test_labels)
    print('Split data Loss:', loss)


    last_row_data = scaler.transform(last_row)

    # Predykcja dla ostatniego miesiąca
    predictions = model.predict(last_row_data)
    print('All data Predictions:', predictions)

    predictions = model_splitted.predict(last_row_data)
    print('Split data Predictions:', predictions)


    predictions_all_data = scaler.inverse_transform(predictions_all_data)
    predictions_splitted_data = scaler.inverse_transform(predictions_splitted_data)
    inflation_actual = scaler.inverse_transform([last_row['inflacja r/r'].values])

    mse_all_data = mean_squared_error(inflation_actual, predictions_all_data)
    mse_splitted_data = mean_squared_error(inflation_actual, predictions_splitted_data)
    r2_all_data = r2_score(inflation_actual, predictions_all_data)
    r2_splitted_data = r2_score(inflation_actual, predictions_splitted_data)

    print('Mean Squared Error on all data:', mse_all_data)
    print('Mean Squared Error on split data:', mse_splitted_data)
    print('R2 Score on all data:', r2_all_data)
    print('R2 Score on split data:', r2_splitted_data)

    # Zapisanie modelu do pliku
    model.save('model.h5')

if __name__ == '__main__':
    main()
