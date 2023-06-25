import pandas as pd
import matplotlib.pyplot as plt
import datetime

def normalize_data(data):
    return (data - data.min()) / (data.max() - data.min())

data = pd.read_csv('../data/Inflacja-dane.csv')
dates = [datetime.datetime(y, m, 1) for m, y in zip(data['miesiąc'], data['rok'])]

# Utworzenie histogramu dla stopy bezrobocia
plt.figure()
plt.hist(data['inflacja r/r'], bins=10)
plt.xlabel('Inflacja')
plt.ylabel('Liczebność')
plt.title('Rozkład inflacji')
plt.grid(True)
plt.savefig('../img/inflacja_histogram.png')

# Utworzenie wykresu zawierającego wszystkie dane (znormalizowane do zakresu (0, 1)
plt.figure(figsize=(20, 10))
plt.scatter(dates, [x/100 - 1 for x in data.iloc[:, 2]], marker='o', label='Inflacja (%)')
plt.scatter(dates, normalize_data(data.iloc[:, 3]), marker='.', label='Stopa bezrobocia')
plt.scatter(dates, normalize_data(data.iloc[:, 4]), marker='.', label='PKB')
plt.scatter(dates, normalize_data(data.iloc[:, 5]), marker='.', label='Podaż pieniądza')
plt.scatter(dates, normalize_data(data.iloc[:, 6]), marker='.', label='Wynagrodzenia przeciętne')
plt.scatter(dates, normalize_data(data.iloc[:, 7]), marker='.', label='HICP (Strefa Euro)')
plt.scatter(dates, normalize_data(data.iloc[:, 8]), marker='.', label='Kurs USD wg NBP')
plt.scatter(dates, normalize_data(data.iloc[:, 9]), marker='.', label='Produkcja przemysłowa')
plt.xlabel('Rok')
plt.ylabel('Wartość')
plt.title('Analiza danych gospodarczych')
plt.legend()
plt.savefig('../img/analiza_danych_gospodarczych.png')

# Utworzenie wykresu loss
with open('../data/loss.txt', 'r') as f:
    loss = [float(x) for x in f.readlines()]
    plt.figure()
    plt.plot(loss)
    plt.xlabel('Epoka')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.grid(True)
    plt.savefig('../img/loss.png')


