import pandas as pd
import matplotlib.pyplot as plt
import datetime

# Wczytanie danych z pliku CSV
data = pd.read_csv('../data/Inflacja-dane.csv')

dates = [datetime.datetime(y, m, 1) for m, y in zip(data['miesiąc'], data['rok'])]


# Utworzenie wykresu dla inflacji r/r
plt.figure(figsize=(10, 5))
plt.plot(dates, data['inflacja r/r'], marker='o')
plt.xlabel('Data')
plt.ylabel('Inflacja r/r')
plt.title('Inflacja na przestrzeni lat')
plt.grid(True)
plt.savefig('../img/inflacja_rr.png')

# Utworzenie histogramu dla stopy bezrobocia
plt.figure(figsize=(10, 5))
plt.hist(data['stopa bezrobocia'], bins=10)
plt.xlabel('Stopa bezrobocia')
plt.ylabel('Liczebność')
plt.title('Rozkład stopy bezrobocia')
plt.grid(True)
plt.savefig('../img/stopa_bezrobocia_histogram.png')

# Utworzenie wykresu liniowego dla stopy bezrobocia
plt.figure(figsize=(10, 5))
plt.plot(data['rok'], data['stopa bezrobocia'], marker='o')
plt.xlabel('Rok')
plt.ylabel('Stopa bezrobocia')
plt.title('Stopa bezrobocia w kolejnych latach')
plt.grid(True)
plt.savefig('../img/stopa_bezrobocia.png')

# Utworzenie wykresu liniowego dla PKB r/r
plt.figure(figsize=(10, 5))
plt.plot(data['rok'], data['PKB r/r'], marker='o')
plt.xlabel('Rok')
plt.ylabel('PKB r/r')
plt.title('Wzrost PKB r/r w kolejnych latach')
plt.grid(True)
plt.savefig('../img/pkb_rr.png')

# Utworzenie wykresu słupkowego dla produkcji przemysłowej r/r
plt.figure(figsize=(10, 5))
plt.bar(data['rok'], data['Produkcja przemysłowa r/r'])
plt.xlabel('Rok')
plt.ylabel('Produkcja przemysłowa r/r')
plt.title('Wzrost produkcji przemysłowej r/r w kolejnych latach')
plt.grid(True)
plt.savefig('../img/produkcja_przemyslowa_rr.png')