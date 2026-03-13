import pandas as pd
import numpy as np

# Zufällige Daten für die Demo erzeugen
np.random.seed(42)
n = 100
df = pd.DataFrame({
    'flaeche': np.random.randint(50, 200, n),
    'alter': np.random.randint(0, 50, n),
    'lage': np.random.randint(1, 11, n)
})
# Preis berechnen mit etwas Rauschen
df['preis'] = (df['flaeche'] * 3200) - (df['alter'] * 700) + (df['lage'] * 12000) + 50000 + np.random.normal(0, 15000, n)
df.to_csv('hauspreise.csv', index=False)