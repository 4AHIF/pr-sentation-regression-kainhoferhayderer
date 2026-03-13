import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. DATEN LADEN
df = pd.read_csv('hauspreise.csv')
print("--- Erste 5 Zeilen der Daten ---")
print(df.head())

# 2. FEATURES & ZIELVARIABLE TRENNEN
# X sind die Einflussfaktoren, y ist das, was wir wissen wollen
X = df[['flaeche', 'alter', 'lage']]
y = df['preis']

# 3. SPLIT IN TRAINING & TEST (Die 'Schularbeit-Metapher')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. DAS MODELL TRAINIEREN (Das 'Mischpult' einstellen)
model = LinearRegression()
model.fit(X_train, y_train)
print("\nModell wurde erfolgreich trainiert!")

# --- 5. INTERPRETATION (Der Algorithmus erklärt seine Logik) ---
print("\n")
print("="*40)
print("🤖 ANALYSE DES KI-MODELLS")
print("="*40)
print("\n")

print(f"Basispreis (wenn alles 0 wäre): {model.intercept_:,.2f} €")
print("-"*40)

features = ['Wohnfläche (pro m²)', 'Alter (pro Jahr)', 'Lage (pro Punkt)']
for name, coef in zip(features, model.coef_):
    richtung = "Steigerung" if coef > 0 else "Verlust"
    print(f"➡️ {name}: {abs(coef):,.2f} € {richtung}")

# --- 6. DIE GÜTE (Wie vertrauenswürdig ist die KI?) ---
r2 = model.score(X_test, y_test)
print("-"*40)
print(f"Modell-Genauigkeit (R²): {r2:.2%}")
# .2% verwandelt 0.85 in "85.00%"

if r2 > 0.8:
    print("✅ Urteil: Sehr zuverlässiges Modell.")
else:
    print("⚠️ Urteil: Modell hat noch hohe Unsicherheiten.")

# --- 7. LIVE-TEST (Interaktion) ---
print("-"*40)
# Hier kannst du Werte aus der Klasse einsetzen (Fläche, Alter, Lage)
test_haus = [[120, 10, 8]]
preis = model.predict(test_haus)[0]

print(f"🏠 PROGNOSE für das Beispiel-Haus:")
print(f"Geschätzter Marktwert: {preis:,.2f} €")
print("="*40)