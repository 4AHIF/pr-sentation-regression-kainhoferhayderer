import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Daten einlesen
df = pd.read_csv('hauspreise.csv')
print(df.head())

X = df[['flaeche', 'alter', 'lage']]  # Ausgansparameter
y = df['preis']  # Zielparameter

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modell trainieren
model = LinearRegression()
model.fit(X_train, y_train)

print(f"\nBasispreis (Intercept): {model.intercept_:,.2f} €")
print("\nEinfluss der einzelnen Faktoren:")

features = ['Wohnfläche (pro Quadratmeter)', 'Alter (pro Jahr)', 'Lage (pro Punkt)']
for name, koeffizient in zip(features, model.coef_):
    richtung = "teurer" if koeffizient > 0 else "günstiger"
    print(f"  {name}: {abs(koeffizient):,.2f} € {richtung}")

# Wie gut ist das Modell? R^2 liefert Anhaltspunkt
r2 = model.score(X_test, y_test)
print(f"\nR^2-Wert: {r2:.2%}")

if r2 > 0.8:
    print("→ Supa Modell")
else:
    print("→ Schlechtes Modell")

# Testbeispiel
testhaus = [[120, 10, 8]]
preis = model.predict(testhaus)[0]
print(f"\nGeschätzter Preis für das Testhaus: {preis:,.2f} €")