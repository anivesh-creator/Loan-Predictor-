import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

print("🔬 Loading data...")
data = pd.read_csv('loan_data.csv')

features = data[['salary_monthly', 'fico', 'loan_requested', 'years_employed', 'current_debt']]
labels = data['loan_approved']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

scaler_obj = StandardScaler()
X_train_scaled = scaler_obj.fit_transform(X_train)
X_test_scaled = scaler_obj.transform(X_test)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

train_score = clf.score(X_train_scaled, y_train)
test_score = clf.score(X_test_scaled, y_test)

print(f"✅ Trained model. Train score: {train_score:.3f}, Test score: {test_score:.3f}")

joblib.dump(clf, 'loan_model.pkl')
joblib.dump(scaler_obj, 'loan_scaler.pkl')
print("✅ Saved loan_model.pkl and loan_scaler.pkl")
