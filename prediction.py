import joblib
import numpy as np

class LoanChecker:
    def __init__(self):
        print("📦 Loading AI brain...")
        self.model = joblib.load('loan_model.pkl')
        self.scaler = joblib.load('loan_scaler.pkl')
        print("✅ AI ready!")

    def predict(self, salary_monthly, fico, loan_requested, years_employed, current_debt):
        details = np.array([[salary_monthly, fico, loan_requested, years_employed, current_debt]])
        details_scaled = self.scaler.transform(details)
        prediction = self.model.predict(details_scaled)[0]
        probs = self.model.predict_proba(details_scaled)[0]
        prob_percent = probs[prediction] * 100
        approved = (prediction == 1)

        reasons = []
        if fico >= 750:
            reasons.append("✅ Excellent credit score!")
        elif fico >= 650:
            reasons.append("✅ Good credit score")
        else:
            reasons.append("❌ Low credit score (need 650+)")

        monthly_payment = loan_requested / 240
        if monthly_payment < salary_monthly * 0.4:
            reasons.append("✅ Can afford monthly payments")
        else:
            reasons.append("❌ Monthly payment too high for salary")

        if years_employed >= 3:
            reasons.append("✅ Good job experience")
        elif years_employed >= 1:
            reasons.append("✅ Has job experience")
        else:
            reasons.append("❌ Need at least 1 year job experience")

        if current_debt < salary_monthly * 0.3:
            reasons.append("✅ Low existing debt")
        elif current_debt < salary_monthly * 0.4:
            reasons.append("⚠️ Moderate existing debt")
        else:
            reasons.append("❌ Too much existing debt")

        return {
            'approved': approved,
            'confidence': prob_percent,
            'reasons': reasons
        }

if __name__ == "__main__":
    checker = LoanChecker()
    print("\n" + "="*50)
    print("TEST 1: Good Applicant")
    print("="*50)
    result = checker.predict(salary_monthly=100000, fico=750, loan_requested=2000000, years_employed=5, current_debt=20000)
    print(f"\n{'✅ APPROVED' if result['approved'] else '❌ REJECTED'}")
    print(f"Confidence: {result['confidence']:.1f}%\n")
    print("Reasons:")
    for r in result['reasons']:
        print(f"  {r}")

    print("\n" + "="*50)
    print("TEST 2: Weak Applicant")
    print("="*50)
    result = checker.predict(salary_monthly=30000, fico=580, loan_requested=3000000, years_employed=0, current_debt=15000)
    print(f"\n{'✅ APPROVED' if result['approved'] else '❌ REJECTED'}")
    print(f"Confidence: {result['confidence']:.1f}%\n")
    print("Reasons:")
    for r in result['reasons']:
        print(f"  {r}")
