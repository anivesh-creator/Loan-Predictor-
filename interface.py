import gradio as gr
from prediction import LoanChecker

checker = LoanChecker()

def evaluate_application(salary_monthly, fico, loan_requested, years_employed, current_debt):
    result = checker.predict(salary_monthly, fico, loan_requested, years_employed, current_debt)
    if result['approved']:
        output = f"# ✅ LOAN APPROVED!\n\n## Confidence: {result['confidence']:.1f}%\n\n### Why you got approved:\n"
    else:
        output = f"# ❌ LOAN REJECTED\n\n## Confidence: {result['confidence']:.1f}%\n\n### Why you got rejected:\n"
    for reason in result['reasons']:
        output += f"\n{reason}"
    return output

demo = gr.Interface(
    fn=evaluate_application,
    inputs=[
        gr.Slider(25000, 200000, 80000, 5000, label="💰 Monthly Salary (₹)"),
        gr.Slider(300, 850, 700, 10, label="💳 Credit Score"),
        gr.Slider(100000, 5000000, 2000000, 100000, label="🏠 Loan Amount Wanted (₹)"),
        gr.Slider(0, 30, 5, 1, label="💼 Job Experience (years)"),
        gr.Slider(0, 100000, 20000, 5000, label="📊 Existing Debt (₹)")
    ],
    outputs=gr.Markdown(label="Result"),
    title="🏦 Loan Approval Checker",
    description="Enter your details below and check if you'll get the loan!",
    examples=[
        [100000, 750, 2000000, 5, 20000],
        [30000, 580, 3000000, 0, 15000],
        [150000, 820, 4000000, 10, 10000],
    ]
)

if __name__ == "__main__":
    print("\n🌐 Starting website...")
    print("📱 URL: http://localhost:7860\n")
    print("💡 Press Ctrl+C to stop")
    demo.launch(share=False, inbrowser=True, server_port=7860)
