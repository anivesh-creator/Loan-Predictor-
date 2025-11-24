"""
interface.py

Gradio interface that uses LoanPredictor to evaluate loan applications.
"""

from __future__ import annotations
import logging
from typing import Tuple

import gradio as gr

from prediction import LoanPredictor, PredictorConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

predictor = LoanPredictor(PredictorConfig(model_dir="model"))


def evaluate_application(monthly_salary: int, fico_score: int, requested_loan_amount: int,
                         employment_years: int, existing_debt: int) -> str:
    """
    Wraps predictor.predict and formats a markdown response for Gradio.
    """
    try:
        result = predictor.predict(
            monthly_salary=monthly_salary,
            fico_score=fico_score,
            requested_loan_amount=requested_loan_amount,
            employment_years=employment_years,
            existing_debt=existing_debt
        )
    except Exception as exc:
        logger.exception("Prediction failed")
        return f"# ❌ Error\n\n{str(exc)}"

    approved = result["approved"]
    confidence = result["confidence"]
    reasons = result["reasons"]

    if approved:
        header = f"# ✅ LOAN APPROVED!\n\n## Confidence: {confidence:.1f}%\n\n### Why you got approved:\n"
    else:
        header = f"# ❌ LOAN REJECTED\n\n## Confidence: {confidence:.1f}%\n\n### Why you got rejected:\n"

    body = "\n".join(f"- {r}" for r in reasons)
    return header + "\n" + body


title = "🏦 Loan Approval Checker (Refactor)"
description = "Enter your details and get an indicative model response. This demo uses a trained sklearn pipeline."

demo = gr.Interface(
    fn=evaluate_application,
    inputs=[
        gr.Slider(20_000, 300_000, 80_000, step=5_000, label="💰 Monthly Salary (₹)"),
        gr.Slider(300, 850, 700, step=10, label="💳 FICO Score"),
        gr.Slider(50_000, 5_000_000, 2_000_000, step=50_000, label="🏠 Requested Loan Amount (₹)"),
        gr.Slider(0, 40, 5, step=1, label="🧾 Employment Years"),
        gr.Slider(0, 2_000_000, 20_000, step=5_000, label="💸 Existing Debt (₹)"),
    ],
    outputs=gr.Markdown(label="Result"),
    title=title,
    description=description,
    examples=[
        [100_000, 750, 2_000_000, 5, 20_000],
        [30_000, 580, 3_000_000, 0, 15_000],
        [150_000, 820, 4_000_000, 10, 10_000],
    ],
)

if __name__ == "__main__":
    logger.info("Starting Gradio demo")
    demo.launch(share=False, inbrowser=True, server_port=7860)
