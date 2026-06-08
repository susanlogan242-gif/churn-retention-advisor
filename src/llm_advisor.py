"""Grounded LLM retention advisor (M4).

Given a customer's profile, churn probability, £-at-risk and SHAP drivers,
ask Claude for a structured retention play (JSON):
  { risk_summary, drivers_in_plain_english[], retention_actions[{action, why, offer}],
    suggested_message, priority }

Design rules:
- Ground every recommendation in the supplied drivers (no invented data).
- Force structured JSON output.
- Cache the static system prompt (prompt caching) to keep cost ~£0.
- Guardrail: reject/flag any action not tied to a real driver.
- Use Haiku for batch, Sonnet for the live showcase customer.
"""

# TODO M4: anthropic client, system prompt (cached), build_user_message(customer, drivers),
#          get_retention_play(...) -> dict, plus the grounding guardrail check.

if __name__ == "__main__":
    raise SystemExit("Not implemented yet — see notebooks/churn_advisor.ipynb")
