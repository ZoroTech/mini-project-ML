"""Streamlit UI tests for ML Bench using streamlit.testing.v1.AppTest.

Run:  python tests/app_test.py

Drives the real app: initial load, demo loading, page navigation, the
one-click guided demo (which trains all classification models), the
Predictions and Comparison pages, and Reset.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def find_button(at, label_part: str, sidebar: bool = False):
    root = at.sidebar if sidebar else at
    for button in root.button:
        if label_part in button.label:
            return button
    raise AssertionError(f"button containing '{label_part}' not found (sidebar={sidebar})")


def assert_clean(at, where: str) -> None:
    assert not at.exception, f"{where}: exception in app — {at.exception[0].value if at.exception else ''}"
    print(f"[PASS] UI: {where} renders without exceptions")


# 1. Cold start ----------------------------------------------------------------
at = AppTest.from_file(APP, default_timeout=300).run()
assert_clean(at, "cold start on Dashboard")
assert at.session_state["df"] is None, "no dataset should be loaded at cold start"
print("[PASS] UI: cold start has no dataset")

# 2. Load Demo Dataset from the sidebar --------------------------------------
find_button(at, "Load Demo Dataset", sidebar=True).click().run()
assert_clean(at, "after Load Demo Dataset")
assert at.session_state["df"] is not None and len(at.session_state["df"]) == 600, "student demo should load"
assert at.session_state["target"] == "Performance", f"target = {at.session_state['target']}"
print("[PASS] UI: demo dataset loaded with its predefined target")

# 3. Visit every page with a dataset loaded -----------------------------------
for page in ["Dataset Explorer", "Model Training", "Model Comparison", "Predictions", "About / Help", "Dashboard"]:
    at.session_state["page"] = page
    at.run()
    assert_clean(at, f"page '{page}'")

# 4. Train from the Training page ---------------------------------------------
at.session_state["page"] = "Model Training"
at.run()
find_button(at, "Train models").click().run()
assert_clean(at, "after Train models")
outcome = at.session_state["outcome"]
assert outcome is not None, "training should store an outcome"
ok = [n for n, r in outcome.results.items() if r.status == "ok"]
assert len(ok) >= 4, f"expected most models to train, got ok={ok}"
print(f"[PASS] UI: training from the UI trained {len(ok)} models: {ok}")

# 5. Comparison + feature importance + Predictions render ----------------------
at.session_state["page"] = "Model Comparison"
at.run()
assert_clean(at, "Comparison with results")
find_button(at, "Compute feature importance").click().run()
assert_clean(at, "after Compute feature importance")

at.session_state["page"] = "Predictions"
at.run()
assert_clean(at, "Predictions with trained models")

# 6. Guided demo from a cold session -------------------------------------------
at2 = AppTest.from_file(APP, default_timeout=600).run()
find_button(at2, "Run Complete Demo", sidebar=True).click().run()
assert_clean(at2, "after Run Complete Demo")
assert at2.session_state["outcome"] is not None, "guided demo should leave a trained outcome"
assert at2.session_state["page"] == "Model Comparison", f"page = {at2.session_state['page']}"
ok = [n for n, r in at2.session_state["outcome"].results.items() if r.status == "ok"]
assert len(ok) >= 2, f"guided demo must train at least two models, got {ok}"
print(f"[PASS] UI: guided demo trained {len(ok)} models and landed on Model Comparison")

# 7. Reset returns to the initial state ----------------------------------------
find_button(at2, "Reset Demo", sidebar=True).click().run()
assert_clean(at2, "after Reset Demo")
assert at2.session_state["df"] is None and at2.session_state["outcome"] is None
print("[PASS] UI: Reset Demo cleared dataset and results")

print("\nAll UI tests passed.")
