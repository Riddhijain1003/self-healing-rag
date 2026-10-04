"""Tests the critic in isolation: hand-written answers, no retrieval, no generation.

Run:  python3 test_critic.py
"""
from main import grade

CONTEXT = [
    "We do not offer refunds on digital products once downloaded.\n"
    "Damaged or defective items must be reported within 48 hours of delivery with photos.",
    "Electronics come with a 1-year manufacturer warranty.\n"
    "Warranty does not cover accidental damage or water damage.",
]

# (question, candidate answer, expected grounded, expected answered)
CASES = [
    ("Can I get a refund on a gift card?", "Yes, gift cards are refundable within 30 days.", False, True),
    ("How long is the warranty?", "The warranty lasts 2 years.", False, True),
    ("Does the warranty cover water damage?", "No, water damage is not covered.", True, True),
    ("Can I get a refund on a gift card?", "I don't know.", True, False),
]

passed = 0
for question, answer, exp_grounded, exp_answered in CASES:
    state = {"question": question, "documents": CONTEXT, "generation": answer, "retry_count": 0}
    out = grade(state)
    ok = out["grounded"] == exp_grounded and out["answered"] == exp_answered
    passed += ok
    print(f"{'PASS' if ok else 'FAIL'} | {answer!r} -> grounded={out['grounded']}, answered={out['answered']}")

print(f"\n{passed}/{len(CASES)} critic checks passed")