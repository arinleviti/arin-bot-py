"""
Eval script for arin-bot-py.

Runs a fixed set of known questions against the bot's answer logic and checks
each response against what a "correct" answer should look like. This is the
regression test for arin-bot-py: run it before AND after any change to the
prompt, retrieval, or retry logic, and compare the pass count.

HOW TO RUN (from the arin-bot-py project root, with your venv active):
    python -m app.eval.run_eval

This must be run from the project root (not from inside eval/) because
answer.py connects to ChromaDB using the relative path "data/chroma_db".
"""

# sys.path manipulation so this script can import from "app" when run as a
# module from the project root. TS equivalent: none needed there since Node
# resolves relative to the project root by default — Python needs to be told.
from app.rag.answer import question_main_logic, ANSWER_NOT_FOUND_MARKER


# --- Eval cases ---
# Each case is a dict describing one question and what a correct answer must
# (and must not) look like. Add more cases here over time as you find new
# ones worth locking in as regression tests.
#
# Fields:
#   id                 short unique name for this case (shown in output)
#   question            the question to send to the bot
#   must_contain_any    list of strings — answer must contain AT LEAST ONE
#                        of these (case-insensitive). Use [] to skip this check.
#   must_not_contain    list of strings — answer must contain NONE of these
#                        (case-insensitive). Use [] to skip this check.
#   notes               human-readable reminder of why this case exists
EVAL_CASES = [
    {
        "id": "react_direct_hit",
        "question": "Is he familiar with the framework created by Facebook?",
        "must_contain_any": ["react"],
        "must_not_contain": [ANSWER_NOT_FOUND_MARKER],
        "notes": "Should answer directly from context, no retry needed.",
    },
    {
        "id": "rust_skill_gap",
        "question": "Does he know Rust?",
        "must_contain_any": ["self-taught", "self taught", "learning", "quickly"],
        "must_not_contain": [ANSWER_NOT_FOUND_MARKER],
        "notes": "Genuine skill gap — should land on the friendly fallback with the learning-velocity framing, never the raw marker.",
    },
    {
        "id": "moon_nonsense",
        "question": "How many times has Arin walked on the moon?",
        "must_contain_any": [],
        "must_not_contain": [ANSWER_NOT_FOUND_MARKER],
        "notes": "Nonsense/unanswerable question — should exhaust retries and land on the friendly fallback, never the raw marker.",
    },
    {
        "id": "apple_pie_nonsense",
        "question": "How much flour does he use for his apple pie?",
        "must_contain_any": [],
        "must_not_contain": [ANSWER_NOT_FOUND_MARKER],
        "notes": "Same as moon case — unanswerable, should not leak the marker.",
    },
    {
        "id": "pumpkin_obscured",
        "question": "Does he have a hobby where he waits months to see if something turns orange?",
        "must_contain_any": ["pumpkin"],
        "must_not_contain": [ANSWER_NOT_FOUND_MARKER],
        "notes": "Real fact, deliberately obscured phrasing — known to require retry/reformulation to surface. If this starts failing, retrieval or the retry logic regressed.",
    },
    {
        "id": "chatbot_model",
        "question": "What's the exact AI model that powers his chatbot's brain?",
        "must_contain_any": ["gpt-oss", "groq"],
        "must_not_contain": [ANSWER_NOT_FOUND_MARKER],
        "notes": "Factual accuracy check. NOTE: as of writing, the knowledge base may still say 'Llama 3.3 70B', which is stale — this case will fail until data/knowledge is updated to match the real model in answer.py.",
    },
]


def check_case(case: dict) -> tuple[bool, str, str]:
    """
    Runs one eval case and returns (passed, answer_text, reason).
    reason is a short explanation, only meaningful when passed is False.
    """
    answer = question_main_logic(case["question"])
    answer_lower = answer.lower()

    # dict.get(key, default) returns the value of "key" if it exists in the dict, or the
    # given default if it doesn't. We pass [] as the default so must_contain_any and
    # must_not_contain are always real lists — never None — even if a case
    # forgets to include one of these fields.
    must_contain_any = case.get("must_contain_any", [])
    if must_contain_any:
        # any(...) is Python's "at least one of these is true" — same idea as
        # Array.some(...) in TS.
        found = any(phrase.lower() in answer_lower for phrase in must_contain_any)
        if not found:
            return False, answer, f"expected one of {must_contain_any} in the answer, found none"

    must_not_contain = case.get("must_not_contain", [])
    for phrase in must_not_contain:
        if phrase.lower() in answer_lower:
            return False, answer, f"found forbidden text '{phrase}' in the answer"

    return True, answer, ""


def run_eval():
    print(f"Running {len(EVAL_CASES)} eval cases...\n")
    results = []

    for case in EVAL_CASES:
        passed, answer, reason = check_case(case)
        results.append(passed)

        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {case['id']}")
        print(f"  question: {case['question']}")
        if not passed:
            print(f"  reason:   {reason}")
            print(f"  answer:   {answer}")
        print()

    total = len(results)
    passed_count = sum(results)
    print("-" * 40)
    print(f"Score: {passed_count}/{total} passed")


if __name__ == "__main__":
    run_eval()