"""
Standalone email-sending tool. This file has ONE job: given a message about a
visitor's interest, send a real email to Arin via Resend. It knows nothing
about the LLM, tool-calling, or the chatbot — that separation matters, because
it means we can test "does sending an email actually work" completely
independently from "does the model decide to call this correctly."
"""

import os
import resend
from dotenv import load_dotenv

load_dotenv()

# Resend's Python SDK works by setting this module-level attribute once,
# rather than creating a client object like Groq's `Groq(api_key=...)`.
resend.api_key = os.environ["RESEND_API_KEY"]

# The address that receives the notification — you, the site owner.
# Kept as an env var rather than hardcoded so it's not committed to git,
# and so it can differ between local testing and production if you ever want that.
NOTIFY_EMAIL = os.environ["NOTIFY_EMAIL"]


def notify_arin(visitor_message: str, visitor_contact: str = None) -> bool:
    """
    Sends Arin an email about a visitor's hiring interest / contact request.

    visitor_message: what the visitor said (the chatbot message that triggered this)
    visitor_contact: an email/contact detail the visitor provided — REQUIRED in
                      practice (see guard below), even though Python can't enforce
                      that at the signature level here (see note below).

    Returns True if the email was sent successfully, False otherwise.
    We return a bool (rather than letting exceptions bubble up) because later,
    when this is called BY the LLM as a tool, we need a clean success/failure
    signal to hand back to the model — not a crashed request.
    """
    # This is the REAL enforcement of "don't send without contact info" — not
    # the function signature. The caller (answer.py) always passes
    # visitor_contact as a keyword argument, even when its value is None
    # (args.get("visitor_contact") returns None if the key is missing).
    # Removing the "= None" default above wouldn't help: the argument is
    # never actually OMITTED from the call, only its VALUE can be empty.
    # So we check the value here instead — a real backstop in case the LLM
    # ever ignores the prompt instructions and calls this without contact info.
    if not visitor_contact:
        print("[send_email] refused to send — no visitor contact info provided")
        return False

    contact_line = f"<p><strong>Contact info provided:</strong> {visitor_contact}</p>"

    try:
        resend.Emails.send({
            # onboarding@resend.dev is Resend's shared test sender — fine for
            # prototyping. Once you verify your own domain in Resend, swap this
            # to something like "bot@arinleviti.site" for a more polished look.
            "from": "onboarding@resend.dev",
            "to": [NOTIFY_EMAIL],
            "subject": "Someone on your portfolio site wants to get in touch",
            "html": (
                f"<p><strong>Visitor's message:</strong> {visitor_message}</p>"
                f"{contact_line}"
            ),
        })
        return True
    except Exception as e:
        # Printing here for now since we're still testing manually — once this
        # is wired into the bot, this is where you'd want real logging instead.
        print(f"[send_email] failed to send: {e}")
        return False


if __name__ == "__main__":
    # Manual test — run "python -m app.tools.send_email" (adjust the module
    # path to match your project) and check your inbox.
    print("Test 1: with contact info (should send)")
    success = notify_arin(
        visitor_message="Hi, I saw your portfolio and I'm very interested in hiring you!",
        visitor_contact="recruiter@example.com",
    )
    print(f"Email sent: {success}\n")

    print("Test 2: without contact info (should be refused, no email)")
    success = notify_arin(
        visitor_message="I'd like to talk to Arin sometime.",
        visitor_contact=None,
    )
    print(f"Email sent: {success}")