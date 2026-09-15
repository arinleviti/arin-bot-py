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
#loads env file
load_dotenv()

# Resend's Python SDK works by setting this module-level attribute once,
# rather than creating a client object like Groq's `Groq(api_key=...)`.
# os.environ lets Python access environment variables
resend.api_key = os.environ["RESEND_API_KEY"]

# The address that receives the notification — you, the site owner.
# Kept as an env var rather than hardcoded so it's not committed to git,
# and so it can differ between local testing and production if you ever want that.
NOTIFY_EMAIL = os.environ["NOTIFY_EMAIL"]


def notify_arin(visitor_message: str, visitor_contact: str = None) -> bool:
    """
    Sends Arin an email about a visitor's hiring interest / contact request.

    visitor_message: what the visitor said (the chatbot message that triggered this)
    visitor_contact: optional — an email/contact detail the visitor provided, if any

    Returns True if the email was sent successfully, False otherwise.
    We return a bool (rather than letting exceptions bubble up) because later,
    when this is called BY the LLM as a tool, we need a clean success/failure
    signal to hand back to the model — not a crashed request.
    """
    contact_line = (
        f"<p><strong>Contact info provided:</strong> {visitor_contact}</p>"
        if visitor_contact
        else "<p><em>No contact info was provided by the visitor.</em></p>"
    )

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
    # Manual test — run "python -m app_tools.send_email" (or wherever this
    # ends up living in your project) and check your inbox.
    success = notify_arin(
        visitor_message="Hi, I saw your portfolio and I'm very interested in hiring you!",
        visitor_contact="recruiter@example.com",
    )
    print(f"Email sent: {success}")