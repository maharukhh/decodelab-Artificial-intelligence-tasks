"""
chatbot.py
-----------
Project 1 -- Rule-Based AI Chatbot.

This is the "Logic Engine" from the slide deck: a fully deterministic,
white-box conversational program. Every response is traceable straight
back to an explicit rule -- there is no model, no probability, no
hallucination risk. Per the "Strategic Necessity of the White Box"
slide, this is exactly the property that makes rule-based systems
essential for compliance-heavy domains (finance, healthcare) and for
the "AI guardrail" layer that sits in front of real LLMs in production.

Architecture (matches "The Blueprint: IPO Model" slide):
    INPUT   -> get_input() + sanitize()      (sanitization & normalization)
    PROCESS -> get_response()                (intent matching via dict, not if-elif)
    OUTPUT  -> print the reply               (response generation)
    LOOP    -> run_interactive()             (the "Heartbeat": while True + exit break)

Design choice -- dictionary lookup over an if-elif ladder:
    The slides make a specific, deliberate point here (the "Anti-Pattern:
    The If-Elif Ladder" + "The Pivot: Hash Maps & Dictionaries" +
    "Algorithmic Efficiency" slides): a long if-elif chain is O(n) --
    every additional rule makes every lookup slower and the code harder
    to maintain. A dictionary lookup via .get() is O(1) -- constant time
    no matter how many rules exist -- and handles the lookup AND the
    fallback in a single atomic expression. This file uses .get()
    exclusively; there is no if-elif ladder anywhere in the intent logic.
"""

# ---------------------------------------------------------------------
# Knowledge base, Phase 1: many raw phrasings -> one canonical intent.
# This is still a plain dictionary lookup (O(1)), just a two-stage one --
# it lets the bot recognize "hi", "hello" and "hey" as the same intent
# without writing three separate response entries (or an if-elif chain).
# ---------------------------------------------------------------------
INTENT_MAP = {
    # Greetings
    "hi": "greeting", "hello": "greeting", "hey": "greeting",
    "hiya": "greeting", "good morning": "greeting", "good evening": "greeting",

    # Exit commands
    "bye": "exit", "goodbye": "exit", "exit": "exit", "quit": "exit", "q": "exit",

    "how are you": "wellbeing", "how are you doing": "wellbeing",

    "what is your name": "identity", "who are you": "identity",

    "what can you do": "capabilities", "help": "capabilities",

    "thanks": "gratitude", "thank you": "gratitude", "thx": "gratitude",

    "what is decodelabs": "about_org",
}

# ---------------------------------------------------------------------
# Knowledge base, Phase 2: canonical intent -> the actual reply.
# 6 distinct intents (+ fallback) -- clears the brief's "5+ intents"
# requirement with room to spare.
# ---------------------------------------------------------------------
RESPONSES = {
    "greeting": "Hi there! I'm DecodeBot, your rule-based training assistant. Type 'help' to see what I can do.",
    "wellbeing": "I'm just a set of if-else... well, dictionary rules, so I'm always running at 100%! How about you?",
    "identity": "I'm DecodeBot -- a deterministic, rule-based chatbot built for Project 1 of the AI Industrial Training Kit.",
    "capabilities": "I can greet you, answer a few basic questions, and say goodbye. Every reply I give comes from an explicit rule -- no guessing.",
    "gratitude": "You're welcome!",
    "about_org": "DecodeLabs runs the Industrial Training Kit this project is part of.",
    "exit": "Goodbye! Thanks for chatting with DecodeBot.",
}

FALLBACK_RESPONSE = "I do not understand. Type 'help' to see what I can talk about, or 'bye' to exit."

EXIT_INTENT = "exit"


def sanitize(raw_input):
    """
    Phase 1 -- Input & Sanitization (per the slide of the same name).
    Normalizes case and strips stray whitespace so "Hello", "hello ",
    and "HELLO" all match the same rule.
    """
    return raw_input.lower().strip()


def get_response(raw_input):
    """
    Phase 2 -- Process (the Logic Skeleton).

    Two chained O(1) dictionary lookups, each with its own fallback,
    handled by .get() -- no if-elif ladder. Returns (reply_text, is_exit).
    """
    clean_input = sanitize(raw_input)
    intent = INTENT_MAP.get(clean_input)  # None if no phrasing matched
    reply = RESPONSES.get(intent, FALLBACK_RESPONSE)
    is_exit = intent == EXIT_INTENT
    return reply, is_exit


def run_interactive():
    """
    Phase 3 -- Output & The Heartbeat (the infinite loop).
    Runs the chatbot in a continuous while-True cycle until the user
    issues an exit command -- the "Kill Command" from the slides.
    """
    print("DecodeBot: Hello! I'm a rule-based chatbot. Type 'help' any time, or 'bye' to exit.")
    while True:
        raw_input_text = input("You: ")
        reply, is_exit = get_response(raw_input_text)
        print(f"DecodeBot: {reply}")
        if is_exit:
            break


if __name__ == "__main__":
    run_interactive()
