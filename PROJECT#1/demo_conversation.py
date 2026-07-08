"""
demo_conversation.py
----------------------
Runs a scripted conversation through the exact same get_response()
logic used by the interactive chatbot in chatbot.py -- no duplicated
logic, just a different "Input" source (a fixed script instead of
input()). This makes the chatbot's behavior verifiable and repeatable
without needing a human to type at a terminal, and produces a saved
transcript as proof the knowledge base + fallback + exit path all work.

Deliberately includes:
    - Multiple phrasings of the same intent (case/whitespace variety)
    - Every one of the 6 defined intents
    - An unrecognized message (to prove the FALLBACK_RESPONSE fires)
    - The exit command last (to prove the loop's "Kill Command" works)
"""

from chatbot import get_response

SCRIPT = [
    "Hello",
    "  HeLLo  ",
    "how are you",
    "what is your name",
    "help",
    "thanks",
    "what is decodelabs",
    "asdkjfh this makes no sense",   # unrecognized -> fallback
    "bye",
]


def run_demo():
    transcript_lines = []
    transcript_lines.append("DecodeBot: Hello! I'm a rule-based chatbot. Type 'help' any time, or 'bye' to exit.")

    for message in SCRIPT:
        reply, is_exit = get_response(message)
        transcript_lines.append(f"You: {message}")
        transcript_lines.append(f"DecodeBot: {reply}")
        if is_exit:
            break

    transcript_text = "\n".join(transcript_lines)
    print(transcript_text)

    with open("output/demo_transcript.txt", "w") as f:
        f.write(transcript_text + "\n")
    print("\n[Transcript saved to output/demo_transcript.txt]")


if __name__ == "__main__":
    run_demo()
