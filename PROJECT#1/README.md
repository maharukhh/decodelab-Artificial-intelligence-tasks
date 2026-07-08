# Project 1 — Rule-Based AI Chatbot
### DecodeLabs Artificial Intelligence Internship — Industrial Training Kit

## 🎯 Goal
Create a simple rule-based chatbot that responds to predefined user
inputs — handling greetings and exit commands, using dictionary-based
decision logic (not an if-elif ladder), and running in a continuous
loop.

## 📁 Folder contents

```
project1/
├── README.md                    ← this file
├── chatbot.py                   ← the chatbot itself (run this to chat)
├── demo_conversation.py         ← scripted, non-interactive proof run
└── output/
    └── demo_transcript.txt      ← generated after running the demo
```

> **Why is there both `chatbot.py` and `demo_conversation.py`?**
> `chatbot.py` is the real thing — an interactive `while True` loop you
> can actually type into. `demo_conversation.py` calls the *exact same*
> `get_response()` function with a fixed script of test messages instead
> of live keyboard input, so there's a repeatable, saved transcript
> proving every intent, the fallback, and the exit command all work —
> without needing anyone to sit and type at a terminal to verify it.

No external dependencies — pure Python standard library.

## 🧠 How it matches the slide deck's architecture

### The IPO Model
- **Input** (`sanitize()`) — lowercases and strips the raw input, so
  `"Hello"`, `"hello "`, and `"HELLO"` all match the same rule (the
  "Phase 1: Input & Sanitization" slide's exact example).
- **Process** (`get_response()`) — looks up the cleaned input in a
  dictionary-based knowledge base to find the matching intent, then
  looks up that intent's reply.
- **Output** — prints the reply back to the user.
- **The Heartbeat** (`run_interactive()`) — a `while True:` loop that
  only ever stops on the exit command's "Kill Command" (`break`),
  exactly like the slide's own infinite-loop diagram.

### Dictionary lookup, not an if-elif ladder
The brief's slides make a specific point of this
("The Anti-Pattern: The If-Elif Ladder" vs. "The Pivot: Hash Maps &
Dictionaries" vs. "Algorithmic Efficiency"): a chain of `if/elif`
checks is **O(n)** — every new rule you add makes every single lookup
slower, and one misplaced `elif` can cascade into a bug. A dictionary's
`.get()` is **O(1)** constant time, no matter how many rules exist, and
handles the "no match" fallback in the same expression:

```python
reply = RESPONSES.get(intent, FALLBACK_RESPONSE)
```

This project uses `.get()` exclusively — there is no if-elif chain
anywhere in `chatbot.py`'s intent-matching logic.

### Two-stage lookup (still O(1), just friendlier)
`chatbot.py` actually chains two dictionaries:
1. `INTENT_MAP` — many raw phrasings (`"hi"`, `"hello"`, `"hey"`, ...)
   all map to one canonical intent (`"greeting"`).
2. `RESPONSES` — each canonical intent maps to its one reply.

This lets the bot recognize several ways of saying the same thing
without writing duplicate response entries or, worse, an if-elif chain
of string comparisons.

### Knowledge base — 6 intents (brief asks for 5+)
| Trigger phrases | Intent | 
|---|---|
| hi, hello, hey, hiya, good morning, good evening | `greeting` |
| how are you, how are you doing | `wellbeing` |
| what is your name, who are you | `identity` |
| what can you do, help | `capabilities` |
| thanks, thank you, thx | `gratitude` |
| what is decodelabs | `about_org` |
| bye, goodbye, exit, quit, q | `exit` (breaks the loop) |
| *(anything else)* | falls back to `FALLBACK_RESPONSE` |

## ▶️ How to run

**Interactive chat:**
```bash
python3 chatbot.py
```
Then type things like `hi`, `help`, `thanks`, or `bye` to exit.

**Scripted proof-of-behavior demo:**
```bash
python3 demo_conversation.py
```

### Sample transcript (`output/demo_transcript.txt`)
```
DecodeBot: Hello! I'm a rule-based chatbot. Type 'help' any time, or 'bye' to exit.
You: Hello
DecodeBot: Hi there! I'm DecodeBot, your rule-based training assistant. Type 'help' to see what I can do.
You:   HeLLo  
DecodeBot: Hi there! I'm DecodeBot, your rule-based training assistant. Type 'help' to see what I can do.
You: how are you
DecodeBot: I'm just a set of if-else... well, dictionary rules, so I'm always running at 100%! How about you?
...
You: asdkjfh this makes no sense
DecodeBot: I do not understand. Type 'help' to see what I can talk about, or 'bye' to exit.
You: bye
DecodeBot: Goodbye! Thanks for chatting with DecodeBot.
```
Note how `"Hello"` and `"  HeLLo  "` both produce the same reply — the
sanitization step doing its job — and the unrecognized message correctly
falls through to `FALLBACK_RESPONSE` instead of crashing or matching the
wrong rule.

## 🔧 Tuning knobs
| Setting | File | Effect |
|---|---|---|
| `INTENT_MAP` | `chatbot.py` | Add more phrasings that map to an existing intent |
| `RESPONSES` | `chatbot.py` | Add a brand-new intent + its reply |
| `FALLBACK_RESPONSE` | `chatbot.py` | What the bot says when nothing matches |
| `SCRIPT` | `demo_conversation.py` | Which messages the scripted demo sends |

## 🔧 Where this goes next
The slide deck's own "Conceptual Bridge" and "Hybrid Architecture"
sections point at exactly where Project 1 leads: this dictionary is a
**discrete, exact-match** knowledge base (a typo or unseen phrasing
always falls back). Project 2 replaces exact string keys with
**semantic embeddings** (continuous vectors), so "how's it going"
would match the same intent as "how are you" without needing to be
listed explicitly. In production, the hybrid pattern is: try the
rule-based dictionary first (fast, 100% predictable) and only fall
through to an LLM when there's no rule match — the same "guardrail in
front of the probabilistic core" idea from the "Modern Application: AI
Guardrails" slide.

## 📝 What to submit
1. `chatbot.py` + `demo_conversation.py`
2. `output/demo_transcript.txt`
3. A short note on which intent you found trickiest to phrase-map, and
   one idea for a 7th intent you'd add
