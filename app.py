import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import ollama

from memory import smart_search


HOST = "127.0.0.1"
PORT = 8501
OLLAMA_MODEL = "qwen2.5:3b-instruct"
TOP_K = 5


HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MeetMind</title>

<style>
* {
    box-sizing: border-box;
}

body {
    margin: 0;
    min-height: 100vh;
    background: #101114;
    color: #eeeeee;
    font-family: Arial, sans-serif;
}

.header {
    height: 68px;
    display: flex;
    align-items: center;
    padding: 0 28px;
    background: #15161a;
    border-bottom: 1px solid #292b31;
}

.logo {
    font-size: 24px;
    font-weight: 700;
}

.status {
    margin-left: 18px;
    color: #8d939d;
    font-size: 13px;
}

.dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    margin-right: 7px;
    border-radius: 50%;
    background: #62d46f;
}

.container {
    width: 100%;
    max-width: 960px;
    margin: 0 auto;
    padding: 45px 20px 150px;
}

.welcome {
    text-align: center;
    margin: 55px 0 50px;
}

.welcome h1 {
    margin: 0 0 12px;
    font-size: 42px;
}

.welcome p {
    margin: 0;
    color: #969ba5;
}

.messages {
    display: flex;
    flex-direction: column;
    gap: 20px;
}

.message {
    padding: 18px 20px;
    border-radius: 16px;
    line-height: 1.6;
    overflow-wrap: anywhere;
}

.user {
    align-self: flex-end;
    max-width: 75%;
    background: #292c33;
}

.assistant {
    width: 100%;
    background: #191b20;
    border: 1px solid #2c2f36;
}

.answer {
    white-space: pre-wrap;
}

.loading {
    color: #9297a0;
    font-style: italic;
}

.error {
    color: #ff9a9a;
}

.evidence-title {
    margin-top: 22px;
    padding-top: 16px;
    border-top: 1px solid #30333a;
    color: #aeb3bc;
    font-size: 13px;
    font-weight: bold;
}

.evidence {
    margin-top: 12px;
    padding: 14px;
    background: #111318;
    border: 1px solid #30343c;
    border-radius: 10px;
}

.meta {
    margin-bottom: 8px;
    color: #8f949d;
    font-size: 12px;
}

.evidence-text {
    color: #e7e7e7;
    font-size: 14px;
}

.composer {
    position: fixed;
    left: 0;
    right: 0;
    bottom: 0;
    padding: 18px 20px 22px;
    background: linear-gradient(transparent, #101114 30%);
}

.input-wrapper {
    max-width: 900px;
    margin: 0 auto;
    display: flex;
    gap: 10px;
    padding: 9px;
    background: #202329;
    border: 1px solid #3a3e47;
    border-radius: 16px;
}

textarea {
    flex: 1;
    min-height: 48px;
    max-height: 180px;
    resize: none;
    padding: 11px 12px;
    background: transparent;
    border: none;
    outline: none;
    color: white;
    font-size: 15px;
    font-family: inherit;
}

textarea::placeholder {
    color: #777c86;
}

button {
    align-self: flex-end;
    min-width: 82px;
    height: 46px;
    border: none;
    border-radius: 11px;
    background: #eeeeee;
    color: #111111;
    font-weight: bold;
    cursor: pointer;
}

button:hover {
    background: white;
}

button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}
</style>
</head>

<body>

<header class="header">
    <div class="logo">meetmind</div>
    <div class="status">
        <span class="dot"></span>
        Local AI
    </div>
</header>

<main class="container">

    <section class="welcome" id="welcome">
        <h1>MeetMind</h1>
        <p>Ask questions about your meetings.</p>
    </section>

    <section class="messages" id="messages"></section>

</main>

<div class="composer">
    <div class="input-wrapper">

        <textarea
            id="question"
            rows="1"
            placeholder="Ask about your meetings..."
        ></textarea>

        <button id="sendButton">
            Ask
        </button>

    </div>
</div>

<script>
const input = document.getElementById("question");
const button = document.getElementById("sendButton");
const messages = document.getElementById("messages");
const welcome = document.getElementById("welcome");

let busy = false;

function escapeHtml(value) {
    const element = document.createElement("div");
    element.textContent = String(value ?? "");
    return element.innerHTML;
}

function scrollDown() {
    window.scrollTo({
        top: document.body.scrollHeight,
        behavior: "smooth"
    });
}

function addUserMessage(text) {
    const div = document.createElement("div");
    div.className = "message user";
    div.textContent = text;
    messages.appendChild(div);
    scrollDown();
}

function addLoading() {
    const div = document.createElement("div");
    div.id = "loading";
    div.className = "message assistant loading";
    div.textContent = "Searching your meeting memory...";
    messages.appendChild(div);
    scrollDown();
}

function removeLoading() {
    const div = document.getElementById("loading");
    if (div) {
        div.remove();
    }
}

function addError(message) {
    const div = document.createElement("div");
    div.className = "message assistant";
    div.innerHTML =
        '<div class="error"><strong>Error</strong><br><br>' +
        escapeHtml(message) +
        '</div>';
    messages.appendChild(div);
    scrollDown();
}

function addAssistant(data) {
    const div = document.createElement("div");
    div.className = "message assistant";

    let html =
        '<div class="answer">' +
        escapeHtml(data.answer) +
        '</div>';

    if (data.evidence && data.evidence.length > 0) {
        html += '<div class="evidence-title">Meeting evidence</div>';

        for (let i = 0; i < data.evidence.length; i++) {
            const item = data.evidence[i];

            html +=
                '<div class="evidence">' +
                '<div class="meta">' +
                '<strong>Excerpt ' + (i + 1) + '</strong><br>' +
                'Meeting: ' + escapeHtml(item.source) +
                ' &nbsp;•&nbsp; Speaker: ' +
                escapeHtml(item.speaker) +
                ' &nbsp;•&nbsp; Time: ' +
                escapeHtml(item.start) + 's - ' +
                escapeHtml(item.end) + 's' +
                ' &nbsp;•&nbsp; Relevance: ' +
                escapeHtml(item.score) +
                '</div>' +
                '<div class="evidence-text">' +
                escapeHtml(item.text) +
                '</div>' +
                '</div>';
        }
    }

    div.innerHTML = html;
    messages.appendChild(div);
    scrollDown();
}

async function askMeetMind() {
    if (busy) {
        return;
    }

    const question = input.value.trim();

    if (!question) {
        return;
    }

    busy = true;
    button.disabled = true;
    welcome.style.display = "none";

    addUserMessage(question);
    input.value = "";
    addLoading();

    try {
        const response = await fetch("/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                question: question
            })
        });

        const data = await response.json();

        removeLoading();

        if (!response.ok || data.error) {
            addError(
                data.error ||
                "The MeetMind server returned an error."
            );
        } else {
            addAssistant(data);
        }

    } catch (error) {
        removeLoading();
        addError(error.message || String(error));

    } finally {
        busy = false;
        button.disabled = false;
        input.focus();
    }
}

button.addEventListener(
    "click",
    askMeetMind
);

input.addEventListener(
    "keydown",
    function(event) {
        if (
            event.key === "Enter" &&
            !event.shiftKey
        ) {
            event.preventDefault();
            askMeetMind();
        }
    }
);
</script>

</body>
</html>
"""


def ask_ollama(question, evidence):
    """Generate an answer using only retrieved meeting evidence."""

    evidence_blocks = []

    for number, (text, source, score) in enumerate(
        evidence,
        start=1,
    ):
        if isinstance(text, dict):
            meeting = text.get("source", source)
            speaker = text.get("speaker", "UNKNOWN")
            start = text.get("start", "?")
            end = text.get("end", "?")
            transcript = text.get("text", "")

            block = (
                f"Evidence {number}\n"
                f"Meeting: {meeting}\n"
                f"Speaker: {speaker}\n"
                f"Time: {start}s - {end}s\n"
                f"Relevance: {score:.3f}\n"
                f"Transcript: {transcript}"
            )

        else:
            block = (
                f"Evidence {number}\n"
                f"Source: {source}\n"
                f"Relevance: {score:.3f}\n"
                f"Transcript: {text}"
            )

        evidence_blocks.append(block)

    evidence_text = "\n\n".join(evidence_blocks)

    system_prompt = """
You are MeetMind, an AI meeting-memory assistant.

Answer questions using ONLY the meeting evidence provided
to you.

Rules:
- Do not invent facts.
- Do not use outside knowledge.
- Do not claim something was said unless the evidence supports it.
- If the evidence is insufficient, say so clearly.
- Be concise but useful.
- Mention speakers or timestamps when they are useful.
"""

    user_prompt = f"""
User question:

{question}

Retrieved meeting evidence:

{evidence_text}

Answer the user's question using only the evidence above.
"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    message = response.get("message", {})
    answer = message.get("content", "")

    if not answer.strip():
        raise RuntimeError(
            "Ollama returned an empty answer."
        )

    return answer.strip()


def make_evidence_json(evidence):
    """Convert search results into JSON-safe objects."""

    output = []

    for text, source, score in evidence:
        if isinstance(text, dict):
            output.append(
                {
                    "text": str(text.get("text", "")),
                    "source": str(
                        text.get("source", source)
                    ),
                    "speaker": str(
                        text.get("speaker", "UNKNOWN")
                    ),
                    "start": text.get("start", "?"),
                    "end": text.get("end", "?"),
                    "score": round(float(score), 3),
                }
            )
        else:
            output.append(
                {
                    "text": str(text),
                    "source": str(source),
                    "speaker": "UNKNOWN",
                    "start": "?",
                    "end": "?",
                    "score": round(float(score), 3),
                }
            )

    return output


class MeetMindHandler(BaseHTTPRequestHandler):

    def log_message(self, format_string, *args):
        print(
            "[MeetMind]",
            format_string % args,
            flush=True,
        )

    def send_json(self, data, status=200):
        body = json.dumps(
            data,
            ensure_ascii=False,
        ).encode("utf-8")

        self.send_response(status)

        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )

        self.send_header(
            "Content-Length",
            str(len(body)),
        )

        self.send_header(
            "Cache-Control",
            "no-store",
        )

        self.end_headers()

        self.wfile.write(body)

    def do_GET(self):
        if self.path != "/":
            self.send_error(404)
            return

        body = HTML.encode("utf-8")

        self.send_response(200)

        self.send_header(
            "Content-Type",
            "text/html; charset=utf-8",
        )

        self.send_header(
            "Content-Length",
            str(len(body)),
        )

        self.end_headers()

        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/ask":
            self.send_error(404)
            return

        try:
            content_length = int(
                self.headers.get(
                    "Content-Length",
                    "0",
                )
            )

            if content_length <= 0:
                self.send_json(
                    {
                        "error": "Empty request.",
                    },
                    400,
                )
                return

            raw_body = self.rfile.read(
                content_length
            )

            payload = json.loads(
                raw_body.decode("utf-8")
            )

            question = str(
                payload.get("question", "")
            ).strip()

            if not question:
                self.send_json(
                    {
                        "error": "Please enter a question.",
                    },
                    400,
                )
                return

            print(
                "",
                flush=True,
            )

            print(
                f"Question: {question}",
                flush=True,
            )

            print(
                "Searching FAISS...",
                flush=True,
            )

            evidence = smart_search(
                question,
                k=TOP_K,
            )

            print(
                f"Found {len(evidence)} meeting chunks.",
                flush=True,
            )

            if not evidence:
                self.send_json(
                    {
                        "answer": (
                            "I couldn't find relevant "
                            "meeting evidence for that question."
                        ),
                        "evidence": [],
                    }
                )
                return

            print(
                f"Sending evidence to {OLLAMA_MODEL}...",
                flush=True,
            )

            answer = ask_ollama(
                question,
                evidence,
            )

            print(
                "Answer generated.",
                flush=True,
            )

            self.send_json(
                {
                    "answer": answer,
                    "evidence": make_evidence_json(
                        evidence
                    ),
                }
            )

        except Exception as exc:
            print(
                f"ERROR: {type(exc).__name__}: {exc}",
                flush=True,
            )

            self.send_json(
                {
                    "error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                },
                500,
            )


def check_ollama():
    """Check whether the requested Ollama model exists."""

    try:
        result = ollama.list()

        models = result.get(
            "models",
            [],
        )

        names = []

        for model in models:
            name = model.get("name")

            if name:
                names.append(name)

        if OLLAMA_MODEL in names:
            print(
                f"Ollama OK: {OLLAMA_MODEL}",
                flush=True,
            )
            return True

        print(
            f"WARNING: {OLLAMA_MODEL} not found.",
            flush=True,
        )

        print(
            "Available models:",
            flush=True,
        )

        for name in names:
            print(
                f"  {name}",
                flush=True,
            )

        return False

    except Exception as exc:
        print(
            "WARNING: Could not connect to Ollama.",
            flush=True,
        )

        print(
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        return False


def main():
    print("=" * 60)
    print("MeetMind + Ollama")
    print("=" * 60)
    print(
        f"URL: http://{HOST}:{PORT}"
    )
    print(
        f"AI model: {OLLAMA_MODEL}"
    )
    print(
        f"Meeting retrieval: top {TOP_K}"
    )
    print("=" * 60)
    print()

    check_ollama()

    try:
        server = ThreadingHTTPServer(
            (HOST, PORT),
            MeetMindHandler,
        )

    except OSError as exc:
        print()
        print(
            f"Could not start server on port {PORT}."
        )
        print(
            f"{type(exc).__name__}: {exc}"
        )
        return

    print()
    print("MeetMind server is ready.")
    print(
        f"Open: http://{HOST}:{PORT}"
    )
    print("Press Ctrl+C to stop.")
    print()

    threading.Timer(
        1.0,
        lambda: webbrowser.open(
            f"http://{HOST}:{PORT}"
        ),
    ).start()

    try:
        server.serve_forever()

    except KeyboardInterrupt:
        print()
        print("Stopping MeetMind...")

    finally:
        server.server_close()


if __name__ == "__main__":
    main()