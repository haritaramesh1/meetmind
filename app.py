import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import ollama

from memory import smart_search


# ============================================================
# CONFIG
# ============================================================

HOST = "127.0.0.1"
PORT = 8501

OLLAMA_MODEL = "qwen2.5:3b-instruct"
TOP_K = 5


# ============================================================
# HTML / CSS / JS
# ============================================================

HTML = r"""
<!DOCTYPE html>
<html lang="en">

<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MeetMind</title>

<style>

:root {
    --bg: #080808;
    --bg2: #101010;
    --glass: rgba(24, 24, 24, 0.72);
    --glass2: rgba(31, 31, 31, 0.88);

    --border: rgba(255,255,255,0.08);
    --border-hover: rgba(255,107,26,0.28);

    --text: #f5f5f5;
    --muted: #858585;
    --muted2: #5e5e5e;

    --orange: #ff6b1a;
    --orange2: #ff8a45;
    --orange-dark: #c94700;

    --green: #62d77b;
}

* {
    box-sizing: border-box;
}

html,
body {
    margin: 0;
    min-height: 100%;
}

body {
    background:
        radial-gradient(
            circle at 78% 5%,
            rgba(255,107,26,0.13),
            transparent 28%
        ),
        radial-gradient(
            circle at 10% 90%,
            rgba(255,107,26,0.055),
            transparent 30%
        ),
        #080808;

    color: var(--text);

    font-family:
        Inter,
        ui-sans-serif,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}

body::before {
    content: "";
    position: fixed;

    width: 450px;
    height: 450px;

    top: -220px;
    right: -120px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            rgba(255,107,26,0.12),
            transparent 68%
        );

    filter: blur(25px);

    pointer-events: none;
}


/* ============================================================
   APP
   ============================================================ */

.app {
    min-height: 100vh;
}


/* ============================================================
   SIDEBAR
   ============================================================ */

.sidebar {
    position: fixed;

    left: 0;
    top: 0;
    bottom: 0;

    width: 255px;

    padding: 20px 15px;

    background:
        rgba(13,13,13,0.78);

    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);

    border-right:
        1px solid var(--border);

    z-index: 20;

    display: flex;
    flex-direction: column;
}

.logo {
    display: flex;
    align-items: center;
    gap: 11px;

    padding: 8px 10px 25px;
}

.logo-mark {
    width: 36px;
    height: 36px;

    border-radius: 11px;

    display: flex;
    align-items: center;
    justify-content: center;

    background:
        linear-gradient(
            145deg,
            var(--orange2),
            var(--orange-dark)
        );

    color: #111;

    font-weight: 900;

    box-shadow:
        0 0 30px rgba(255,107,26,0.25);
}

.logo-text {
    font-size: 19px;
    font-weight: 750;
    letter-spacing: -0.5px;
}

.logo-text span {
    color: var(--orange);
}


/* NAV */

.nav {
    display: flex;
    flex-direction: column;
    gap: 5px;
}

.nav-item {
    width: 100%;

    padding: 11px 12px;

    border-radius: 11px;

    display: flex;
    align-items: center;
    gap: 11px;

    color: #858585;

    font-size: 13px;

    cursor: pointer;

    transition:
        background 0.2s,
        color 0.2s;
}

.nav-item:hover {
    background: rgba(255,255,255,0.04);
    color: #ddd;
}

.nav-item.active {
    background:
        linear-gradient(
            90deg,
            rgba(255,107,26,0.15),
            rgba(255,107,26,0.025)
        );

    color: #fff;

    border:
        1px solid rgba(255,107,26,0.12);
}

.nav-icon {
    width: 20px;
    text-align: center;
}


/* DIVIDER */

.sidebar-divider {
    height: 1px;

    margin: 22px 8px;

    background:
        rgba(255,255,255,0.055);
}


/* RECENT */

.sidebar-label {
    padding: 0 11px 9px;

    color: #5c5c5c;

    font-size: 10px;

    font-weight: 700;

    text-transform: uppercase;

    letter-spacing: 1.2px;
}

.recent {
    display: flex;
    flex-direction: column;
    gap: 3px;
}

.recent-item {
    padding: 9px 11px;

    border-radius: 9px;

    color: #707070;

    font-size: 12px;

    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;

    cursor: pointer;
}

.recent-item:hover {
    background: rgba(255,255,255,0.035);
    color: #aaa;
}


/* BOTTOM */

.sidebar-bottom {
    margin-top: auto;

    padding-top: 15px;

    border-top:
        1px solid rgba(255,255,255,0.055);
}

.local-card {
    padding: 12px;

    border-radius: 12px;

    background:
        rgba(255,255,255,0.025);

    border:
        1px solid rgba(255,255,255,0.055);
}

.local-title {
    display: flex;
    align-items: center;
    gap: 7px;

    font-size: 11px;

    color: #aaa;

    margin-bottom: 5px;
}

.local-dot {
    width: 6px;
    height: 6px;

    border-radius: 50%;

    background: var(--green);

    box-shadow:
        0 0 9px rgba(98,215,123,0.65);
}

.local-sub {
    color: #555;
    font-size: 10px;
}


/* ============================================================
   MAIN
   ============================================================ */

.main {
    width: calc(100% - 255px);

    margin-left: 255px;

    min-height: 100vh;
}


/* TOPBAR */

.topbar {
    height: 68px;

    padding: 0 30px;

    display: flex;
    align-items: center;
    justify-content: space-between;

    background:
        rgba(8,8,8,0.45);

    border-bottom:
        1px solid rgba(255,255,255,0.055);

    backdrop-filter: blur(20px);

    position: sticky;
    top: 0;

    z-index: 15;
}

.page-title {
    color: #aaa;
    font-size: 13px;
}

.top-actions {
    display: flex;
    gap: 8px;
}

.top-pill {
    padding: 7px 11px;

    border-radius: 8px;

    border:
        1px solid rgba(255,255,255,0.07);

    background:
        rgba(255,255,255,0.025);

    color: #777;

    font-size: 11px;
}


/* ============================================================
   CONTENT
   ============================================================ */

.content {
    max-width: 1000px;

    margin: 0 auto;

    padding:
        65px 28px 170px;
}


/* HERO */

.hero {
    text-align: center;

    margin-bottom: 42px;
}

.hero-orb {
    width: 53px;
    height: 53px;

    margin: 0 auto 20px;

    border-radius: 17px;

    display: flex;
    align-items: center;
    justify-content: center;

    background:
        linear-gradient(
            145deg,
            var(--orange2),
            var(--orange-dark)
        );

    color: #121212;

    font-size: 22px;

    font-weight: 900;

    box-shadow:
        0 0 55px rgba(255,107,26,0.22);
}

.hero h1 {
    margin: 0;

    font-size:
        clamp(36px, 5vw, 50px);

    line-height: 1.05;

    letter-spacing: -2.2px;

    font-weight: 780;
}

.hero h1 span {
    color: var(--orange);
}

.hero p {
    max-width: 530px;

    margin: 15px auto 0;

    color: #707070;

    font-size: 14px;

    line-height: 1.65;
}


/* ============================================================
   SUGGESTIONS
   ============================================================ */

.suggestions {
    display: grid;

    grid-template-columns:
        repeat(3, 1fr);

    gap: 10px;

    margin-bottom: 38px;
}

.suggestion {
    min-height: 82px;

    padding: 14px;

    border-radius: 13px;

    background:
        rgba(255,255,255,0.025);

    border:
        1px solid rgba(255,255,255,0.07);

    cursor: pointer;

    transition:
        transform 0.2s,
        border 0.2s,
        background 0.2s;
}

.suggestion:hover {
    transform: translateY(-2px);

    background:
        rgba(255,107,26,0.05);

    border-color:
        rgba(255,107,26,0.2);
}

.suggestion-icon {
    color: var(--orange);

    font-size: 14px;

    margin-bottom: 9px;
}

.suggestion-text {
    color: #888;

    font-size: 12px;

    line-height: 1.45;
}


/* ============================================================
   MESSAGES
   ============================================================ */

.messages {
    display: flex;
    flex-direction: column;

    gap: 18px;
}

.message {
    animation:
        messageIn 0.25s ease-out;
}

@keyframes messageIn {
    from {
        opacity: 0;
        transform: translateY(8px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}


/* USER */

.user-message {
    display: flex;
    justify-content: flex-end;
}

.user-bubble {
    max-width: 72%;

    padding:
        13px 17px;

    border-radius:
        15px 15px 4px 15px;

    background:
        linear-gradient(
            135deg,
            #ff7022,
            #d94b08
        );

    color: white;

    font-size: 14px;

    line-height: 1.55;

    box-shadow:
        0 10px 35px
        rgba(255,88,15,0.12);
}


/* ASSISTANT */

.assistant-card {
    padding: 22px;

    border-radius: 17px;

    background:
        linear-gradient(
            135deg,
            rgba(29,29,29,0.8),
            rgba(17,17,17,0.74)
        );

    border:
        1px solid rgba(255,255,255,0.075);

    box-shadow:
        0 25px 80px rgba(0,0,0,0.4);

    backdrop-filter:
        blur(25px);
}

.assistant-header {
    display: flex;
    align-items: center;
    gap: 9px;

    margin-bottom: 15px;

    color: #858585;

    font-size: 11px;
}

.ai-dot {
    width: 7px;
    height: 7px;

    border-radius: 50%;

    background: var(--orange);

    box-shadow:
        0 0 12px rgba(255,107,26,0.65);
}

.answer {
    color: #e8e8e8;

    font-size: 15px;

    line-height: 1.75;

    white-space: pre-wrap;
}


/* ============================================================
   EVIDENCE
   ============================================================ */

.evidence-section {
    margin-top: 22px;

    padding-top: 18px;

    border-top:
        1px solid rgba(255,255,255,0.07);
}

.evidence-heading {
    display: flex;

    align-items: center;
    justify-content: space-between;

    margin-bottom: 10px;

    color: #777;

    font-size: 10px;

    font-weight: 750;

    letter-spacing: 1px;

    text-transform: uppercase;
}

.evidence-heading span:last-child {
    color:
        rgba(255,107,26,0.8);

    letter-spacing: 0;

    text-transform: none;
}

.evidence {
    margin-top: 9px;

    padding: 13px 14px;

    border-radius: 11px;

    background:
        rgba(0,0,0,0.24);

    border:
        1px solid rgba(255,255,255,0.055);

    transition:
        border 0.2s,
        background 0.2s;
}

.evidence:hover {
    border-color:
        rgba(255,107,26,0.18);

    background:
        rgba(255,107,26,0.025);
}

.evidence-meta {
    display: flex;

    flex-wrap: wrap;

    align-items: center;

    gap: 7px;

    margin-bottom: 8px;

    color: #666;

    font-size: 10px;
}

.evidence-tag {
    padding: 3px 7px;

    border-radius: 5px;

    background:
        rgba(255,107,26,0.08);

    color:
        rgba(255,150,94,0.9);
}

.evidence-text {
    color: #a9a9a9;

    font-size: 12px;

    line-height: 1.6;
}


/* ============================================================
   LOADING
   ============================================================ */

.loading-card {
    display: flex;
    align-items: center;
    gap: 10px;

    color: #777;

    font-size: 13px;
}

.loading-orb {
    width: 22px;
    height: 22px;

    border-radius: 7px;

    background:
        linear-gradient(
            135deg,
            var(--orange),
            var(--orange-dark)
        );

    animation:
        pulse 1.2s infinite;
}

@keyframes pulse {
    0%, 100% {
        opacity: 0.45;
        transform: scale(0.9);
    }

    50% {
        opacity: 1;
        transform: scale(1);
    }
}


/* ============================================================
   ERROR
   ============================================================ */

.error {
    color: #ff9090;

    font-size: 13px;

    line-height: 1.6;
}


/* ============================================================
   COMPOSER
   ============================================================ */

.composer {
    position: fixed;

    left: 255px;
    right: 0;
    bottom: 0;

    padding:
        18px 28px 24px;

    background:
        linear-gradient(
            transparent,
            rgba(8,8,8,0.97) 30%
        );

    z-index: 30;
}

.composer-inner {
    max-width: 860px;

    margin: 0 auto;

    padding: 7px;

    display: flex;
    gap: 8px;

    border-radius: 17px;

    background:
        rgba(27,27,27,0.86);

    border:
        1px solid rgba(255,255,255,0.09);

    box-shadow:
        0 20px 70px rgba(0,0,0,0.55);

    backdrop-filter:
        blur(25px);
}

.composer-inner:focus-within {
    border-color:
        rgba(255,107,26,0.35);

    box-shadow:
        0 20px 70px rgba(0,0,0,0.55),
        0 0 35px rgba(255,107,26,0.06);
}

textarea {
    flex: 1;

    min-height: 46px;
    max-height: 150px;

    resize: none;

    padding:
        12px 13px;

    background: transparent;

    border: none;
    outline: none;

    color: #eee;

    font-family: inherit;

    font-size: 14px;

    line-height: 1.5;
}

textarea::placeholder {
    color: #5e5e5e;
}

.send {
    width: 47px;
    height: 47px;

    align-self: flex-end;

    border: none;

    border-radius: 12px;

    background:
        linear-gradient(
            145deg,
            var(--orange2),
            var(--orange-dark)
        );

    color: #151515;

    cursor: pointer;

    font-size: 18px;

    font-weight: 900;

    transition:
        transform 0.15s,
        filter 0.15s;
}

.send:hover {
    filter: brightness(1.12);
    transform: translateY(-1px);
}

.send:disabled {
    opacity: 0.4;
    cursor: not-allowed;
}


/* ============================================================
   RESPONSIVE
   ============================================================ */

@media (max-width: 850px) {

    .sidebar {
        width: 70px;
        padding: 15px 10px;
    }

    .logo {
        justify-content: center;
        padding-bottom: 20px;
    }

    .logo-text,
    .nav-item span,
    .sidebar-label,
    .recent,
    .local-card {
        display: none;
    }

    .nav-item {
        justify-content: center;
        padding: 12px;
    }

    .main {
        width: calc(100% - 70px);
        margin-left: 70px;
    }

    .composer {
        left: 70px;
    }

    .suggestions {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 600px) {

    .topbar {
        padding: 0 15px;
    }

    .content {
        padding:
            45px 14px 150px;
    }

    .hero h1 {
        font-size: 35px;
    }

    .user-bubble {
        max-width: 88%;
    }

    .composer {
        padding:
            12px 12px 16px;
    }
}

</style>
</head>


<body>

<div class="app">


    <!-- SIDEBAR -->

    <aside class="sidebar">

        <div class="logo">

            <div class="logo-mark">
                M
            </div>

            <div class="logo-text">
                meet<span>mind</span>
            </div>

        </div>


        <nav class="nav">

            <div class="nav-item active">
                <div class="nav-icon">◈</div>
                <span>Ask meetings</span>
            </div>

            <div class="nav-item">
                <div class="nav-icon">◫</div>
                <span>Meetings</span>
            </div>

            <div class="nav-item">
                <div class="nav-icon">◌</div>
                <span>Insights</span>
            </div>

            <div class="nav-item">
                <div class="nav-icon">◎</div>
                <span>Search memory</span>
            </div>

        </nav>


        <div class="sidebar-divider"></div>


        <div class="sidebar-label">
            Recent
        </div>


        <div class="recent">

            <div class="recent-item">
                Meeting 1
            </div>

            <div class="recent-item">
                Floating Farm discussion
            </div>

            <div class="recent-item">
                Project planning
            </div>

        </div>


        <div class="sidebar-bottom">

            <div class="local-card">

                <div class="local-title">
                    <span class="local-dot"></span>
                    Local AI
                </div>

                <div class="local-sub">
                    Qwen · Ollama
                </div>

            </div>

        </div>

    </aside>


    <!-- MAIN -->

    <main class="main">

        <header class="topbar">

            <div class="page-title">
                Meeting intelligence
            </div>

            <div class="top-actions">

                <div class="top-pill">
                    FAISS memory
                </div>

                <div class="top-pill">
                    Local
                </div>

            </div>

        </header>


        <section class="content">


            <div
                class="hero"
                id="welcome"
            >

                <div class="hero-orb">
                    M
                </div>

                <h1>
                    Ask your
                    <span>meetings.</span>
                </h1>

                <p>
                    Search your meeting memory and get
                    grounded answers with the original
                    transcript evidence.
                </p>

            </div>


            <!-- SUGGESTIONS -->

            <div
                class="suggestions"
                id="suggestions"
            >

                <div
                    class="suggestion"
                    onclick="useSuggestion('What decisions were made in the meeting?')"
                >

                    <div class="suggestion-icon">
                        ◈
                    </div>

                    <div class="suggestion-text">
                        What decisions were made?
                    </div>

                </div>


                <div
                    class="suggestion"
                    onclick="useSuggestion('What did they discuss about Floating Farm?')"
                >

                    <div class="suggestion-icon">
                        ◇
                    </div>

                    <div class="suggestion-text">
                        What did they discuss about Floating Farm?
                    </div>

                </div>


                <div
                    class="suggestion"
                    onclick="useSuggestion('What action items were discussed?')"
                >

                    <div class="suggestion-icon">
                        ◆
                    </div>

                    <div class="suggestion-text">
                        What were the action items?
                    </div>

                </div>

            </div>


            <div
                class="messages"
                id="messages"
            ></div>


        </section>

    </main>

</div>


<!-- COMPOSER -->

<div class="composer">

    <div class="composer-inner">

        <textarea
            id="question"
            rows="1"
            placeholder="Ask anything about your meetings..."
        ></textarea>

        <button
            class="send"
            id="sendButton"
            title="Ask MeetMind"
        >
            ↑
        </button>

    </div>

</div>


<script>

const input =
    document.getElementById("question");

const sendButton =
    document.getElementById("sendButton");

const messages =
    document.getElementById("messages");

const welcome =
    document.getElementById("welcome");

const suggestions =
    document.getElementById("suggestions");

let busy = false;


/* ============================================================
   ESCAPE
   ============================================================ */

function escapeHtml(value) {

    const element =
        document.createElement("div");

    element.textContent =
        String(value ?? "");

    return element.innerHTML;
}


/* ============================================================
   SCROLL
   ============================================================ */

function scrollDown() {

    window.scrollTo({
        top: document.body.scrollHeight,
        behavior: "smooth"
    });
}


/* ============================================================
   SUGGESTION
   ============================================================ */

function useSuggestion(text) {

    input.value = text;

    input.focus();

    autoResize();
}


/* ============================================================
   USER
   ============================================================ */

function addUserMessage(text) {

    const wrapper =
        document.createElement("div");

    wrapper.className =
        "message user-message";

    wrapper.innerHTML = `
        <div class="user-bubble">
            ${escapeHtml(text)}
        </div>
    `;

    messages.appendChild(wrapper);

    scrollDown();
}


/* ============================================================
   LOADING
   ============================================================ */

function addLoading() {

    const wrapper =
        document.createElement("div");

    wrapper.id =
        "loading-message";

    wrapper.className =
        "message assistant-message";

    wrapper.innerHTML = `
        <div class="assistant-card">

            <div class="loading-card">

                <div class="loading-orb"></div>

                Searching your meeting memory...

            </div>

        </div>
    `;

    messages.appendChild(wrapper);

    scrollDown();
}


function removeLoading() {

    const loading =
        document.getElementById(
            "loading-message"
        );

    if (loading) {
        loading.remove();
    }
}


/* ============================================================
   ERROR
   ============================================================ */

function addError(message) {

    const wrapper =
        document.createElement("div");

    wrapper.className =
        "message assistant-message";

    wrapper.innerHTML = `
        <div class="assistant-card">

            <div class="error">

                <strong>
                    MeetMind couldn't complete that request.
                </strong>

                <br><br>

                ${escapeHtml(message)}

            </div>

        </div>
    `;

    messages.appendChild(wrapper);

    scrollDown();
}


/* ============================================================
   ASSISTANT
   ============================================================ */

function addAssistant(data) {

    const wrapper =
        document.createElement("div");

    wrapper.className =
        "message assistant-message";


    let html = `
        <div class="assistant-card">

            <div class="assistant-header">

                <span class="ai-dot"></span>

                MeetMind

            </div>

            <div class="answer">
                ${escapeHtml(data.answer)}
            </div>
    `;


    if (
        Array.isArray(data.evidence) &&
        data.evidence.length
    ) {

        html += `
            <div class="evidence-section">

                <div class="evidence-heading">

                    <span>
                        Supporting evidence
                    </span>

                    <span>
                        ${data.evidence.length} excerpts
                    </span>

                </div>
        `;


        for (
            let i = 0;
            i < data.evidence.length;
            i++
        ) {

            const item =
                data.evidence[i];


            html += `
                <div class="evidence">

                    <div class="evidence-meta">

                        <span class="evidence-tag">
                            Excerpt ${i + 1}
                        </span>

                        <span>
                            ${escapeHtml(item.source)}
                        </span>

                        <span>
                            ${escapeHtml(item.speaker)}
                        </span>

                        <span>
                            ${escapeHtml(item.start)}s -
                            ${escapeHtml(item.end)}s
                        </span>

                        <span>
                            score ${escapeHtml(item.score)}
                        </span>

                    </div>

                    <div class="evidence-text">
                        ${escapeHtml(item.text)}
                    </div>

                </div>
            `;
        }


        html += `
            </div>
        `;
    }


    html += `
        </div>
    `;


    wrapper.innerHTML =
        html;

    messages.appendChild(wrapper);

    scrollDown();
}


/* ============================================================
   ASK
   ============================================================ */

async function askMeetMind() {

    if (busy) {
        return;
    }


    const question =
        input.value.trim();


    if (!question) {
        return;
    }


    busy = true;

    sendButton.disabled = true;


    welcome.style.display = "none";

    suggestions.style.display = "none";


    addUserMessage(question);


    input.value = "";

    autoResize();


    addLoading();


    try {

        const response =
            await fetch(
                "/ask",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        question: question
                    })
                }
            );


        const data =
            await response.json();


        removeLoading();


        if (
            !response.ok ||
            data.error
        ) {

            addError(
                data.error ||
                "MeetMind returned an error."
            );

        } else {

            addAssistant(data);
        }


    } catch (error) {

        removeLoading();

        addError(
            error.message ||
            String(error)
        );


    } finally {

        busy = false;

        sendButton.disabled = false;

        input.focus();
    }
}


/* ============================================================
   TEXTAREA
   ============================================================ */

function autoResize() {

    input.style.height = "auto";

    input.style.height =
        Math.min(
            input.scrollHeight,
            150
        ) + "px";
}


/* ============================================================
   EVENTS
   ============================================================ */

sendButton.addEventListener(
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


input.addEventListener(
    "input",
    autoResize
);

</script>

</body>
</html>
"""


# ============================================================
# OLLAMA
# ============================================================

def ask_ollama(question, evidence):

    evidence_blocks = []

    for number, (text, source, score) in enumerate(
        evidence,
        start=1,
    ):

        if isinstance(text, dict):

            meeting = text.get(
                "source",
                source,
            )

            speaker = text.get(
                "speaker",
                "UNKNOWN",
            )

            start = text.get(
                "start",
                "?",
            )

            end = text.get(
                "end",
                "?",
            )

            transcript = text.get(
                "text",
                "",
            )

            block = (
                f"Evidence {number}\n"
                f"Meeting: {meeting}\n"
                f"Speaker: {speaker}\n"
                f"Time: {start}s - {end}s\n"
                f"Relevance: {float(score):.3f}\n"
                f"Transcript: {transcript}"
            )

        else:

            block = (
                f"Evidence {number}\n"
                f"Source: {source}\n"
                f"Relevance: {float(score):.3f}\n"
                f"Transcript: {text}"
            )

        evidence_blocks.append(block)


    evidence_text = "\n\n".join(
        evidence_blocks
    )


    system_prompt = """
You are MeetMind, an AI meeting-memory assistant.

Answer questions using ONLY the meeting evidence
provided to you.

Rules:
- Do not invent facts.
- Do not use outside knowledge.
- Do not claim something was said unless the evidence supports it.
- If the evidence is insufficient, say so clearly.
- Be concise but useful.
- Mention speakers or timestamps when useful.
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


    message = response.get(
        "message",
        {},
    )


    answer = message.get(
        "content",
        "",
    )


    if not answer.strip():

        raise RuntimeError(
            "Ollama returned an empty answer."
        )


    return answer.strip()


# ============================================================
# EVIDENCE JSON
# ============================================================

def make_evidence_json(evidence):

    output = []

    for text, source, score in evidence:

        if isinstance(text, dict):

            output.append(
                {
                    "text": str(
                        text.get(
                            "text",
                            "",
                        )
                    ),

                    "source": str(
                        text.get(
                            "source",
                            source,
                        )
                    ),

                    "speaker": str(
                        text.get(
                            "speaker",
                            "UNKNOWN",
                        )
                    ),

                    "start": text.get(
                        "start",
                        "?",
                    ),

                    "end": text.get(
                        "end",
                        "?",
                    ),

                    "score": round(
                        float(score),
                        3,
                    ),
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
                    "score": round(
                        float(score),
                        3,
                    ),
                }
            )

    return output


# ============================================================
# HTTP HANDLER
# ============================================================

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
                        "error":
                            "Empty request.",
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
                payload.get(
                    "question",
                    "",
                )
            ).strip()


            if not question:

                self.send_json(
                    {
                        "error":
                            "Please enter a question.",
                    },
                    400,
                )

                return


            print()
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
                        "answer":
                            (
                                "I couldn't find "
                                "relevant meeting "
                                "evidence for that "
                                "question."
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

                    "evidence":
                        make_evidence_json(
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
                    "error":
                        f"{type(exc).__name__}: {exc}",
                },
                500,
            )


# ============================================================
# OLLAMA CHECK
# ============================================================

def check_ollama():

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


# ============================================================
# MAIN
# ============================================================

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
            (
                HOST,
                PORT,
            ),
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
    print(
        "MeetMind server is ready."
    )

    print(
        f"Open: http://{HOST}:{PORT}"
    )

    print(
        "Press Ctrl+C to stop."
    )

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
        print(
            "Stopping MeetMind..."
        )


    finally:

        server.server_close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()