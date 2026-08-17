import gradio as gr

from ask_meeting import ask_meeting


CSS = """
:root {
    --orange: #ff6b35;
    --orange-light: #ff8a5b;
    --bg: #0b0b0d;
    --panel: #151518;
    --panel-2: #1c1c20;
    --border: #2b2b31;
    --text: #f5f5f5;
    --muted: #9b9ba3;
}

.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

body {
    background: var(--bg) !important;
}

#hero {
    padding: 42px 10px 25px 10px;
    text-align: center;
}

#logo {
    font-size: 48px;
    font-weight: 800;
    letter-spacing: -2px;
    color: white;
}

#logo span {
    color: var(--orange);
}

#tagline {
    color: var(--muted);
    font-size: 17px;
    margin-top: 8px;
}

#status {
    background: #171719 !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    color: var(--orange-light) !important;
}

.panel {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 18px !important;
    padding: 20px !important;
}

.stat {
    background: var(--panel-2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px;
    text-align: center;
}

.stat-number {
    font-size: 25px;
    font-weight: 750;
    color: var(--orange);
}

.stat-label {
    font-size: 12px;
    color: var(--muted);
    margin-top: 4px;
}

textarea,
input {
    background: #101012 !important;
    border: 1px solid #33333a !important;
    color: white !important;
    border-radius: 14px !important;
}

textarea:focus,
input:focus {
    border-color: var(--orange) !important;
    box-shadow: 0 0 0 2px rgba(255, 107, 53, 0.15) !important;
}

#ask-btn {
    background: var(--orange) !important;
    color: white !important;
    border: none !important;
    border-radius: 13px !important;
    font-weight: 700 !important;
    min-height: 48px !important;
}

#ask-btn:hover {
    background: var(--orange-light) !important;
}

.example-btn {
    border-radius: 10px !important;
}

#answer {
    min-height: 280px;
}

footer {
    display: none !important;
}

.small-muted {
    color: var(--muted);
    font-size: 13px;
}
"""


def show_searching():
    return "🟠 **Searching your meeting memory...**"


def show_thinking():
    return "🧠 **Relevant passages found. MeetMind is thinking...**"


def answer_question(question):
    if not question.strip():
        return "Please enter a question."

    try:
        return ask_meeting(question)
    except Exception as e:
        return f"❌ **Something went wrong:** `{e}`"


with gr.Blocks(
    title="MeetMind",
    theme=gr.themes.Base(
        primary_hue="orange",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
    css=CSS,
) as demo:

    gr.HTML(
        """
        <div id="hero">
            <div id="logo">meet<span>mind</span></div>
            <div id="tagline">
                Your meeting, remembered.
            </div>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML(
                """
                <div class="stat">
                    <div class="stat-number">787</div>
                    <div class="stat-label">MEETING CHUNKS</div>
                </div>
                """
            )

        with gr.Column(scale=1):
            gr.HTML(
                """
                <div class="stat">
                    <div class="stat-number">AI</div>
                    <div class="stat-label">GROUNDED ANSWERS</div>
                </div>
                """
            )

        with gr.Column(scale=1):
            gr.HTML(
                """
                <div class="stat">
                    <div class="stat-number">24/7</div>
                    <div class="stat-label">MEETING MEMORY</div>
                </div>
                """
            )

    gr.Markdown("")

    with gr.Column(elem_classes="panel"):

        gr.Markdown(
            """
            ### Ask your meeting

            Search the conversation using natural language.
            """
        )

        question = gr.Textbox(
            label="",
            placeholder="e.g. What did we discuss about Floating Farm?",
            lines=3,
        )

        with gr.Row():
            ask_button = gr.Button(
                "Ask MeetMind  →",
                elem_id="ask-btn",
                variant="primary",
            )

        status = gr.Markdown(
            "🟢 **Ready**",
            elem_id="status",
        )

    gr.Markdown("### Try asking")

    gr.Examples(
        examples=[
            ["What did they discuss about Floating Farm?"],
            ["What decisions were made?"],
            ["Who had action items?"],
            ["What should happen next?"],
            ["What was the main goal of the meeting?"],
        ],
        inputs=question,
        label="",
    )

    gr.Markdown("### MeetMind Answer")

    answer = gr.Markdown(
        "Ask a question above to search your meeting memory.",
        elem_id="answer",
    )

    # Immediately update the UI so the user knows something is happening.
    ask_event = ask_button.click(
        fn=show_searching,
        inputs=None,
        outputs=status,
        show_progress="hidden",
    )

    ask_event = ask_event.then(
        fn=show_thinking,
        inputs=None,
        outputs=status,
        show_progress="hidden",
    )

    ask_event.then(
        fn=answer_question,
        inputs=question,
        outputs=answer,
        show_progress="full",
    ).then(
        fn=lambda: "🟢 **Answer ready**",
        inputs=None,
        outputs=status,
        show_progress="hidden",
    )

    question.submit(
        fn=show_searching,
        inputs=None,
        outputs=status,
        show_progress="hidden",
    ).then(
        fn=show_thinking,
        inputs=None,
        outputs=status,
        show_progress="hidden",
    ).then(
        fn=answer_question,
        inputs=question,
        outputs=answer,
        show_progress="full",
    ).then(
        fn=lambda: "🟢 **Answer ready**",
        inputs=None,
        outputs=status,
        show_progress="hidden",
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch()