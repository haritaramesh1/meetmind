import json
from pathlib import Path
import gradio as gr
from pipeline import main as run_pipeline
from memory import smart_search

def process(video):
    if not video:
        return "Upload a meeting first.", "", "", ""
    run_pipeline(video)
    minutes = json.loads(Path("outputs/minutes.json").read_text())
    transcript = json.loads(Path("outputs/speaker_transcript.json").read_text())
    fusion = json.loads(Path("outputs/fusion.json").read_text())

    summary = "\n".join(f"• {x}" for x in minutes["summary"])
    decisions = "\n".join(f"• {x}" for x in minutes["decisions"])
    actions = "\n".join(
        f'• {a["owner"]}: {a["task"]}' + (f' (due {a["due"]})' if a.get("due") else "")
        for a in minutes["action_items"]
    )
    explained = f"""## Meeting explained

### Summary
{summary or "No summary."}

### Decisions
{decisions or "No decisions detected."}

### Action items
{actions or "No action items detected."}

### Voice → face fusion
{json.dumps(fusion["mapping"], indent=2)}
"""
    transcript_md = "\n".join(
        f'**[{x["start"]:.1f}s] {x["speaker"]}:** {x["text"]}'
        for x in transcript
    )
    return explained, transcript_md, json.dumps(fusion["mapping"], indent=2), "Pipeline complete."

def ask(question):
    results = smart_search(question, k=3)
    if not results:
        return "No matching meeting memory found."
    return "\n\n".join(f"**{source}**\n{text}" for text, source, _ in results)

with gr.Blocks(title="MeetMind") as demo:
    gr.Markdown("# MeetMind\n### The meeting that explains itself")
    with gr.Tab("Process meeting"):
        video = gr.File(label="Meeting video", file_types=[".mp4", ".mov", ".m4a", ".wav", ".mp3"], type="filepath")
        button = gr.Button("Analyze meeting", variant="primary")
        status = gr.Markdown()
        explained = gr.Markdown()
        with gr.Tab("Transcript"):
            transcript = gr.Markdown()
        with gr.Tab("Voice ↔ Face"):
            mapping = gr.Code(language="json")
        button.click(process, inputs=video, outputs=[explained, transcript, mapping, status])
    with gr.Tab("Meeting memory"):
        question = gr.Textbox(label="Ask your meetings")
        answer = gr.Markdown()
        question.submit(ask, inputs=question, outputs=answer)

if __name__ == "__main__":
    demo.launch()
