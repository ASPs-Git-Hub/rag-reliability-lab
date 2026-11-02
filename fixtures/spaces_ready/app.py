# Fixture: the same app made Spaces-ready.
import os
import gradio as gr

api_key = os.environ.get("OPENAI_API_KEY", "")
faq = open("data/faq.txt").read()


def answer(q):
    return "stub"


demo = gr.Interface(answer, "text", "text")
demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
