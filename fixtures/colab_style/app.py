# Fixture: a prototype that worked in Colab. Intentionally broken for Spaces.
from google.colab import drive
drive.mount('/content/drive')
import gradio as gr

api_key = "sk-FAKE-FAKE-FAKE-FAKE-FAKE-FAKE"
faq = open("/content/drive/MyDrive/faq.txt").read()
notes = open("data/faq.txt").read()


def answer(q):
    return "stub"


demo = gr.Interface(answer, "text", "text")
demo.launch(server_name="127.0.0.1", server_port=8000)
