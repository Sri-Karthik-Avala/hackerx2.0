import threading
import http.server
import socketserver
import os
import yaml
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import gradio as gr
from utils.upload_file import UploadFile
from utils.chatbot import ChatBot
from utils.ui_settings import UISettings
from utils.load_config import LoadConfig
from pyprojroot import here

# Load the app config
with open(here("configs/app_config.yml")) as cfg:
    app_config = yaml.load(cfg, Loader=yaml.FullLoader)

PORT = app_config["serve"]["port"]
DIRECTORY1 = app_config["directories"]["data_directory"]
DIRECTORY2 = app_config["directories"]["data_directory_2"]

# ================================
# Part 1: Reference Serve Code
# ================================
class MultiDirectoryHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Serve files from multiple directories."""
    def translate_path(self, path):
        parts = path.split('/', 2)
        if len(parts) > 1:
            first_directory = parts[1]
            if first_directory == os.path.basename(DIRECTORY1):
                path = os.path.join(DIRECTORY1, *parts[2:])
            elif first_directory == os.path.basename(DIRECTORY2):
                path = os.path.join(DIRECTORY2, *parts[2:])
            else:
                file_path1 = os.path.join(DIRECTORY1, first_directory)
                file_path2 = os.path.join(DIRECTORY2, first_directory)
                if os.path.isfile(file_path1):
                    return file_path1
                elif os.path.isfile(file_path2):
                    return file_path2
        return super().translate_path(path)

def start_reference_server():
    with socketserver.TCPServer(("", PORT), MultiDirectoryHTTPRequestHandler) as httpd:
        print(f"Serving at port {PORT}")
        httpd.serve_forever()

# ================================
# Part 2: LLM Serve Code
# ================================
APPCFG = LoadConfig()

app = Flask(__name__)

# Load the LLM and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    APPCFG.llm_engine, token=APPCFG.gemma_token, device=APPCFG.device)
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
                                             token=APPCFG.gemma_token,
                                             torch_dtype=torch.float16,
                                             device_map=APPCFG.device)
app_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

@app.route("/generate_text", methods=["POST"])
def generate_text():
    data = request.json
    prompt = data.get("prompt", "")
    max_new_tokens = data.get("max_new_tokens", 1000)
    do_sample = data.get("do_sample", True)
    temperature = data.get("temperature", 0.1)
    top_k = data.get("top_k", 50)
    top_p = data.get("top_p", 0.95)

    tokenized_prompt = app_pipeline.tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True)
    outputs = app_pipeline(
        tokenized_prompt,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )

    return jsonify({"response": outputs[0]["generated_text"][len(tokenized_prompt):]})

def start_llm_server():
    app.run(debug=False, port=8888)

# ================================
# Part 3: Gradio Chatbot Code
# ================================
def start_gradio_app():
    with gr.Blocks() as demo:
        with gr.Tabs():
            with gr.TabItem("Med-App"):
                # First row
                with gr.Row() as row_one:
                    with gr.Column(visible=False) as reference_bar:
                        ref_output = gr.Markdown()
                    with gr.Column() as chatbot_output:
                        chatbot = gr.Chatbot(
                            [], elem_id="chatbot", bubble_full_width=False, height=500,
                            avatar_images=("images/test.png", "images/Gemma-logo.png")
                        )
                        chatbot.like(UISettings.feedback, None, None)

                # Second row
                with gr.Row():
                    input_txt = gr.Textbox(
                        lines=4, scale=8, placeholder="Enter text and press enter, or upload PDF files"
                    )

                # Third row
                with gr.Row() as row_two:
                    text_submit_btn = gr.Button(value="Submit text")
                    btn_toggle_sidebar = gr.Button(value="References")
                    upload_btn = gr.UploadButton(
                        "📁 Upload PDF or doc files", file_types=['.pdf', '.doc'], file_count="multiple"
                    )
                    clear_button = gr.ClearButton([input_txt, chatbot])
                    rag_with_dropdown = gr.Dropdown(
                        label="RAG with", choices=["Preprocessed doc", "Upload doc: Process for RAG"], value="Preprocessed doc"
                    )

                # Fourth row
                with gr.Row() as row_four:
                    temperature_bar = gr.Slider(
                        minimum=0.1, maximum=1, value=0.1, step=0.1, label="Temperature",
                        info="Increasing the temperature will make the model answer more creatively."
                    )
                    top_k = gr.Slider(
                        minimum=0.0, maximum=100.0, step=1, label="top_k", value=50,
                        info="A lower value (e.g. 10) will result in more conservative answers."
                    )
                    top_p = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.01, label="top_p", value=0.95,
                        info="A lower value will generate more focused and conservative text."
                    )

                # Process uploaded files and text
                file_msg = upload_btn.upload(
                    fn=UploadFile.process_uploaded_files, inputs=[upload_btn, chatbot, rag_with_dropdown],
                    outputs=[input_txt, chatbot], queue=False
                )
                txt_msg = input_txt.submit(
                    fn=ChatBot.respond, inputs=[chatbot, input_txt, rag_with_dropdown, temperature_bar, top_k, top_p],
                    outputs=[input_txt, chatbot, ref_output], queue=False
                ).then(lambda: gr.Textbox(interactive=True), None, [input_txt], queue=False)
                text_submit_btn.click(
                    fn=ChatBot.respond, inputs=[chatbot, input_txt, rag_with_dropdown, temperature_bar, top_k, top_p],
                    outputs=[input_txt, chatbot, ref_output], queue=False
                ).then(lambda: gr.Textbox(interactive=True), None, [input_txt], queue=False)

    demo.launch(share=True)

# ================================
# Main: Running all services concurrently
# ================================
if __name__ == "__main__":
    # Start all services in separate threads
    reference_server_thread = threading.Thread(target=start_reference_server)
    llm_server_thread = threading.Thread(target=start_llm_server)
    gradio_app_thread = threading.Thread(target=start_gradio_app)

    reference_server_thread.start()
    llm_server_thread.start()
    gradio_app_thread.start()

    # Keep the main thread alive
    reference_server_thread.join()
    llm_server_thread.join()
    gradio_app_thread.join()
