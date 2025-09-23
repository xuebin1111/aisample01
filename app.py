import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load DeepSeek model
model_id = "deepseek-ai/deepseek-llm-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

# Inference function
def generate_response(prompt, temperature, top_p, max_tokens, repetition_penalty):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_tokens,
        repetition_penalty=repetition_penalty
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Gradio UI
iface = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Textbox(label="Prompt", lines=6, placeholder="Ask something..."),
        gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p"),
        gr.Slider(32, 2048, value=512, step=32, label="Max New Tokens"),
        gr.Slider(1.0, 2.0, value=1.1, step=0.1, label="Repetition Penalty")
    ],
    outputs="text",
    title="🧠 DeepSeek LLM Chat with Parameter Tuning",
    theme="soft"
)

if __name__ == "__main__":
    iface.launch()