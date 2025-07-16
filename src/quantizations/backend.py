from flask import Flask, request, send_file, jsonify, send_from_directory
from flask_cors import CORS
import torch
from io import BytesIO
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

app = Flask(__name__)
CORS(app)  # Enable CORS for easier development

# Load the Stable Diffusion image model
def load_image_model():
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = "sdxl_lightning_8step_unet.safetensors"
    unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", torch.float16)
    unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
    pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to("cuda")
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    return pipe

image_model = load_image_model()

# Load the GEMM-2B text model with BitsAndBytesConfig
def load_text_model():
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", 
                                                 quantization_config=quantization_config)
    return model, tokenizer

text_model, text_tokenizer = load_text_model()

@app.route('/generate-image', methods=['POST'])
def generate_image():
    try:
        content = request.json
        prompt = content.get('prompt', '')
        if not prompt:
            return jsonify({"error": "Prompt is required for image generation."}), 400

        image = image_model(prompt, num_inference_steps=4, guidance_scale=7.5).images[0]
        byte_arr = BytesIO()
        image.save(byte_arr, format='PNG')
        byte_arr.seek(0)
        return send_file(byte_arr, mimetype='image/png', attachment_filename='generated_image.png')
    except Exception as e:
        return jsonify({"error": f"Image generation error: {str(e)}"}), 500

@app.route('/generate-text', methods=['POST'])
def generate_text():
    try:
        content = request.json
        user_input = content.get('prompt', '')
        if not user_input:
            return jsonify({"error": "Prompt is required for text generation."}), 400

        prompt_template = '''You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. {user_input}'''
        full_prompt = prompt_template.format(user_input=user_input)
        inputs = text_tokenizer(full_prompt, return_tensors="pt")
        outputs = text_model.generate(**inputs)
        response_text = text_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({"response": response_text})
    except Exception as e:
        return jsonify({"error": f"Text generation error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
