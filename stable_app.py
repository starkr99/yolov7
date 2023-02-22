from flask import Flask, request, jsonify
import base64
from io import BytesIO
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()

def get_inputs(prompt, batch_size=1):
    generator = [torch.Generator("cuda").manual_seed(i) for i in range(batch_size)]
    prompts = batch_size * [prompt]
    num_inference_steps = 20

    return {"prompt": prompts, "generator": generator, "num_inference_steps": num_inference_steps}

app = Flask(__name__)

@app.route('/inference', methods=['POST'])
def inference():
    data = request.get_json()
    prompt = data.get('prompt', '')
    count = data.get('count', 1)

    images = pipe(**get_inputs(prompt, count)).images
    strings = list(map((lambda x: imageToBase64String(x)), images))
    response = {'results': strings}
    return jsonify(response)

def imageToBase64String(img):
  buffered = BytesIO()
  img.save(buffered, format="JPEG")
  return base64.b64encode(buffered.getvalue()).decode('ascii')

@app.route('/echo', methods=['GET'])
def echo():
    text = request.args.get('text', '')
    return text

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
