from flask import Flask, request, jsonify
from detect import Detect
import os
import wget

app = Flask(__name__)

# if yolov7.pt file is not exists root directory, download it
# !wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
if not os.path.exists('yolov7.pt'):
    wget.download("https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt")

detect = Detect(weights='yolov7.pt', img_size=640, conf_thres=0.25, iou_thres=0.45, device='cpu')

@app.route('/inference', methods=['POST'])
def inference():
    data = request.get_json()
    datas = data.get('datas', [])
    
    results = detect.detectFromString(datas)  # replace with actual inference results
    response = {'results': results}
    return jsonify(response)

@app.route('/echo', methods=['GET'])
def echo():
    text = request.args.get('text', '')
    return text

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
