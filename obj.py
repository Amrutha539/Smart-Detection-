from flask import Flask, render_template_string, Response, request, jsonify
from ultralytics import YOLO
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
from openai import OpenAI

# ---------------- CHATGPT API ----------------
client = OpenAI(api_key="")
# ----------------------------------------------------

app = Flask(__name__)

# Create folders
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/output", exist_ok=True)

# YOLOv8 model
model_yolo = YOLO("yolov8n.pt")

# ResNet50 classification model
model_cls = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model_cls.eval()
imagenet_labels = models.ResNet50_Weights.DEFAULT.meta["categories"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Object descriptions dictionary
object_descriptions = {
    "person": "A human being, possibly walking or performing an action.",
    "car": "A four-wheeled motor vehicle used for transportation.",
    "dog": "A domestic animal known for loyalty and companionship.",
    "cat": "A small domesticated carnivorous mammal with soft fur.",
    "chair": "A piece of furniture designed for sitting.",
    "bottle": "A container used to store liquids.",
    "bicycle": "A two-wheeled vehicle powered by pedaling.",
    "cell phone": "A portable electronic device used for communication.",
    "laptop": "A portable personal computer suitable for mobile use.",
    "book": "A collection of written or printed pages bound together.",
    "truck": "A large motor vehicle for transporting goods.",
    "bus": "A large vehicle designed to carry many passengers.",
    "horse": "A large domesticated animal used for riding or transport.",
    "sheep": "A woolly farm animal raised for wool or meat.",
    "cow": "A large domesticated farm animal that produces milk.",
    "train": "A series of connected railway cars pulled by a locomotive.",
    "knife": "A utensil with a blade used for cutting.",
    "apple": "A sweet edible fruit that grows on apple trees.",
    "banana": "A long curved fruit that grows in clusters and has soft flesh.",
    "cup": "A small bowl-shaped container used for drinking.",
    "sofa": "A comfortable seating furniture that can fit multiple people."
}

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Smart Detection</title>
<style>
body, html { margin:0; padding:0; height:100%; font-family: Arial, sans-serif; overflow-x:hidden; }
#loginContainer video#bgVideo { position: fixed; top:0; left:0; min-width: 100%; min-height: 100%; object-fit: cover; z-index: -1; }
#loginContainer { position: relative; width: 100%; height: 100vh; }
.heading-container { position: absolute; top: 2%; width: 100%; text-align: center; z-index: 2; }
.heading-container span { display: inline-block; font-size: 70px; font-weight: bold; color: white; text-shadow: 3px 3px 8px rgba(0,0,0,0.7); animation: slideAcross 10s linear infinite; }
@keyframes slideAcross { 0% { transform: translateX(120vw); } 100% { transform: translateX(-120vw); } }
#loginForm { position: absolute; top: 20%; left: 5%; z-index: 3; background: rgba(255,255,255,0.95); padding: 40px; border-radius: 10px; width: 350px; box-shadow: 0 0 20px rgba(0,0,0,0.2); }
#loginForm h2 { text-align: center; color: #007bff; margin-bottom: 20px; font-size: 32px; text-shadow: 2px 2px 5px rgba(0,0,0,0.7); }
#loginForm input, #loginForm button { width: 100%; padding: 10px; margin: 10px 0; border-radius: 5px; border: 1px solid #ccc; }
#loginForm button { background: #007bff; color: white; border: none; cursor: pointer; font-size: 16px; }
#loginForm button:hover { background:#0056b3; }
#detectionContainer { display:none; min-height:100vh; background: url('/static/img1.avif') no-repeat center center fixed; background-size: cover; color:white; text-align:center; padding:20px; }
#detectionContainer h1 { font-size:60px; text-shadow: 3px 3px 8px rgba(0,0,0,0.7); }
.upload-section input[type=file], .upload-section button { border-radius:8px; font-size:16px; border:none; background:#007bff; color:white; cursor:pointer; padding:10px 20px; margin:5px; }
.upload-section button:hover { background:#0056b3; }
#output-section { margin-top:20px; display:none; }
#output-section img, #video-stream { width:700px; height:auto; border-radius:15px; border:2px solid #007bff; box-shadow:0 4px 15px rgba(0,0,0,0.3); }
</style>
</head>
<body>

<div id="loginContainer">
    <video id="bgVideo" autoplay muted loop>
        <source src="static/animation.mp4" type="video/mp4">
    </video>
    <div class="heading-container"><span>Smart Detection</span></div>
    <form id="loginForm">
        <h2>Login</h2>
        <input type="text" name="username" placeholder="Username" required>
        <input type="password" name="password" placeholder="Password" required>
        <button type="submit">Login</button>
    </form>
</div>

<div id="detectionContainer">
    <h1>Smart Detection</h1>
    <div class="upload-section">
        <h3>Upload an Image</h3>
        <form id="uploadForm" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" required>
            <button type="submit">Detect</button>
        </form>
        <br>
        <button id="toggleWebcam">Allow Webcam</button>
    </div>
    <div id="output-section">
        <h3 id="output-title"></h3>
        <div id="output-content"></div>
    </div>
</div>

<script>
const loginForm = document.getElementById('loginForm');
loginForm.addEventListener('submit', (e)=>{
    e.preventDefault();
    if(loginForm.username.value==='admin' && loginForm.password.value==='1234'){
        document.getElementById('loginContainer').style.display='none';
        document.getElementById('detectionContainer').style.display='block';
    } else alert('Invalid username or password.');
});

const uploadForm = document.getElementById("uploadForm");
const toggleWebcamBtn = document.getElementById("toggleWebcam");
const outputSection = document.getElementById("output-section");
const outputTitle = document.getElementById("output-title");
const outputContent = document.getElementById("output-content");
let webcamOn=false;

uploadForm.addEventListener("submit", async (e)=>{
    e.preventDefault();
    stopWebcam();
    outputSection.style.display="block";
    outputTitle.innerText="Processing...";
    outputContent.innerHTML="";

    const formData = new FormData(uploadForm);
    const response = await fetch("/upload",{method:"POST", body:formData});
    const data = await response.json();

    if(data.error){
        outputTitle.innerText="Error";
        outputContent.innerHTML=`<p style='color:red;'>${data.error}</p>`;
    } else {
        outputTitle.innerHTML=`Detected Objects`;
        let descHTML = `<img src="${data.output_image}" alt="Detected Image"><br><br>`;
        if(data.descriptions.length > 0){
            descHTML += `<h3>Objects and Descriptions:</h3><ul>`;
            data.descriptions.forEach(item => {
                descHTML += `<li><b>${item.object}:</b> ${item.description}</li>`;
            });
            descHTML += "</ul>";
        }
        if(data.gpt_summary){
    descHTML += `<h2 style="color:blue;">Image Summary</h2>
                 <p style="color:black; background:white; padding:10px; border-radius:8px; width:80%; margin:auto;">${data.gpt_summary}</p>`;
}

        outputContent.innerHTML = descHTML;
    }
});

toggleWebcamBtn.addEventListener("click", ()=>{
    if(!webcamOn) startWebcam();
    else stopWebcam();
});

function startWebcam(){
    outputSection.style.display="block";
    outputTitle.innerText="Live Webcam Detection";
    outputContent.innerHTML=`<img id="video-stream" src="/video_feed" alt="Live Stream">`;
    toggleWebcamBtn.textContent="Stop Webcam";
    webcamOn=true;
}
function stopWebcam(){
    fetch("/stop_video");
    toggleWebcamBtn.textContent="Allow Webcam";
    outputContent.innerHTML="";
    outputSection.style.display="none";
    webcamOn=false;
}
</script>
</body>
</html>
"""

# ---------------- FLASK ROUTES ----------------
@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error':'No image uploaded'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error':'Empty filename'}), 400

    file_path = os.path.join("static/uploads", file.filename)
    file.save(file_path)

    results = model_yolo(file_path)
    output_path = os.path.join("static/output", f"det_{file.filename}")
    results[0].save(filename=output_path)

    detected_objects = [results[0].names[int(box.cls)] for box in results[0].boxes]

    img = Image.open(file_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        preds = model_cls(img_tensor)
        class_label = imagenet_labels[preds.argmax().item()]

    all_objects = list(set(detected_objects + [class_label]))

    description_data = []
    for obj in all_objects:
        desc = object_descriptions.get(obj.lower())
        if desc:
            description_data.append({"object": obj, "description": desc})

    try:
        chat_prompt = f"The detected objects are: {', '.join(all_objects)}. Describe this image in 3-4 lines."
        gpt_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful AI image describer."},
                {"role": "user", "content": chat_prompt}
            ]
        )
        gpt_description = gpt_response.choices[0].message.content.strip()
    except Exception as e:
        gpt_description = f"Description unavailable (ChatGPT error: {str(e)})"

    return jsonify({
        'output_image': output_path,
        'descriptions': description_data,
        'gpt_summary': gpt_description
    })

# ---------------- WEBCAM FIXED ----------------
cap = None
def generate_frames():
    global cap
    cv2.destroyAllWindows()
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot access webcam")
        return

    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (640, 480))
        if frame_count % 5 == 0:
            results = model_yolo.predict(source=frame, stream=False, verbose=False)
            annotated_frame = results[0].plot()
        frame_count += 1

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_video')
def stop_video():
    global cap
    if cap and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    return jsonify({"status": "stopped"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
