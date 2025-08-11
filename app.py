from flask import Flask, render_template, request, Response
import os
import cv2
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from tqdm import tqdm

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
FRAMES_FOLDER = os.path.join("static", "frames")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model once
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def extract_frames(video_path, output_folder, interval=2):
    """Extract frames every `interval` seconds from video."""
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)

    count = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
        count += 1
    cap.release()
    return saved_count

def caption_images_batch(image_paths):
    """Generate captions for a batch of image paths."""
    images = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    out = model.generate(**inputs)
    captions = [processor.decode(o, skip_special_tokens=True) for o in out]
    return captions

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    if 'video' not in request.files:
        return "No file uploaded", 400

    file = request.files['video']
    if file.filename == '':
        return "No file selected", 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Clear old frames
    for f in os.listdir(FRAMES_FOLDER):
        os.remove(os.path.join(FRAMES_FOLDER, f))

    num_frames = extract_frames(filepath, FRAMES_FOLDER, interval=2)
    batch_size = 4

    def generate_stream():
        for start in tqdm(range(0, num_frames, batch_size), desc="Captioning batches"):
            batch_paths = [
                os.path.join(FRAMES_FOLDER, f"frame_{i}.jpg")
                for i in range(start, min(start + batch_size, num_frames))
            ]
            batch_captions = caption_images_batch(batch_paths)
            for fname, cap in zip(batch_paths, batch_captions):
                yield f"data: {os.path.basename(fname)}|{cap}\n\n"

    return Response(generate_stream(), mimetype='text/event-stream')

if __name__ == "__main__":
    app.run(debug=True)
