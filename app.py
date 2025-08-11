from flask import Flask, render_template, request, Response
import os
import cv2
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
FRAMES_FOLDER = os.path.join("static", "frames")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # Max 200MB upload limit

# Load model once
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

DEFAULT_INTERVAL = 2  # seconds between frames
DEFAULT_BATCH_SIZE = 4

def extract_frames(video_path, output_folder, interval=DEFAULT_INTERVAL):
    """Extract frames every `interval` seconds from video."""
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps:  # fallback if fps is 0 or NaN
        fps = 30
    frame_interval = max(int(fps * interval), 1)

    count = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_count}.jpg")
            if not cv2.imwrite(frame_filename, frame):
                print(f"Warning: failed to save frame {saved_count}")
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
    """Generate captions for uploaded video frames with streaming response."""
    if 'video' not in request.files:
        return "No file uploaded", 400

    file = request.files['video']
    if file.filename == '':
        return "No file selected", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Clear old frames
    for f in os.listdir(FRAMES_FOLDER):
        try:
            os.remove(os.path.join(FRAMES_FOLDER, f))
        except Exception as e:
            print(f"Failed to delete old frame {f}: {e}")

    num_frames = extract_frames(filepath, FRAMES_FOLDER, interval=DEFAULT_INTERVAL)

    def generate_stream():
        for start in range(0, num_frames, DEFAULT_BATCH_SIZE):
            batch_paths = [
                os.path.join(FRAMES_FOLDER, f"frame_{i}.jpg")
                for i in range(start, min(start + DEFAULT_BATCH_SIZE, num_frames))
            ]
            captions = caption_images_batch(batch_paths)
            for fname, cap in zip(batch_paths, captions):
                yield f"data: {os.path.basename(fname)}|{cap}\n\n"

        # Delete uploaded video after processing to save disk space
        try:
            os.remove(filepath)
        except Exception as e:
            print(f"Failed to remove uploaded video file: {e}")

    return Response(generate_stream(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no'  # For nginx to disable buffering if used
    })

if __name__ == "__main__":
    app.run(debug=True)
