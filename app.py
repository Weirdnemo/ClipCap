from flask import Flask, render_template, request, redirect, url_for
import os, cv2


def extract_frames(video_path, output_folder, interval=2):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps*interval)

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

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'video' not in request.files:
            return "No file part", 400
        
        file = request.files['video']
        if file.filename == '':
            return "No selected file", 400
        
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        frames_folder = os.path.join("static", "frames")
        os.makedirs(frames_folder, exist_ok=True)
        
        num_frames = extract_frames(filepath, frames_folder, interval=2)
        return f"Extracted {num_frames} frames from video"
    

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

