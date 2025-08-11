
# AI Video Captioning Tool

🎥 **Real-time video frame captioning with state-of-the-art vision-language AI**

---

## About

After diving into Reinforcement Learning for a couple of months, I decided to explore something new — building an AI-powered video captioning tool! This Flask app extracts frames from uploaded videos, generates captions using a powerful image captioning model, and streams the captions live to the browser.

Ideal for accessibility demos (e.g., smart glasses for the visually impaired) or just exploring vision-language AI in action.

---

## Features

- Upload a video and extract frames every 2 seconds  
- Batch caption frames with a pretrained BLIP model  
- Stream captions live to the frontend with a sleek dark-themed UI  
- Responsive grid layout with smooth caption animations

---

## Demo

![Demo GIF](path/to/demo.gif)

---

## Usage

1. Clone the repo  
2. Create and activate a Python environment  
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```  
4. Run the Flask app:  
   ```
   python app.py
   ```  
5. Open `http://127.0.0.1:5000` in your browser  
6. Upload a video and watch captions stream live!

---

## Tech Stack

- Python, Flask  
- PyTorch, Transformers (Salesforce BLIP model)  
- OpenCV for frame extraction  
- HTML/CSS/JS frontend with Fetch API and Server-Sent Events

---

## To Do

- Add transcript download feature  
- Improve frame extraction intervals  
- Explore integration with RL-based attention mechanisms for caption improvement

---

## License

MIT License
