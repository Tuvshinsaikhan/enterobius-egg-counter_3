# E. vermicularis — YOLO26s + SAHI video demo

## Files
- `index.html`: static browser player and detection overlay
- `process_video.py`: offline video inference, exports `detections.json`
- `requirements.txt`: Python dependencies

## Google Colab workflow
1. Enable GPU runtime, upload your trained `best.pt` and `video.mp4` into `/content`.
2. Upload `process_video.py` and `requirements.txt`.
3. Install dependencies: `!pip install -r /content/requirements.txt`
4. Run: `!python /content/process_video.py --model /content/best.pt --video /content/video.mp4 --output /content/detections.json --device cuda:0 --interval 0.5 --slice 640 --overlap 0.2`
5. Download `detections.json` and place it next to `index.html` and the **same** `video.mp4`.
6. Optionally add `PhData.jpg`, `sict.png`, `Ashius.png` logos.

## Run locally
From the web folder: `python -m http.server 8000` and open `http://localhost:8000/`.
Do not open `index.html` with a `file://` URL, as browsers may block JSON fetch.

## Deploy
Upload `index.html`, `video.mp4`, `detections.json` and optional logos to a GitHub Pages repository or Vercel static site. Video must be encoded in a browser-compatible format (typically H.264 MP4). Ensure JSON and video originate from the same video version, with no edits after inference.

## Interpretation
- The counter is the number of predicted eggs in the **current sampled frame**, not the unique egg total across the video.
- The confidence slider filters exported predictions; it cannot restore boxes omitted during inference.
- Detection annotations should be manually audited and evaluation conducted on held-out, independent specimens. Not for clinical diagnosis.
- `best.pt` is not included: use your own trained weights.
- Some SAHI versions may require an updated version for YOLO26 support. If loading fails, check SAHI/Ultralytics compatibility.
