import sys
import os
from ultralytics import YOLO
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip
from pathlib import Path
from tqdm import tqdm

def run_phone_detection(
    video_path,
    weights_path="runs/detect/train/weights/best.pt",
    conf=0.35,
    save_dir="results"
):
    # Load YOLO model
    model = YOLO(weights_path)
    
    # Prepare output paths
    video_path = Path(video_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    out_video_path = save_dir / (video_path.stem + "_yolo.mp4")
    summary_path = save_dir / (video_path.stem + "_summary.csv")

    # Extract audio from original video (MoviePy)
    clip = VideoFileClip(str(video_path))
    audio = clip.audio
    fps = clip.fps

    # Run YOLO inference (frame by frame)
    cap = cv2.VideoCapture(str(video_path))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(str(out_video_path), fourcc, fps, (W, H))
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    usage_timestamps = []

    for idx in tqdm(range(frame_count), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break
        # YOLO expects RGB
        results = model.predict(frame, conf=conf, verbose=False)
        boxes = results[0].boxes if hasattr(results[0], "boxes") else results[0].pred
        phone_detected = False
        # Draw boxes and check if phone detected
        if len(boxes) > 0:
            phone_detected = True
            for box in boxes:
                # xyxy format
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf_score = float(box.conf[0])
                label = f"Phone {conf_score:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        out_video.write(frame)
        # If phone is detected, store timestamp in seconds
        if phone_detected:
            ts = idx / fps
            usage_timestamps.append(ts)
    cap.release()
    out_video.release()

    # Merge audio with annotated video (MoviePy)
    annotated_clip = VideoFileClip(str(out_video_path))
    annotated_clip = annotated_clip.set_audio(audio)
    final_video_path = save_dir / (video_path.stem + "_with_audio.mp4")
    annotated_clip.write_videofile(str(final_video_path), audio_codec='aac', codec='libx264')

    # Remove temp video without audio
    os.remove(out_video_path)

    # Generate summary
    usage_intervals = []
    if usage_timestamps:
        start = prev = usage_timestamps[0]
        for t in usage_timestamps[1:]:
            if t - prev > 1:  # if next detected frame is more than 1s away, new interval
                usage_intervals.append((start, prev))
                start = t
            prev = t
        usage_intervals.append((start, prev))
    
    # Write summary to CSV
    import csv
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Only add title if there is at least one interval
        if usage_intervals:
            start_time = usage_intervals[0][0]
            end_time = usage_intervals[-1][1]
            writer.writerow([f"Phone Usage Summary: Start={start_time:.2f}, End={end_time:.2f}"])
        else:
            writer.writerow(["Phone Usage Summary: No usage detected"])
        writer.writerow(["Start (sec)", "End (sec)"])
        for interval in usage_intervals:
            writer.writerow([f"{interval[0]:.2f}", f"{interval[1]:.2f}"])


    print(f"\nAnnotated video with audio: {final_video_path}")
    print(f"Phone usage summary CSV: {summary_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("-w", "--weights", default="best.pt", help="Path to YOLO weights")
    parser.add_argument("-c", "--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("-o", "--outdir", default="results", help="Directory to save outputs")
    args = parser.parse_args()
    run_phone_detection(args.video_path, args.weights, args.conf, args.outdir)
