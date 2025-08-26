# Handheld Phone Usage Detection from Video

Detect and summarize handheld phone usage in video files using YOLOv8.  
This repo supports automatic inference, retains original audio, and generates a CSV summary of detected phone usage intervals.

![Demo Animation](results/20250715_142638_e37e7821.gif)

## Folder Structure

├── results/ # Output folder for annotated videos and CSV summaries  
├── best.pt # Trained YOLOv8 weights file  
├── main.ipynb # Jupyter notebook for interactive experiments  
├── requirements.txt # List of required Python packages  
└── test.py # Main script for batch phone detection


## Quick Start

### 1. Install Requirements

```
pip install -r requirements.txt
```

### 2. Run Detection on video

```
python test.py <video_path> -w best.pt -c 0.35 -o results
```

### 3. Output

- Annotated video (with boxes, original audio):
`results/<video_name>_with_audio.mp4`
- CSV summary of phone usage intervals:
`results/<video_name>_summary.csv`

### Jupyter Notebook

- Use main.ipynb for step-by-step experiments or interactive visualization.