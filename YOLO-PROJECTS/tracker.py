from ultralytics import YOLO
import numpy as np
model = YOLO("yolo26n.pt")
results = model.track("https://youtu.be/LNwODJXcvt4", show=True,tracker='bytetrack.yaml')