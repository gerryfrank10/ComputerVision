from ultralytics import YOLO
model = YOLO("yolo26n-seg.pt")
results = model.predict('https://ultralytics.com/images/bus.jpg')
results[0].show()