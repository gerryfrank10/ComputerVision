from ultralytics import YOLO
import matplotlib.pyplot as plt
model = YOLO("yolo26n.pt")
annotated_image = model.predict('https://ultralytics.com/images/bus.jpg')[0].show()
# This one by using matplotlib
# annotated_image = model.predict('https://ultralytics.com/images/bus.jpg')[0].plot()
# plt.imshow(annotated_image[:, :, ::-1])
# plt.axis('off')
# plt.show()