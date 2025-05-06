import numpy as np
import cv2
import matplotlib.pyplot as plt

def show_img(img, ax=None, figsize=(10, 8)):
    if not ax: _, ax  = plt.subplots(1,1, figsize=figsize)
    if len(img.shape) == 2: img = np.tile(img[:,:,None], 3)
    ax.imshow(img[:, :, ::-1])
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    return ax

# cars = cv2.imread('../04-files/cars.jpg')
cars = cv2.imread('IMG_1601.jpeg')
# show_img(cars)

roi_points = []
items = []
segmentation_history = []

def select_roi(event, x, y, flags, param):
    global roi_points

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        roi_points.append((x, y))

        # Draw a rectangle around the ROI
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        cv2.rectangle(cars, roi_points[0], roi_points[1], color, -1)

        cv2.imshow('Image', cars)

        # Calculate and print dimensions (in pixels)
        width = abs(roi_points[1][0] - roi_points[0][0])
        height = abs(roi_points[1][1] - roi_points[0][1])
        segmentation_history.append((cars.copy(), roi_points.copy()))
        items.append((width, height))
        print(f'Width: {width} pixels, Height: {height} pixels')

def undo_segmentation():
    global roi_points
    if len(segmentation_history) > 0:
        cars, _ = segmentation_history.pop()
        cv2.imshow('Image', cars)
    else:
        print('No ROI to undo.')

cv2.namedWindow('Image')
cv2.setMouseCallback('Image', select_roi)
cv2.imshow('Image', cars)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('u'):
        undo_segmentation()
    elif key == ord('q'):
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
print(items)
print(len(items))