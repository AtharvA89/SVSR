import cv2
import numpy as np
from google.colab.patches import cv2_imshow   

# Read image
img = cv2.imread('input.png')   # make sure file exists in Colab
img = cv2.resize(img, (600, 600))
output = img.copy()

# Convert to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# -------- Color Ranges --------
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

lower_green = np.array([36, 50, 70])
upper_green = np.array([89, 255, 255])

lower_blue = np.array([90, 50, 70])
upper_blue = np.array([128, 255, 255])

lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([35, 255, 255])

# -------- Shape Detection --------
def detect_shape(cnt):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

    if len(approx) == 3:
        return "Triangle"
    elif len(approx) == 4:
        return "Rectangle"
    else:
        return "Circle"

# -------- Process Function --------
def detect(mask, color_name):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 700:
            shape = detect_shape(cnt)
            x, y, w, h = cv2.boundingRect(cnt)

            cv2.drawContours(output, [cnt], -1, (0, 0, 0), 2)
            cv2.putText(output, color_name + " " + shape, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

# -------- Masks --------
mask_red = cv2.bitwise_or(
    cv2.inRange(hsv, lower_red1, upper_red1),
    cv2.inRange(hsv, lower_red2, upper_red2)
)

mask_green = cv2.inRange(hsv, lower_green, upper_green)
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

# -------- Detection --------
detect(mask_red, "Red")
detect(mask_green, "Green")
detect(mask_blue, "Blue")
detect(mask_yellow, "Yellow")

# -------- Show Output (FIXED) --------
cv2_imshow(output)