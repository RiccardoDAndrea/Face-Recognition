import cv2 
from matplotlib import pyplot as plt
import time

start_time = time.time()

img = cv2.imread("Scripts/data/Pictures/group_pic.jpeg")

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

stop_data = cv2.CascadeClassifier("Scripts/models/haarcascade_frontalface_default.xml")
fullbody = cv2.CascadeClassifier("Scripts/models/haarcascade_fullbody.xml")

found = stop_data.detectMultiScale(img_gray, minSize =(20, 20))

amount = len(found)

if amount != 0:
    for (x, y, width, height) in found:
        # Länge der offenen Stellen an den Seiten
        offset = 20
        
        # Dicke der Linien (zum Beispiel 2 Pixel)
        thickness = 1
        
        # Linien oben und unten
        cv2.line(img_rgb, (x, y), (x + width // 2 - offset, y), (0, 255, 0), thickness)
        cv2.line(img_rgb, (x + width // 2 + offset, y), (x + width, y), (0, 255, 0), thickness)
        cv2.line(img_rgb, (x, y + height), (x + width // 2 - offset, y + height), (0, 255, 0), thickness)
        cv2.line(img_rgb, (x + width // 2 + offset, y + height), (x + width, y + height), (0, 255, 0), thickness)
        
        # Linien links und rechts
        cv2.line(img_rgb, (x, y), (x, y + height // 2 - offset), (0, 255, 0), thickness)
        cv2.line(img_rgb, (x, y + height // 2 + offset), (x, y + height), (0, 255, 0), thickness)
        cv2.line(img_rgb, (x + width, y), (x + width, y + height // 2 - offset), (0, 255, 0), thickness)
        cv2.line(img_rgb, (x + width, y + height // 2 + offset), (x + width, y + height), (0, 255, 0), thickness)

# Endzeit messen und Laufzeit berechnen
end_time = time.time()
execution_time = end_time - start_time
print(f"Der Code lief in {execution_time:.4f} Sekunden durch.")



plt.subplot(1, 1, 1)
plt.imshow(img_rgb)
plt.show()