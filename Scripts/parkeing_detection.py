import cv2
import numpy as np
#####
## Das Ziel dieses Skript wird in zwei aufgaben unterteilt die erreicht werden sollen
## 1. Erkenne die Parkplätze auf dem Bild
## 2. Erkenne ob ein Parkplatz besetzt ist oder nicht

## Was ist zu tun:
## 1. Wir müssen feststellen durch hochladen eines Bildes wo die Parkplätze sind.
#### 1.1. Wir werden durch die Linien auf den Boden die Parkplätze erkennen
#### 1.2. Die Parkplätze müssen dann Markiert werden das es ein Parkplatz ist

## 2. Wir müssen erkennen ob das Auto darauf ist oder nicht
#### 2.1. Die Autos müssen erkannt werden ob sie auf dem Parkplatz sind oder nicht
#### 2.2. Die Autos müssen dann markiert werden das sie auf dem Parkplatz sind
#### 2.3. Die Autos müssen dann markiert werden das sie nicht auf dem Parkplatz sind



path = "Scripts/data/Pictures/Parked cars/Cars-parked-in-parking-lot.jpg"
path_without_cars = "Scripts/data/Pictures/Parked cars/image.png"
img = cv2.imread(filename=path_without_cars)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)


rho = 1                         # distance resolution in pixels of the Hough grid
theta = np.pi / 180             # angular resolution in radians of the Hough grid
threshold = 15                  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 50            # minimum number of pixels making up a line
max_line_gap = 20               # maximum gap in pixels between connectable line segments
line_image = np.copy(img) * 0   # creating a blank to draw lines on

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                    min_line_length, max_line_gap)

for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)


lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)





cv2.imshow("Result", lines_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()