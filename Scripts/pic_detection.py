import cv2 
from matplotlib import pyplot as plt

# List of image file paths
images = [
    "/Users/riccardo/Desktop/Github/Python_Face_Recognition/Scripts/data/Pictures/Face_1.jpeg",
    "/Users/riccardo/Desktop/Github/Python_Face_Recognition/Scripts/data/Pictures/Face_2.jpeg",
    "/Users/riccardo/Desktop/Github/Python_Face_Recognition/Scripts/data/Pictures/Face_3.jpeg"
]

# Load the cascade classifier for detecting stop signs (assuming stop_data.xml is correct)
stop_data = cv2.CascadeClassifier("/Users/riccardo/Desktop/Github/Python_Face_Recognition/Scripts/models/haarcascade_frontalface_default.xml")

# Read and process each image
for i, image_path in enumerate(images):
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect stop signs in the image
    found = stop_data.detectMultiScale(img_gray, minSize=(20, 20))

    # Draw rectangles around detected stop signs
    for (x, y, width, height) in found:
        cv2.rectangle(img_rgb, (x, y), (x + width, y + height), (0, 255, 0), 5)

    # Display the image in a subplot
    plt.subplot(1, len(images), i + 1)
    plt.imshow(img_rgb)
    plt.title(f"Image {i + 1}")
    plt.axis('off')

# Show all images
plt.show()
