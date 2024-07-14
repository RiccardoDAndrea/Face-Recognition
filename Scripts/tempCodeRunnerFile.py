    plt.subplot(1, len(images), i + 1)
    plt.imshow(img_rgb)
    plt.title(f"Face {i+1}")
    plt.axis('off')

plt.show()