import cv2
import sys
import os


def view_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # Use cv2 for display
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return

    window_name = "Input Preview (Press any key to continue)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img)

    # Wait until a key is pressed
    print(
        f"Displaying {image_path}. Focus the window and press any key to allow the main script to proceed..."
    )
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 view_image.py <image_path>")
        sys.exit(1)

    view_image(sys.argv[1])
