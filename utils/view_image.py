import cv2
import sys
import os


def view_image(image_path, window_title="Image Viewer"):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # Use cv2 for display
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return

    window_name = window_title
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img)

    # Wait until window is closed
    print(
        f"Displaying {image_path} with title '{window_title}'. Please click the window's close button (X) when finished."
    )

    while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
        cv2.waitKey(100)  # Process GUI events, but ignore key presses (do not break)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 view_image.py <image_path> [window_title]")
        sys.exit(1)

    image_path = sys.argv[1]
    title = sys.argv[2] if len(sys.argv) > 2 else "Image Viewer"

    view_image(image_path, title)
