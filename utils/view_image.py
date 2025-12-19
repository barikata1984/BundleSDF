import cv2
import sys
import os
import numpy as np


def view_image(image_source, window_title="Image Viewer", from_stdin=False):
    if from_stdin:
        # Read all bytes from stdin
        image_data = image_source.read()
        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        source_desc = "stdin"
    else:
        if not os.path.exists(image_source):
            print(f"Error: Image not found at {image_source}")
            return
        img = cv2.imread(image_source)
        source_desc = image_source

    if img is None:
        print(f"Error: Could not load image from {source_desc}")
        return

    window_name = window_title
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img)

    # Wait until window is closed
    print(
        f"Displaying image from {source_desc} with title '{window_title}'. Please click the window's close button (X) when finished."
    )

    while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
        cv2.waitKey(100)  # Process GUI events, but ignore key presses (do not break)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 view_image.py <image_path|stdin> [window_title]")
        sys.exit(1)

    input_arg = sys.argv[1]
    title = sys.argv[2] if len(sys.argv) > 2 else "Image Viewer"

    if input_arg == "stdin":
        view_image(sys.stdin.buffer, title, from_stdin=True)
    else:
        view_image(input_arg, title, from_stdin=False)
