"""
========================================================
Real-Time Image Processing Lab
Digital Image Processing - Semester Project
========================================================
Libraries: OpenCV, NumPy
Run webcam mode:
    python main.py

Run image file mode:
    python main.py --image input.jpg

Controls:
    Press keys shown on screen to toggle filters
    Press W to save processed result
========================================================
"""

import cv2
import numpy as np
import argparse
import os
from datetime import datetime


class ImageProcessor:
    def __init__(self):
        self.filters = {
            'grayscale': False,
            'edge_sobel': False,
            'edge_prewitt': False,
            'edge_laplacian': False,
            'blur': False,
            'gaussian_blur': False,
            'median_filter': False,
            'sharpen': False,
            'histogram_eq': False,
            'negative': False,
            'threshold': False,
            'sepia': False,
            'emboss': False,
            'fft_lowpass': False,
        }

        self.threshold_val = 128
        self.blur_ksize = 5
        self.gaussian_sigma = 1.5

    def apply_filters(self, frame):
        result = frame.copy()

        # -------------------------------------------------
        # Spatial Domain Operation 1: Grayscale Conversion
        # -------------------------------------------------
        if self.filters['grayscale']:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        # -------------------------------------------------
        # Histogram Processing: Histogram Equalization
        # -------------------------------------------------
        if self.filters['histogram_eq']:
            yuv = cv2.cvtColor(result, cv2.COLOR_BGR2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            result = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

        # -------------------------------------------------
        # Color Transformation: Sepia
        # -------------------------------------------------
        if self.filters['sepia']:
            kernel = np.array([
                [0.131, 0.534, 0.272],
                [0.168, 0.686, 0.349],
                [0.189, 0.769, 0.393]
            ])
            result = cv2.transform(result, kernel)
            result = np.clip(result, 0, 255).astype(np.uint8)

        # -------------------------------------------------
        # Spatial Domain Operation 2: Box Blur
        # -------------------------------------------------
        if self.filters['blur']:
            result = cv2.blur(result, (self.blur_ksize, self.blur_ksize))

        # -------------------------------------------------
        # Spatial Domain Operation 3: Gaussian Blur
        # -------------------------------------------------
        if self.filters['gaussian_blur']:
            result = cv2.GaussianBlur(result, (5, 5), self.gaussian_sigma)

        # -------------------------------------------------
        # Spatial Domain Operation 4: Median Filtering
        # -------------------------------------------------
        if self.filters['median_filter']:
            result = cv2.medianBlur(result, 5)

        # -------------------------------------------------
        # Spatial Domain Operation 5: Sharpening
        # -------------------------------------------------
        if self.filters['sharpen']:
            kernel = np.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ])
            result = cv2.filter2D(result, -1, kernel)

        # -------------------------------------------------
        # Frequency Domain Operation: FFT Low-Pass Filter
        # -------------------------------------------------
        if self.filters['fft_lowpass']:
            result = self.fft_lowpass_filter(result)

        # -------------------------------------------------
        # Edge Detection: Sobel
        # -------------------------------------------------
        if self.filters['edge_sobel']:
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

            magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
            magnitude = np.uint8(np.clip(magnitude, 0, 255))

            result = cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)

        # -------------------------------------------------
        # Edge Detection: Prewitt
        # -------------------------------------------------
        if self.filters['edge_prewitt']:
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).astype(np.float64)

            kernelx = np.array([
                [-1, 0, 1],
                [-1, 0, 1],
                [-1, 0, 1]
            ])

            kernely = np.array([
                [-1, -1, -1],
                [0, 0, 0],
                [1, 1, 1]
            ])

            gx = cv2.filter2D(gray, -1, kernelx)
            gy = cv2.filter2D(gray, -1, kernely)

            magnitude = np.sqrt(gx ** 2 + gy ** 2)
            magnitude = np.uint8(np.clip(magnitude, 0, 255))

            result = cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)

        # -------------------------------------------------
        # Edge Detection: Laplacian
        # -------------------------------------------------
        if self.filters['edge_laplacian']:
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian = np.uint8(np.clip(np.abs(laplacian), 0, 255))

            result = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)

        # -------------------------------------------------
        # Spatial Domain Operation: Emboss
        # -------------------------------------------------
        if self.filters['emboss']:
            kernel = np.array([
                [-2, -1, 0],
                [-1, 1, 1],
                [0, 1, 2]
            ])

            embossed = cv2.filter2D(result, cv2.CV_16S, kernel)
            embossed = embossed + 128
            result = np.clip(embossed, 0, 255).astype(np.uint8)

        # -------------------------------------------------
        # Thresholding
        # -------------------------------------------------
        if self.filters['threshold']:
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

            _, binary = cv2.threshold(
                gray,
                self.threshold_val,
                255,
                cv2.THRESH_BINARY
            )

            result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        # -------------------------------------------------
        # Negative Transformation
        # -------------------------------------------------
        if self.filters['negative']:
            result = 255 - result

        return result

    def fft_lowpass_filter(self, image):
        """
        Frequency-domain low-pass filtering using FFT.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2

        dft = np.fft.fft2(gray)
        dft_shift = np.fft.fftshift(dft)

        mask = np.zeros((rows, cols), np.uint8)
        radius = max(10, min(rows, cols) // 8)
        cv2.circle(mask, (ccol, crow), radius, 1, -1)

        filtered_shift = dft_shift * mask

        inverse_shift = np.fft.ifftshift(filtered_shift)
        img_back = np.fft.ifft2(inverse_shift)
        img_back = np.abs(img_back)

        img_back = np.clip(img_back, 0, 255).astype(np.uint8)

        return cv2.cvtColor(img_back, cv2.COLOR_GRAY2BGR)


def calculate_mse(original, processed):
    """Mean Squared Error between original and processed image."""
    if original.shape != processed.shape:
        processed = cv2.resize(processed, (original.shape[1], original.shape[0]))

    original_float = original.astype(np.float64)
    processed_float = processed.astype(np.float64)

    mse = np.mean((original_float - processed_float) ** 2)

    return mse


def calculate_psnr(original, processed):
    """Peak Signal-to-Noise Ratio between original and processed image."""
    mse = calculate_mse(original, processed)

    if mse == 0:
        return float('inf')

    psnr = 20 * np.log10(255.0 / np.sqrt(mse))

    return psnr


def create_histogram_panel(image, target_height=480, panel_width=320):
    """Create a live histogram display panel with proper sizing."""
    
    # Create panel with exact target height
    panel = np.zeros((target_height, panel_width, 3), dtype=np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten()

    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    min_intensity = np.min(gray)
    max_intensity = np.max(gray)

    cv2.putText(panel, "Histogram Analysis", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    cv2.putText(panel, f"Mean: {mean_intensity:.2f}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    cv2.putText(panel, f"Std Dev: {std_intensity:.2f}", (20, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    cv2.putText(panel, f"Min: {min_intensity}  Max: {max_intensity}", (20, 104),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    margin_left = 30
    margin_right = 15
    margin_top = 130
    margin_bottom = 35

    plot_w = panel_width - margin_left - margin_right
    plot_h = max(50, target_height - margin_top - margin_bottom)

    x_axis_y = margin_top + plot_h

    # Draw axes
    cv2.line(panel, (margin_left, margin_top), (margin_left, x_axis_y), (180, 180, 180), 1)
    cv2.line(panel, (margin_left, x_axis_y), (margin_left + plot_w, x_axis_y), (180, 180, 180), 1)

    # Normalize histogram for display
    if hist.max() > 0:
        hist_norm = hist / hist.max() * plot_h
    else:
        hist_norm = hist

    # Draw histogram curve
    for i in range(1, 256):
        x1 = margin_left + int((i - 1) * plot_w / 255)
        y1 = x_axis_y - int(hist_norm[i - 1])

        x2 = margin_left + int(i * plot_w / 255)
        y2 = x_axis_y - int(hist_norm[i])

        cv2.line(panel, (x1, y1), (x2, y2), (0, 255, 100), 1)

    cv2.putText(panel, "0", (margin_left - 5, x_axis_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    cv2.putText(panel, "255", (margin_left + plot_w - 25, x_axis_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    return panel


def draw_ui(frame, processor, source_text):
    """Draw filter status overlay on the original frame."""
    overlay = frame.copy()

    cv2.rectangle(overlay, (10, 10), (360, 455), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, "DIP Processing Lab", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    cv2.putText(frame, source_text, (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    cv2.putText(frame, "Press keys to toggle filters:", (20, 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    controls = [
        ('G', 'grayscale', 'Grayscale'),
        ('S', 'edge_sobel', 'Sobel Edge'),
        ('P', 'edge_prewitt', 'Prewitt Edge'),
        ('L', 'edge_laplacian', 'Laplacian Edge'),
        ('B', 'blur', 'Box Blur'),
        ('V', 'gaussian_blur', 'Gaussian Blur'),
        ('M', 'median_filter', 'Median Filter'),
        ('H', 'sharpen', 'Sharpen'),
        ('E', 'histogram_eq', 'Histogram Eq.'),
        ('N', 'negative', 'Negative'),
        ('T', 'threshold', 'Threshold'),
        ('A', 'sepia', 'Sepia'),
        ('D', 'emboss', 'Emboss'),
        ('F', 'fft_lowpass', 'FFT Low Pass'),
    ]

    for i, (key, filter_name, label) in enumerate(controls):
        y = 115 + i * 22

        active = processor.filters[filter_name]
        color = (0, 255, 100) if active else (110, 110, 110)
        status = "ON" if active else "OFF"
        text = f"[{key}] {label}: {status}"

        cv2.putText(frame, text, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.putText(frame, "[W] Save Processed Image", (20, 425),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 255, 150), 1)

    cv2.putText(frame, "[R] Reset All   [Q] Quit", (20, 445),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 255), 1)

    return frame


def draw_metrics(combined, start_x, mse, psnr):
    """Draw quantitative evaluation values on display."""
    cv2.rectangle(combined, (start_x + 10, 10), (start_x + 300, 95), (0, 0, 0), -1)

    cv2.putText(combined, "Quantitative Evaluation", (start_x + 20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

    cv2.putText(combined, f"MSE: {mse:.2f}", (start_x + 20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if np.isinf(psnr):
        psnr_text = "PSNR: infinite"
    else:
        psnr_text = f"PSNR: {psnr:.2f} dB"

    cv2.putText(combined, psnr_text, (start_x + 20, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return combined


def resize_to_target_height(image, target_height):
    """Resize image to match target height while maintaining aspect ratio."""
    h, w = image.shape[:2]
    if h == target_height:
        return image
    
    ratio = target_height / float(h)
    new_w = int(w * ratio)
    return cv2.resize(image, (new_w, target_height))


def save_processed_image(processed, output_dir="outputs"):
    """Save processed image to disk."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"processed_{timestamp}.png"
    path = os.path.join(output_dir, filename)

    success = cv2.imwrite(path, processed)

    if success:
        print(f"  [SAVE] Processed image saved to: {path}")
    else:
        print("  [ERROR] Could not save image.")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Real-Time Digital Image Processing Lab")

    parser.add_argument("--image", "-i", type=str, default=None,
                        help="Path to image file. If not provided, webcam mode is used.")

    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index for webcam mode. Default is 0.")

    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Directory where processed images will be saved.")

    parser.add_argument("--max-width", type=int, default=640,
                        help="Maximum display width for loaded image files.")

    return parser.parse_args()


def main():
    args = parse_arguments()

    processor = ImageProcessor()

    cap = None
    source_image = None
    source_text = ""

    # Target height for all panels
    TARGET_HEIGHT = 480

    # -------------------------------------------------
    # Image File Input
    # -------------------------------------------------
    if args.image is not None:
        source_image = cv2.imread(args.image)

        if source_image is None:
            print(f"Error: Could not load image from disk: {args.image}")
            return

        source_image = resize_to_target_height(source_image, TARGET_HEIGHT)

        source_text = f"Source: Image File - {args.image}"

        print("=" * 60)
        print("  Real-Time DIP Processing Lab")
        print("  Mode: Image File")
        print(f"  Loaded image: {args.image}")
        print("  Press W to save processed result")
        print("  Press Q to quit")
        print("=" * 60)

    # -------------------------------------------------
    # Webcam Input
    # -------------------------------------------------
    else:
        cap = cv2.VideoCapture(args.camera)

        if not cap.isOpened():
            print("Error: Cannot open webcam!")
            return

        source_text = f"Source: Webcam Camera {args.camera}"

        print("=" * 60)
        print("  Real-Time DIP Processing Lab")
        print("  Mode: Webcam")
        print("  Press keys to toggle filters")
        print("  Press W to save processed result")
        print("  Press Q to quit")
        print("=" * 60)

    cv2.namedWindow("DIP Processing Lab", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("DIP Processing Lab", 1600, 520)

    last_processed = None

    while True:
        if source_image is not None:
            frame = source_image.copy()
        else:
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame.")
                break
            
            # Resize webcam frame to target height
            frame = resize_to_target_height(frame, TARGET_HEIGHT)

        # Apply selected filters
        processed = processor.apply_filters(frame)

        last_processed = processed.copy()

        # Quantitative evaluation
        mse = calculate_mse(frame, processed)
        psnr = calculate_psnr(frame, processed)

        # Draw UI on original image
        display_original = draw_ui(frame.copy(), processor, source_text)

        # Create histogram panel with exact target height
        histogram_panel = create_histogram_panel(processed, 
                                                  target_height=TARGET_HEIGHT, 
                                                  panel_width=320)

        # Get dimensions
        h1, w1 = display_original.shape[:2]
        h2, w2 = processed.shape[:2]
        h3, w3 = histogram_panel.shape[:2]

        # Resize all panels to same height
        display_original = resize_to_target_height(display_original, TARGET_HEIGHT)
        processed = resize_to_target_height(processed, TARGET_HEIGHT)
        histogram_panel = resize_to_target_height(histogram_panel, TARGET_HEIGHT)

        # Combine original, processed, and histogram display
        combined = np.hstack([display_original, processed, histogram_panel])

        # Labels
        cv2.putText(combined, "ORIGINAL", (w1 // 2 - 50, TARGET_HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.putText(combined, "PROCESSED", (w1 + w2 // 2 - 70, TARGET_HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.putText(combined, "HISTOGRAM", (w1 + w2 + 95, TARGET_HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)

        # Draw quantitative metrics on processed side
        combined = draw_metrics(combined, w1, mse, psnr)

        cv2.imshow("DIP Processing Lab", combined)

        key = cv2.waitKey(1) & 0xFF

        if key == 255:
            continue

        try:
            key_char = chr(key).lower()
        except ValueError:
            key_char = ""

        key_map = {
            'g': 'grayscale',
            's': 'edge_sobel',
            'p': 'edge_prewitt',
            'l': 'edge_laplacian',
            'b': 'blur',
            'v': 'gaussian_blur',
            'm': 'median_filter',
            'h': 'sharpen',
            'e': 'histogram_eq',
            'n': 'negative',
            't': 'threshold',
            'a': 'sepia',
            'd': 'emboss',
            'f': 'fft_lowpass',
        }

        if key_char in key_map:
            filter_name = key_map[key_char]
            processor.filters[filter_name] = not processor.filters[filter_name]
            status = "ON" if processor.filters[filter_name] else "OFF"
            print(f"  [{filter_name.upper()}] -> {status}")

        elif key_char == 'r':
            for filter_name in processor.filters:
                processor.filters[filter_name] = False
            print("  [RESET] All filters disabled")

        elif key_char == 'w':
            if last_processed is not None:
                save_processed_image(last_processed, args.output_dir)

        elif key_char == 'q':
            break

    if cap is not None:
        cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()