# 🖼️ Real-Time Image Processing Lab

A real-time Digital Image Processing application built with Python, OpenCV, and NumPy. Toggle 14 filters live on webcam or image input, with a side-by-side histogram panel and MSE/PSNR metrics.

---

## Requirements

- Python 3.8+
- OpenCV
- NumPy

Install dependencies:

```bash
pip install opencv-python numpy
```

---

## Running the App

**Webcam mode (default):**
```bash
python main.py
```

**Image file mode:**
```bash
python main.py --image input.jpg
```

**All options:**
```bash
python main.py --image input.jpg --camera 0 --output-dir results/ --max-width 640
```

| Argument | Default | Description |
|---|---|---|
| `--image` / `-i` | None | Path to an image file. Omit to use webcam. |
| `--camera` | `0` | Webcam index (if no `--image` provided) |
| `--output-dir` | `outputs/` | Folder where saved images are written |
| `--max-width` | `640` | Max display width for image file mode |

---

## Keyboard Controls

| Key | Filter | Category |
|---|---|---|
| `G` | Grayscale | Spatial |
| `S` | Sobel Edge Detection | Spatial |
| `P` | Prewitt Edge Detection | Spatial |
| `L` | Laplacian Edge Detection | Spatial |
| `B` | Box Blur | Spatial |
| `V` | Gaussian Blur | Spatial |
| `M` | Median Filter | Spatial |
| `H` | Sharpen | Spatial |
| `E` | Histogram Equalization | Histogram |
| `N` | Negative | Intensity |
| `T` | Threshold | Intensity |
| `A` | Sepia | Color |
| `D` | Emboss | Spatial |
| `F` | FFT Low-Pass Filter | Frequency |
| `W` | **Save** processed image to disk | — |
| `R` | **Reset** all filters off | — |
| `Q` | **Quit** | — |

Filters can be combined — press multiple keys to stack effects.

---

## Display Layout

The window shows three panels side by side:

```
[ ORIGINAL + UI overlay ]  |  [ PROCESSED output ]  |  [ HISTOGRAM panel ]
```

MSE and PSNR values are overlaid on the processed panel in real time.

---

## Output

Saved images are written to the `outputs/` folder (or your `--output-dir`) with a timestamp:

```
outputs/processed_20240531_143022.png
```

---

## Project Structure

```
main.py          # Main application — all code in one file
outputs/         # Auto-created when you first save (W key)
README.md
```
