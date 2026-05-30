# OpenCV Computer Vision — Complete Engineering Notes

> **Scope:** Production-grade, mechanism-first notes covering the full OpenCV bootcamp curriculum. Every section explains *why* something works, not just *how* to call it. Python code included throughout.

---

## Table of Contents

1. [Foundations — How Images Are Represented](#1-foundations--how-images-are-represented)
2. [Reading and Writing Images](#2-reading-and-writing-images)
3. [Working with Video Files](#3-working-with-video-files)
4. [Exploring Color Spaces](#4-exploring-color-spaces)
5. [Color Thresholding](#5-color-thresholding)
6. [Image Resizing, Scaling, and Interpolation](#6-image-resizing-scaling-and-interpolation)
7. [Flip, Rotate, and Crop](#7-flip-rotate-and-crop)
8. [OpenCV Coordinate System](#8-opencv-coordinate-system)
9. [Drawing Lines and Shapes](#9-drawing-lines-and-shapes)
10. [Adding Text to Images](#10-adding-text-to-images)
11. [Affine and Perspective Transformations](#11-affine-and-perspective-transformations)
12. [Image Filters](#12-image-filters)
13. [Blur Filters — Average, Gaussian, Median](#13-blur-filters--average-gaussian-median)
14. [Edge Detection — Sobel, Canny, Laplacian](#14-edge-detection--sobel-canny-laplacian)
15. [Calculating and Plotting Histograms](#15-calculating-and-plotting-histograms)
16. [Histogram Equalization](#16-histogram-equalization)
17. [CLAHE — Contrast Limited Adaptive Histogram Equalization](#17-clahe--contrast-limited-adaptive-histogram-equalization)
18. [Contours](#18-contours)
19. [Image Segmentation](#19-image-segmentation)
20. [Haar Cascade Face Detection](#20-haar-cascade-face-detection)
21. [Tradeoffs, Failure Modes, and Production Considerations](#21-tradeoffs-failure-modes-and-production-considerations)

---

## 1. Foundations — How Images Are Represented

### Internal Representation

An image in OpenCV is a **NumPy ndarray**. This is non-negotiable to internalize — every OpenCV function returns or operates on ndarrays.

```
Grayscale image:  ndarray of shape (H, W)       — dtype uint8 (0–255)
Color image:      ndarray of shape (H, W, 3)    — dtype uint8, channels = BGR
BGRA image:       ndarray of shape (H, W, 4)    — with alpha channel
```

**Critical OpenCV-specific convention:** OpenCV stores color channels in **BGR** order, not RGB. This differs from matplotlib, PIL, PyTorch, and nearly everything else. Forgetting this causes color inversion bugs.

### Pixel Value Mechanics

```python
import cv2
import numpy as np

img = cv2.imread("image.jpg")

# Shape
print(img.shape)    # (480, 640, 3)  → H=480, W=640, C=3
print(img.dtype)    # uint8

# Access a single pixel at row=100, col=200
pixel = img[100, 200]          # array([B, G, R])
blue  = img[100, 200, 0]       # B channel
green = img[100, 200, 1]       # G channel
red   = img[100, 200, 2]       # R channel

# Slicing a region (ROI = Region of Interest)
roi = img[50:150, 100:300]     # rows 50-149, cols 100-299
```

### Memory Layout

Images are stored in row-major (C-contiguous) order:

```
img[0,0,0], img[0,0,1], img[0,0,2],  img[0,1,0], img[0,1,1], ...
```

This matters for performance. Operations along rows are cache-friendly. Passing non-contiguous arrays to OpenCV can cause errors — use `np.ascontiguousarray()` if needed.

### Data Type Considerations

| dtype | Range | Use Case |
|-------|-------|----------|
| `uint8` | 0–255 | Standard 8-bit images |
| `float32` | any | Filter outputs, neural net inputs |
| `float64` | any | High-precision intermediate computations |
| `int16` | -32768–32767 | Sobel, Laplacian (can go negative) |

Overflow wraps around with `uint8`. `255 + 1 = 0`. Use `cv2.add()` for saturating arithmetic instead of `+`.

```python
# Dangerous: numpy overflow
img1 + img2       # wraps around at 255

# Safe: saturates at 255
cv2.add(img1, img2)
```

---

## 2. Reading and Writing Images

### Mechanism

`cv2.imread()` decodes compressed image files (JPEG, PNG, BMP, TIFF, etc.) into uncompressed BGR ndarrays in memory. Under the hood it uses libjpeg/libpng/libtiff depending on format.

```python
import cv2

# Read full color (default)
img = cv2.imread("image.jpg")                        # BGR, uint8
img = cv2.imread("image.jpg", cv2.IMREAD_COLOR)      # explicit
img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)  # single channel
img = cv2.imread("image.jpg", cv2.IMREAD_UNCHANGED)  # keep alpha channel

# CRITICAL: imread returns None on failure, not an exception
if img is None:
    raise FileNotFoundError("Image not found or corrupt")
```

### Writing Images

```python
# Write to disk (format inferred from extension)
cv2.imwrite("output.jpg", img)
cv2.imwrite("output.png", img)

# JPEG quality control (0-100, higher = larger file, less compression)
cv2.imwrite("output.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])

# PNG compression (0-9, higher = smaller file, slower)
cv2.imwrite("output.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
```

### Display

```python
cv2.imshow("Window Title", img)
cv2.waitKey(0)        # 0 = wait indefinitely, n = wait n milliseconds
cv2.destroyAllWindows()
```

### BGR → RGB for Matplotlib

```python
import matplotlib.pyplot as plt

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.axis('off')
plt.show()
```

### Complete Pipeline

```python
import cv2
import matplotlib.pyplot as plt

def load_and_display(path: str) -> None:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    
    print(f"Shape: {img.shape}, dtype: {img.dtype}")
    print(f"Min: {img.min()}, Max: {img.max()}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)
    plt.title(f"Image: {path}")
    plt.axis('off')
    plt.show()
```

### Failure Modes

- **imread returns None**: path wrong, permissions, corrupt file, unsupported format
- **JPEG compression artifacts**: use PNG for lossless intermediate storage in pipelines
- **Large images**: a single 4K image is ~25MB uncompressed in RAM

---

## 3. Working with Video Files

### Mechanism

Video files are sequences of compressed frames. `VideoCapture` decodes frames one at a time from a file or camera stream. Each `cap.read()` call decodes one frame into a BGR ndarray.

```python
import cv2

# Open video file
cap = cv2.VideoCapture("video.mp4")

# Open webcam (device index 0 = first camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open video source")
```

### Reading Frames

```python
while True:
    ret, frame = cap.read()  # ret = success bool, frame = BGR ndarray
    
    if not ret:
        break  # End of video or read error
    
    # Process frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow("Frame", frame)
    
    # 'q' to quit; waitKey(1) = 1ms delay (required for display refresh)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Video Properties

```python
fps    = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"FPS: {fps}, Size: {width}x{height}, Total frames: {frames}")
```

### Writing Video

```python
fourcc = cv2.VideoWriter_fourcc(*'mp4v')   # codec
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    processed = your_processing_function(frame)
    out.write(processed)

cap.release()
out.release()
```

### Common Codecs

| Codec | fourcc | Container | Notes |
|-------|--------|-----------|-------|
| H.264 | `avc1` | .mp4 | Widely supported, needs extra lib |
| MPEG-4 | `mp4v` | .mp4 | Good compatibility |
| XVID | `XVID` | .avi | Open source |
| MJPEG | `MJPG` | .avi | High quality, large files |

### Seek to Specific Frame

```python
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)  # Jump to frame 100
ret, frame = cap.read()
```

### Production Considerations

- `waitKey(1)` is mandatory in display loops — without it the window won't render
- For real-time processing, check if processing time per frame < 1/FPS
- For batch processing (no display), skip `imshow` and `waitKey` — they add ~1ms overhead each
- Thread the capture and processing separately for high-FPS pipelines

---

## 4. Exploring Color Spaces

### What Is a Color Space?

A color space is a mathematical model mapping physical color perception to numerical coordinates. Different color spaces are better suited for different tasks.

### BGR (OpenCV default)

Three channels: Blue, Green, Red. Each 0–255. Additive model — (0,0,0) is black, (255,255,255) is white.

```python
b, g, r = cv2.split(img)
img_bgr = cv2.merge([b, g, r])
```

### Grayscale

Single channel luminance approximation. OpenCV uses the weighted formula:

```
Y = 0.114*B + 0.587*G + 0.299*R
```

These weights reflect human perception — eyes are most sensitive to green, least to blue.

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Shape: (H, W)
```

### HSV — Hue, Saturation, Value

**This is the most important color space for color-based object detection.**

- **Hue**: Color type (0–180 in OpenCV, representing 0°–360° mapped to half)
- **Saturation**: Color purity/intensity (0–255)
- **Value**: Brightness (0–255)

Why HSV? It separates *color information* (hue) from *lighting* (value). This makes color detection robust to shadows and lighting changes — you threshold on hue, not RGB.

```python
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

# OpenCV HSV ranges:
# Hue:        0–180 (red wraps: 0–10 and 170–180)
# Saturation: 0–255
# Value:      0–255
```

**HSV color reference:**

| Color | H range (OpenCV) |
|-------|-----------------|
| Red | 0–10, 170–180 |
| Orange | 10–25 |
| Yellow | 25–35 |
| Green | 35–85 |
| Cyan | 85–100 |
| Blue | 100–130 |
| Violet/Purple | 130–160 |
| Pink | 160–170 |

### LAB — Perceptually Uniform

- **L**: Lightness (0–100 mapped to 0–255)
- **A**: Green–Red axis
- **B**: Blue–Yellow axis

LAB is perceptually uniform — equal numerical distances correspond to equal perceived color differences. Used for color comparison, color transfer, and skin detection.

```python
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
```

### YCrCb — Used in JPEG, Video Compression

- **Y**: Luma (brightness)
- **Cr**: Red-difference chrominance
- **Cb**: Blue-difference chrominance

Human vision is more sensitive to luma than chroma, so video compression downsamples Cr/Cb channels (chroma subsampling).

```python
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
```

### Conversion Reference

```python
# All conversions through cv2.cvtColor
# Common codes:
cv2.COLOR_BGR2GRAY
cv2.COLOR_BGR2HSV
cv2.COLOR_BGR2LAB
cv2.COLOR_BGR2YCrCb
cv2.COLOR_BGR2RGB
cv2.COLOR_HSV2BGR
cv2.COLOR_GRAY2BGR   # Expand single channel to 3-channel
```

### Visualizing All Channels

```python
import cv2
import matplotlib.pyplot as plt

def visualize_channels(img_bgr: np.ndarray, colorspace: str = 'hsv') -> None:
    if colorspace == 'hsv':
        converted = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        names = ['Hue', 'Saturation', 'Value']
    elif colorspace == 'lab':
        converted = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        names = ['L (Lightness)', 'A (Green-Red)', 'B (Blue-Yellow)']
    
    channels = cv2.split(converted)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, ch, name in zip(axes, channels, names):
        ax.imshow(ch, cmap='gray')
        ax.set_title(name)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
```

---

## 5. Color Thresholding

### Mechanism

Thresholding converts a continuous-valued image to a binary mask based on pixel value ranges. Color thresholding in HSV isolates objects by color.

### Binary Thresholding (Grayscale)

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# thresh_val: pixels > thresh_val → max_val, else → 0
ret, binary = cv2.threshold(gray, thresh_val=127, maxval=255, type=cv2.THRESH_BINARY)

# Inverse
ret, binary_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# Otsu's method: automatically finds optimal threshold
ret, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"Otsu threshold: {ret}")
```

### Otsu's Algorithm

Otsu's method finds the threshold that minimizes intra-class variance (equivalently, maximizes inter-class variance) between foreground and background. It works best on **bimodal histograms**. Computed in O(n) where n = number of gray levels.

```
σ²_between(t) = w₀(t)·w₁(t)·[μ₀(t) - μ₁(t)]²
Maximize over t ∈ [0, 255]
```

### Adaptive Thresholding

For non-uniform illumination (shadows, uneven lighting), a global threshold fails. Adaptive thresholding computes a local threshold for each pixel based on its neighborhood.

```python
# Threshold = mean of block_size×block_size neighborhood minus C
adaptive_mean = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    blockSize=11,   # must be odd
    C=2             # constant subtracted from mean
)

# Gaussian-weighted neighborhood (usually better)
adaptive_gauss = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    blockSize=11,
    C=2
)
```

### HSV Color Masking (Most Useful)

```python
def color_mask(img_bgr: np.ndarray, 
               lower_hsv: tuple, 
               upper_hsv: tuple) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))
    result = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    return result, mask

# Detect green objects
lower_green = (35, 50, 50)
upper_green = (85, 255, 255)
result, mask = color_mask(img, lower_green, upper_green)

# Detect red (wraps around hue axis — need two ranges)
mask1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
mask2 = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
red_mask = cv2.bitwise_or(mask1, mask2)
```

### Morphological Post-Processing

Raw masks are noisy. Standard cleanup:

```python
kernel = np.ones((5, 5), np.uint8)

# Remove small noise (erosion then dilation)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Fill small holes (dilation then erosion)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
```

### Threshold Type Reference

| Type | Description |
|------|-------------|
| `THRESH_BINARY` | pixel > t → max, else 0 |
| `THRESH_BINARY_INV` | pixel > t → 0, else max |
| `THRESH_TRUNC` | pixel > t → t, else unchanged |
| `THRESH_TOZERO` | pixel > t → unchanged, else 0 |
| `THRESH_TOZERO_INV` | pixel > t → 0, else unchanged |

---

## 6. Image Resizing, Scaling, and Interpolation

### Why Interpolation Matters

When you resize an image, you're mapping a source grid to a destination grid of different size. Pixels don't align perfectly — interpolation determines how to compute values for in-between positions.

```
Source pixel at (x_src, y_src) → maps to non-integer position (x_dst, y_dst) in dest
Interpolation: how to estimate the color at that fractional position
```

### Resize

```python
# By explicit dimensions
resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

# By scale factor
resized = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
# fx, fy: scale factors for width and height
```

### Interpolation Methods

#### INTER_NEAREST — Nearest Neighbor

Fastest. Takes the value of the closest source pixel. No smoothing.

```
f(x, y) = f(round(x), round(y))
```

- **Use when**: pixel art, label maps, semantic segmentation masks (you don't want to interpolate class IDs)
- **Cost**: O(1) per pixel
- **Artifacts**: Blocky/jagged edges

#### INTER_LINEAR — Bilinear

Weighted average of the 4 surrounding pixels. Default for most operations.

```
f(x, y) = (1-a)(1-b)·f(x₀,y₀) + a(1-b)·f(x₁,y₀) + (1-a)b·f(x₀,y₁) + ab·f(x₁,y₁)
where a = x - x₀,  b = y - y₀
```

- **Use when**: General upscaling
- **Cost**: O(4) per pixel

#### INTER_CUBIC — Bicubic

Uses 4×4 = 16 surrounding pixels. Fits a cubic polynomial. Sharper than bilinear.

```
Piecewise cubic kernel:  W(x) = (a+2)|x|³ - (a+3)|x|² + 1  for |x| ≤ 1
```

- **Use when**: High-quality upscaling
- **Cost**: O(16) per pixel — ~4× slower than bilinear

#### INTER_AREA — Area-Based

Averages the source pixels that map to each destination pixel. Best for **downscaling**.

- **Use when**: Shrinking images — prevents moiré artifacts
- **Equivalent to**: Proper decimation with antialiasing

#### INTER_LANCZOS4 — Lanczos

8×8 neighborhood, sinc-based kernel. Highest quality, highest cost.

- **Use when**: Maximum quality upscaling
- **Cost**: O(64) per pixel

### Choosing Interpolation

```
Downscaling:  INTER_AREA (antialiasing, no moiré)
Upscaling:    INTER_LINEAR (fast) or INTER_CUBIC (quality)
Masks/labels: INTER_NEAREST (no value mixing)
Max quality:  INTER_LANCZOS4
```

```python
def resize_maintain_aspect(img: np.ndarray, 
                            target_width: int = None, 
                            target_height: int = None) -> np.ndarray:
    h, w = img.shape[:2]
    
    if target_width and not target_height:
        scale = target_width / w
        new_dim = (target_width, int(h * scale))
    elif target_height and not target_width:
        scale = target_height / h
        new_dim = (int(w * scale), target_height)
    else:
        new_dim = (target_width, target_height)
    
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, new_dim, interpolation=interp)
```

---

## 7. Flip, Rotate, and Crop

### Flip

Flipping is a free operation — it's just array indexing.

```python
# flipCode: 0=vertical, 1=horizontal, -1=both
flipped_h = cv2.flip(img, 1)   # left-right
flipped_v = cv2.flip(img, 0)   # up-down
flipped_b = cv2.flip(img, -1)  # both

# Equivalent numpy (for understanding internals)
flipped_h = img[:, ::-1]       # reverse columns
flipped_v = img[::-1, :]       # reverse rows
```

### Rotate (90° increments)

```python
# 90° counterclockwise
rot_90  = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
# 90° clockwise
rot_270 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
# 180°
rot_180 = cv2.rotate(img, cv2.ROTATE_180)
```

### Rotate (Arbitrary Angle) — Rotation Matrix

For arbitrary angles, OpenCV uses a 2×3 affine rotation matrix:

```
M = [ cos(θ)  -sin(θ)  tx ]
    [ sin(θ)   cos(θ)  ty ]
```

where tx, ty are translation components to rotate around a chosen center.

```python
def rotate_image(img: np.ndarray, angle: float, scale: float = 1.0) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, scale)
    
    # Compute new bounding box dimensions to avoid clipping
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    
    # Adjust translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    return cv2.warpAffine(img, M, (new_w, new_h))
```

### Crop (Region of Interest)

Cropping is pure numpy slicing — no OpenCV function needed.

```python
# img[y_start:y_end, x_start:x_end]
roi = img[100:300, 150:400]   # rows 100-299, cols 150-399

# This is a VIEW (no copy) — modifying roi modifies img
# For independent copy:
roi = img[100:300, 150:400].copy()
```

### Practical: Crop with Padding

```python
def crop_with_padding(img: np.ndarray, 
                      x: int, y: int, w: int, h: int, 
                      pad: int = 0) -> np.ndarray:
    """Crop region with optional padding, clamped to image boundaries."""
    H, W = img.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(W, x + w + pad)
    y2 = min(H, y + h + pad)
    return img[y1:y2, x1:x2]
```

---

## 8. OpenCV Coordinate System

### Critical: Two Different Systems in One Library

OpenCV is inconsistent across its API:

```
Array indexing:   img[row, col]   → img[y, x]
OpenCV functions: (x, y)          → (col, row)
```

This is a constant source of bugs.

```python
# Array indexing: [row, col] = [y, x]
pixel = img[100, 200]   # row=100 (y=100), col=200 (x=200)

# cv2.circle: center=(x, y) = (col, row)
cv2.circle(img, center=(200, 100), radius=10, color=(0,255,0), thickness=2)
#                        (x=200, y=100)
```

### Coordinate Origin

```
(0,0) is TOP-LEFT
X increases rightward  →
Y increases downward   ↓
```

This is opposite to standard Cartesian math (where Y increases upward).

```
(0,0)────────────────► X (col)
  │
  │
  │
  ▼
  Y (row)
```

### Bounding Box Conventions

OpenCV bounding boxes: `(x, y, w, h)` where (x,y) is **top-left corner**.

```python
# From cv2.boundingRect()
x, y, w, h = cv2.boundingRect(contour)
# Crop using bounding rect
roi = img[y:y+h, x:x+w]
```

---

## 9. Drawing Lines and Shapes

All OpenCV drawing functions modify the image **in-place**. Always pass a copy if you want to preserve the original.

### Core Drawing Functions

```python
import cv2
import numpy as np

canvas = np.zeros((500, 700, 3), dtype=np.uint8)  # Black canvas

# Line
# pt1=(x1,y1), pt2=(x2,y2), color=BGR, thickness (px), lineType
cv2.line(canvas, (50, 50), (650, 50), color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

# Rectangle: top-left to bottom-right
cv2.rectangle(canvas, (100, 100), (300, 200), color=(255, 0, 0), thickness=2)
cv2.rectangle(canvas, (350, 100), (550, 200), color=(0, 0, 255), thickness=-1)  # -1 = filled

# Circle
cv2.circle(canvas, center=(350, 350), radius=80, color=(0, 255, 255), thickness=3)
cv2.circle(canvas, center=(100, 350), radius=50, color=(255, 0, 255), thickness=-1)  # filled

# Ellipse
# center, axes=(major, minor), angle=rotation, startAngle, endAngle
cv2.ellipse(canvas, (500, 350), (100, 50), angle=30, startAngle=0, endAngle=360,
            color=(255, 255, 0), thickness=2)

# Polylines (open or closed polygon)
pts = np.array([[200, 400], [250, 350], [300, 400], [280, 450], [220, 450]], np.int32)
pts = pts.reshape((-1, 1, 2))  # Required shape for cv2.polylines
cv2.polylines(canvas, [pts], isClosed=True, color=(255, 255, 255), thickness=2)

# Filled polygon
cv2.fillPoly(canvas, [pts], color=(100, 150, 200))
```

### Line Types

| Type | Description |
|------|-------------|
| `cv2.LINE_4` | 4-connected (faster, jagged) |
| `cv2.LINE_8` | 8-connected (default) |
| `cv2.LINE_AA` | Anti-aliased (smooth, use for display) |

### Arrow

```python
cv2.arrowedLine(canvas, (50, 300), (200, 300), color=(0,165,255), thickness=2, tipLength=0.3)
```

---

## 10. Adding Text to Images

### Basic Text

```python
cv2.putText(
    img=canvas,
    text="OpenCV Notes",
    org=(50, 50),              # bottom-left corner of text (x, y)
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    fontScale=1.0,
    color=(255, 255, 255),
    thickness=2,
    lineType=cv2.LINE_AA       # always use for smooth text
)
```

### Font Reference

| Font | Code |
|------|------|
| Simplex | `cv2.FONT_HERSHEY_SIMPLEX` |
| Plain | `cv2.FONT_HERSHEY_PLAIN` |
| Duplex | `cv2.FONT_HERSHEY_DUPLEX` |
| Complex | `cv2.FONT_HERSHEY_COMPLEX` |
| Triplex | `cv2.FONT_HERSHEY_TRIPLEX` |
| Script | `cv2.FONT_HERSHEY_SCRIPT_SIMPLEX` |
| Italic modifier | `+ cv2.FONT_ITALIC` |

### Measuring Text (Critical for Centering)

```python
def draw_centered_text(img: np.ndarray, text: str, 
                        font=cv2.FONT_HERSHEY_SIMPLEX, 
                        scale: float = 1.0, 
                        color: tuple = (255,255,255),
                        thickness: int = 2) -> np.ndarray:
    img = img.copy()
    
    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    
    h, w = img.shape[:2]
    x = (w - text_w) // 2
    y = (h + text_h) // 2
    
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    return img
```

### Text with Background Box

```python
def put_text_with_background(img: np.ndarray, text: str, 
                              org: tuple, font_scale: float = 0.6,
                              text_color: tuple = (255,255,255),
                              bg_color: tuple = (0,0,0),
                              padding: int = 5) -> np.ndarray:
    img = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    
    # Draw filled rectangle background
    cv2.rectangle(img, 
                  (x - padding, y - h - padding),
                  (x + w + padding, y + baseline + padding),
                  bg_color, -1)
    
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    return img
```

---

## 11. Affine and Perspective Transformations

### Linear Algebra Foundation

Both transformations apply matrix multiplication to map source pixel coordinates to destination coordinates.

### Affine Transformation

An affine transform is any transformation that preserves **parallel lines**. It encodes: translation, rotation, scaling, shearing.

**Mathematical form:**
```
[x']   [a  b  tx] [x]
[y'] = [c  d  ty] [y]
[1 ]   [0  0   1] [1]

Or compactly: dst = M · src
M is a 2×3 matrix
```

**What it preserves:**
- Parallelism of lines
- Ratios of distances along lines
- Collinearity

**Degrees of freedom:** 6 (2×3 matrix) → requires **3 point correspondences** to solve.

```python
# Affine from 3 point pairs
src_pts = np.float32([[50, 50], [200, 50], [50, 200]])
dst_pts = np.float32([[10, 100], [200, 50], [100, 250]])

M = cv2.getAffineTransform(src_pts, dst_pts)
# M shape: (2, 3)

h, w = img.shape[:2]
warped = cv2.warpAffine(img, M, (w, h))
```

### Perspective Transformation (Homography)

Perspective transform models a **pinhole camera projection** — parallel lines can converge. Used for: document scanning, bird's-eye view, removing perspective distortion.

**Mathematical form (homogeneous coordinates):**
```
[x']     [h00 h01 h02] [x]
[y']  ~  [h10 h11 h12] [y]
[w']     [h20 h21 h22] [1]

Actual coordinates: (x'/w', y'/w')
H is a 3×3 matrix (8 DOF, since scale is free)
```

Requires **4 point correspondences** to solve (8 equations, 8 unknowns).

```python
# Classic: warp a document to top-down view
# src_pts: detected corners in original image (TL, TR, BR, BL)
src_pts = np.float32([[73, 239], [356, 117], [475, 265], [187, 443]])

# dst_pts: where those corners should map to (a rectangle)
dst_pts = np.float32([[0, 0], [400, 0], [400, 600], [0, 600]])

H = cv2.getPerspectiveTransform(src_pts, dst_pts)
# H shape: (3, 3)

warped = cv2.warpPerspective(img, H, (400, 600))
```

### Practical: Document Scanner

```python
def four_point_transform(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Warp a quadrilateral region to a top-down rectangle.
    pts: (4, 2) array of corners [TL, TR, BR, BL]
    """
    (tl, tr, br, bl) = pts
    
    # Compute output dimensions
    width_top    = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left  = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    
    max_width  = int(max(width_top, width_bottom))
    max_height = int(max(height_left, height_right))
    
    src = np.float32([tl, tr, br, bl])
    dst = np.float32([[0, 0], [max_width-1, 0], 
                      [max_width-1, max_height-1], [0, max_height-1]])
    
    H = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, H, (max_width, max_height))
```

### Translation Matrix

```python
# Shift image by (tx, ty)
tx, ty = 100, 50
M_translate = np.float32([[1, 0, tx],
                           [0, 1, ty]])
shifted = cv2.warpAffine(img, M_translate, (w, h))
```

### Border Mode

```python
# What to fill when pixels go out of bounds
cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
```

---

## 12. Image Filters

### Convolution — The Fundamental Operation

Image filtering is based on **discrete 2D convolution**:

```
(f * g)[x,y] = Σᵢ Σⱼ f[x-i, y-j] · g[i, j]

f: input image
g: kernel (filter)
Output: new pixel value at (x,y)
```

Intuitively: the kernel slides over the image, and at each position, computes a weighted sum of the neighborhood. The kernel determines what the filter does.

```python
import cv2
import numpy as np

# Apply a custom kernel
kernel = np.array([[ 0, -1,  0],
                   [-1,  5, -1],
                   [ 0, -1,  0]], dtype=np.float32)   # Sharpening kernel

sharpened = cv2.filter2D(img, ddepth=-1, kernel=kernel)
# ddepth=-1: output same depth as input
```

### Common Kernels and What They Do

**Identity (no-op):**
```
[0 0 0]
[0 1 0]
[0 0 0]
```

**Sharpen:**
```
[ 0 -1  0]
[-1  5 -1]
[ 0 -1  0]
```
Amplifies high-frequency (edges) relative to low-frequency (smooth areas).

**Box blur (3×3, normalized):**
```
[1/9 1/9 1/9]
[1/9 1/9 1/9]
[1/9 1/9 1/9]
```

**Emboss:**
```
[-2 -1  0]
[-1  1  1]
[ 0  1  2]
```

### Practical Custom Filters

```python
# Sharpen
sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
sharpened = cv2.filter2D(img, -1, sharpen_kernel)

# Emboss
emboss_kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], np.float32)
embossed = cv2.filter2D(img, -1, emboss_kernel)

# Edge detection (simple Laplacian approximation)
laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)
edges = cv2.filter2D(img, cv2.CV_64F, laplacian_kernel)

# Unsharp masking: sharpen = original + alpha*(original - blurred)
blurred = cv2.GaussianBlur(img, (5,5), 1.0)
sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
```

### Separable Kernels (Performance)

Many kernels are **separable** — they can be written as the outer product of two 1D vectors:

```
Gaussian(5×5) = gaussian_1D(5) × gaussian_1D(5)ᵀ
```

This reduces convolution from O(k²·H·W) to O(2k·H·W) — significant speedup for large kernels. OpenCV uses separable kernels internally for Gaussian and box filters.

---

## 13. Blur Filters — Average, Gaussian, Median

Blurring suppresses high-frequency noise. Different blur methods make different assumptions about noise.

### Average (Box) Blur

Replaces each pixel with the unweighted average of its k×k neighborhood.

```
Output[x,y] = (1/k²) · Σ Input[x+i, y+j]  for i,j ∈ [-k/2, k/2]
```

```python
blurred = cv2.blur(img, ksize=(5, 5))

# Equivalent using filter2D:
kernel = np.ones((5,5), np.float32) / 25
blurred = cv2.filter2D(img, -1, kernel)
```

**Properties:**
- Fast (uses integral images internally → O(1) per pixel regardless of kernel size)
- Blurs edges badly
- Not noise-model-aware
- Use case: quick preview, where quality doesn't matter

### Gaussian Blur

Weights neighborhood pixels by a Gaussian distribution — closer pixels contribute more.

```
G(x,y) = (1/2πσ²) · exp(-(x²+y²)/(2σ²))
```

```python
# ksize must be odd, positive
# sigmaX: Gaussian std in X direction (sigmaY=sigmaX if not specified)
blurred = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=1.0)
blurred = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=2.0)  # kernel auto-computed from sigma
```

**Sigma vs. Kernel Size relationship:**
```
ksize ≈ 6σ + 1  (captures 3σ on each side)
σ=1 → ksize=7, σ=2 → ksize=13
```

**Properties:**
- Smooths noise while better preserving edges than box blur
- Computationally efficient (separable kernel)
- Blurs edges — not edge-preserving
- **Use case**: Preprocessing before edge detection (Canny uses it internally), noise reduction

**Frequency domain interpretation:** Gaussian in spatial domain = Gaussian in frequency domain. It's an ideal low-pass filter with no ringing artifacts (Gibbs phenomenon).

### Median Blur

Replaces each pixel with the **median** value of its k×k neighborhood.

```python
blurred = cv2.medianBlur(img, ksize=5)   # ksize must be odd
```

**Properties:**
- **Edge-preserving**: edges are sharp features — the median preserves them
- **Excellent at removing salt-and-pepper noise** (random black/white pixels)
- Nonlinear — not a convolution
- Computationally heavier: O(k² log k) per pixel vs O(k²) for linear filters
- **Use case**: Document images, removing scanner noise, preprocessing for OCR

### Bilateral Filter — Edge-Preserving Smooth

The most sophisticated built-in blur. Blurs based on both spatial proximity AND color similarity.

```
BF[p] = (1/Wp) · Σq G_σs(||p-q||) · G_σr(|I_p - I_q|) · I_q

G_σs: spatial Gaussian (how close are the pixels spatially)
G_σr: range Gaussian (how similar are the pixel values)
```

```python
# d: diameter of pixel neighborhood
# sigmaColor: range sigma (larger = more colors blurred together)
# sigmaSpace: spatial sigma (larger = more distant pixels influence)
bilateral = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
```

**Properties:**
- Smooths flat regions while preserving edges
- Very expensive: O(d² · H · W)
- **Use case**: Photo enhancement, skin smoothing, HDR tone mapping

### Comparison Summary

| Method | Edge Preservation | Noise Type | Cost |
|--------|-----------------|------------|------|
| Box | Poor | Any | O(1)/px |
| Gaussian | Moderate | Gaussian | O(k)/px |
| Median | Good | Salt & pepper | O(k² log k)/px |
| Bilateral | Excellent | Any | O(k²)/px |

---

## 14. Edge Detection — Sobel, Canny, Laplacian

### What Is an Edge?

An edge is a location of **rapid intensity change** in an image. Mathematically, it corresponds to a large gradient magnitude.

```
Gradient: ∇I = [∂I/∂x, ∂I/∂y]
Magnitude: |∇I| = sqrt((∂I/∂x)² + (∂I/∂y)²)
Direction: θ = arctan(∂I/∂y / ∂I/∂x)
```

### Sobel Operator

Approximates image derivatives using 3×3 kernels:

```
Sobel-X (horizontal gradient):    Sobel-Y (vertical gradient):
[-1  0  1]                        [-1 -2 -1]
[-2  0  2]                        [ 0  0  0]
[-1  0  1]                        [ 1  2  1]
```

The center row/column has weight 2 for smoothing — reduces noise sensitivity.

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Compute x and y gradients
# ddepth=cv2.CV_64F: use float to preserve negative values
sobelx = cv2.Sobel(gray, cv2.CV_64F, dx=1, dy=0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, dx=0, dy=1, ksize=3)

# Gradient magnitude
magnitude = np.sqrt(sobelx**2 + sobely**2)
magnitude = np.uint8(np.clip(magnitude, 0, 255))

# Direction (in radians)
direction = np.arctan2(sobely, sobelx)

# Combined visualization (absolute values)
sobelx_abs = cv2.convertScaleAbs(sobelx)
sobely_abs = cv2.convertScaleAbs(sobely)
sobel_combined = cv2.addWeighted(sobelx_abs, 0.5, sobely_abs, 0.5, 0)
```

**Why CV_64F?** Derivatives can be negative (dark-to-light vs light-to-dark transitions). `uint8` would clamp negatives to 0, losing half the edges.

### Laplacian Operator

Second-order derivative — detects edges as **zero-crossings**.

```
∇²I = ∂²I/∂x² + ∂²I/∂y²

Approximation kernel:
[0  1  0]
[1 -4  1]
[0  1  0]
```

```python
# Apply Gaussian first (Laplacian is noise-sensitive)
blurred = cv2.GaussianBlur(gray, (3,3), 0)
laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
laplacian_abs = cv2.convertScaleAbs(laplacian)
```

**Laplacian of Gaussian (LoG):**
Gaussian smoothing + Laplacian edge detection combined. Better noise robustness. OpenCV doesn't have a direct LoG function, but:

```python
# Approximate LoG: Difference of Gaussians
g1 = cv2.GaussianBlur(gray, (5,5), 1.0)
g2 = cv2.GaussianBlur(gray, (9,9), 2.0)
dog = cv2.subtract(g1, g2)   # Difference of Gaussians ≈ LoG
```

### Canny Edge Detector

The gold standard for edge detection. Multi-step algorithm:

**Step 1: Gaussian smoothing** — reduces noise before differentiation  
**Step 2: Sobel gradients** — compute magnitude and direction at every pixel  
**Step 3: Non-maximum suppression** — thin edges to 1 pixel wide by suppressing pixels that aren't local maxima along the gradient direction  
**Step 4: Double thresholding** — classify pixels as strong edges, weak edges, or non-edges  
**Step 5: Edge tracking by hysteresis** — keep weak edges connected to strong edges

```python
# threshold1: lower threshold (weak edges below this are rejected)
# threshold2: upper threshold (edges above this are always strong)
# Typical ratio: 1:2 or 1:3
edges = cv2.Canny(gray, threshold1=50, threshold2=150)

# With pre-smoothing (usually not needed, Canny does it internally)
blurred = cv2.GaussianBlur(gray, (5,5), 0)
edges = cv2.Canny(blurred, 30, 100)

# With L2 gradient (more accurate, slower)
edges = cv2.Canny(gray, 50, 150, L2gradient=True)
```

**Threshold selection heuristic:**
- Low: ~1/3 of high threshold
- High: can use `np.median` + sigma rule:

```python
def auto_canny(img_gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    v = np.median(img_gray)
    low  = int(max(0,   (1.0 - sigma) * v))
    high = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(img_gray, low, high)
```

### Comparison

| Method | Derivative Order | Output | Noise Sensitivity | Use Case |
|--------|----------------|--------|------------------|----------|
| Sobel | 1st | Gradient map (thick) | Moderate | Gradient-based features |
| Laplacian | 2nd | Zero-crossings | High | Requires pre-smoothing |
| Canny | 1st + NMS + hysteresis | Thin binary edges | Low | General edge detection |

---

## 15. Calculating and Plotting Histograms

### What Is an Image Histogram?

A histogram counts the frequency of each intensity value in an image. For an 8-bit image: 256 bins, each bin counts pixels with that value.

```
hist[v] = count(pixels with intensity == v)
```

Histograms reveal:
- Brightness distribution (dark/bright image)
- Contrast range
- Color dominance
- Bimodal distributions (foreground/background separation)

### Computing Histograms

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Grayscale histogram
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hist = cv2.calcHist(
    images=[gray],      # list of images
    channels=[0],       # channel index
    mask=None,          # optional mask (compute hist only for masked region)
    histSize=[256],     # number of bins
    ranges=[0, 256]     # value range
)
# hist shape: (256, 1)

# BGR color histograms
colors = ('b', 'g', 'r')
for i, col in enumerate(colors):
    hist_ch = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist_ch, color=col)

# Numpy alternative
hist_np, bins = np.histogram(gray.ravel(), 256, [0, 256])
```

### 2D Histogram (H-S from HSV)

Useful for color-based object detection and comparison.

```python
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hist_2d = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
# hist_2d shape: (180, 256) — Hue × Saturation
```

### Plotting

```python
def plot_histogram(img: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # Image
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Image")
    axes[0].axis('off')
    
    # Histogram
    axes[1].set_title("Color Histogram")
    axes[1].set_xlim([0, 256])
    for i, (col, name) in enumerate(zip(['blue','green','red'], ['B','G','R'])):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        axes[1].plot(hist, color=col, label=name)
    axes[1].legend()
    axes[1].set_xlabel("Pixel Intensity")
    axes[1].set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()
```

### Histogram Comparison

Compare two images by their histograms — useful for image retrieval and detection:

```python
hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])

# Correlation: 1 = identical, -1 = inverse
score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

# Chi-square: 0 = identical
score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)

# Intersection: higher = more similar
score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)

# Bhattacharyya: 0 = identical, 1 = completely different
score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
```

---

## 16. Histogram Equalization

### Problem: Low-Contrast Images

An image with poor contrast has a histogram concentrated in a narrow intensity range. Equalization spreads it across the full 0–255 range.

### Mathematical Mechanism

The equalization transform is the **Cumulative Distribution Function (CDF)** of pixel intensities, scaled to [0, 255]:

```
CDF(v) = (1/N) · Σᵥ'≤ᵥ hist[v']    (normalized CDF)

Equalized value: T(v) = round(CDF(v) · 255)
```

This guarantees a **uniform distribution** of output pixel values — each intensity level appears roughly equally often.

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
equalized = cv2.equalizeHist(gray)

# Visualize before/after
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0,0].imshow(gray, cmap='gray');      axes[0,0].set_title("Original")
axes[0,1].imshow(equalized, cmap='gray'); axes[0,1].set_title("Equalized")
axes[1,0].hist(gray.ravel(), 256, [0,256]);      axes[1,0].set_title("Hist: Original")
axes[1,1].hist(equalized.ravel(), 256, [0,256]); axes[1,1].set_title("Hist: Equalized")
plt.tight_layout(); plt.show()
```

### Color Image Equalization

**Wrong:** Equalize each BGR channel independently — changes colors.  
**Right:** Convert to HSV or YCrCb, equalize only the luminance channel:

```python
# Method 1: HSV
def equalize_color_hsv(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_eq = cv2.equalizeHist(v)
    hsv_eq = cv2.merge([h, s, v_eq])
    return cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)

# Method 2: YCrCb (more photorealistic)
def equalize_color_ycrcb(img_bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    ycrcb_eq = cv2.merge([y_eq, cr, cb])
    return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
```

### Limitations

- Global operation — uses one histogram for the whole image
- Fails on images with non-uniform illumination (dark corners, bright center)
- Can over-enhance noise in flat regions
- Solution: **CLAHE** (next section)

---

## 17. CLAHE — Contrast Limited Adaptive Histogram Equalization

### Problem with Global Equalization

Global equalization computes one CDF for the entire image and applies it uniformly. If illumination varies (e.g., an image with a bright region and a dark shadow), this produces over-amplified noise in already-bright areas and under-enhances dark areas.

### CLAHE Algorithm

1. **Divide image into tiles** (default 8×8 grid)
2. **Compute histogram per tile** separately
3. **Clip histogram** at a threshold (`clipLimit`) — redistributes excess counts uniformly. This prevents noise amplification.
4. **Compute equalization per tile** from clipped histogram
5. **Bilinear interpolation between tile equalization maps** — avoids tile boundary artifacts

```
clipLimit controls the maximum slope of the CDF per tile
→ Higher clipLimit = more contrast enhancement
→ Lower clipLimit = closer to no-op
```

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create CLAHE object
clahe = cv2.createCLAHE(
    clipLimit=2.0,          # contrast limit (typically 2.0-4.0)
    tileGridSize=(8, 8)     # grid of tiles
)

clahe_result = clahe.apply(gray)

# Color image: apply to luminance only
def apply_clahe_color(img_bgr: np.ndarray, 
                       clip_limit: float = 2.0, 
                       tile_size: tuple = (8, 8)) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    l_clahe = clahe.apply(l)
    
    lab_clahe = cv2.merge([l_clahe, a, b])
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
```

**Use LAB (not HSV or YCrCb) for color CLAHE** — LAB's L channel is the most perceptually accurate representation of lightness.

### CLAHE vs. Equalization Comparison

| Aspect | Global Equalization | CLAHE |
|--------|--------------------|-|
| Scope | Entire image | Per tile |
| Noise amplification | High | Controlled via clipLimit |
| Uniform illumination | Works well | Overkill |
| Non-uniform illumination | Fails | Designed for this |
| Speed | Fast | Slower |
| Typical use | Low-contrast global images | Medical imaging, night photos |

### Applications

- **Medical imaging**: X-rays, CT scans, MRIs — non-uniform tissue contrast
- **Night photography**: varying illumination
- **Satellite imagery**: shadows and bright areas in same scene
- **Fingerprint enhancement**: subtle ridge patterns

---

## 18. Contours

### What Are Contours?

A contour is a list of points tracing the boundary of a connected region of same-intensity pixels. In practice, you compute contours on binary images (after thresholding or edge detection).

### Finding Contours

```python
# CRITICAL: Find contours on binary image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# mode: how contour hierarchy is reported
# method: how many points to use for each contour
contours, hierarchy = cv2.findContours(
    binary, 
    mode=cv2.RETR_EXTERNAL,      # retrieval mode
    method=cv2.CHAIN_APPROX_SIMPLE  # compression method
)

print(f"Number of contours: {len(contours)}")
# Each contour: ndarray of shape (N, 1, 2) — N points, each (x, y)
```

### Retrieval Modes

| Mode | Description |
|------|-------------|
| `RETR_EXTERNAL` | Only outermost contours (no holes) |
| `RETR_LIST` | All contours, no hierarchy |
| `RETR_CCOMP` | 2-level hierarchy (objects and holes) |
| `RETR_TREE` | Full hierarchy tree |

### Approximation Methods

| Method | Description | Points |
|--------|-------------|--------|
| `CHAIN_APPROX_NONE` | All boundary points | Many (expensive) |
| `CHAIN_APPROX_SIMPLE` | Compress horizontal/vertical/diagonal runs | Fewer |
| `CHAIN_APPROX_TC89_L1` | Teh-Chin chain approximation | Even fewer |

### Drawing Contours

```python
# Draw all contours on a copy
canvas = img.copy()
cv2.drawContours(canvas, contours, 
                 contourIdx=-1,           # -1 = all, or specific index
                 color=(0, 255, 0), 
                 thickness=2)

# Draw single contour
cv2.drawContours(canvas, contours, 
                 contourIdx=0,            # first contour
                 color=(0, 0, 255), 
                 thickness=-1)            # filled
```

### Contour Properties

```python
for cnt in contours:
    # Area (in pixels²)
    area = cv2.contourArea(cnt)
    
    # Perimeter / arc length
    perimeter = cv2.arcLength(cnt, closed=True)
    
    # Bounding rectangle (axis-aligned)
    x, y, w, h = cv2.boundingRect(cnt)
    
    # Minimum enclosing circle
    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
    
    # Minimum area rotated rectangle
    rect = cv2.minAreaRect(cnt)          # (center, (w,h), angle)
    box  = cv2.boxPoints(rect)           # 4 corners
    box  = np.int0(box)
    cv2.drawContours(canvas, [box], 0, (255,0,0), 2)
    
    # Moments (centroid, area)
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
```

### Contour Approximation

Reduce number of points while preserving shape:

```python
# Douglas-Peucker algorithm
# epsilon: maximum distance from contour to approximated curve
epsilon = 0.02 * cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, closed=True)

# approx has fewer points than cnt
# For shapes:
if len(approx) == 3: shape = "Triangle"
elif len(approx) == 4: shape = "Rectangle/Square"
elif len(approx) == 5: shape = "Pentagon"
else: shape = "Circle/Ellipse"
```

### Convex Hull

```python
hull = cv2.convexHull(cnt)

# Convexity defects (deviation from convex hull)
hull_indices = cv2.convexHull(cnt, returnPoints=False)
defects = cv2.convexityDefects(cnt, hull_indices)
# defects[i, 0] = [start_idx, end_idx, farthest_idx, distance]
```

### Shape Descriptors

```python
# Circularity: 1.0 = perfect circle, < 1 = more irregular
circularity = 4 * np.pi * area / (perimeter ** 2)

# Aspect ratio
aspect_ratio = w / h

# Extent: ratio of contour area to bounding box area
extent = area / (w * h)

# Solidity: ratio of contour area to convex hull area
hull_area = cv2.contourArea(cv2.convexHull(cnt))
solidity = area / hull_area

# Hu Moments: shape-invariant descriptors (translation, scale, rotation invariant)
hu_moments = cv2.HuMoments(M).flatten()
```

### Filtering Contours

```python
# Filter small noise contours by area
min_area = 500
filtered = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

# Filter by aspect ratio
def is_square(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    ar = w / h
    return 0.9 < ar < 1.1

squares = [cnt for cnt in contours if is_square(cnt)]
```

---

## 19. Image Segmentation Using OpenCV

### Segmentation Methods in OpenCV

Segmentation partitions an image into meaningful regions. OpenCV provides several approaches, ranging from simple thresholding to GrabCut and watershed.

### 1. Threshold-Based Segmentation

Simplest form — already covered. Works when objects have clearly distinct intensity from background.

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

### 2. Watershed Algorithm

For segmenting touching or overlapping objects. Based on topographic distance transform.

**Concept:** Treat grayscale image as a topographic map (peaks = bright, valleys = dark). "Flood" from marked seed points. Where flooding from different sources meets = segment boundary.

```python
import cv2
import numpy as np

def watershed_segment(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Remove noise
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Background region
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Foreground region (distance transform)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Unknown region (boundary)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labeling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1                     # background → 1
    markers[unknown == 255] = 0              # unknown → 0
    
    # Apply watershed
    markers = cv2.watershed(img, markers)
    
    # Mark boundaries
    img_copy = img.copy()
    img_copy[markers == -1] = [255, 0, 0]   # boundaries in blue
    return img_copy, markers
```

### 3. GrabCut

Interactive/semi-automatic foreground segmentation. Iterates between graph cut optimization and Gaussian mixture model fitting.

**Algorithm:**
1. Initialize foreground/background models from user rectangle or mask
2. Fit Gaussian Mixture Models (GMMs) to foreground and background pixels
3. Construct graph (pixels as nodes, capacities from GMM probabilities)
4. Run min-cut to separate foreground/background
5. Update GMMs and repeat until convergence

```python
def grabcut_segment(img: np.ndarray, rect: tuple = None) -> np.ndarray:
    """
    rect: (x, y, w, h) bounding rectangle around the foreground object
    """
    mask = np.zeros(img.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    if rect:
        cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 
                    iterCount=5, mode=cv2.GC_INIT_WITH_RECT)
    
    # mask: 0=bg, 1=fg, 2=probable bg, 3=probable fg
    # Extract foreground
    fg_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    result = img * fg_mask[:, :, np.newaxis]
    return result, fg_mask
```

### 4. Mean Shift Segmentation

Non-parametric, clusters pixels by color and spatial proximity.

```python
# Spatially smooth the image using mean shift
# sp: spatial window radius
# sr: color window radius
# maxLevel: image pyramid level for speedup
shifted = cv2.pyrMeanShiftFiltering(img, sp=20, sr=40, maxLevel=1)

# Then segment the smoothed image
gray_shifted = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray_shifted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

### 5. K-Means Color Segmentation

Cluster pixels by color. Fast, but ignores spatial information.

```python
def kmeans_segment(img: np.ndarray, k: int = 4) -> np.ndarray:
    # Reshape: (H*W, 3) — each row is a pixel's BGR value
    pixels = img.reshape(-1, 3).astype(np.float32)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 
                                     attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()]
    return segmented.reshape(img.shape), labels.reshape(img.shape[:2])
```

### Segmentation Method Selection Guide

| Method | Requires | Strength | Weakness |
|--------|----------|----------|----------|
| Thresholding | Clear intensity gap | Simple, fast | Fails with complex scenes |
| K-Means | Number of clusters | Color-based | Ignores spatial structure |
| Mean Shift | Bandwidth params | No k needed | Slow |
| GrabCut | User rectangle | High quality fg/bg | Semi-manual |
| Watershed | Distance transform | Splits touching objects | Sensitive to markers |

---

## 20. Haar Cascade Face Detection

### How Haar Cascades Work

Haar cascade is a **boosted classifier** trained to detect objects by evaluating rectangular features at multiple scales. Introduced by Viola-Jones (2001). It was the state-of-the-art pre-deep-learning approach and is still useful for real-time, CPU-bound applications.

### Algorithm Details

#### 1. Haar Features

Rectangular features computed as the sum of pixels in one rectangle minus the sum in another:

```
Feature value = Σ(white region pixels) - Σ(dark region pixels)

Examples of Haar features:
Edge features:   □□□    Line features:  □■□    Diagonal:  □■
                 ■■■                    □■□              ■□
```

#### 2. Integral Image

Computing sum over a rectangle in O(1) using precomputed integral image:

```
II[x,y] = Σ_{x'≤x, y'≤y} I[x',y']

Sum of rect (x1,y1) to (x2,y2):
= II[x2,y2] - II[x1-1,y2] - II[x2,y1-1] + II[x1-1,y1-1]
```

This makes evaluating thousands of Haar features feasible in real time.

#### 3. AdaBoost Selection

During training, AdaBoost selects the ~6000 most discriminative features from 180,000 candidates. Weak classifiers based on individual features are combined into a strong classifier.

#### 4. Cascade of Classifiers

The cascade is the key to real-time performance. Rather than evaluating all features on every window:

1. Stage 1 (few features): Quick rejection of obvious non-face windows
2. Stage 2: Rejects more windows
3. Stage N: Very high precision on remaining candidates

**~99.9% of windows are rejected at early stages.** Only potential faces reach later stages.

```
Window → Stage 1 → Reject (fast) → discard
                 → Pass → Stage 2 → Reject → discard
                                  → Pass → Stage 3 → ... → Accept: FACE
```

### Implementation

```python
import cv2

# Load pre-trained Haar cascade
# OpenCV ships these in: cv2.data.haarcascades
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

def detect_faces(img: np.ndarray, 
                 scale_factor: float = 1.1,
                 min_neighbors: int = 5,
                 min_size: tuple = (30, 30)) -> list:
    """
    scale_factor: how much image size is reduced at each scale (1.1 = 10% reduction)
    min_neighbors: minimum number of overlapping detections to count as face
    min_size: minimum face size in pixels
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces  # list of (x, y, w, h) tuples

def draw_detections(img: np.ndarray, faces) -> np.ndarray:
    result = img.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Detect eyes within each face ROI
        roi_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
        for (ex, ey, ew, eh) in eyes:
            cv2.circle(result, (x + ex + ew//2, y + ey + eh//2), 
                      ew//2, (0, 0, 255), 2)
    return result
```

### Real-Time Video Face Detection

```python
def realtime_face_detection():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)   # improve contrast for detection
        
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Face", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
        cv2.imshow("Face Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
```

### Available Haar Cascade XML Files

```python
import os
haarcascades_path = cv2.data.haarcascades
print(os.listdir(haarcascades_path))

# Common ones:
# haarcascade_frontalface_default.xml  — frontal face
# haarcascade_frontalface_alt.xml      — frontal face (alternative)
# haarcascade_profileface.xml          — side face
# haarcascade_eye.xml                  — eye
# haarcascade_eye_tree_eyeglasses.xml  — eye with glasses
# haarcascade_fullbody.xml             — full body
# haarcascade_upperbody.xml            — upper body
# haarcascade_smile.xml                — smile
# haarcascade_car.xml                  — car
# haarcascade_russian_plate_number.xml — license plate (RU)
```

### Parameter Tuning

| Parameter | Effect |
|-----------|--------|
| `scaleFactor` ↑ | Faster, misses more faces |
| `scaleFactor` ↓ | Slower, detects more scales |
| `minNeighbors` ↑ | Fewer false positives, more false negatives |
| `minNeighbors` ↓ | More detections, more false positives |
| `minSize` ↑ | Faster, ignores small faces |

---

## 21. Tradeoffs, Failure Modes, and Production Considerations

### Haar Cascade Limitations

Haar cascades are **not competitive with modern deep learning** for accuracy:

| Aspect | Haar Cascade | Deep Learning (MTCNN, RetinaFace) |
|--------|-------------|-----------------------------------|
| Accuracy | ~80-90% | ~95-99% |
| Side profiles | Poor | Good |
| Occlusion | Fails | Handles partially |
| Lighting robustness | Poor | Better |
| Speed on CPU | Very fast | Slower (but ONNX/TFLite competitive) |
| Training data | Classical features | Data-driven |

**When to still use Haar:**
- Constrained environment (controlled lighting, frontal faces)
- No GPU, no ML framework, edge devices
- Need millisecond latency on very weak hardware
- Already integrated in legacy OpenCV pipeline

### Edge Detection Limitations

- Canny requires parameter tuning per image type
- All gradient-based detectors struggle with: texture-heavy backgrounds, specular highlights, compression artifacts
- Illumination changes create spurious edges

### Color Thresholding Limitations

- HSV ranges must be tuned for each lighting condition
- Light source color temperature shifts hues
- Use LAB or adaptive methods for robustness

### General Production Checklist

```python
# 1. Always validate inputs
assert img is not None, "Image load failed"
assert img.dtype == np.uint8, f"Expected uint8, got {img.dtype}"
assert len(img.shape) in [2, 3], "Invalid dimensions"

# 2. Use copies for drawing operations
display = img.copy()
cv2.drawContours(display, contours, -1, (0,255,0), 2)

# 3. Release video resources
try:
    cap = cv2.VideoCapture(0)
    # ... processing ...
finally:
    cap.release()
    cv2.destroyAllWindows()

# 4. Handle BGR/RGB explicitly in any cross-library code
img_for_pytorch = img[:, :, ::-1].copy()  # BGR → RGB
# Or: cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 5. Normalize before neural network input
img_float = img.astype(np.float32) / 255.0
```

### Computational Cost Reference

| Operation | Relative Cost |
|-----------|--------------|
| `img[y,x]` pixel access | Baseline |
| `cv2.resize` (nearest) | Very fast |
| `cv2.resize` (bilinear) | Fast |
| `cv2.GaussianBlur` (5×5) | Fast |
| `cv2.bilateralFilter` | Slow |
| `cv2.findContours` | Moderate |
| `cv2.watershed` | Moderate |
| `cv2.grabCut` | Slow |
| Haar `detectMultiScale` | Moderate (varies by image) |

### Performance Optimization Tips

```python
# 1. Resize to smallest workable resolution before processing
small = cv2.resize(img, (320, 240))
faces = detector.detect(small)
# Scale results back up

# 2. Convert grayscale early if color not needed
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. Use numpy operations instead of Python loops for pixel access
# WRONG (slow):
for y in range(h):
    for x in range(w):
        img[y,x] = transform(img[y,x])

# RIGHT (fast):
img = np.clip(img.astype(np.float32) * 1.5, 0, 255).astype(np.uint8)

# 4. Process video frames in a thread (overlap IO and compute)
import threading
from queue import Queue

# 5. Profile with cProfile or line_profiler to find actual bottlenecks
```

---

## Quick Reference — OpenCV Function Cheatsheet

```python
# I/O
cv2.imread(path, flags)
cv2.imwrite(path, img, params)
cv2.VideoCapture(source)
cv2.VideoWriter(path, fourcc, fps, size)

# Color
cv2.cvtColor(img, code)
cv2.split(img)
cv2.merge(channels)
cv2.inRange(img, lower, upper)

# Geometry
cv2.resize(img, dsize, fx, fy, interpolation)
cv2.flip(img, flipCode)
cv2.rotate(img, rotateCode)
cv2.warpAffine(img, M, dsize)
cv2.warpPerspective(img, H, dsize)
cv2.getRotationMatrix2D(center, angle, scale)
cv2.getAffineTransform(src, dst)
cv2.getPerspectiveTransform(src, dst)

# Draw
cv2.line, cv2.rectangle, cv2.circle, cv2.ellipse
cv2.polylines, cv2.fillPoly
cv2.putText
cv2.drawContours

# Filtering
cv2.filter2D(img, ddepth, kernel)
cv2.blur, cv2.GaussianBlur, cv2.medianBlur, cv2.bilateralFilter
cv2.Sobel, cv2.Laplacian, cv2.Canny

# Morphology
cv2.morphologyEx(img, op, kernel)  # MORPH_OPEN, CLOSE, GRADIENT, TOPHAT, BLACKHAT
cv2.erode, cv2.dilate

# Histogram
cv2.calcHist, cv2.equalizeHist, cv2.createCLAHE
cv2.compareHist

# Contours
cv2.findContours, cv2.drawContours
cv2.contourArea, cv2.arcLength, cv2.boundingRect
cv2.moments, cv2.convexHull, cv2.approxPolyDP

# Thresholding
cv2.threshold, cv2.adaptiveThreshold

# Segmentation
cv2.watershed, cv2.grabCut, cv2.pyrMeanShiftFiltering, cv2.kmeans

# Detection
cv2.CascadeClassifier.detectMultiScale
```

---

*These notes cover the complete foundational curriculum of OpenCV computer vision — from pixel representation to segmentation and detection. Every mechanism is explained from first principles to support both implementation and debugging in production contexts.*