"""Histogram of oriented gradients (OpenCV 5 dropped HOGDescriptor from the python build)."""
import cv2
import numpy as np


def hog(img, cell=16, bins=9):
    """Per-cell orientation histograms (unsigned, `bins` bins), L2-normalised over 2x2 cell blocks."""
    g = img.astype(np.float32)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=1), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=1)
    mag, ang = np.hypot(gx, gy), np.degrees(np.arctan2(gy, gx)) % 180
    b = np.minimum((ang / (180 / bins)).astype(int), bins - 1)
    ch, cw = g.shape[0] // cell, g.shape[1] // cell
    cells = np.stack([(mag * (b == i))[:ch * cell, :cw * cell].reshape(ch, cell, cw, cell).sum((1, 3)) for i in range(bins)], -1)
    blocks = [cells[y:y + 2, x:x + 2].ravel() for y in range(ch - 1) for x in range(cw - 1)]
    return np.concatenate([v / (np.linalg.norm(v) + 1e-6) for v in blocks])
