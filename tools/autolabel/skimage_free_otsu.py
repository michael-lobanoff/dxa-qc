import numpy as np
def otsu(x):
    hist, edges = np.histogram(x.ravel(), bins=256, range=(0, 256)); p = hist / hist.sum()
    w0 = np.cumsum(p); mu = np.cumsum(p * np.arange(256)); mt = mu[-1]
    sb = (mt * w0 - mu) ** 2 / (w0 * (1 - w0) + 1e-12)
    return float(np.argmax(sb))
