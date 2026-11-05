"""Perspective rectification using the annotation's semantic corner order."""
import cv2
import numpy as np


def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot read image: {path}')
    return image


def align_document(image, points, max_side=1000):
    """Rectify TL, TR, BR, BL corners; scale the longest side to max_side."""
    if image is None or image.ndim not in (2, 3):
        raise ValueError('Invalid image')
    if not isinstance(max_side, int) or max_side < 2:
        raise ValueError('max_side must be an integer >= 2')
    src = np.asarray(points, dtype=np.float32)
    if src.shape != (4, 2) or not np.isfinite(src).all():
        raise ValueError('Expected four finite coordinate pairs')
    h, w = image.shape[:2]
    if src[:, 0].max() < 0 or src[:, 0].min() >= w or src[:, 1].max() < 0 or src[:, 1].min() >= h:
        raise ValueError('Document lies outside image')
    if not cv2.isContourConvex(src) or abs(cv2.contourArea(src)) < 4:
        raise ValueError('Corners must form a non-degenerate convex quadrilateral in order')
    width = max(np.linalg.norm(src[1]-src[0]), np.linalg.norm(src[2]-src[3]))
    height = max(np.linalg.norm(src[3]-src[0]), np.linalg.norm(src[2]-src[1]))
    scale = max_side / max(width, height)
    out_w, out_h = max(2, round(width*scale)), max(2, round(height*scale))
    dst = np.array([[0, 0], [out_w-1, 0], [out_w-1, out_h-1], [0, out_h-1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError('Degenerate perspective transform')
    return cv2.warpPerspective(image, matrix, (out_w, out_h))
