import numpy as np


def cosine_similarity(vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    vec_norm = np.linalg.norm(vec)
    mat_norm = np.linalg.norm(mat, axis=1)
    norms = vec_norm * mat_norm
    norms[norms == 0] = 1e-10
    return np.dot(mat, vec) / norms
