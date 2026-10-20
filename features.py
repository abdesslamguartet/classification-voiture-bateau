"""
features.py - Extraction des caracteristiques d'une image RGB (IMG_SIZE x IMG_SIZE).
Ce meme fichier est utilise par le notebook (entrainement) ET par la page web (prediction),
afin que les caracteristiques soient calculees exactement de la meme facon.
"""
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

CLASSES = ["Voiture", "Bateau"]  # etiquette 0 = Voiture (dossier obj_car), 1 = Bateau (dossier obj_ship)
IMG_SIZE = 32                    # toutes les images sont redimensionnees en IMG_SIZE x IMG_SIZE


def _to_uint8(img):
    """Accepte une image float normalisee [0,1] ou uint8 [0,255] et renvoie du uint8."""
    if img.dtype != np.uint8:
        img = np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)
    return img


def rgb_stats(img):
    """Moyenne et ecart-type de chaque canal R, G, B -> 6 valeurs."""
    f = img.astype(np.float32)
    return np.concatenate([f.mean(axis=(0, 1)), f.std(axis=(0, 1))])


def hsv_hist(img, bins=8):
    """Histogramme normalise de H, S, V (8 classes chacun) -> 24 valeurs."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    ranges = [180, 256, 256]  # H va de 0 a 179 dans OpenCV
    feats = []
    for c in range(3):
        h = cv2.calcHist([hsv], [c], None, [bins], [0, ranges[c]]).flatten()
        feats.append(h / (h.sum() + 1e-9))
    return np.concatenate(feats)


def glcm_feats(gray):
    """Texture GLCM (distance 1, 4 angles) : 5 proprietes moyennees -> 5 valeurs."""
    glcm = graycomatrix(gray, distances=[1],
                        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=256, symmetric=True, normed=True)
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
    return np.array([graycoprops(glcm, p).mean() for p in props])


def lbp_hist(gray, P=8, R=1):
    """Histogramme LBP 'uniform' normalise -> P+2 = 10 valeurs."""
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    h, _ = np.histogram(lbp.ravel(), bins=P + 2, range=(0, P + 2))
    return h / (h.sum() + 1e-9)


def hu_moments(gray):
    """7 moments de Hu (echelle logarithmique) -> 7 valeurs."""
    hu = cv2.HuMoments(cv2.moments(gray)).flatten()
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)


def extract_features(img):
    """Vecteur de caracteristiques d'une image RGB. Taille totale : 52."""
    img = _to_uint8(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return np.concatenate([
        rgb_stats(img),     # 6
        hsv_hist(img),      # 24
        glcm_feats(gray),   # 5
        lbp_hist(gray),     # 10
        hu_moments(gray),   # 7
    ]).astype(np.float64)


def extract_matrix(images):
    """Matrice (n_images, 52) : une ligne = une image, une colonne = une caracteristique."""
    return np.array([extract_features(im) for im in images])
