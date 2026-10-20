import gradio as gr
import joblib
import numpy as np
import cv2
from features import extract_features, CLASSES, IMG_SIZE

model = joblib.load("model_svm_rbf.joblib")

def classifier(image):
    if image is None:
        return None
    # même taille que pour l'entraînement
    petite = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    x = extract_features(petite).reshape(1, -1)
    p = model.predict_proba(x)[0]
    return {CLASSES[i]: float(p[i]) for i in range(len(CLASSES))}

demo = gr.Interface(
    fn=classifier,
    inputs=gr.Image(type="numpy", image_mode="RGB", label="Importer une image (voiture ou bateau)"),
    outputs=gr.Label(num_top_classes=2, label="Classe prédite"),
    title="Classification d'images : Voiture vs Bateau",
    description=("Modèle SVM (noyau RBF) entraîné sur la base DB2C (voitures et bateaux) avec des caractéristiques extraites "
                 "(RGB, HSV, GLCM, LBP, moments de Hu). Le modèle ne connaît que 2 classes : "
                 "toute autre image sera forcément classée Voiture ou Bateau."),
)

if __name__ == "__main__":
    demo.launch()
