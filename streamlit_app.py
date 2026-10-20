import cv2
import joblib
import numpy as np
import streamlit as st

from features import extract_features, CLASSES, IMG_SIZE

st.set_page_config(page_title="Voiture vs Bateau", page_icon="🚗")


@st.cache_resource
def charger_modele():
    return joblib.load("model_svm_rbf.joblib")


model = charger_modele()

st.title("Classification d'images : Voiture vs Bateau")
st.write(
    "Modèle SVM (noyau RBF) entraîné sur la base DB2C avec des caractéristiques extraites "
    "(RGB, HSV, GLCM, LBP, moments de Hu). Le modèle ne connaît que 2 classes : "
    "toute autre image sera forcément classée Voiture ou Bateau."
)

fichier = st.file_uploader("Importer une image (voiture ou bateau)",
                           type=["jpg", "jpeg", "png", "bmp", "webp"])

if fichier is not None:
    data = np.frombuffer(fichier.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Impossible de lire cette image.")
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(rgb, caption="Image importée", width=320)

        # même taille que pour l'entraînement
        petite = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        x = extract_features(petite).reshape(1, -1)
        p = model.predict_proba(x)[0]
        k = int(np.argmax(p))

        st.subheader(f"Classe prédite : {CLASSES[k]}")
        for i, nom in enumerate(CLASSES):
            st.progress(float(p[i]), text=f"{nom} : {p[i] * 100:.1f} %")
