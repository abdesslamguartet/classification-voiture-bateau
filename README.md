# classification-voiture-bateau
# Classification d'images : Voiture vs Bateau (SVM + caractéristiques)

Atelier de classification supervisée (Pr. Abdelouahed Sabri).
Les images (base **DB2C** : dossiers `obj_car` et `obj_ship`) sont décrites par 52 caractéristiques
(statistiques RGB, histogramme HSV, GLCM, LBP, moments de Hu), puis classées par un SVM à noyau RBF.
Cinq autres modèles (SVM linéaire, SVM sigmoïde, KNN k=3, KNN k=5, arbre de décision) sont comparés.

**Page web de test :** [https://classification-voiture-bateau-2rtpnq7cxmcxdbzbfuvcxq.streamlit.app/]

## Contenu du dépôt

| Fichier | Rôle |
|---|---|
| `Atelier_Classification_Voiture_Bateau.ipynb` | Notebook complet (étapes A à F) |
| `features.py` | Extraction des caractéristiques (utilisé par le notebook et la page web) |
| `app.py` | Page web Gradio : on importe une image, le modèle donne Voiture ou Bateau |
| `model_svm_rbf.joblib` | Modèle SVM RBF entraîné |
| `requirements.txt` | Bibliothèques nécessaires |

## Lancer la page web en local

```bash
pip install -r requirements.txt
python app.py
```
Puis ouvrir l'adresse affichée (par défaut http://127.0.0.1:7860).

## Réentraîner le modèle

Placer le dossier `DB2C` à côté du notebook et exécuter toutes les cellules.

## Limite

Le modèle ne connaît que deux classes : toute autre image sera classée Voiture ou Bateau.
 
# classification-voiture-bateau
 
