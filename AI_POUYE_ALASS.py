import os
from pathlib import Path
from typing import List, Dict, Tuple, Any

from PIL import Image, UnidentifiedImageError
import numpy as np
from tqdm import tqdm
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class SeaClassifierPipeline:
    """
    Pipeline complet de classification d'images (Computer Vision)
    Architecture : Feature Engineering (RGB Split) -> Data Augmentation -> Naive Bayes
    """

    def __init__(self, img_width: int = 640, img_height: int = 480):
        self.size = (img_width, img_height)
        self.model = GaussianNB()
        self.is_trained = False

    # ==========================================
    # 1. FEATURE ENGINEERING (Extraction)
    # ==========================================
    def _resize(self, img: Image.Image) -> Image.Image:
        return img.resize(self.size)

    def _compute_histograms(self, img: Image.Image) -> np.ndarray:
        """Calcule et fusionne les 3 histogrammes (Normal, Focus Bleu, Focus Rouge/Vert)"""
        r_im = self._resize(img.convert("RGB"))
        grey_im = r_im.convert('L')
        r, g, b = r_im.split()

        # 1. Histo classique
        histo_normal = np.array(r_im.histogram())

        # 2. Histo Focus Bleu (R et G en gris)
        blue_focus = Image.merge("RGB", (grey_im, grey_im, b))
        histo_blue = np.array(blue_focus.histogram())

        # 3. Histo Sans Bleu (B en gris)
        not_blue_focus = Image.merge("RGB", (r, g, grey_im))
        histo_notblue = np.array(not_blue_focus.histogram())

        # Fusion (Concatenation des features)
        return np.hstack([histo_normal, histo_blue, histo_notblue])

    # ==========================================
    # 2. CHARGEMENT ET PRÉPARATION DES DONNÉES
    # ==========================================
    def load_dataset(self, path: str, label: int = None, n_max: int = None, desc: str = "Chargement") -> List[Dict]:
        """Charge un dossier d'images avec barre de progression."""
        dataset = []
        folder = Path(path)

        if not folder.exists():
            print(f" Attention : Le dossier {path} n'existe pas. Ignoré.")
            return dataset

        files = list(folder.glob('*'))
        if n_max: files = files[:n_max]

        for filepath in tqdm(files, desc=desc, unit="img"):
            try:
                with Image.open(filepath) as im:
                    dataset.append({
                        'name_path': filepath.name,
                        'image': self._resize(im.copy()), # On garde l'image en RAM pour l'augmentation
                        'features': self._compute_histograms(im),
                        'y_true': label,
                        'y_pred': None
                    })
            except UnidentifiedImageError:
                pass
        return dataset

    def augment_training_data(self, dataset: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Data Augmentation par effet miroir (Flip Horizontal). Appliqué UNIQUEMENT sur le Train Set."""
        X, y = [], []

        for data in dataset:
            # 1. Donnée originale
            X.append(data['features'])
            y.append(data['y_true'])

            # 2. Donnée augmentée (Miroir)
            # Utilisation de la syntaxe moderne de PIL (Image.Transpose)
            flip_img = data['image'].transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            X.append(self._compute_histograms(flip_img))
            y.append(data['y_true'])

        return np.array(X), np.array(y)

    # ==========================================
    # 3. ÉVALUATION ET ANALYSE VISUELLE
    # ==========================================
    def plot_confusion_matrix(self, y_true, y_pred, title="Matrice de Confusion (Hold-Out)"):
        """Génère une carte de chaleur visuelle des performances."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu",
                    xticklabels=["Ailleurs (-1)", "Mer (1)"],
                    yticklabels=["Ailleurs (-1)", "Mer (1)"])
        plt.title(title, fontweight='bold')
        plt.ylabel("Vraie Classe")
        plt.xlabel("Prédiction de l'IA")
        plt.tight_layout()
        plt.savefig("analyse_performances.png", dpi=300)
        plt.close()

    # ==========================================
    # 4. ORCHESTRATION DU PIPELINE
    # ==========================================
    def run_full_pipeline(self, path_mer: str, path_ailleurs: str, path_cc2: str):
        print("\n" + "="*60)
        print(" PIPELINE IA : CLASSIFICATION D'ENVIRONNEMENTS MARINS")
        print("="*60)

        # 1. Chargement
        S_mer = self.load_dataset(path_mer, label=1, desc="Océan")
        S_ail = self.load_dataset(path_ailleurs, label=-1, desc="Ailleurs")
        S_total = S_mer + S_ail

        # 2. Séparation stricte (Hold-Out 80/20) pour éviter le Data Leakage
        S_train, S_test = train_test_split(S_total, test_size=0.20, random_state=42)

        # 3. Augmentation UNIQUEMENT sur l'entraînement
        print("\n  Génération des données augmentées (Train Set uniquement)...")
        X_train_aug, y_train_aug = self.augment_training_data(S_train)

        X_test = np.array([s['features'] for s in S_test])
        y_test = np.array([s['y_true'] for s in S_test])

        # 4. Entraînement et Évaluation Hold-out
        print(" Entraînement du modèle Naive Bayes...")
        self.model.fit(X_train_aug, y_train_aug)

        y_pred_test = self.model.predict(X_test)
        er_holdout = 1 - accuracy_score(y_test, y_pred_test)

        # Génération du graphique
        self.plot_confusion_matrix(y_test, y_pred_test)

        # 5. Validation Croisée (Cross-Validation)
        print(" Lancement de la Validation Croisée (8-Folds)...")
        X_all = np.array([s['features'] for s in S_total])
        y_all = np.array([s['y_true'] for s in S_total])
        cv_scores = cross_val_score(GaussianNB(), X_all, y_all, cv=8)
        er_cv = 1 - np.mean(cv_scores)

        # 6. Modèle Ultime (Entraîné sur 100% des données augmentées pour le CC2)
        print("\n Entraînement du modèle ultime pour la production...")
        X_final_aug, y_final_aug = self.augment_training_data(S_total)
        final_model = GaussianNB()
        final_model.fit(X_final_aug, y_final_aug)

        ee_final = 1 - accuracy_score(y_final_aug, final_model.predict(X_final_aug))

        # 7. Prédictions Inconnues (CC2)
        S_cc2 = self.load_dataset(path_cc2, label=None, desc="Base Inconnue (CC2)")
        if S_cc2:
            X_cc2 = np.array([s['features'] for s in S_cc2])
            predictions_cc2 = final_model.predict(X_cc2)
            for i, s in enumerate(S_cc2):
                s['y_pred'] = predictions_cc2[i]

        # 8. Sauvegarde MLOps
        joblib.dump(final_model, 'sea_classifier_model.pkl')

        # 9. Écriture du rapport final
        print("\n  Génération du rapport officiel...")
        with open("l'autodidaque.pouye'.txt", "w", encoding="utf-8") as f:
            f.write("#  POUYE Alassane\n")
            f.write("# Architecture : Feature Fusion (RGB+Blue Focus) + Data Augmentation\n\n")

            if S_cc2:
                for s in S_cc2:
                    f.write(f"{s['name_path']} {s['y_pred']}\n")
            else:
                f.write("# (Base CC2 introuvable lors de l'exécution)\n")

            f.write("\n# === STATISTIQUES D'APPRENTISSAGE ===\n")
            f.write(f"# Erreur Empirique (Modèle Final) : {ee_final:.4f}\n")
            f.write(f"# Erreur Réelle (Hold-Out 80/20)  : {er_holdout:.4f}\n")
            f.write(f"# Erreur Réelle (CV 8-Folds)      : {er_cv:.4f}\n")

            f.write("\n# === RAPPORT DÉTAILLÉ (HOLD-OUT) ===\n")
            f.write(classification_report(y_test, y_pred_test, target_names=["Ailleurs (-1)", "Mer (1)"]))

        print("✅ Terminé ! Consultez le fichier texte et l'image générée.")


# ==========================================
# POINT D'ENTRÉE DU PROGRAMME
# ==========================================
if __name__ == "__main__":
    pipeline = SeaClassifierPipeline()
    # Mettez les vrais chemins de vos dossiers ici
    pipeline.run_full_pipeline("Init/Mer", "Init/Ailleurs", "Data CC2")