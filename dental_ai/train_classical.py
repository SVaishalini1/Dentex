"""Trains the 6 traditional models on the DENTEX disease data (same recipe as your reference notebook):
GroupShuffleSplit(test 0.3, seed 42), StratifiedGroupKFold(3), GridSearchCV(f1_macro), class_weight=balanced.
Run in Colab after downloading DENTEX:   python train_classical.py
Output: models/classical/classical_models.joblib  (copy to the app's models/classical/)"""
import glob, os, json, sys, joblib, numpy as np, pandas as pd, cv2, sklearn
from skimage.feature import hog
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/content"
js = glob.glob(f"{ROOT}/**/train_quadrant_enumeration_disease.json", recursive=True)
assert js, "disease JSON not found - pass the dataset folder: python train_classical.py /path/to/dataset"
data = json.load(open(js[0])); folder = os.path.dirname(js[0])
paths = {os.path.basename(p): p for p in glob.glob(folder + "/**/*", recursive=True) if p.lower().endswith((".png", ".jpg"))}
classes = {c["id"]: c["name"] for c in data["categories_3"]}
imgs = {im["id"]: im for im in data["images"]}
rows = [dict(image_id=a["image_id"], file=imgs[a["image_id"]]["file_name"], W=imgs[a["image_id"]]["width"],
             H=imgs[a["image_id"]]["height"], bbox=a["bbox"], label=a["category_id_3"]) for a in data["annotations"]]
df = pd.DataFrame(rows).sort_values("file").reset_index(drop=True)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
feats, cur, img = [], None, None
for r in df.itertuples():
    if r.file != cur: img, cur = cv2.imread(paths[r.file], cv2.IMREAD_GRAYSCALE), r.file
    x, y, w, h = [int(v) for v in r.bbox]; x, y = max(x, 0), max(y, 0)
    c = clahe.apply(cv2.resize(img[y:y + max(h, 1), x:x + max(w, 1)], (64, 64)))
    feats.append(np.concatenate([hog(c, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)),
                                 [(x + w / 2) / r.W, (y + h / 2) / r.H, w / r.W, h / r.H]]))
X, y, g = np.array(feats), df["label"].values, df["image_id"].values
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42).split(X, groups=g))

models = {
    "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")),
    "ridge": make_pipeline(StandardScaler(), RidgeClassifier(class_weight="balanced")),
    "rf": RandomForestClassifier(class_weight="balanced", random_state=42),
    "svm": make_pipeline(StandardScaler(), PCA(n_components=0.95, random_state=42), SVC(class_weight="balanced")),
    "knn": make_pipeline(StandardScaler(), PCA(random_state=42), KNeighborsClassifier()),
    "dt": DecisionTreeClassifier(class_weight="balanced", random_state=42),
}
grids = {
    "logreg": {"logisticregression__C": [0.01, 0.1, 1, 10]},
    "ridge": {"ridgeclassifier__alpha": [0.1, 1, 10, 100]},
    "rf": {"n_estimators": [200, 400], "max_depth": [None, 20]},
    "svm": {"svc__C": [1, 10], "svc__gamma": ["scale"]},
    "knn": {"pca__n_components": [0.90, 0.95], "kneighborsclassifier__n_neighbors": [3, 5, 9]},
    "dt": {"max_depth": [5, 10, 20, None], "min_samples_leaf": [1, 5]},
}
cv, best, metrics = StratifiedGroupKFold(n_splits=3), {}, []
for n in models:
    gs = GridSearchCV(models[n], grids[n], cv=cv, scoring="f1_macro", n_jobs=-1).fit(X[tr], y[tr], groups=g[tr])
    p = gs.predict(X[te]); best[n] = gs.best_estimator_
    metrics.append(dict(model=n, accuracy=round(accuracy_score(y[te], p), 3),
                        precision=round(precision_score(y[te], p, average="macro", zero_division=0), 3),
                        recall=round(recall_score(y[te], p, average="macro", zero_division=0), 3),
                        f1=round(f1_score(y[te], p, average="macro", zero_division=0), 3), params=str(gs.best_params_)))
    print(n, "| cv f1:", round(gs.best_score_, 3), "| test f1:", metrics[-1]["f1"], "|", gs.best_params_)

os.makedirs("models/classical", exist_ok=True)
joblib.dump({"models": best, "classes": classes, "metrics": metrics, "sklearn": sklearn.__version__},
            "models/classical/classical_models.joblib")
print("saved models/classical/classical_models.joblib | scikit-learn", sklearn.__version__)
