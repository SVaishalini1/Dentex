
# Dental OPG Analyzer: Tooth Detection, FDI Numbering and Abnormality Detection on Panoramic X-rays
 
An end-to-end project on the **DENTEX** dataset. Given a panoramic dental X-ray (OPG), the system
1. divides the jaw into its **4 quadrants**,
2. finds every tooth and gives it its **FDI number**,
3. finds the **abnormalities** (Caries, Deep Caries, Periapical Lesion, Impacted tooth),
and compares what **six traditional machine-learning models** (Logistic Regression, Ridge, Random Forest, SVM, KNN, Decision Tree) predict for the same abnormalities. Everything is served through a web app.
 
> **Disclaimer.** This is an academic / research project. It is **not** a medical device and must not be used for diagnosis or treatment decisions.
 
---
 
## Table of contents
1. [Project overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Repository structure](#3-repository-structure)
4. [Key concepts](#4-key-concepts)
5. [Part A: YOLO detection](#5-part-a-yolo-detection)
6. [Part B: Traditional ML models](#6-part-b-traditional-ml-models)
7. [Results](#7-results)
8. [Part C: Web application](#8-part-c-web-application)
9. [Limitations and future work](#9-limitations-and-future-work)
10. [Reproducibility](#10-reproducibility)
11. [Acknowledgements and license](#11-acknowledgements-and-license)
---
 
## 1. Project overview
 
The project has three parts that share one dataset:
 
| Part | What it does | Where |
|---|---|---|
| **A. YOLO** | Locates quadrants, teeth (with tooth number) and abnormalities on the full X-ray | `YOLO.ipynb` |
| **B. Traditional ML** | Classifies a cropped tooth / region with HOG features and six classical models | the six model notebooks in the repo root |
| **C. Web app** | Upload an OPG, see the three annotated images, and compare the traditional models | `dental_ai/` |
 
### Objectives
 
| # | Objective | Done by |
|---|---|---|
| 01 | **Tooth localization**: locate individual teeth / quadrants in an OPG | YOLO (Part A) |
| 02 | **Quadrant prediction and tooth enumeration**: classify a tooth into one of four quadrants and give its number | YOLO (Part A) and traditional models (Part B) |
| 04 | **Pathology detection**: identify caries, deep caries, periapical lesions and impacted teeth | YOLO (Part A) and traditional models (Part B) |
 
### Pipeline
 
```mermaid
flowchart LR
    A[Upload OPG] --> B[YOLO quadrant model]
    A --> C[YOLO tooth-number model]
    A --> D[YOLO disease model]
    B --> E[Assign each tooth to a quadrant]
    C --> E
    E --> F["FDI = quadrant x 10 + tooth number"]
    D --> G[Attach each abnormality to the tooth it overlaps]
    F --> G
    G --> H[3 annotated images + table]
    D --> I[Crop each abnormality]
    I --> J[HOG features]
    J --> K[6 traditional models]
    K --> L[Comparison page]
```
 
---
 
## 2. Dataset
 
**DENTEX** (Dental Enumeration and Diagnosis on Panoramic X-rays), from the MICCAI 2023 challenge. The training data is downloaded from the Hugging Face dataset `keycxl/DENTEX` (`DENTEX/training_data.zip`). Please check the dataset's own license and citation requirements before redistributing anything derived from it.
 
| Subset | Images | Annotations | Labels |
|---|---|---|---|
| `quadrant` | 693 | 2,772 quadrant regions | quadrant (1 to 4) |
| `quadrant_enumeration` | 634 | 18,095 teeth | quadrant + tooth number (1 to 8) |
| `quadrant-enumeration-disease` | 705 | 3,529 abnormal teeth | quadrant + tooth number + abnormality |
| `unlabelled` | 1,571 | none | not used |
 
Things worth knowing about the data:
- Annotations are in **COCO format** (`bbox` plus polygon `segmentation`). Category ids are **not** the quadrant numbers, so names must always be mapped through the `categories` lists.
- The disease file has **no healthy class**. Only abnormal teeth are annotated, about 5 per X-ray against roughly 28 teeth in a full mouth. A classifier trained on it can say *which* abnormality a tooth has, not *whether* it is healthy.
- The four abnormality classes are heavily **imbalanced**:
| Abnormality | Teeth | Share |
|---|---|---|
| Caries | 2,189 | 62% |
| Impacted | 604 | 17% |
| Deep Caries | 578 | 16% |
| Periapical Lesion | 158 | 4% |
 
- The official DENTEX challenge test set has no public labels and is **not used**. All evaluation here uses held-out parts of the labelled training data, always split **by X-ray**.
- About 66 abnormal teeth (about 2%) carry two abnormality labels. They are kept as separate rows.
---
 
## 3. Repository structure
 
```
Dentex/
├── README.md
├── YOLO.ipynb                                       # Part A: quadrant, tooth-number and disease YOLO models
├── KNN.ipynb                                        # Part B: K-Nearest Neighbours
├── random_forest(1).ipynb                           # Part B: Random Forest
├── SVM.ipynb                                        # Part B: Support Vector Machine
├── Decision_Tree.ipynb                              # Part B: Decision Tree
├── dentex_logistic_hog_json_70_30_(1).ipynb         # Part B: Logistic Regression
├── dentex_ridge_hog_json_70_30_(1) (1).ipynb        # Part B: Ridge
├── .gitignore                                       # keeps weights and results out of git
└── dental_ai/                                       # Part C: the web application (FastAPI)
    ├── run.py   config.py   requirements.txt   train_classical.py
    ├── models/
    │   ├── quadrant.pt  enum.pt  disease.pt         # YOLO weights (not stored in git)
    │   ├── imgsz.txt                                # optional: training image size
    │   └── classical/classical_models.joblib        # the 6 traditional models (not stored in git)
    ├── app/
    │   ├── main.py                                  # API and page routes
    │   └── services/
    │       ├── yolo_engine.py    mock_engine.py     postprocess.py
    │       ├── annotate.py       classical.py
    ├── static/   index.html  compare.html  style.css
    ├── storage/results/                             # one folder per analysis
    └── tests/test_api.py
```
 
Model weights and `.joblib` files are large binaries. They are excluded by `.gitignore`; share them through a GitHub Release, Git LFS or a drive link (see [8.2](#82-setup)).
 
---
 
## 4. Key concepts
 
**OPG (orthopantomogram).** A panoramic X-ray of both jaws. By convention the **patient's right side appears on the left of the image**.
 
**Quadrants.** The mouth is divided into four: 1 = upper right, 2 = upper left, 3 = lower left, 4 = lower right (patient's point of view).
 
**FDI notation.** A tooth is named with two digits: *quadrant* then *tooth number*. Tooth numbers run 1 (central incisor), 2 (lateral incisor), 3 (canine), 4 and 5 (premolars), 6 to 8 (molars, 8 = wisdom tooth). Tooth 36 is therefore the first molar of the lower-left quadrant. There are 32 valid numbers (11 to 18, 21 to 28, 31 to 38, 41 to 48).
 
**Abnormalities.**
- *Caries*: tooth decay.
- *Deep caries*: decay that has reached close to the pulp.
- *Periapical lesion*: inflammation at the root tip, seen as a dark area around the apex.
- *Impacted tooth*: a tooth that cannot erupt normally, most often a wisdom tooth.
**Object detection (YOLO).** A neural network that finds objects on the whole image and returns a bounding box, a class and a confidence for each. Used here because traditional models cannot find teeth by themselves; they can only classify a crop that is already given.
 
**HOG (Histogram of Oriented Gradients).** A hand-crafted feature that describes the shape and edge directions inside an image patch. A crop is divided into small cells, the gradient directions in each cell are counted, and the counts form a feature vector. It is the input of all traditional models here.
 
**CLAHE / autocontrast.** Contrast enhancement applied to each crop before HOG, because dental X-rays have low and uneven contrast.
 
**Position features.** The normalized centre and size of a box on the X-ray. Quadrant and tooth number depend strongly on *where* a tooth is, which appearance alone cannot tell (a tooth and its mirror image look alike).
 
**Class imbalance.** Caries is 62% of the abnormal teeth. Models are therefore evaluated with macro-averaged metrics, and `class_weight="balanced"` is used where the algorithm supports it (KNN has no such option).
 
**Group-aware splitting.** One X-ray contains many teeth. If crops from the same X-ray fall in both training and test sets, the model recognises the patient's anatomy and the score is inflated (data leakage). All splits therefore keep every crop of one X-ray on the same side:
- `GroupShuffleSplit(test_size=0.3, random_state=42)` for the train/test split.
- `StratifiedGroupKFold` for cross-validation (keeps X-rays together and class proportions similar).
**Hyperparameter tuning.** `GridSearchCV` with `scoring="f1_macro"` is run on the training part only; the test part is used once at the end.
 
**Macro-averaged metrics.** Precision, recall and F1 are computed per class and averaged with equal weight, so rare classes count as much as common ones. Accuracy alone is misleading on imbalanced data (always guessing "Caries" would score about 62%).
 
**Hierarchical FDI.** Instead of one 32-class model, predict the quadrant first (about 97 to 99% accurate), then choose the tooth number among 8 candidates: `FDI = quadrant x 10 + tooth number`.
 
---
 
## 5. Part A: YOLO detection
 
`YOLO.ipynb` trains **three separate YOLOv8 models** with Ultralytics, one per task:
 
| Model | Detects | Classes |
|---|---|---|
| `quadrant` | the 4 quadrant regions | 1, 2, 3, 4 |
| `enum` | individual teeth with their tooth number | 1 to 8 |
| `disease` | abnormal teeth | Impacted, Caries, Periapical Lesion, Deep Caries |
 
**Pipeline in the notebook**
1. Download DENTEX from Hugging Face and extract it.
2. Convert the COCO annotations (polygons or boxes) to YOLO label format, one dataset per task.
3. Split **by image** into train / validation / test (seed 42), so no X-ray appears in two sets.
4. Train each model, then evaluate on the held-out images.
**Training settings** (verify against the notebook before publishing): YOLOv8s, 50 epochs, image size 1024, batch 4, seed 42, **horizontal flip disabled** (`fliplr=0.0`). Flipping an OPG swaps the patient's left and right and would corrupt the quadrant and FDI labels.
 
**Detection metrics:** *(add mAP@0.5, precision and recall per model from the notebook's output here.)*
 
**How FDI is produced from three models.** The tooth-number model only gives 1 to 8. Each detected tooth is assigned to the quadrant box that contains most of it (ties are broken by distance to the box centre), and `FDI = quadrant x 10 + tooth number`. If two boxes get the same FDI number, only the most confident is kept. Each abnormality box is attached to the tooth it overlaps most (at least 30% of the abnormality box must lie inside the tooth).
 
---
 
## 6. Part B: Traditional ML models
 
Traditional models cannot localize anything. They classify an **already cropped** region (a quadrant region, a tooth, or an abnormal tooth) from its HOG features. All six follow the same experimental protocol so the comparison is fair.
 
### 6.1 Three tasks
 
| Task | Crops come from | Target | Classes |
|---|---|---|---|
| **Quadrant** | `quadrant` subset (2,772 regions) | quadrant | 4 |
| **Tooth numbering** | `quadrant_enumeration` (18,095 teeth) | tooth number 1 to 8 (FDI = quadrant x 10 + tooth number) | 8 |
| **Abnormality** | `quadrant-enumeration-disease` (3,529 teeth) | abnormality type | 4 |
 
### 6.2 Shared protocol
1. Read the COCO JSON, map category ids to names, build a table of boxes and labels.
2. Crop each box from the X-ray, enhance contrast, resize, and extract **HOG** features.
3. Split with `GroupShuffleSplit(test_size=0.3, random_state=42)` grouped by X-ray file.
4. Tune with `StratifiedGroupKFold` + `GridSearchCV(scoring="f1_macro")` on the training part.
5. Evaluate once on the held-out 30%: accuracy, macro precision, macro recall, macro F1, confusion matrix.
6. Save the best model with `joblib`.
### 6.3 Models and search grids
 
| Model | Pipeline | Searched hyperparameters | Notes |
|---|---|---|---|
| **KNN** | `StandardScaler` > `PCA` > `KNeighborsClassifier` | PCA components {0.90, 0.95}; k {3, 5, 9} | Distance-based; PCA reduces the ~1,800 HOG features. No `class_weight` option. |
| **Random Forest** | `RandomForestClassifier(class_weight="balanced")` | `n_estimators` {200, 400}; `max_depth` {None, 20} | Handles the position columns without scaling. |
| **Logistic Regression** | `StandardScaler` > `LogisticRegression` | `C` {0.005, 0.01, 0.1, 1, 10}; `max_iter` {100, 150, 300, 500, 1000, 2000} | `class_weight="balanced"` for the abnormality task. Strong L2 regularization (small C) wins. |
| **Ridge** | `StandardScaler` > `RidgeClassifier(solver="lsqr")` | `alpha` {1 ... 100000}; `max_iter` {50, 200, 1000} | `RidgeClassifier` is the classification form of ridge regression (it regresses on class targets). `class_weight="balanced"` for abnormality. |
| **SVM** | `StandardScaler` > `PCA` > `SVC` | same structure as the KNN / RF notebooks | See `SVM.ipynb`. |
| **Decision Tree** | `DecisionTreeClassifier(class_weight="balanced")` | same structure as the KNN / RF notebooks | See `Decision_Tree.ipynb`. |
 
### 6.4 Best hyperparameters found
 
| Model | Quadrant | Tooth numbering | Abnormality |
|---|---|---|---|
| KNN | k=9, PCA 0.90 | k=9, PCA 0.95 | k=3, PCA 0.90 |
| Random Forest | 400 trees, depth None | 400 trees, depth 20 | 200 trees, depth 20 |
| Logistic Regression | C=0.01, max_iter=150 | C=0.005, max_iter=150 | C=0.005, max_iter=150 |
| Ridge | alpha=100, max_iter=50 | alpha=1000, max_iter=50 | alpha=1000, max_iter=50 |
| SVM | *(fill in from notebook)* | *(fill in)* | *(fill in)* |
| Decision Tree | *(fill in)* | *(fill in)* | *(fill in)* |
 
For Logistic Regression and Ridge several settings tie exactly (for example every `max_iter` gives the same score), which means the solver converges well before the iteration cap and `max_iter` is not an influential choice here. Some best values sit at the **edge of their grid** (for example the smallest `C`, the smallest `max_iter`), so a wider grid might help slightly.
 
### 6.5 Feature differences between the notebooks
 
The notebooks are not byte-for-byte identical in preprocessing, so results should be compared with care:
 
| | KNN / RF (and SVM / DT) notebooks | Logistic / Ridge notebooks |
|---|---|---|
| Crop size | 64 x 64 | 64 x 64 (quadrant), 48 x 96 (teeth) |
| Contrast | CLAHE | `ImageOps.autocontrast` |
| Position | 4 features (normalized centre x, y, width, height) | 2 centre features computed, plus **one-hot true quadrant** (tooth task) and **one-hot true quadrant + tooth number** (abnormality task) |
| HOG | 9 orientations, 8 x 8 cells, 2 x 2 blocks | same |
 
In the Logistic / Ridge tooth and abnormality tasks the quadrant (and tooth number) are given as **ground-truth inputs**. In a real pipeline these would come from the upstream predictions, so those scores are an optimistic upper bound for that setup.
 
---
 
## 7. Results
 
### 7.1 Cross-validation macro-F1 (training part, X-ray-grouped folds)
 
| Model | Quadrant | Tooth numbering | Abnormality |
|---|---|---|---|
| KNN | 0.968 | 0.502 | 0.423 |
| Random Forest | 0.992 | 0.555 | 0.344 |
| Logistic Regression | 0.996 | 0.565 | 0.534 |
| Ridge | 0.998 | 0.571 | 0.507 |
| SVM | *(fill in)* | *(fill in)* | *(fill in)* |
| Decision Tree | *(fill in)* | *(fill in)* | *(fill in)* |
 
### 7.2 Held-out test results (30% of X-rays, unseen)
 
Test sizes for Logistic / Ridge: 832 quadrant regions (208 X-rays), 5,434 teeth, 1,027 abnormal teeth.
 
| Model | Task | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---|---|---|---|
| Logistic Regression | Quadrant | 0.995 | 0.995 | 0.995 | 0.995 |
| Logistic Regression | Tooth numbering | 0.583 | 0.587 | 0.584 | 0.585 |
| Logistic Regression | Abnormality | 0.639 | 0.553 | 0.595 | 0.564 |
| Ridge | Quadrant | 1.000 | 1.000 | 1.000 | 1.000 |
| Ridge | Tooth numbering | 0.588 | 0.591 | 0.588 | 0.588 |
| Ridge | Abnormality | 0.635 | 0.541 | 0.626 | 0.562 |
| KNN, RF, SVM, Logistic Regression | Quadrant | 0.995 | 0.995 | 0.995 | 0.995 |
| KNN, RF, SVM, Logistic Regression | Tooth Numbering | 0.604 | 0.609 | 0.611 | 0.608|
| KNN, RF, SVM, Logistic Regression | Abnormality | 0.767 | 0.697 | 0.504 | 0.493 |

### 7.3 What the results show
- **Quadrant is almost solved** (above 99% for the linear models). The label is closely tied to where the region sits on the image, so it is largely a position problem. Perfect scores on a 832-region test set should be read with that in mind.
- **Tooth numbering is the hard part** (about 0.58 to 0.59 accuracy for the linear models). The confusion matrices show errors **between neighbouring numbers** (1 vs 2, 4 vs 5, 6 vs 7 vs 8), because adjacent teeth look alike.
- **Abnormality detection is weak on the rare classes.** The models mostly separate *Caries* and *Impacted* and confuse *Deep Caries* and *Periapical Lesion* with Caries. Periapical Lesion has only 158 examples in total, so its metrics are noisy.
- **Accuracy and macro-F1 disagree** on the abnormality task (accuracy about 0.64, macro-F1 about 0.56), which is why macro metrics are reported.
---
 
## 8. Part C: Web application
 
A FastAPI backend with a plain HTML/CSS/JavaScript frontend (no external libraries or CDN).
 
### 8.1 Pages
1. **YOLO analysis (`/`)**: upload an OPG. Three tabs show the same X-ray with:
   1. the 4 quadrants,
   2. the FDI number of each tooth,
   3. the abnormalities with FDI, type and confidence.
   A side panel lists the findings and counts. Images open full size on click and can be downloaded.
2. **Traditional models (`/compare`)**: every abnormality found by YOLO is cropped and passed to the six classical models. Each finding gets a card with the crop, YOLO's label, and each model's prediction (green outline = agrees with YOLO), plus per-model agreement bars and the held-out test scores from training. YOLO is the **reference, not ground truth**.
### 8.2 Setup
 
```bash
cd dental_ai
pip install -r requirements.txt
```
 
Place the weights:
 
```
dental_ai/models/quadrant.pt
dental_ai/models/enum.pt
dental_ai/models/disease.pt
dental_ai/models/classical/classical_models.joblib      # for the comparison page
dental_ai/models/imgsz.txt                              # only if you trained at a size other than 1024
```
 
- Copy each YOLO run's `weights/best.pt` and rename it as above. *(Add the link to where the weights are shared, for example a GitHub Release.)*
- Create `classical_models.joblib` by running `python train_classical.py <dataset folder>` (it trains all six models on the abnormality task and also stores their test metrics). Install the **same scikit-learn version** that trained it, otherwise loading can fail.
### 8.3 Run
 
```bash
python run.py                  # http://localhost:8000
```
 
The app checks for the three `.pt` files at startup:
- **All present:** real YOLO inference.
- **Any missing:** **demo mode** with simulated boxes and a visible warning banner. The boxes in demo mode do not follow the X-ray.
- Override with `USE_MOCK=1` (force demo) or `USE_MOCK=0` (force real).
Run the tests (they use the demo detector and a tiny dummy classical model):
 
```bash
pytest
```
 
### 8.4 API
 
| Method | Route | Purpose |
|---|---|---|
| `GET` | `/` | YOLO analysis page |
| `GET` | `/compare` | Traditional-models page |
| `GET` | `/api/health` | Status: demo mode, classical models loaded, colours |
| `POST` | `/api/analyze` | Upload an image (`file`); returns teeth, FDI, findings, image URLs, comparison |
| `GET` | `/api/results/{id}` | Saved result of an earlier analysis |
 
Each analysis is stored in `storage/results/<id>/` (original, three annotated images, crops, `result.json`).
 
### 8.5 How the backend is organized
 
| File | Role |
|---|---|
| `app/services/yolo_engine.py` | Loads the three YOLO models and runs them at the training image size |
| `app/services/mock_engine.py` | Demo detector used when weights are missing |
| `app/services/postprocess.py` | Tooth to quadrant assignment, FDI numbers, duplicate removal, linking abnormalities to teeth, summary counts |
| `app/services/annotate.py` | Draws the three annotated images |
| `app/services/classical.py` | Same HOG + position features as training, then predictions from the six models |
| `config.py` | Paths, confidence thresholds, image size, class colours |
 
### 8.6 Troubleshooting
 
| Problem | Likely cause |
|---|---|
| Yellow "Demo mode" banner | One of the three `.pt` files is missing or misnamed |
| Boxes form a fixed grid | You are in demo mode |
| Few or misplaced detections | `imgsz.txt` does not match the size used for training |
| Comparison page asks for `classical_models.joblib` | The file is not in `models/classical/`, or the app was not restarted |
| Error mentioning `sklearn` or unpickling | scikit-learn version differs from the one that trained the models |
| Slow first upload | Three YOLO models are loading |
 
---
 
## 9. Limitations and future work
 
- **Rare abnormalities** (Deep Caries, Periapical Lesion) are poorly separated by HOG + classical models. Small crops and hand-crafted features hold little information about lesion depth.
- **No healthy class.** The disease annotations contain only abnormal teeth. A "probably healthy" class could be derived from teeth listed in the enumeration data but absent from the disease data, with the caveat that some of them have untagged disease or restorations.
- **Ground-truth inputs.** Several classical experiments use the true quadrant or tooth number as input. End-to-end performance with predicted values will be lower.
- **Detection recall.** The comparison page only sees abnormalities that YOLO finds; anything YOLO misses is not classified.
- **Dataset size.** Few examples for the rare classes make their metrics unstable between folds.
- **Small, single-source data.** Results may not transfer to other X-ray machines or populations.
- Ideas: tooth-context features (rank from the midline within a quadrant), wider hyperparameter grids where best values sit at an edge, oversampling inside cross-validation folds for rare classes, CNN classifiers on the crops, semi-supervised use of the 1,571 unlabelled images, and nested cross-validation for fully unbiased tuning.
---
 
## 10. Reproducibility
 
- Random seed **42** everywhere (splits, models).
- Splits are always by X-ray, never by crop.
- Tuning uses the training part only; the held-out part is evaluated once.
- Record the library versions you used (`pip freeze`), in particular `scikit-learn`, `scikit-image`, `ultralytics` and `torch`.
- Colab sessions are wiped on disconnect. Save weights and `.joblib` files outside `/content` as soon as training ends.
---
 
## 11. Acknowledgements and license
 
- Dataset: **DENTEX**, MICCAI 2023 challenge (cite the dataset authors according to their instructions).
- Libraries: Ultralytics YOLO, scikit-learn, scikit-image, OpenCV, FastAPI, Uvicorn, pandas, NumPy, joblib.
- Team: *(add names and roles: YOLO part, traditional models, web application)*.
- License: *(choose a license for your code, for example MIT; the dataset keeps its own license)*.
 
