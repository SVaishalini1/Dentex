# Dental OPG Analyzer (YOLO + traditional models)

Upload an OPG -> 3 YOLO models (quadrant, tooth number, disease) -> 3 images:
1. OPG divided into 4 quadrants  2. FDI number of each tooth  3. abnormalities per tooth.
Second page: what Logistic Regression, Ridge, Random Forest, SVM, KNN and Decision Tree predict
for the same abnormality crops.

## Structure
```
dental_ai/
├── run.py  config.py  requirements.txt  train_classical.py
├── models/
│   ├── quadrant.pt  enum.pt  disease.pt        <- best.pt of your 3 YOLO runs, renamed
│   └── classical/classical_models.joblib       <- made by train_classical.py
├── app/
│   ├── main.py                    API + page routes
│   └── services/
│       ├── yolo_engine.py         your 3 YOLO models (imgsz 1024, as trained)
│       ├── mock_engine.py         demo detector (USE_MOCK=1)
│       ├── postprocess.py         tooth -> quadrant -> FDI, abnormality -> tooth
│       ├── annotate.py            draws the 3 images
│       └── classical.py           HOG+position features, runs the 6 models
├── static/  index.html (YOLO page)  compare.html (traditional models)  style.css
├── storage/results/               one folder per analysis
└── tests/test_api.py
```

## Setup (quick)
Run `Train_Everything_Colab.ipynb` on Colab (T4 GPU), download `models_bundle.zip`, extract it INTO this app's `models/` folder, then `python run.py`.
The app switches to the real YOLO models automatically when the weights are present; otherwise it shows a DEMO MODE banner.

## Setup (manual)
1. YOLO weights: copy `runs/quadrant/weights/best.pt` -> `models/quadrant.pt`, same for `enum` -> `models/enum.pt`
   and `disease` -> `models/disease.pt`.
2. Traditional models (Colab, after the dataset is downloaded): `python train_classical.py /content`
   then copy `models/classical/classical_models.joblib` into the app. Install the same scikit-learn version it prints.
3. `pip install -r requirements.txt`

## Run
```
python run.py                  # demo mode (simulated detections): http://localhost:8000
USE_MOCK=0 python run.py       # your YOLO models
pytest
```

## How FDI is built
Tooth-number model gives 1-8, the quadrant model's box containing the tooth gives 1-4;
FDI = quadrant x 10 + tooth number. A disease box is attached to the tooth it overlaps most.
