
Deployment package - Retail Sales Forecasting (no Docker)
========================================================

Structure:
- backend/
  - main.py
  - model_utils.py
  - requirements.txt
  - models/   <-- place your model_{STORE}.pkl files here (one per store)
- ui/
  - app.py
  - requirements.txt

Model artifact spec:
Each model file must be a pickled dict with keys:
  - 'model' : trained sklearn/xgboost regressor with .predict(X)
  - 'preprocessor' : trained ColumnTransformer (preprocessor)
  - 'features' : list of feature names (order matters)

Save example (during training):
>>> artifact = {'model': best_model, 'preprocessor': preprocessor, 'features': X_train.columns.tolist()}
>>> import pickle
>>> with open('backend/models/model_CA_1.pkl', 'wb') as f:
...     pickle.dump(artifact, f)

Run backend:
$ cd backend
$ python -m pip install -r requirements.txt
$ uvicorn main:app --reload --port 8000

Run UI (in a new terminal):
$ cd ui
$ python -m pip install -r requirements.txt
$ streamlit run app.py

Notes:
- Ensure Python 3.9+
- The preprocessor must be picklable (sklearn transformers are picklable)
- Keep models local for reliability (don't fetch from external sources at inference time)