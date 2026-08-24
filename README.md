# CIFAR-10 Image Classifier

A CNN trained on CIFAR-10 (10 object classes: airplane, automobile, bird, cat,
deer, dog, frog, horse, ship, truck), served through a Streamlit web app.

## Project structure
```
cifar10-streamlit/
├── app.py                # Streamlit app
├── cifra_10.ipynb        # training notebook (from Colab)
├── cifar10_model.h5      # trained model (you export this from Colab, see below)
├── requirements.txt
├── .gitignore
└── README.md
```

## 1. Export the trained model from Colab
Your notebook trains `model` but never saves it. Add this cell **after training**,
before you leave Colab:
```python
model.save("cifar10_model.h5")
from google.colab import files
files.download("cifar10_model.h5")
```
Save the downloaded `cifar10_model.h5` into this project folder.

> Note: `.h5` files are excluded by `.gitignore` since trained models are often
> too large for a normal GitHub push (100MB hard limit). If your file is under
> ~25MB you can remove `*.h5` from `.gitignore` and commit it directly.
> Otherwise, use [Git LFS](https://git-lfs.com/) or host the model separately
> (Google Drive / Hugging Face Hub) and download it at app startup.

## 2. Run the app locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open the local URL Streamlit prints (usually http://localhost:8501).

## 3. Deploy for free (Streamlit Community Cloud)
1. Push this repo to GitHub (steps below).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click "New app", pick this repo, branch `main`, and file `app.py`.
4. Deploy. If your model is too large for GitHub, add code in `app.py` to
   download it from a hosted link on first run.
