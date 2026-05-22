# MedTwin

Interpretable breast cancer risk stratification system based on routine clinical data.

MedTwin is a dual-end breast cancer risk management demo system. It provides a patient-side form, doctor-side case queue, and XGBoost-based risk prediction API.

## Local Run

```powershell
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:8000/`.

## Deploy

This project is designed for a Python web host such as Render. GitHub Pages can host only static files, so it cannot run the XGBoost prediction backend.

On Render, use:

- Build command: `pip install -r requirements.txt`
- Start command: `python app.py`
- Health check path: `/api/meta`
