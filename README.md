# FloodSense AI

FloodSense AI is a machine learning project that predicts flood occurrence using environmental, geographic, and infrastructure-related features.

## Dataset
- 10,000 rows
- Includes rainfall, humidity, temperature, water level, elevation, land cover, soil type, and more
- Target: Flood Occurred

## Project Structure
- `src/train_model.py` trains the model
- `src/preprocess.py` handles preprocessing
- `src/predict.py` tests prediction
- `src/app.py` is the Streamlit app

## Model
Current model: Random Forest Classifier