FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install flask flask-cors joblib scikit-learn==1.6.1 numpy
COPY . .
CMD ["python", "app.py"]