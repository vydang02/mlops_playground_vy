FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY scripts/ ./scripts/
COPY housing_linear.joblib housing_linear.joblib
ENV PYTHONPATH=/app
EXPOSE 3000
CMD ["python3", "scripts/session_3/api.py"]