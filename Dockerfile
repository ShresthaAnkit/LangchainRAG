FROM python:3.11-slim

WORKDIR /app

# RUN useradd -m appuser

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install uv \
    && uv pip install --system -r requirements.txt

COPY . .

# RUN chown -R appuser:appuser /app

# USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
