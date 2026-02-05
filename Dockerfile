FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#COPY mcmc.py .
COPY mmm/ ./mmm/
# COPY run.py .
COPY app.py .
COPY mmm_contributions.csv .
COPY mmm_roas_summary.csv .

RUN mkdir -p /app/output

# Run MMM once at build time OR container start time.
# For take-home: better to run at container start so results are fresh.
EXPOSE 8000

CMD ["bash", "-lc", "python app.py"]
