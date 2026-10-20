# Hugging Face Space (Docker SDK): serves the ABM interface from example.py on port 7860.
# Local test: docker build -t imbanuma-abm . && docker run -p 7860:7860 imbanuma-abm
FROM python:3.12-slim

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

USER user
ENV HOME=/home/user \
    EXAMPLE_UI=abm \
    MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1

EXPOSE 7860
CMD ["solara", "run", "example.py", "--host", "0.0.0.0", "--port", "7860", "--production"]
