# app/Dockerfile

FROM python:3.11



RUN apt-get update && \
    apt-get install -y libhdf5-serial-dev netcdf-bin libnetcdf-dev 
# libgdal-de

WORKDIR /app
COPY requirements.txt .

RUN pip install -r requirements.txt --verbose

COPY . .

ENTRYPOINT ["streamlit", "run", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableWebsocketCompression=false", "--server.enableXsrfProtection=false", "--server.headless=true", "vestavind.py"]

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

