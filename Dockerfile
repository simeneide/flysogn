# app/Dockerfile

FROM python:3.12-slim



RUN apt-get update && \
    apt-get install -y libhdf5-serial-dev netcdf-bin libnetcdf-dev 
# libgdal-de

WORKDIR /app
COPY requirements.txt .

RUN pip install -r requirements.txt --verbose

COPY . .

ENTRYPOINT ["streamlit", "run", "--server.port=80", "--server.address=0.0.0.0", "--server.headless=true", "vestavind.py"]

EXPOSE 80

#HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

