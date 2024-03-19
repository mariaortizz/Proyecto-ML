FROM python:3.11.5 
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN mkdir -p app
COPY ./App app
WORKDIR app
EXPOSE 5000
CMD ["python3", "app.py", "--host", "0.0.0.0", "port", "5000"]

# docker build -t nombre_imagen .

# docker run -d --name nombre-app -p 5001:5001 nombre_imagen 