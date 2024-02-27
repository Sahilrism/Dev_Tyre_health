FROM checkexploreai/global_tyre:v1

RUN pip3 install PyMySQL

COPY eval.py eval.py
COPY backbone.py backbone.py
COPY yolact.py yolact.py
COPY detect.py detect.py
COPY utils utils
COPY data data
COPY scripts scripts
COPY models models
COPY layers layers
COPY runs runs
COPY utils utils
COPY ./app /app
COPY ./uploaded_files /uploaded_files

CMD [ "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8091" ]

# CMD [ "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "443" , "--ssl-keyfile","app/bridgestoneprod.checkexplore.com.key" , "--ssl-certfile","app/certificate.crt"]


