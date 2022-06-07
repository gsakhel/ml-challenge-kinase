FROM python:3.9.2

WORKDIR python-Dockerfile

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip3 install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+cpu.html

COPY . .

CMD [ "python3", "app.py"]