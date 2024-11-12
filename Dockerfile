# python 3.13 and above hit build errors for spaCy dependencies
FROM python:3.12

WORKDIR /usr/src/app

RUN python -m pip install --upgrade pip

# spaCy 3.8+ build runs into errors
RUN python -m pip install spacy==3.7.5

# download preprint spreadsheet
COPY project.yml .
RUN python -m weasel assets

# install dependencies
COPY requirements.txt .
RUN python -m weasel run dependencies:install

# allow these to be cached
COPY scripts scripts
COPY corpus corpus
COPY datasets datasets
COPY configs configs

# prepare the data
RUN python -m weasel run preprints:download
RUN python -m weasel run preprints:clean

# train the text categorizer
RUN python -m weasel run train_textcat

# run visualizer and expose on port 8080
EXPOSE 8080
CMD ["python", "-m", "streamlit", "run", "scripts/visualize.py", "--server.port", "8080"]
