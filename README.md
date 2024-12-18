<!-- WEASEL: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 Weasel Project: Affiliation Extraction and Parsing from OpenALEX Preprints

## Design
This project uses spaCy to extract and parse affiliations from preprints in PDF format. The pipeline consists of several components:

- **PDF-to-text conversion**: Extracts plain text blocks from PDF files using [pyMuPDF](https://pymupdf.readthedocs.io/en/latest/).
- **Text categorization**: Predicts text blocks containing affiliations using a binary text categorization model so they can be extracted separately.
- **Named entity recognition**: Parses the extracted text blocks to identify named entities like organizations, people, and locations.
- **Relation extraction**: Builds a graph of affiliations and their relationships to authors and institutions.

For a full overview of the project, including possible next steps, see [the report](https://docs.google.com/document/d/1kdqEFBh0IolQUq-xybDajOPejoLRKy1cF4b7LycTv60/edit?usp=sharing).

## Setup
To use the workflow in this project, you need to start by installing spaCy, ideally in a new virtual environment:
```sh
pip install spacy
```
Then, before running any of the workflows, make sure to install the dependencies by running the following command:
```sh
spacy project run dependencies:install
```
This will ensure you have the latest copies of pretrained models as well as the `thinc-apple-ops` package if you are running on an M-series Mac, which significantly speeds up training times.

## Data
Preprint data was sourced from [sul-dlss-labs/preprints-evaluation-dataset](https://github.com/sul-dlss-labs/preprints-evaluation-dataset). You can pull down a copy of the spreadsheet listing all preprints with:
```sh
spacy project assets
```
This command will check the CSV file's checksum to ensure you are using the same copy of the data as the project was developed with. To fetch the actual PDF files and their metadata and convert them to plaintext, run:
```sh
spacy project run preprints:download
spacy project run preprints:clean
```
Note that **you need to be on Stanford VPN** to fetch files from SDR.

## Annotating
Annotated training datasets are stored in the [datasets/](datasets/) directory. They have also been pre-exported in the binary format used by spaCy in the [corpus/](corpus/) directory and are already configured as training data in the `config.cfg` files in the `configs/[ner|textcat]` directory.

If you have a local copy of [prodigy](https://prodi.gy/), you can use it to annotate more data for training. To recreate the dataset for text categorization, run:
```sh
weasel run dataset:textcat:create
```
Then, you can annotate the data using:
```sh
weasel run annotate:textcat
```
To recreate the dataset for named entity recognition, run:
```sh
weasel run dataset:ner:create
```
And then annotate the data using:
```sh
weasel run annotate:ner
```
The annotation tasks will open in your browser, and you can use the Prodigy UI to annotate more data.

## Training
You can train the text categorization model using:
```sh
weasel run train_textcat
```
And the named entity recognition model using:
```sh
weasel run train_ner
```
These commands are parameterized by the `embedding` and `transformer_model_name` variables in the project.yml file. You can choose to use spaCy's built-in token-to-vector embedding layer (`tok2vec`) or a transformer-based model (`transformer`). If you choose to use a transformer-based model, you can specify the model name (`transformer_model_name`) from the Hugging Face model hub.

If you change these parameters and re-run the training command, spaCy saves the results in the `training/` directory. The best-scoring run and the last run are always saved to `training/[ner|textcat]/model-best` and `training/[ner|textcat]/model-last`, respectively.

You can further customize the training configuration by editing the `config_[embedding].cfg` file in the `configs/[ner|textcat]` directory. If you do this, it's usually worth it to run:
```sh
spacy debug config configs/[ner|textcat]/config_[embedding].cfg
```
This will check the configuration file for errors and print out a summary of the settings.

## Visualizing
There are several interfaces built with [Streamlit](https://streamlit.io/) to help debug the various parts of the pipeline. You can view these with:
```sh
streamlit run scripts/visualize.py
```
This will open a browser window with the Streamlit interface, allowing you to preview different data and model parameters.

## API
There is an example API built with [FastAPI](https://github.com/fastapi) that processes a PDF file sent as form data and returns predicted affiliations as JSON. You can start a development server with:
```sh
fastapi dev scripts/api.py
```
Then, POST a PDF file using:
```sh
curl -F file=@assets/preprints/pdf/W2901173781.pdf "http://localhost:8000/analyze"
```


## 📋 project.yml

The [`project.yml`](project.yml) defines the data assets required by the
project, as well as the available commands and workflows. For details, see the
[Weasel documentation](https://github.com/explosion/weasel).

### ⏯ Commands

The following commands are defined by the project. They
can be executed using [`weasel run [name]`](https://github.com/explosion/weasel/tree/main/docs/cli.md#rocket-run).
Commands are only re-run if their inputs have changed.

| Command | Description |
| --- | --- |
| `dependencies:install` | Install dependencies |
| `preprints:download` | Download preprint PDFs and metadata (**Needs Stanford VPN**) |
| `preprints:clean` | Extract plain text from preprint PDFs |
| `dataset:textcat:create` | Create a dataset for annotating text categorization training data |
| `annotate:textcat` | Annotate binary training data for text categorization |
| `dataset:ner:create` | Create a dataset for annotating NER data |
| `annotate:ner` | Annotate training data for NER by correcting and updating an existing model |
| `train_textcat` | Train spaCy text categorization pipeline for affiliation extraction |
| `train_ner` | Train spaCy NER pipeline for affiliation parsing |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`weasel run [name]`](https://github.com/explosion/weasel/tree/main/docs/cli.md#rocket-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `prepare` | `preprints:download` &rarr; `preprints:clean` &rarr; `dataset:textcat:create` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`weasel assets`](https://github.com/explosion/weasel/tree/main/docs/cli.md#open_file_folder-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/preprints.csv` | Git | List of preprint documents in the training dataset |

<!-- WEASEL: AUTO-GENERATED DOCS END (do not remove) -->
