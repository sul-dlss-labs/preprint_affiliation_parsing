<!-- WEASEL: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 Weasel Project: Affiliation Extraction and Parsing from OpenALEX Preprints

## Setup
To use the workflow in this project, you need to start by installing spaCy, ideally in a new virtual environment:
```sh
pip install spacy
```
Then, before running any of the workflows, make sure to install the dependencies by running the following command:
```sh
weasel run dependencies:install
```
This will ensure you have the latest copies of pretrained models as well as the `thinc-apple-ops` package if you are running on an M-series Mac, which significantly speeds up training times.

## Data
Preprint data was sourced from [sul-dlss-labs/preprints-evaluation-dataset](https://github.com/sul-dlss-labs/preprints-evaluation-dataset). You can pull down a copy of the spreadsheet listing all preprints with:
```sh
weasel assets
```
This command will check the CSV file's checksum to ensure you are using the same copy of the data as the project was developed with. To fetch the actual PDF files and their metadata and convert them to plaintext, run:
```sh
weasel run preprints:download
weasel run preprints:clean
```
Note that **you need to be on Stanford VPN** to fetch files from SDR.

## Annotating
If you have a local copy of [prodigy](https://prodi.gy/), you can use it to annotate the data for training. To create a dataset for text categorization, run:
```sh
weasel run dataset:textcat:create
```
Then, you can annotate the data using:
```sh
weasel run annotate:textcat
```
To create a dataset for named entity recognition, run:
```sh
weasel run dataset:ner:create
```
And then annotate the data using:
```sh
weasel run annotate:ner
```
The annotation tasks will open in your browser, and you can use the Prodigy UI to annotate the data.

## Training
Once you have annotated data, you can train a text categorization model using:
```sh
weasel run train_textcat
```
And a named entity recognition model using:
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
There are several interfaces built with [Streamlit](https://streamlit.io/) to help debug the various parts of the pipeline. You can run these with:
```sh
streamlit run scripts/visualize_blocks.py # pdf-to-text conversion
streamlit run scripts/visualize_textcat.py # text categorization
streamlit run scripts/visualize_ner.py # named entity recognition
streamlit run scripts/visualize_relations.py # affiliation relations/graph
```
These scripts will open a browser window with the Streamlit interface, allowing you to interact with the data and the models.


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
