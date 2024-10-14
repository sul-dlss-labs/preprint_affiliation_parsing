<!-- WEASEL: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 Weasel Project: Affiliation Extraction and Parsing from OpenALEX Preprints

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
| `preprints:download` | Download preprint PDFs (**Needs Stanford VPN**) |
| `preprints:clean` | Extract plain text from preprint PDFs |
| `dataset:create` | Create a dataset for annotating affiliations |
| `annotate:ner` | Annotate training data for NER by correcting and updating an existing model |
| `annotate:spans` | Manually annotate training data for spans and relations |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`weasel run [name]`](https://github.com/explosion/weasel/tree/main/docs/cli.md#rocket-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `prepare` | `preprints:download` &rarr; `preprints:clean` &rarr; `dataset:create` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`weasel assets`](https://github.com/explosion/weasel/tree/main/docs/cli.md#open_file_folder-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/preprints.csv` | Git | List of preprint documents in the training dataset |

<!-- WEASEL: AUTO-GENERATED DOCS END (do not remove) -->
