# Project Overview

This repoository contains the code related to the FYP ***"Textual Decoding of EEG Patterns"***.

---

## Folder Structure and Contents

### `psychopy/`

This folder contains standalone scripts used to run the experimental procedures and collect the raw data. These files are independent of the data processing and analysis scripts. Properly setting up and running the experiment involves two steps: first, install [PsychoPy](https://www.psychopy.org/download.html), then open and execute the `data_collection.psyexp` using PsychoPy.


### `MI/`, `SI/`, `VEP/`

These folders correspond to their respective datasets:

* **`MI/`**: Contains code for the Motor Imagery experiments. Dataset available [here](https://www.physionet.org/content/eegmmidb/1.0.0/#files-panel).
* **`SI/`**: Contains code for the Speech Imagery experiments. Dataset available [here](https://github.com/JeremyFarrugia/Speech-Imagery-Dataset).
* **`VEP/`**: Contains code for the Visually Evoked Potentials experiments. Dataset available [here](https://data.mendeley.com/datasets/g9shp2gxhy/2).

While the code within these folders is nearly identical in its overall structure and purpose, each folder is kept separate. This is due to **differences in how the raw data from each experiment is formatted**, which necessitates specific handling and processing steps unique to each dataset.

### `visualisation/`

This folder contains scripts specifically designed for **visualising the data and results**. Although these files are intended to be used in conjunction with the `SI/` (Speech Imagery) dataset (should be placed in the same folder), they are kept in a separate folder. This separation makes it easier to locate the visualisation tools without cluttering the main `SI/` directory, while showing distinction between the purpose of the different files.

---