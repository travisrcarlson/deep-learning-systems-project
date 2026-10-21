# Telling Categories of Unexploded Ordnance Apart in Photos

A convolutional neural network (CNN) in PyTorch that sorts crops of unexploded ordnance (UXO), cut from photos of real scenes in the CTX-UXO dataset, into five categories: projectile, grenade, mortar bomb, cartridge and aviation bomb. A second configuration changes one thing, random erasing during training, and both are tested on clear crops and on crops with part of the object hidden.

Photos of suspected items could come from drones flying over an area, from survey teams, or from local people, and a model could pass them to an EOD technician with the beginnings of an understanding, to be validated. Such a tool would generally be used only in war-torn areas or on former battlefields. Every result has to be verified by a technician: items can have nuanced, slight differences between them that change how they function, and so the hazards involved, and the features that tell them apart are often in a part of the item obscured from the photo, covered by brush or partly buried. That is why this project tests what happens when part of each object is hidden. Some false alarms or "unknown" answers are fine; specialists confirming or rejecting the model's answers could also help refine it. This is the fourth project of my AI capstone on humanitarian mine action.

## Result
On the test photos, the baseline reached a macro-F1 of 0.509 across the five categories (accuracy 0.646), and the model trained with random erasing 0.550 (0.678). The random-erasing model stayed ahead with 10% to 40% of each crop hidden, but its lead did not grow as more was hidden, the baseline scored higher on the validation photos, and the 95% interval for the difference, from resampling the test photos, included zero in three of the four test conditions. An earlier run that differed only in floating-point rounding moved the gap by up to 0.045, so the experiment gives no reliable evidence that random erasing helps. Grenades were the easiest category for both models (F1 about 0.8), and most cartridges were missed. With a confidence cut-off set on the validation photos, either model gave a confident answer for about 15% of the test crops, 97% to 98% of them right, and sent the rest to "unknown", together with 98% of the crops from the seven categories it had never seen.

## What's in this repository
- `deep_learning.ipynb`: the problem and data; checks of the data files, the photos, the labels and data quality; preparing the crops; the baseline CNN and its training; the second configuration; the evaluation and comparison on clear and partly hidden crops, with 95% intervals, a check without near-duplicate photos, the "unknown" band, examples and the categories the models never saw; and a summary.
- `data/ctx_uxo_crops_64.zip`: the 15,449 object crops, 64 × 64 pixels, made from the dataset's photos by the code in section 3.1 of the notebook.
- `data/crops_manifest.csv`: one row per crop, with its split, category, source photo and box.
- `data/photos_manifest.csv`: one row per photo, with its split, size, number of objects and image hash.
- `data/sample_photos/`: six of the dataset's training photos, shrunk to 800 pixels on the longer side, for the notebook's first figure.
- `figures/`: the ten figures, written by the notebook.
- `Deep_Learning_Systems_Analysis_Report.pdf` and `module_summary.pdf`: the report. The two files are identical; the brief uses both names.
- `requirements.txt`: the Python environment, from `pip freeze`.

## Dataset
*CTX-UXO: A Comprehensive Dataset for Detection and Identification of UneXploded Ordnances*, by Gheorghe Marian Craioveanu and Grigore Stamatescu of the National University of Science and Technology Politehnica Bucharest, version 1, published on Zenodo on 2025-09-04: https://zenodo.org/records/17052675. The record also shows the DOI 10.21227/cwnm-de53, which leads to the dataset's IEEE DataPort entry.

- Contents: 3,520 photos, mostly from mobile phones, split by the creators into training (2,461), validation (530) and test (529) folders, with 15,449 labelled objects in 12 categories. The record says that some scenes use replicas.
- Download: the archive `CTX_UXO_DS_v2_2025.zip` (3,895,410,390 bytes; MD5 `4191257476e65a50f72dfb9d3ad3213d`, as listed on the record), downloaded on 2026-10-20 from https://zenodo.org/records/17052675/files/CTX_UXO_DS_v2_2025.zip?download=1. It is too large for this repository, so the repository ships the crops made from it.
- Crops: the notebook's section 3.1 holds the code that made them. Put the archive in `data/` and run the notebook, and that section rebuilds the crops into `rebuilt_crops/` and checks that they are identical to the shipped files; two builds gave byte-identical files. The notebook checks the shipped files' SHA-256 checksums when it starts:
  - `ctx_uxo_crops_64.zip`: `9f3cbc88609089864fba9b418c39ec8eb9a97a2451ffdddb2116b1a1695c87c1`
  - `crops_manifest.csv`: `4a4322ff01b9390b14e5a7d416676cb1d6e457cf725e5e2ac10aba04f7e8c306`
  - `photos_manifest.csv`: `53c0c67e9b3d7e704735513f731ab9a437c803e3fd30871d21c385e730eee166`
- Licence: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The crops and the sample photos are adapted from the dataset under that licence: the objects were cut out of the photos and resized, and the sample photos were shrunk.
- Use: none of the dataset's 12 categories is battlefield debris, which is not UXO, mines or other explosive hazards but is found with them on the former battlefields where such a tool would be used, so these models cannot say whether an object is ordnance at all, only which of five categories it looks most like. Their answer is a first indication for an EOD technician to confirm, never a statement that an object is safe, and an answer below the confidence cut-off is "unknown".

## How to run
Needs Python 3.13 and Git. `requirements.txt` was generated on Python 3.13.0 with the CPU build of PyTorch, which comes from PyTorch's own package index; the install line below adds that index.

```
git clone https://github.com/travisrcarlson/deep-learning-systems-project.git
cd deep-learning-systems-project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
jupyter notebook deep_learning.ipynb
```

On macOS or Linux, activate with `source .venv/bin/activate`; on macOS, also change `torch==2.14.1+cpu` in `requirements.txt` to `torch==2.14.1` before installing, since PyTorch publishes no `+cpu` build for macOS. If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once. In Jupyter, choose **Kernel > Restart Kernel and Run All Cells**.

A full run took about 31 minutes on a two-core machine; training the two models takes most of it, and the notebook prints how long each took. Training runs on the CPU with two threads, fixed seeds and deterministic algorithms, so a rerun on the same computer gives the same results. On another computer the scores can differ somewhat, because training amplifies small differences in how processors round sums; section 7.4 of the notebook shows how far such differences moved the results between two runs.
