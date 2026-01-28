# BlueNuclei: Detect and Classify Live/Dead Transfected Neurons

<img src="assets/logo.png"
     width="320"
     alt="BlueNuclei logo"
     align="right"
     vspace="40">

BlueNuclei is a machine-learning–based software designed for neurobiologists without programming experience. It runs fully offline and is distributed as a stand-alone desktop application for Windows and macOS.

A schematic overview of the algorithmic workflow is provided in [`BlueNuclei_workflow.svg`](BlueNuclei_workflow.svg). Full methodological details are described in the accompanying manuscript.

## Supported input
Under the current version (v1.0), only Zeiss .czi files are supported. The input file should be a dual-channel image with DAPI channel (nuclei) and GFP channel (cytoplasm).
## Installation & Usage
### Download
Download the BlueNuclei app (Windows or macOS) from the [GitHub Releases page](https://github.com/ZHAN-ZHA/BlueNuclei/releases), along with the provided sample image toy_image.czi.
### Launch
Windows: double-click `BlueNuclei.exe`

Mac: double-click `BlueNuclei.app`

(On Mac, if a message pops up saying "BlueNuclei not opened: Apple can't verify BlueNuclei is free of malware...": click `Done` - go to `System Settings` on your Mac - go to `Privacy & Security` - scroll down to `"BlueNuclei" was blocked to protect your Mac` - click `Open Anyway`)

After a brief initialization, BlueNuclei launches automatically in your default web browser.
<p align="center">
  <img src="assets/ui.jpg" alt="BlueNuclei UI" width="800">
</p>

### Analyze images
- Click “Run BlueNuclei” in the left menu
- Select one or more .czi image files (or the provided sample)
- Click “Analyze”

For a typical 5000 × 5000 pixel image, analysis completes in ~30 seconds on a standard laptop.
### Visualization & Export
- Click “Save results” to export live/dead counts as a CSV file
- Click “Visualize” to inspect results interactively (synchronized zoom/pan across DAPI and GFP channels). You can move around and zoom in/out. Each predicted nucleus is marked together with its corresponding cytoplasm, with calculated feature values and a confidence score (the number in parenthesis following "Class") displayed as text. The text will appear in red if a nucleus has a confidence score close to the machine learning model's decision boundry (i.e. edge case between "live" and "dead").

<p align="center">
  <img src="assets/ui_2.jpg" alt="BlueNuclei UI" width="800">
</p>

## For developers/advanced users
### Codebase
BlueNuclei is written in Python, built on:
- Flask (local web UI)
- ImageJ / NumPy / SciPy (image processing)
- scikit-learn (SVM)
- PyInstaller (packaging)
The core algorithm is implemented in:
`app/BlueNuclei_utils.py`
### Algorithm overview
BlueNuclei consists of two integrated modules:

“Hyades” module – Nucleus Detection
- Identifies cytoplasmic territories of transfected neurons in the GFP channel using thresholding and shape-based filtering
- Projects these regions onto the DAPI channel
- Extracts enclosed nuclear contours using high-contrast edge pixels

“Pleiades” module – Live/Dead Classification

A linear Support Vector Machine (SVM) classifier trained on five sub-nuclear features designed to mimic expert visual assessment:
- Spottiness
- Spot distribution
- Edge gradient
- Area
- Intensity

Training and evaluation data are available upon request
(contact: zzha2@u.rochester.edu or daniel.taliun@mcgill.ca).
### Build BlueNuclei from source (MacOS example)
```bash
cd your_folder_path

python3 -m venv build_env 
source build_env/bin/activate

pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
pip install pyinstaller waitress

pyinstaller --noconfirm --clean BlueNuclei.spec
```
### Train a custom SVM model
#### Step 1: Annotate nuclei in QuPath
Use [QuPath](https://qupath.github.io/) (an open-source image annotation software) to label and annotate the nuclei in your images with the “wand” or “polygon” tool.
#### Step 2: Export annotated nuclei from QuPath
Run the script [`extras/QuPath_script_1.txt`](extras/QuPath_script_1.txt) in QuPath’s script editor. This generates:
- Image tiles
- Integer-encoded mask images
- roi_mapping.csv
in a standardized ground_truth/ folder structure for each image
#### Step 3: Train the SVM
Run the notebook [`extras/BlueNuclei_train_SVM.ipynb`](extras/BlueNuclei_train_SVM.ipynb). This produces:
- svm_model.pkl
- std_scaler.pkl
- minmax_scaler.pkl
- label_encoder.pkl
- svm_threshold.pkl

These files can be used to rebuild the BlueNuclei application with your custom SVM model.
## Citation
If you use BlueNuclei in your research, please cite our accompanying manuscript (details forthcoming).
## Contact
For questions, data access, or collaboration:
- Zhan Zha – zzha2@u.rochester.edu
- Daniel Taliun – daniel.taliun@mcgill.ca

## License
This repository is released under the Apache 2.0 License for academic and research use.
Commercial support and enterprise licensing are available separately.

