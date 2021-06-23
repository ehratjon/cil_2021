# Computational Intelligence Lab

## TODO
- Do project
- While doing the project, go through all lectures
- Write some summary of the lecture notes here
- Stonks

## Exam
- Mi 11.08.	15:30-17:30	263-0008-00S Computational Intelligence Lab
- The written exam takes 120 minutes. The language of examination is English. NO WRITTEN AIDS ALLOWED.

## Sources
- Website: http://www.vvz.ethz.ch/Vorlesungsverzeichnis/lerneinheit.view?semkez=2021S&ansicht=ALLE&lerneinheitId=149117&lang=de
- Course catalogue: http://www.vvz.ethz.ch/Vorlesungsverzeichnis/lerneinheit.view?semkez=2021S&ansicht=ALLE&lerneinheitId=149117&lang=de
- Recordings: https://polybox.ethz.ch/index.php/s/SnwEeciNNmrgLPc ; PW: CIL2021
- Piazza: https://piazza.com/ethz.ch/spring2021/263000800l
- Lecture notes and exercises on website
- Old exams on website
- Reading material on website

# Road Segmentation (deadline 31. July 23:59)

## TODO
- ~~Download data~~
  - ~~Download data from kaggle~~
  - ~~Add gitignore as to not upload data~~
- Implement Baseline
  - write data loader
    - ~~install pytorch, pandas, sckit_image and pip freeze those~~
    - ~~push everything then delte .vscode folder locally~~
    - ~~how to handle that training images = 400x400pixels and test images = 600x6000?~~
    - ~~no rescaling -> this might result in unwated artifacts...~~
    - implement transformers as in: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    - additionally go through these links to have a full data loader finished
      - https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html
      - https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
  - write model training
  - first research: What do you need to make a first submission (do while coding)
  - Make a notebook that loads,trains,tests and submits code
  - "implement baseline"
  - write model testing
  - use `kaggle competitions submit -c cil-road-segmentation-2021 -f submission.csv -m "Message"`
  - look at mask_to_submission.py and submission_to_mask.py
  - write submission
  - submit project
- More Research
  - Understand submission format, maybe can adapt evaluation to that
    - read make_submission code
    - don't quite understand: A simple baseline is to partition an image into a set of patches and classify every patch according to some simple features (average intensity). Although this can produce reasonable results for simple images, natural images typically require more complex procedures that reason abut the entire image or very large windows. 
    - patches are 16x16 pixels. maybe can be used to optimise (don't need to classify each pixel itself but rather each patch)
    - TA wrote we can use https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html with average="samples" 
  - Search for similar stuff online (spatial u-nets etc.)
  - Read through lecture notes and try to improve
  - Read through all piazza posts and try to improve
- Find novel solution
  - debug
  - optimise to data 
- Find baselines
  - You must compare your solution to at least two baseline algorithms. For the baselines, you can use the implementations you developed as part of the programming exercises or come up with your own relevant baselines.
- Improve novel solution
  - read https://docs.google.com/document/d/1T5EjTYempPQng1BecGolbLtN5LtCL_xwq2PmmM-mAJ0/edit again to see what might be missing
- Write report
  - see https://docs.google.com/document/d/1T5EjTYempPQng1BecGolbLtN5LtCL_xwq2PmmM-mAJ0/edit to get an overview of what it needs to include
  - additionally, it says in the file above we should receive a pdf on how to write papers

## Pipeline

### Ideas
- ?

## Sources

### Project info
- Overall project requirements: https://docs.google.com/document/d/1T5EjTYempPQng1BecGolbLtN5LtCL_xwq2PmmM-mAJ0/edit
- Road segmentation project: https://docs.google.com/document/d/1HDgadE9_5mBJTQGHFZScMYSvLENh-RcR-HWFobdra3I/edit
- kaggle: https://www.kaggle.com/c/cil-road-segmentation-2021
- kaggle join link: https://www.kaggle.com/t/c5b92ef46fff4ec7b67f619c8e21d1bd
- repository: https://github.com/ehratjon/cil_2021

## Code

### Setup
- create a python environment wiih: `python -m venv cil_venv` or `python3 -m venv cil_venv`
- install requirements: `pip install -r requirements.txt`

### Start coding
- activate venv: `source ../cil_venv/bin/activate`
- start a notebook: `jupyter-notebook kaggle_intro.ipynb`
- add packages: `pip install numpy` and afterwards: `pip freeze > requirements.txt`

## Data

### Training data
- 100 aerial images acquired from GoogleMaps
- Ground truth: image with 0=bg, 1=road

## Hand in
- Novel solution by combining and extending previous work.
- Compare your novel solution to at least two baseline algorithms.
- Methodology and experimental results in the form of a short scientific paper. (max 4 pages)

### Evaluation
- Prediction accuracy, measured by fraction of correctly predicted patches (= correctly predicted patches / number of patches)
- Patches = 16x16 pixels      

### Submission details
- You have 8 submissions every 24 hours
- The first column corresponds to the image id followed by the x and y top-left coordinate of the image patch (16x16 pixels)
- The second column is the label assigned to the image patch
- 135736 predictions
- csv file wanted (or archive thereof)