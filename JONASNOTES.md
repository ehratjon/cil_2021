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
  - ~~how to handle that training images = 400x400pixels and test images = 608x608?~~
  - ~~no rescaling -> this might result in unwated artifacts...~~
  - ~~implement test dataset~~
  - ~~implement transformers as in: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html~~
  - ~~make sure all transformers also work if groundtruth is none!~~
  - ~~additionally go through these links to have a full data loader finished~~
    - ~~https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html~~
    - ~~https://pytorch.org/tutorials/beginner/basics/data_tutorial.html~~
  - ~~implement flip~~
- ~~test data loader~~
  - ~~maybe rescale needs real images instead of numpy images~~
  - ~~(https://stackoverflow.com/questions/26681756/how-to-convert-a-python-numpy-array-to-an-rgb-image-with-opencv-2-4)~~
- ~~write model training~~
  - ~~read through https://pytorch.org/tutorials/beginner/basics/intro.html~~
  - ~~how to split data into train/eval for neural network with dataloader~~
  - ~~does this all really work???????????????????????????~~
  - ~~write model that always guesses black~~
  - ~~write model with one node or so (added to next todo list)~~
  - ~~set all global seeds (pytorch, numpy, etc...)~~
  - ~~choose a good simplest model for first baseline~~
  - ~~explain why this model~~
- ~~create own branch, mike said so~~
- ~~MAKE DATA FLOAT FROM BEGINNING OR ELSE SHIET~~
  - ~~add in make_tensor transformer~~
  - ~~or add make_float_tensor as transformer~~
  - ~~then remove everything u made to float~~
- ~~find out why gradient not working~~
- ~~print all params with values + grads~~
  - ~~#for name, param in model.named_parameters():~~
    ~~#print(name, param.data, param.grad)~~
    ~~#print(f"weights {model.weights[0][0]:>5f}, {model.weights[1][0]:>5f}, {model.weights[2][0]:>5f}")~~
    ~~#print(model.weights.grad)~~
    ~~print(f"batch number: {batch_number:>3d} loss for this batch: {loss.item():>7f}")~~
  - make it print multiple eval images by giving it as arugment
- reintroduce correct loss functions
  - make sure to test it out completely!!!!
- make sure loss functions work with batches and not with single sample
- before continuing with test cases start with new model
  - need to find out how loss function needs to work such that we optimize correclty
  - create one node model
- lightning files afoh
- "schlechti" bilder usem dataset usenäh
  - die wo keini strosse druf hend????
  - colouring anderst mache bi wasser
- google neural network to fix heatmaps
  - als zweits neurals netzwerk wo dheatmaps fixed
- alli neural networks vom mike code übernäh
- write test cases to test every component alone
  - really seems like convert_into_label_list_for_patches works strangely...
  - can be used to then compute torch.eq(a,b) to compute accuracy
  - can be used to simplify the loss function
  - leave old loss function alone and try to write a new one that is simpler and 
    works with only tensor operations
- write more intuitive model with grad, since testing with no grad is difficult
  - only one node
- write test cases for new model and better visualizations to get a glimpse of things
- remove test.py and include these things in train.py
- go through train.py again and check if the code makes sense
- we want our predictions to be between 0 and 1. make sure that is the case
  - with 0.25 being the threshhold
- first research: What do you need to make a first submission (do while coding)
- Make a notebook that loads,trains,tests and submits code
- "implement baseline"
  - choose a good simplest model for first baseline
  - explain why this model
- write model testing
- use `kaggle competitions submit -c cil-road-segmentation-2021 -f submission.csv -m "Message"`
- look at mask_to_submission.py and submission_to_mask.py
- write submission
- submit project
- Change from pytorch to pytorch lightning
- Comment everything
- More Research
  - Understand submission format, maybe can adapt evaluation to that
    - read make_submission code
    - don't quite understand: A simple baseline is to partition an image into a set of patches and classify every patch according to some simple features (average intensity). Although this can produce reasonable results for simple images, natural images typically require more complex procedures that reason abut the entire image or very large windows. 
    - patches are 16x16 pixels. maybe can be used to optimise (don't need to classify each pixel itself but rather each patch)
    - TA wrote we can use https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html with average="samples" 
  - Search for similar stuff online (spatial u-nets etc.)
  - Read through lecture notes and try to improve
  - Read through all piazza posts and try to improve
  - go through other tutorials to add more visualization or simplicity etc...
    - like https://pytorch.org/tutorials/beginner/basics/intro.html
- Find novel solution
  - debug
  - optimise to data 
  - how to handle that training images = 400x400pixels and test images = 608x608?
    - find out: looks like houses in training set are a bit larger than in the testing set, but could be also same size
  - find other data augmentations than flip etc. 
    - look at skimage.transform.___ might have some interesting warps etc.
    - look at https://www.analyticsvidhya.com/blog/2019/09/9-powerful-tricks-for-working-image-data-skimage-python/
- Find baselines
  - You must compare your solution to at least two baseline algorithms. For the baselines, you can use the implementations you developed as part of the programming exercises or come up with your own relevant baselines.
- Improve novel solution
  - read https://docs.google.com/document/d/1T5EjTYempPQng1BecGolbLtN5LtCL_xwq2PmmM-mAJ0/edit again to see what might be missing
  - use pytorch lightning instead of pytorch
- Write report
  - see https://docs.google.com/document/d/1T5EjTYempPQng1BecGolbLtN5LtCL_xwq2PmmM-mAJ0/edit to get an overview of what it needs to include
  - additionally, it says in the file above we should receive a pdf on how to write papers

## Pipeline

### Ideas
- how to handle that training images = 400x400pixels and test images = 608x608?
  - use window size of 16x16 pixels (like the patches)
    - postprocessing is needed
    - first neural network that computes to many roads on each patch
    - postprocessing removes all roads that do not align over different patches
  - use window size of 400x400 pixels
    - no postprocessing for training
    - take best of 4 for test images

## Sources

### Project info
- Overall project requirements: https://docs.google.com/document/d/1T5EjTYempPQng1BecGolbLtN5LtCL_xwq2PmmM-mAJ0/edit
- Road segmentation project: https://docs.google.com/document/d/1HDgadE9_5mBJTQGHFZScMYSvLENh-RcR-HWFobdra3I/edit
- kaggle: https://www.kaggle.com/c/cil-road-segmentation-2021
- kaggle join link: https://www.kaggle.com/t/c5b92ef46fff4ec7b67f619c8e21d1bd
- repository: https://github.com/ehratjon/cil_2021

### Approaches
- spatial dependent u-nets: https://www.researchgate.net/publication/350311259_Spatially_Dependent_U-Nets_Highly_Accurate_Architectures_for_Medical_Imaging_Segmentation
- application thereof: https://github.com/aschneuw/road-segmentation-unet

## Code

### Setup
- create a python environment wiih: `python -m venv cil_venv` or `python3 -m venv cil_venv`
- install requirements: `pip install -r requirements.txt`

### Start coding
- activate venv: `source ../cil_venv/bin/activate`
- start a notebook: `jupyter-notebook kaggle_intro.ipynb`
- add packages: `pip install numpy` and afterwards: `pip freeze > requirements.txt`

### Download data
- create and download api token: https://www.kaggle.com/docs/api
- download data into folder `kaggle competitions download -c cil-road-segmentation-2021`
- unzip data `unzip cil-road-segmentation-2021.zip -d cil_data` (take care that this folder might already exist in our repo since it also contains files)
- if already present this can be done with
``` 
unzip cil-road-segmentation-2021.zip -d cil_data2 
mv cil_data2/test_images cil_data/test_images
mv cil_data2/training cil_data/training
mv cil_data2/sample_submission.csv cil_data/sample_submission.csv
rm -r cil_data2
```

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
