# CIL Street Segmentation 2021

## TODO
- Create git-ignore  
- Everything else  


## Register on kaggle
On the [CIL website](http://da.inf.ethz.ch/teaching/2021/CIL/) under Semester Project there is a link to the [private kaggle competitions](http://da.inf.ethz.ch/teaching/2021/CIL/files/projects.txt). 
The [link for Road Segmentation](https://www.kaggle.com/t/c5b92ef46fff4ec7b67f619c8e21d1bd) is the one we want.
Sign up for the project with your kaggle account. Once every one of us has an account, we can form the team.


### Environment setup
Create a python environment wiih:  
`python -m venv cil_venv` or `python3 -m venv cil_venv`

To activate the environment:  
`source cil_venv/bin/activate`

To install all "necessary" things do (while inside the cil\_venv):  
`pip install -r requirements.txt`

Then you can execute the test-notebook:  
`jupyter-notebook kaggle_intro.ipynb`

If you don't want to work with notebooks use the file:
`test.py`

Put all other code files in the `tools` folder.

### Adding packages
If you install new packages with pip, add them to the requirements.txt:  
`pip freeze > requirements.txt`


