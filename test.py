import numpy as np
import kaggle

# all code files used are stored in a tools folder
# this allows us to directly import those files
import sys
sys.path.append("cil_data")
sys.path.append("models")
sys.path.append("tools")

import data

dataset = data.RoadSegmentationDataset()