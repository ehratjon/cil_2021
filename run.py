import os
import time

models = [
    'UNet', 
    'UNetDilated',
    'UNetSpatial',
    'UNetSpatialDilated',

    'NestedUNet',
    'NestedUNetDilated',
    'NestedUNetSpatial',
    'NestedUNetSpatialDilated',
    
    'UNet_3Plus',
    'UNet_3Plus_Dilated',
    'UNet_3Plus_Spatial',
    'UNet_3Plus_Spatial_Dilated'
]

models_fix = [
    'UNet_fix', 
    'UNetDilated_fix',
    'UNetSpatial_fix',
    'UNetSpatialDilated_fix',
    'NestedUNet_fix',
    'NestedUNetDilated_fix',
    'NestedUNetSpatial_fix',
    'NestedUNetSpatialDilated_fix',
    'UNet_3Plus_fix',
    'UNet_3Plus_Dilated_fix',
    'UNet_3Plus_Spatial_fix',
    'UNet_3Plus_Spatial_Dilated_fix'
]

for model in models:    
    os.system(f'bsub -n 8 -W 12:00 -R "rusage[mem=4096, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -o {model + ".log"} "python {model}.py"')
    time.sleep(1)
