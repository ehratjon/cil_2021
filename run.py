import os


models = [
    'UNet', 
    'UNetDilated'
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

for model in models:    
    os.system(f'bsub -n 8 -W 04:00 -R "rusage[mem=4096, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -o {model} "python {model}.py"')
