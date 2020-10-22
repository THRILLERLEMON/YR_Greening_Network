# from .setting import *
# from .useful_class import *
# from .useful_class import *
from matplotlib.pyplot import xlim
from setting import *
from useful_class import *
from useful_class import *
import numpy as np
import pandas as pd
import igraph
import pyunicorn.core.network as network
import matplotlib as mpl
mpl.use('cairo')
import matplotlib.pyplot as plt

LABEL_DIC = {
    1: 'Forests',
    2: 'Shrublands',
    3: 'Grasslands',
    4: 'Crop',
    5: 'OT',
    6: 'UB',
    7: 'Water Bodies',
    8: 'Desert and Low-vegetated lands',
}
COLOR_DIC = {
    'Forests': '#1B7201',
    'Shrublands': '#50FF00',
    'Grasslands': '#A4D081',
    'Crop': '#F2F100',
    'OT': '#F29B00',
    'UB': '#F00001',
    'Water Bodies': '#0058F0',
    'Desert and Low-vegetated lands': '#9F9F9F'
}
# plt.savefig(BaseConfig.OUT_PATH + 'LUCC_Net_BigClass//hist.png')