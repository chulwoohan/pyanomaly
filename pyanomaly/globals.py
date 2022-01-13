from .datatypes import iterstruct
from .config import *
from .log import *
from .util import *

# weight_scheme
VW = 'vw'  # value-weight
EW = 'ew'  # equal-weight

# data frequency
ANNUAL = 1
QUARTERLY = 4
MONTHLY = 12
DAILY = 365

# pyanomaly frequency to pandas frequency symbols.
freq_map = {
    ANNUAL: 'Y',
    QUARTERLY: 'Q',
    MONTHLY: 'M',
    DAILY: 'D'
}

# Colour codes
# https://github.com/vega/vega/wiki/Scales#scale-range-literals
COLOR10 = iterstruct(
    blue = '#1f77b4',
    orange = '#ff7f0e',
    green = '#2ca02c',
    red = '#d62728',
    purple = '#9467bd',
    brown = '#8c564b',
    pink = '#e377c2',
    grey = '#7f7f7f',
    mustard = '#bcbd22',
    sky = '#17becf'
)

COLOR20 = iterstruct(
    blue1 = '#1f77b4',
    blue0 = '#aec7e8',
    orange1 = '#ff7f0e',
    orange0 = '#ffbb78',
    green1 = '#2ca02c',
    green0 = '#98df8a',
    red1 = '#d62728',
    red0 = '#ff9896',
    purple1 = '#9467bd',
    purple0 = '#c5b0d5',
    brown1 = '#8c564b',
    brown0 = '#c49c94',
    pink1 = '#e377c2',
    pink0 = '#f7b6d2',
    grey1 = '#7f7f7f',
    grey0 = '#c7c7c7',
    mustard1 = '#bcbd22',
    mustard0 = '#dbdb8d',
    sky1 = '#17becf',
    sky0 = '#9edae5',
)

