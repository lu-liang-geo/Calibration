import glob

'''
Set LC Sensor Variables to Use
'''

# Variable Ref:
# https://github.com/bomeara/purpleairpy/blob/master/api.md

lcVars = [
    'current_temp_f',       # Current temperature in F.
    'current_humidity',     # Current Humidity in %.
    # 'current_dewpoint_f',   # Calculated dew point in F.
    # 'pressure',             # Current pressure in millibars.
    # 'pm1_0_atm',            # Channel A ATM PM1.0 particulate mass in ug/m3
    'pm2_5_atm',            # Channel A ATM PM2.5 particulate mass in ug/m3
    # 'pm10_0_atm',           # Channel A ATM PM10.0 particulate mass in ug/m3
    # 'pm1_0_cf_1',           # Channel A CF=1 PM1.0 particulate mass in ug/m3
    'pm2_5_cf_1',           # Channel A CF=1 PM2.5 particulate mass in ug/m3
    # 'pm10_0_cf_1',          # Channel A CF=1 PM10.0 particulate mass in ug/m3
    # 'p_0_3_um',             # Channel A 0.3 um particle counts per deciliter of air
    # 'p_0_5_um',             # Channel A 0.5 um particle counts per deciliter of air
    'p_1_0_um',             # Channel A 1.0 um particle counts per deciliter of air
    'p_2_5_um',             # Channel A 2.5 um particle counts per deciliter of air
    'p_5_0_um',             # Channel A 5.0 um particle counts per deciliter of air
    'p_10_0_um',            # Channel A 10.0 um particle counts per deciliter of air
    # 'pm1_0_atm_b',          # Channel B ATM PM1.0 particulate mass in ug/m3.
    'pm2_5_atm_b',          # Channel B ATM PM2.5 particulate mass in ug/m3
    # 'pm10_0_atm_b',         # Channel B ATM PM10.0 particulate mass in ug/m3
    # 'pm1_0_cf_1_b',         # Channel B CF=1 PM1.0 particulate mass in ug/m3
    'pm2_5_cf_1_b',         # Channel B CF=1 PM2.5 particulate mass in ug/m3
    # 'pm10_0_cf_1_b',        # Channel B CF=1 PM10.0 particulate mass in ug/m3
    # 'p_0_3_um_b',           # Channel B 0.3 um particle counts per deciliter of air
    # 'p_0_5_um_b',           # Channel B 0.5 um particle counts per deciliter of air
    'p_1_0_um_b',           # Channel B 1.0 um particle counts per deciliter of air
    'p_2_5_um_b',           # Channel B 2.5 um particle counts per deciliter of air
    'p_5_0_um_b',           # Channel B 5.0 um particle counts per deciliter of air
    'p_10_0_um_b',          # Channel B 10.0 um particle counts per deciliter of air
]

# Names of sites to keep during sample size analysis
siteToKeep = [
    'AK4',
    'AK1',
    'AZ4',
    'AZ3',
    'CA2',
    'CA13',
    'CO1',
    'FL1',
    'GA1',
    'IA3',
    'IA2',
    'KS1',
    'KS2',
    'NC4',
    'OK1',
    'UNT-GEO',
    'VT1',
    'WA1',
    'WA3',
    'WI2',
    'WI1',
]


''' 
Set Features for Dataset Tests
'''

datasetFeatures = [
    [
        'PM25atm', 'PM25FM'
    ],
    [
        'PM25atm', 'b2.5um', 'PM25FM'
    ],
    [
        'PM25atm', 'RH', 'b2.5um', 'PM25FM'
    ],
    [
        'PM25atm', 'RH', 'TempC', 
        'b2.5um', 'PM25FM'
    ],
    [
        'PM25atm', 'RH', 'TempC',
        'b1um', 'b2.5um', 'b5um', 'b10um', 'PM25FM'
    ]
]


''' 
Set Algorithm Parameters
'''
# Shape of hidden layers
nn_params = [
    [(50,)], 
    [(50, 50)], 
    [(50, 50, 50)],
    [(100,)], 
    [(100, 100)], 
    [(100, 100, 100)],
    [(256,)], 
    [(256, 256)], 
    [(256, 256, 256)],
    [(512,)], 
    [(512, 512)], 
    [(512, 512, 512)]
]

# Kernels (Linear, Polynomial, Radial Basis Function)
svm_params = [
    ['linear'],
    ['poly'],
    ['rbf']
]

# Number of trees
rf_params = [
    [1],
    [2],
    [4],
    [7],
    [20],
    [40],
    [60],
    [80],
    [100],
    [120],
    [140],
    [160],
    [180],
    [200],
]

# Number of neighbors, neighbor weights
# 'Uniform' is all neighbors weighted equally
# 'Distance' is neigbors weighted by inverse distance
knn_params = [
    [1, 'uniform'],
    [2, 'uniform'],
    [3, 'uniform'],
    [4, 'uniform'],
    [5, 'uniform'],
    [7, 'uniform'],
    [10, 'uniform'],
    [1, 'distance'],
    [2, 'distance'],
    [3, 'distance'],
    [4, 'distance'],
    [5, 'distance'],
    [7, 'distance'],
    [10, 'distance']
]

MLR_params = [[]]

dt_params = [[]]

# alpha, the regularization strength
ridge_params = [
    [0.5],
    [0.75],
    [1.0],
    [1.5],
    [2.0]
]

# alpha, the regularization strength
lasso_params = [
    [0.5],
    [0.75],
    [1.0],
    [1.5],
    [2.0]
]

bayes_params = [[]]

# Number of estimators
ada_params = [
    [25],
    [50],
    [75],
    [100],
    [150],
    [200],
    [300]
]

# Number of estimators, 
# subsample (fraction of samples to be used for fitting the individual base learners)
grad_params = [
    [25, 0.7],
    [50, 0.7],
    [75, 0.7],
    [100, 0.7],
    [150, 0.7],
    [200, 0.7],
    [300, 0.7],
    [25, 0.9],
    [50, 0.9],
    [75, 0.9],
    [100, 0.9],
    [150, 0.9],
    [200, 0.9],
    [300, 0.9],
    [25, 1.0],
    [50, 1.0],
    [75, 1.0],
    [100, 1.0],
    [150, 1.0],
    [200, 1.0],
    [300, 1.0]
]