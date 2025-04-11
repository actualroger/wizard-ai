
import json
import sys
import os.path

from PlotUtils import *

# load a results json
runFolder = os.path.abspath(sys.argv[1])
settingsJson = os.path.join(runFolder, 'settings.json')
resultsJson = os.path.join(runFolder, 'training_results.json')

with open(settingsJson, 'r') as f:
    params = json.load(f)
with open(resultsJson, 'r') as f:
    results = json.load(f)

plotTrainingReturns(results['train_returns'], show=False, filename=os.path.join(runFolder, 'training_returns.png'))
plotTrainingLosses(results['train_losses'], show=False, filename=os.path.join(runFolder, 'training_losses.png'))

'''
import json
from PlotUtils import *
with open('../runs/DDQN_hybrid_2x15_2x5x5_2x16_2x64_pooled_priority/training_results.json','r') as f:
    data = json.load(f)

plotTrainingReturns([data['train_returns'][0]], show=True, filename='training_returns_best.png')
'''
