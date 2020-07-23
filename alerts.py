import numpy as np
from labml import tracker, experiment

configs = {
    'fs': 100,  # sample rate
    'f': 2,  # the frequency of the signal
}

x = np.arange(configs['fs'])
y = np.sin(2 * np.pi * configs['f'] * (x / configs['fs']))

experiment.create(name='sign_wave')
experiment.configs(configs)
experiment.start()

for y_i in y:
    tracker.save({'y_i': y_i})
    tracker.add_global_step()
