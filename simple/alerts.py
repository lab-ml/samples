import numpy as np
from labml import tracker, experiment, lab

lab.configure({
    'web_api': 'https://api.lab-ml.com/api/v1/track?labml_token=YOUR_LABML_TOEKN&channel=YOUR_SLACK_CHANNEL',
})

configs = {
    'fs': 100000,  # sample rate
    'f': 1,  # the frequency of the signal
}

x = np.arange(configs['fs'])
y = np.sin(2 * np.pi * configs['f'] * (x / configs['fs']))

experiment.create(name='sin_wave')
experiment.configs(configs)
experiment.start()

for y_i in y:
    tracker.save({'loss': y_i, 'noisy': y_i + np.random.normal(0, 10, 100)})
    tracker.add_global_step()
