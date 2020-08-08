import numpy as np
from labml import tracker, experiment, lab

lab.configure({
    'web_api': {
        'url': 'https://api.lab-ml.com/api/v1/track?labml_token=67783400797e4068a0d2eb276fb82c9a&channel=training',
        'frequency': 10,
        'verify_connection': True
    }
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
