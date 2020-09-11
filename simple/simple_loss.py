import numpy as np

from labml import tracker, experiment

conf = {'batch_size': 20}
n = 0


def train():
    global n
    n += 1
    return 0.999 ** n + np.random.random() / 10, 1 - .999 ** n + np.random.random() / 10


with experiment.record(name='sample', exp_conf=conf, web_api='samples', comment='test'):
    for i in range(100000):
        loss, accuracy = train()
        tracker.save(i, {'loss': loss, 'accuracy': accuracy})
