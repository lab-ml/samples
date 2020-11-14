import torch.distributed
from torch.nn.parallel import DistributedDataParallel

from labml import experiment
from labml.configs import option
from labml_samples.pytorch.mnist.e_labml_helpers import Configs, Net


@option(Configs.model)
def ddp_model(c: Configs):
    return DistributedDataParallel(Net().to(c.device), device_ids=[c.device])


def main(rank, world_size, uuid):
    torch.distributed.init_process_group("gloo", init_method="tcp://localhost:23456", rank=rank, world_size=world_size)
    conf = Configs()
    experiment.create(uuid=uuid, name='cifar_10')
    experiment.distributed(rank, world_size)
    experiment.configs(conf,
                       {'optimizer.optimizer': 'Adam',
                        'optimizer.learning_rate': 1e-4,
                        'device.cuda_device': rank})
    conf.set_seed.set()
    experiment.add_pytorch_models(dict(model=conf.model))
    with experiment.start():
        conf.run()


if __name__ == '__main__':
    # main(0, 1, experiment.generate_uuid())
    torch.multiprocessing.spawn(main, args=(3, experiment.generate_uuid()), nprocs=3, join=True)
