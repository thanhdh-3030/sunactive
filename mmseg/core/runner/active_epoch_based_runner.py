import warnings
import time
from abc import abstractmethod

import torch

import mmcv
from mmcv.runner import EpochBasedRunner, get_host_info, RUNNERS


@RUNNERS.register_module()
class ActiveEpochBasedRunner(EpochBasedRunner):
    """
    This is Epoch Based Runner but customized for Active Learning
    This runner trains model by X epoch, performs validation, then
    performs active learning sample selection to update train dataloader
    It uses the `train` and `val` loop of EpochBasedRunner.
    This runner stops training after K cycles
    Args:
        max_cycles (int): number of cycles for Active Learning
    """
    def __init__(self,
                 max_cycles:int,
                 **kwargs):
        self._max_cycles = max_cycles
        self._cycle = 0
        super(ActiveEpochBasedRunner, self).__init__(**kwargs)

    @property
    def max_cycles(self):
        """ Maximum cycles to run """
        return self._max_cycles

    @property
    def cycle(self):
        """ Current cycle """
        return self._cycle

    def run(self,
            data_loaders,
            workflow=[('train', 1),
                      ('active', 1)],
            max_epochs=None,
            **kwargs):
        """Start running. The base workflow should be train -> active (select sample process)

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        self._data_loaders = data_loaders
        while self.cycle < self.max_cycles:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(self._data_loaders[i], **kwargs) # train 1st, active 2nd


        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    @abstractmethod
    @torch.no_grad()
    def active(self, data_loader, **kwargs):
        """
        This function performs sample selection for next cycle,
        this is different for every Active Learning method, this should be self implemented by users
        just write a sub-class of this base class then implement sample_select of their own
        """
        self.model.eval()
        self.mode = 'active'
        self.data_loader = data_loader
        pass