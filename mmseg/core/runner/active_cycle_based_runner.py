import warnings
import time
import platform
import shutil
import os.path as osp
from abc import abstractmethod

import torch

import mmcv
from mmcv.runner import EpochBasedRunner, get_host_info, RUNNERS, \
    save_checkpoint

from mmseg.datasets import build_dataset, build_dataloader


@RUNNERS.register_module()
class ActiveCycleBasedRunner(EpochBasedRunner):
    """
    This is Epoch Based Runner but customized for Active Learning
    This runner trains model by X epoch, performs validation, then
    performs active learning sample selection to update train dataloader
    It uses the `train` and `val` loop of EpochBasedRunner.
    This runner stops training after K cycles
    Args:
        dataset_cfg (dict): Config for training dataset
        dataloader_cfg (dict): Config for dataloader
        max_cycles (int): Number of cycles for Active Learning
        load_best (bool): Load best model from last cycle
        restart_after_active (bool): Restart model stats after each cycle
    """
    def __init__(self,
                 dataset_cfg,
                 dataloader_cfg,
                 max_cycles:int,
                 load_best=False,
                 restart_after_active=False,
                 **kwargs):
        self.dataset_cfg = dataset_cfg
        self.dataloader_cfg = dataloader_cfg
        self.load_best = load_best
        self._max_cycles = max_cycles
        self._cycle = 0
        self.restart_after_active = restart_after_active
        if self.load_best is True and self.restart_after_active is True:
            warnings.warn(f"Can't set both load_best and restart_after_active in {self.__class__}"
                          f"Setting restart_after_active only. Please consider re-config in the future")
            self.load_best = False
        self.base_optimizer = self.optimizer
        self.base_model = self.model
        super(ActiveCycleBasedRunner, self).__init__(**kwargs)

    @property
    def max_cycles(self):
        """ Maximum cycles to run """
        return self._max_cycles

    @property
    def cycle(self):
        """ Current cycle """
        return self._cycle

    def train(self, data_loader, **kwargs):
        """ This function trains model for all epochs """
        # ----- restart training stats for each training loop -----
        self.optimizer = self.base_optimizer
        if self.restart_after_active:
            self.model = self.base_model
        if self.load_best:
            pass
        # ----- perform training loop -----
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        for epoch in self.max_epochs:
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            self.call_hook('before_train_epoch')
            for i, data_batch in enumerate(self.data_loader):
                self.data_batch = data_batch
                self._inner_iter = i
                self.call_hook('before_train_iter')
                self.run_iter(data_batch, train_mode=True, **kwargs)
                self.call_hook('after_train_iter')
                del self.data_batch
                self._iter += 1
            self.call_hook('after_train_epoch')
            self._epoch += 1

    @torch.no_grad()
    def active(self, data_loader, **kwargs):
        """
        This function performs sample selection for next cycle,
        this is different for every Active Learning method, this should be self implemented by users
        """
        self.model.eval()
        self.mode = 'active'

        for i, data_batch in enumerate(data_loader):
            self.data_batch = data_batch
            # user should self implement their strategy here
            self._strategy(data_batch)
            del self.data_batch

        self._build_training_loader()

    def _build_training_loader(self):
        """ This function rebuilt dataloader every new cycle """
        train_datasets = [build_dataset(self.dataset_cfg)]
        self._data_loaders = [build_dataloader(ds, **self.dataloader_cfg) for ds in train_datasets]

    @abstractmethod
    def _strategy(self, data_batch):
        """
        User should implement their active learning strategy here.
        This function should modify `split` file for training so that it will add new samples to `split` file
        """
        pass

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
                    cycle_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    cycle_runner(self._data_loaders[i], **kwargs) # train 1st, active 2nd

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='cycle_{}_epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)
            # Note: meta.update(self.meta) should be done before
            # meta.update(epoch=self.epoch + 1, iter=self.iter) otherwise
            # there will be problems with resumed checkpoints.
            # More details in https://github.com/open-mmlab/mmcv/pull/1108
        meta.update(epoch=self.epoch + 1, iter=self.iter,
                    cycle=self.cycle)

        filename = filename_tmpl.format(self.cycle, self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)