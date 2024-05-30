

from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS

from mmtrack.core import YOLOXModeSwitchHook


@HOOKS.register_module()
class ProbYOLOXModeSwitchHook(YOLOXModeSwitchHook):
    """Extend the YOLOXModeSwitchHook to support ProbYOLOX.
        Enable box covariance parameters backprop for L1-based regression loss in bbox_head.
        
        Args:
            only_l1 (bool): Whether box covariance parameters are only trained
                when use_l1 is True. If True, box covariance parameters backprop
                is enabled and are trained using regression loss.
                If False, it is assumed their gradient is computed during training.
                Default: False.
    """
    
    def __init__(self, only_l1=False, **kwargs):
        super(ProbYOLOXModeSwitchHook, self).__init__(**kwargs)
        self.only_l1 = only_l1
        
    def before_train_epoch(self, runner):
        """Close mosaic and mixup augmentation and switches to use L1 loss.
        Enable gradient computation of covariance parameters if only_l1 is True."""
        epoch = runner.epoch
        train_loader = runner.data_loader
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        if (epoch + 1) == runner.max_epochs - self.num_last_epochs:
            runner.logger.info('No mosaic and mixup aug now!')
            # The dataset pipeline cannot be updated when persistent_workers
            # is True, so we need to force the dataloader's multi-process
            # restart. This is a very hacky approach.
            train_loader.dataset.update_skip_type_keys(self.skip_type_keys)
            if hasattr(train_loader, 'persistent_workers'
                       ) and train_loader.persistent_workers is True:
                train_loader._DataLoader__initialized = False
                train_loader._iterator = None
                self._restart_dataloader = True
            runner.logger.info('Add additional L1 loss now!')
            if hasattr(model, 'detector'):
                model.detector.bbox_head.use_l1 = True
                #? Enable box covariance backpropagation
                if self.only_l1:
                    if hasattr(model.detector.bbox_head, 'turn_on_bbox_cov_grad'):
                        runner.logger.info('Enable covariance backprop now!')
                        model.detector.bbox_head.turn_on_bbox_cov_grad()
            else:
                model.bbox_head.use_l1 = True
                #? Enable box covariance backpropagation
                if self.only_l1:
                    if hasattr(model.bbox_head, 'turn_on_bbox_cov_grad'):
                        runner.logger.info('Enable covariance backprop now!')
                        model.bbox_head.turn_on_bbox_cov_grad()
            
        else:
            # Once the restart is complete, we need to restore
            # the initialization flag.
            if self._restart_dataloader:
                train_loader._DataLoader__initialized = True