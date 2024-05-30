from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class EpochHook(Hook):

    def __init__(self):
        pass

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        #? Set epoch number for loss attenuation in reid head
        if hasattr(runner.model.module, 'head'):
            runner.model.module.head.current_epoch = runner.epoch
        
        #? Set epoch number for loss attenuation in bbox head
        if hasattr(runner.model.module, 'bbox_head'):
            runner.model.module.bbox_head.current_epoch = runner.epoch

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        #? Set iteration number for loss attenuation in reid head
        if hasattr(runner.model.module, 'head'):
            runner.model.module.head.current_iter = runner.iter
        
        #? Set iteration number for loss attenuation in bbox head
        if hasattr(runner.model.module, 'bbox_head'):
            runner.model.module.bbox_head.current_iter = runner.iter

    def after_iter(self, runner):
        pass