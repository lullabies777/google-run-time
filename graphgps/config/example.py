from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('example')
def set_cfg_example(cfg):
    r'''
    This function sets the default config value for customized options
    :return: customized configuration use by the experiment.
    '''

    # ----------------------------------------------------------------------- #
    # Customized options
    # ----------------------------------------------------------------------- #

    # example argument
    cfg.example_arg = 'example'

    # example argument group
    cfg.example_group = CN()

    # then argument can be specified within the group
    cfg.example_group.example_arg = 'example'

@register_config('more_cfgs')
def set_cfg_example(cfg):
    r'''
    This function sets the default config value for customized options
    :return: customized configuration use by the experiment.
    '''

    # ----------------------------------------------------------------------- #
    # Customized options
    # ----------------------------------------------------------------------- #

    # example argument
    cfg.train.mode = 'custom_tpu'
    
    cfg.device = "cuda"
    
    cfg.source = "nlp"
    
    cfg.search = "random"
    
    cfg.gnn.dim_in = 128
    
    cfg.train.num_sample_config = 32
    
    cfg.margin = 0.1
    
    cfg.heads = 4
    
    cfg.dropout = 0.5