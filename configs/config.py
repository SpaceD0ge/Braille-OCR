import yaml


def merge_dicts(main, update):
    if not isinstance(main, dict):
        return update
    for key in update:
        if isinstance(update[key], dict):
            if key in main:
                main[key] = merge_dicts(main[key], update[key])
            else:
                main[key] = update[key]
        else:
            main[key] = update[key]
    return main


class ConfigReader():
    def __init__(self, root='.'):
        self.root = root

    def _read_config_file(self, cfg_fname):
        with open(self.root + '/' + cfg_fname) as f:
            cfg = yaml.safe_load(f)
        if 'base' in cfg:
            with open(self.root + '/configs/' + cfg['base']) as f:
                main = yaml.safe_load(f)
            return merge_dicts(main, cfg)
        return cfg

    def _check_keys(self, cfg):
        for key in ('task', 'model', 'data'):
            if key not in cfg:
                raise ValueError(f'Config error: key {key} not found')

    def process(self, fname):
        cfg = self._read_config_file(fname)
        self._check_keys(cfg)
        cfg['model']['num_classes'] = cfg['data']['num_classes']
        return cfg
