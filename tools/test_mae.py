import os
import torch
import warnings
import argparse
from mmcv import Config, DictAction
from mmaction.models import build_model
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmaction.datasets import build_dataloader, build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--fuse-conv-bn', action='store_true',
                        help='Whether to fuse conv and bn, this will slightly increase the inference speed')
    parser.add_argument('--options', nargs='+', action=DictAction, default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument('--average-clips', choices=['score', 'prob', None], default=None,
        help='average type when averaging test clips')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.cfg_options)
    cfg.data.test.test_mode = True

    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
                              workers_per_gpu=cfg.data.get('workers_per_gpu', 1), dist=False, shuffle=False)
    dataloader_setting = dict(dataloader_setting, **cfg.data.get('test_dataloader', {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)

    # build the model and load checkpoint
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    results = []
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, **data)
        results.extend(result)
        pass
    pass



if __name__ == '__main__':
    main()
