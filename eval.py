# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""eval of ringmoe_framework"""
import argparse
import logging
import os

from register import RingMoEConfig, ActionDict


def str2bool(b):
    if b.lower() in ["false"]:
        output = False
    elif b.lower() in ["true"]:
        output = True
    else:
        raise Exception("Invalid Bool Value")
    return output


def _aicc_monitor(fn):
    try:
        import aicc_tools as ac
    except Exception:
        return fn
    return ac.aicc_monitor(fn)


def _maybe_obs_register(ac) -> None:
    if ac is None:
        return
    obs_ak = os.environ.get("RINGMOE_OBS_AK") or ""
    obs_sk = os.environ.get("RINGMOE_OBS_SK") or ""
    obs_server = os.environ.get("RINGMOE_OBS_SERVER", "")
    if not obs_ak or not obs_sk:
        logging.warning("OBS credentials not set (RINGMOE_OBS_AK/SK). Cloud features disabled.")
    if obs_ak and obs_sk and obs_server:
        ac.obs_register(ak=obs_ak, sk=obs_sk, server=obs_server)


@_aicc_monitor
def main(args):
    try:
        import aicc_tools as ac
        from mindspore.train.model import Model
        from mindspore.train.serialization import load_checkpoint

        from ringmoe_framework.datasets import build_dataset
        from ringmoe_framework.models import build_model, build_eval_engine
        from ringmoe_framework.parallel_config import build_parallel_config
        from ringmoe_framework.tools.helper import build_context
    except Exception as e:
        raise SystemExit(
            "MindSpore/AICC dependencies are required for the Ascend 910B version.\n"
            "For Linux + NVIDIA A100/H100 use the PyTorch/DeepSpeed refactor: `pytorch_refactor/train.py`.\n"
            "See: RUNNING_LINUX_A100.md\n"
            f"Root cause: {type(e).__name__}: {e}"
        ) from e

    _maybe_obs_register(ac)

    # init context
    cfts, _ = build_context(args)

    # evaluation dataset
    args.logger.info(".........Build Eval Dataset..........")
    eval_dataset = build_dataset(args, is_pretrain=False, is_train=False)

    # build context config
    args.logger.info(".........Build context config..........")
    build_parallel_config(args)
    args.logger.info("context config is:{}".format(args.parallel_config))
    args.logger.info("moe config is:{}".format(args.moe_config))

    # build net
    args.logger.info(".........Build Net..........")
    net = build_model(args)
    eval_engine = build_eval_engine(net, eval_dataset, args)

    # load task ckpt
    resume_ckpt = args.train_config.resume_ckpt
    if resume_ckpt:
        args.logger.info(".........Load Task Checkpoint..........")
        resume_ckpt = cfts.get_checkpoint(resume_ckpt)
        params_dict = load_checkpoint(resume_ckpt, filter_prefix=["adam_m", "adam_v"])
        net_not_load = net.load_pretrained(params_dict)
        args.logger.info(f"===============net_not_load================{net_not_load}")

    args.logger.info(".........Starting Init Eval Model..........")
    model = Model(net, metrics=eval_engine.metric, eval_network=eval_engine.eval_network)
    eval_engine.set_model(model)
    # define Model and begin eval
    args.logger.info(".........Starting Eval Model..........")
    eval_engine.eval()
    output = eval_engine.get_result()
    last_metric = 'Top1 accuracy={:.6f}'.format(float(output))
    args.logger.info(last_metric)


if __name__ == "__main__":
    work_path = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default=os.path.join(work_path, "../config/simmim/aircas/vit/pretrain-simmim-vit-moe-p16-01.yaml"),
        help='YAML config files')
    parser.add_argument('--device_id', default=None, type=int, help='device id')
    parser.add_argument('--seed', default=None, type=int, help='random seed')
    parser.add_argument('--batch_size', default=None, type=int, help='batch size')
    parser.add_argument('--use_parallel', default=None, type=str2bool, help='whether use parallel mode')
    parser.add_argument('--eval_path', default=None, type=str, help='checkpoint path for eval')
    parser.add_argument(
        '--options',
        nargs='+',
        action=ActionDict,
        help='override some settings in the used config, the key-value pair'
             'in xxx=yyy format will be merged into config file')

    args_ = parser.parse_args()
    config = RingMoEConfig(args_.config)
    if args_.device_id is not None:
        config.context.device_id = args_.device_id
    if args_.seed is not None:
        config.seed = args_.seed
    if args_.use_parallel is not None:
        config.use_parallel = args_.use_parallel
    if args_.eval_path is not None:
        config.train_config.resume_ckpt = args_.eval_path
    if args_.batch_size is not None:
        config.train_config.batch_size = args_.batch_size
    if args_.options is not None:
        config.merge_from_dict(args_.options)

    if config.finetune_dataset.eval_offset < 0:
        config.finetune_dataset.eval_offset = config.train_config.epoch % config.finetune_dataset.eval_interval

    main(config)
