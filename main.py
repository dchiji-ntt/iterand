
import torch
import argparse
import commands
import yaml
from pprint import PrettyPrinter

hyperparam_names = {
    'lr': float,
    'weight_decay': float,
    'model.config_name': str,
    'seed': int,
    'conv_sparsity': float,
    'rerand_freq': int,
    'rerand_lambda': float,
}

def load_configs(config):
    with open(config, 'r') as f:
        yml = f.read()
        dic = yaml.load(yml, Loader=yaml.FullLoader)
    return dic

def main(args):
    pp = PrettyPrinter(indent=1)
    print('Experiment: ', args.exp_name)

    # Load config from YAML file
    command = args.command
    cfgs = load_configs(args.config)
    cfg = cfgs[args.exp_name]
    cfg['data_parallel'] = (cfg['num_gpus'] > 1)
    cfg['accum_grad'] = args.accum_grad
    cfg['force_restart'] = args.force_restart

    if args.output_dir is not None:
        cfg['output_dir'] = args.output_dir
    if args.save_best_model is not None:
        if args.save_best_model == 'True' or args.save_best_model == 'true':
            cfg['save_best_model'] = True
        elif args.save_best_model == 'False' or args.save_best_model == 'false':
            cfg['save_best_model'] = False
        else:
            raise NotImplementedError

    if (command == 'train') and cfg['parallel_grid'] is not None:
        print(cfg['parallel_grid'])
        for key in hyperparam_names:
            val = getattr(args, key)
            print(key, val)
            if val is None:
                if key in cfg['parallel_grid']:
                    print(f"[Error] Please specify an option for `{args.exp_name}` experiment: --{key}")
                    return
                else:
                    continue
            else:
                if key not in cfg['parallel_grid']:
                    print(f"[Error] Please specify only options in `parallel_grid` of `{args.exp_name}`; Not supported: --{key}")
                    return
            cfg[key] = val

    #pp.pprint(cfg)
    cfg['__other_configs__'] = cfgs

    # Call command function
    command = getattr(getattr(commands, command), command)
    command(args.exp_name, cfg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', type=str)
    parser.add_argument('config', type=str, help='file path for YAML configure file')
    parser.add_argument('exp_name', type=str, help='specify the name of experiment')
    parser.add_argument('--accum_grad', type=int, default=1)
    parser.add_argument('--force_restart', action='store_true')
    parser.add_argument('--save_best_model', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)

    for key in hyperparam_names:
        parser.add_argument('--' + key, type=hyperparam_names[key], default=None)

    args = parser.parse_args()

    main(args)
