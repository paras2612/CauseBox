import sys
import os
import numpy as np
from subprocess import call
import shutil

def load_config(cfg_file):
    cfg = {}

    with open(cfg_file,'r') as f:
        for l in f:
            l = l.strip()
            if len(l)>0 and not l[0] == '#':
                vs = l.split('=')
                if len(vs)>0:
                    k,v = (vs[0].strip(), eval(vs[1].strip()))
                    if not isinstance(v,list):
                        v = [v]
                    cfg[k] = v
    return cfg

def sample_config(configs):
    cfg_sample = {}
    for k in configs.keys():
        opts = configs[k]
        c = np.random.choice(len(opts),1)[0]
        cfg_sample[k] = opts[c]
    #print(cfg_sample)
    return cfg_sample

def cfg_string(cfg):
    ks = sorted(cfg.keys())
    cfg_str = ','.join(['%s:%s' % (k, str(cfg[k])) for k in ks])
    return cfg_str.lower()

def is_used_cfg(cfg, used_cfg_file):
    cfg_str = cfg_string(cfg)
    try:
        used_cfgs = read_used_cfgs(used_cfg_file)
    except:
        return False
    return cfg_str in used_cfgs

def read_used_cfgs(used_cfg_file):
    used_cfgs = set()
    with open(used_cfg_file, 'r') as f:
        for l in f:
            used_cfgs.add(l.strip())

    return used_cfgs

def save_used_cfg(cfg, used_cfg_file):
    try:
        with open(used_cfg_file, 'a') as f:
            cfg_str = cfg_string(cfg)
            f.write('%s\n' % cfg_str)
    except:
        return

def run(cfg_file, num_runs):
    configs = load_config(cfg_file)

    outdir = configs['outdir'][0]
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    used_cfg_file = '%s/used_configs.txt' % outdir

    if not os.path.isfile(used_cfg_file):
        f = open(used_cfg_file, 'w')
        f.close()

    for i in range(num_runs):
        cfg = sample_config(configs)
        if is_used_cfg(cfg, used_cfg_file):
            print('Configuration used, skipping')
            continue

        save_used_cfg(cfg, used_cfg_file)

        #print('------------------------------')
        #print('Run %d of %d:' % (i+1, num_runs))
        #print('------------------------------')
        print('\n'.join(['%s: %s' % (str(k), str(v)) for k,v in cfg.items() if len(configs[k])>1]))

        flags = ' '.join('--%s %s' % (k,str(v)) for k,v in cfg.items())
        call('python main.py cfrnet %s' % flags, shell=True)
        shutil.rmtree("D:/PycharmProjects/pythonProject/CFRNet/Results/ihdp")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: python cfr_param_search.py <config file> <num runs>')
    else:
        run(sys.argv[1], int(sys.argv[2]))