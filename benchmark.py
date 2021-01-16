import torch
from timeit import default_timer as timer
from datetime import timedelta
from tqdm.auto import tqdm
from models.effdet import DetectionModel
from configs.config import ConfigReader
import subprocess
import argparse


parser = argparse.ArgumentParser(description="Benchmark")
parser.add_argument("-cfg", type=str, default="configs/effdet_infer.yaml")
parser.add_argument("-root", type=str, default=".")
parser.add_argument("-device", type=str, default=None)
parser.add_argument("-trials", type=int, default="10")

args = parser.parse_args()


def get_device_name(dtype='cpu'):
    if dtype=='cpu':
        cpu_string = (subprocess.check_output("lscpu", shell=True).strip()).decode()
        return cpu_string.split('\n')[13].split('  ')[-1]
    else:
        return torch.cuda.get_device_name(0)    


if __name__ == "__main__":
    cfg = ConfigReader(args.root).process(args.cfg)
    if args.device is not None:
        device = args.device
    else:
        device = 'cpu' if cfg['process']['gpus'] == 0 else 'cuda'

    model = DetectionModel(cfg['model']).to(device)
    model.eval()
    print(get_device_name(device))
    times = []
    for _ in tqdm(range(args.trials)):
        image = torch.rand(
            (1,3,cfg['model']['image_size'][0],cfg['model']['image_size'][1])
        ).to(device)
        start = timer()
        with torch.no_grad():
            outs = model({'image':image})
        end = timer()
        delta = timedelta(seconds=end-start)
        times.append(delta)
    avg_time = sum(times, timedelta())/args.trials
    print(f'average processing time of {args.trials} trials is {avg_time}')
    min_time = min(times)
    print(f'minimum time of {min_time}')
