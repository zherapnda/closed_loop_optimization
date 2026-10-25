import csv
import os
from datetime import datetime

HEADERS = ['experiment_num', 'structure_id', 'timestamp', 'amplitude', 'rise', 'dwell', 'fall',
           'duty_cycle_mean', 'duty_cycle_std', 'output', 'source', 'deskew_angle', 'image_path',
           'gp_pred_mean', 'gp_pred_std', 'gp_fallback']

def init_logger(filepath):
    if not os.path.exists(filepath):
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=HEADERS)
            writer.writeheader()

def log_experiment(filepath, experiment_num, structure_id, amplitude, rise, dwell, fall, mean, std, output,
                    source=None, deskew_angle=None, image_path=None,
                    gp_pred_mean=None, gp_pred_std=None, gp_fallback=None):
    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writerow({
            'experiment_num': experiment_num,
            'structure_id': structure_id,
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'amplitude': amplitude,
            'rise': rise,
            'dwell': dwell,
            'fall': fall,
            'duty_cycle_mean': mean,
            'duty_cycle_std': std,
            'output': output,
            'source': source,
            'deskew_angle': deskew_angle,
            'image_path': image_path,
            'gp_pred_mean': gp_pred_mean,
            'gp_pred_std': gp_pred_std,
            'gp_fallback': gp_fallback,
        })
