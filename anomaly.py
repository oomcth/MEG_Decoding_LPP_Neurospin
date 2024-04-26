
from pathlib import Path
import mne
import mne_bids
import torch
from operator import itemgetter
import numpy as np
import matplotlib
matplotlib.use('agg')


def get_file_b(run_id, sub):
    path = "/Volumes/KINGSTON/datasets/MEG/LPP_MEG_auditory/sub-" + sub
    path += "/ses-01/meg/sub-" + sub
    path += "_ses-01_task-listen_run-" + run_id + "_events.tsv"
    return path


def Get_bads(subject, run_id, events_return=False):
    path = "/Volumes/KINGSTON/datasets/MEG/LPP_MEG_auditory"
    task = "listen"
    print(f"\n Epoching for run {run_id}, subject: {subject}\n")
    bids_path = mne_bids.BIDSPath(
        subject=subject,
        session="01",
        task=task,
        datatype="meg",
        root=path,
        run=run_id
    )

    raw = mne_bids.read_raw_bids(bids_path)
    raw.del_proj()
    raw.load_data()
    coil_types = np.array([ch['coil_type'] for ch in raw.info['chs']])
    picks = np.where(coil_types == 3012)[0]
    channel_names = itemgetter(*picks)(raw.ch_names)
    raw = raw.pick_channels(channel_names)

    auto_noisy_chs, auto_flat_chs = mne.preprocessing.find_bad_channels_maxwell(raw)
    return auto_noisy_chs


subjects = [str(i) for i in range(1, 59)]
final_data = None
bads = []
local_bads = []
to_delete = []
for subject in subjects:
    for train_id in ["01", "02", "03", "04", "05", "06", "07", "08", "09"]:
        local_bads.append([])
        if True:
            try:
                path = get_file_b(train_id, subject)
                if Path(path).exists():
                    bad = Get_bads(subject, train_id)
                    local_bads[-1] = list(set(local_bads[-1] + bad))
                    bads = list(set(bads + bad))
                else:
                    to_delete.append(subject + '-' + train_id)
            except Exception:
                to_delete.append(subject + '-' + train_id)
torch.save(bads, "badsAll.pth")
torch.save(local_bads, "local.pth")
torch.save(to_delete, "delete.pth")
