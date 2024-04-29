
from pathlib import Path
import numpy as np
import pandas as pd
import mne
import mne_bids
import warnings
import ast
import os
import textgrid
import copy
import itertools
import torch.nn.functional as F
import torch
from dataset import Segment_Batch
from intervaltree import IntervalTree
from scipy.spatial.distance import cdist

from jra_utils import approx_match_samples
from utils import (
    match_list,
    add_syntax,
    get_code_path,
)


global Decomposition
Decomposition = None


nogo = list(torch.load('nogo.pth'))
for i in range(len(nogo)):
    nogo[i] = tuple(nogo[i])


dropChannel = torch.load('toDrop.pth')

test = [
    (2, 9),
    (11, 9),
    (19, 9),
    (55, 9),
    (35, 9)
]

valid = [
    (2, 8),
    (11, 8),
    (19, 8),
    (55, 8),
    (35, 8)
]

nogo += valid
nogo += test


choices = ['<p:>', 'R', 'a', 't', '@', 'i', 's', 'p', 'k']


TOL_MISSING_DICT = {
    (9, 6): (30, 5),
    (10, 6): (30, 5),
    (12, 5): (30, 5),
    (13, 3): (30, 5),
    (13, 7): (30, 5),
    (14, 9): (30, 5),
    (21, 6): (30, 5),
    (21, 8): (30, 5),
    (22, 4): (30, 5),
    (33, 2): (30, 5),
    (39, 5): (30, 5),
    (40, 2): (30, 5),
    (41, 1): (30, 5),
    (43, 4): (30, 5),
    (43, 5): (30, 5),
    (44, 9): (30, 5),
    (24, 2): (10, 20),
}


number_to_range = {
    "01": "1-3",
    "02": "4-6",
    "03": "7-9",
    "04": "10-12",
    "05": "13-14",
    "06": "15-19",
    "07": "20-22",
    "08": "23-25",
    "09": "26-27"
}


CHAPTER_PATHS = [
    "ch1-3.wav",
    "ch4-6.wav",
    "ch7-9.wav",
    "ch10-12.wav",
    "ch13-14.wav",
    "ch15-19.wav",
    "ch20-22.wav",
    "ch23-25.wav",
    "ch26-27.wav",
]


def equilibrate_phonemes(df):
    phoneme_counts = df['phoneme'].value_counts()
    max_count = phoneme_counts.max()
    duplicated_rows = []

    for phoneme, count in phoneme_counts.items():
        duplications_needed = max_count - count

        phoneme_rows = df[df['phoneme'] == phoneme]

        duplicated = phoneme_rows.sample(n=duplications_needed, replace=True)
        duplicated_rows.append(duplicated)

    df_balanced = pd.concat([df] + duplicated_rows, ignore_index=True)
    df_balanced = df_balanced.reset_index(drop=True)
    return df_balanced


def read_raw(subject, run_id, events_return=False):
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
    raw.pick_types(meg=True, stim=True, misc=True)

    event_file = path + "/" + f"sub-{bids_path.subject}"
    event_file = event_file + "/" + f"ses-{bids_path.session}"
    event_file = event_file + "/" + "meg"
    event_file = str(event_file + "/" + f"sub-{bids_path.subject}")
    event_file += f"_ses-{bids_path.session}"
    event_file += f"_task-{bids_path.task}"
    event_file += f"_run-{bids_path.run}_events.tsv"
    assert Path(event_file).exists()
    events = mne.find_events(raw, stim_channel="STI101", shortest_event=1)

    raw_copy = copy.deepcopy(raw)
    sti101_data = raw_copy.pick_channels(['STI101'])
    sti101_data = sti101_data.get_data()
    first_non_zero_sample = np.where(sti101_data != 0)[1][0]
    Global_Start = first_non_zero_sample / 1000
    del raw_copy, sti101_data, first_non_zero_sample

    error_msg_prefix = (
            f"subject {subject}, session None, run {run_id}\n"
        )
    sound_triggers = mne.find_events(raw, stim_channel="STI101",
                                     shortest_event=1)
    events = []

    for annot in raw.annotations:
        description = annot.pop("description")
        if "BAD_ACQ_SKIP" in description:
            continue
        event = eval(description)
        event["condition"] = "sentence"
        event["type"] = event.pop("kind").capitalize()
        event["start"] = annot["onset"]
        event["duration"] = annot["duration"]
        event["stop"] = annot["onset"] + annot["duration"]
        event["language"] = "french"
        events.append(event)

    try:
        sound_triggers = sound_triggers[sound_triggers[:, 2] == 1]
        start, stop = sound_triggers[:, 0] / raw.info["sfreq"]
        events.append(
            dict(
                type="Sound",
                start=start,
                duration=stop - start,
                filepath=Path(event_file),
            )
        )
    except Exception as e:
        warnings.warn(
            f"No sound triggers found for subject {subject}, run {run_id}: {e}"
        )

    events_df = pd.DataFrame(events).rename(columns=dict(word="text"))
    events_df.loc[events_df["text"] == " ", "text"] = None
    events_df = events_df.dropna(subset=["text"])
    events_df.reset_index(drop=True, inplace=True)

    def extract_kind_word(trial_type):
        return pd.Series({'kind': trial_type['kind'],
                          'word': trial_type['word']})

    metadata = pd.read_csv(event_file, sep="\t")
    metadata['trial_type'] = metadata['trial_type'].apply(ast.literal_eval)
    metadata[['kind', 'word']] = metadata['trial_type'].apply(extract_kind_word)

    rows_events, rows_metadata = match_list(
        [str(word) for word in events_df["text"].values],
        [str(word) for word in metadata["word"].values],
    )
    assert len(rows_events) / len(events_df) > 0.95, (
        error_msg_prefix
        + f"only {len(rows_events) / len(events_df)}"
        + "of the words were found in the metadata"
    )
    events_idx, metadata_idx = (
        events_df.index[rows_events],
        metadata.index[rows_metadata],
    )

    path_syntax = get_code_path() + "/" + "data" + "/" + "syntax_new_no_punct"
    metadata = add_syntax(metadata, path_syntax, int(run_id))
    metadata["sequence_id"] = np.cumsum(metadata.is_last_word.
                                        shift(1, fill_value=False))
    for s, d in metadata.groupby("sequence_id"):
        metadata.loc[d.index, "word_id"] = range(len(d))
    metadata.word = metadata.word.str.replace('"', "")
    metadata["wlength"] = metadata.word.apply(len)
    metadata["run"] = run_id
    metadata["has_trigger"] = False

    events_df["word"] = events_df["text"]
    for col in ["sequence_id", "n_closing", "is_last_word", "pos"]:
        events_df.loc[events_idx, col] = metadata.loc[metadata_idx, col]

    starts = mne.find_events(raw, output="step", shortest_event=1)[:, 0]
    meg_times = np.copy(raw.times)
    meg_triggers = np.zeros_like(meg_times)
    meg_triggers[starts - raw.first_samp] = 1
    words = events_df.loc[events_df.type == "Word"]
    word_triggers = mne.find_stim_steps(raw, stim_channel="STI008")
    word_triggers = word_triggers[word_triggers[:, 2] == 0]

    abs_tol, max_missing = TOL_MISSING_DICT.get(
        (int(subject), int(run_id)), (10, 5)
    )
    i, j = approx_match_samples(
        (words.start * 1000).tolist(),
        word_triggers[:, 0],
        abs_tol=abs_tol,
        max_missing=max_missing,
    )
    print(f"Found {len(i)/len(words)} of the words in the triggers")
    assert len(i)/len(words) > 0.9

    words = words.iloc[i, :]
    events_df.loc[:, "unaligned_start"] = events_df.loc[:, "start"]
    events_df.loc[words.index, "start"] = word_triggers[j, 0] / raw.info["sfreq"]

    uri = f"method:_load_raw?timeline={1}"
    meg = {"filepath": uri, "type": "Meg", "start": 0}
    events_df = pd.concat([pd.DataFrame([meg]), events_df])
    events_df = events_df.sort_values(by="start").reset_index(drop=True)
    event = events_df

    raw.load_data()
    raw = raw.filter(5, 45)
    events_df = events_df.fillna("")

    if events_return:
        return raw, metadata, events_df, get_phonemes(
            run_id, Global_Start, events_df, subject
        )
    else:
        return raw, metadata, get_phonemes(
            run_id, Global_Start, events_df, subject
        )


def getB(run_id, sub):
    df = pd.read_csv(get_file_b(run_id, sub), sep='\t', header=None, names=['start', 'delta', 'data'])
    df = df.iloc[1:]
    df['data'] = df['data'].apply(ast.literal_eval)
    df['start'] = df['start'].apply(float)
    df['delta'] = df['delta'].apply(float)
    df['end'] = df['start'] + df['delta']
    df['word'] = df['data'].apply(lambda x: x.get('word'))
    df['word'] = df['word'].apply(lambda x: 'j\'' if x == 'j' else x)
    df = df.sort_values('start')
    df['phoneme'] = df['word']
    return df[['start', 'end', 'phoneme']]


def map_number_to_range(number):
    return number_to_range.get(number, "error")


def extract_interval_data(interval):
    return interval.minTime, interval.maxTime, interval.mark


def get_file_phonemes(run_id):
    path = "/Volumes/KINGSTON/datasets/MEG/LPP_MEG_visual_neuralset/"
    path += "sourcedata/stimuli/phonemes/ch"
    path += map_number_to_range(run_id) + ".TextGrid"
    return path


def get_file_b(run_id, sub):
    path = "/Volumes/KINGSTON/datasets/MEG/LPP_MEG_auditory/sub-" + sub
    path += "/ses-01/meg/sub-" + sub
    path += "_ses-01_task-listen_run-" + run_id + "_events.tsv"
    return path


def get_phonemes(run_id, Global_Start, data, subject):
    data = data.drop(data.index[0])
    file = get_file_phonemes(run_id)
    assert os.path.exists(file), file
    raw_data = textgrid.TextGrid.fromFile(file)
    extracted_data = map(extract_interval_data, raw_data[-1])
    df_A = pd.DataFrame(extracted_data, columns=['start', 'end', 'phoneme'])
    df_A['x'] = df_A['start']
    df_A = df_A.drop(df_A.index[0])

    raw_data = textgrid.TextGrid.fromFile("1-3.TextGrid")
    extracted_data = map(extract_interval_data, raw_data[-1])
    df_B = getB(run_id, subject)
    df_B['x'] = df_B['start']
    df_B = df_B.assign(index_seq=range(len(df_B)))
    Local_Start = df_B.loc[1, 'start']

    result = pd.merge_asof(df_A[['x', 'start', 'end', 'phoneme']],
                           df_B[['x', 'start', 'phoneme', 'index_seq']],
                           on='x', direction='nearest', suffixes=('_A', '_B'))
    result = result.reset_index(drop=True)
    result['delta'] = result['end'] - result['start_A']
    result['start_A'] = result['start_A'] - result['start_B']
    treshold = 0.1
    result = result.loc[(result['start_A'].abs() <= treshold)]

    result = result.groupby('index_seq').apply(
        lambda x: x[['x', 'start_A',
                     'phoneme_A', 'delta',
                     'start_B', 'phoneme_B']].values.tolist()
    )
    arr = []
    result = np.array(result)
    for i in range(len(result)):
        arr += [result[i][0][5]]
    words = np.array((data['text']))
    starts = np.array((data['unaligned_start']))
    i, j = match_list(arr, words)
    words = words[j]
    starts = starts[j]
    result = result[i]
    for i in range(len(result)):
        for item in result[i]:
            item[4] = starts[i]
    flattened_data = list(itertools.chain(*result))
    result = pd.DataFrame(
        flattened_data, columns=['jsp', 'start', 'phoneme',
                                 'delta', 'start_B', 'pB']
    )
    result['start'] += result['start_B'] - result.loc[0, 'start_B']
    result['start'] += Global_Start + Local_Start

    result = result[['start', 'phoneme', 'delta']]
    result['end'] = result['start'] + result['delta']
    tree = IntervalTree()
    for index, row in result.iterrows():
        tree[row['start']:row['end']] = index
    to_drop = set()
    for interval in tree:
        if len(list(tree[interval.begin:interval.end])) > 1:
            for idx in list(tree[interval.begin:interval.end]):
                to_drop.add(idx[2])
    result = result.drop(index=to_drop).reset_index(drop=True)
    return result


def loc(tuple):
    a, b = tuple
    return (a-1) * 9 + b - 1


def get_index(loc):
    return (int((loc // 9) + 1), int((loc % 9) + 1))


dataloc = [get_index(i) for i in range(0, 58*9) if get_index(i) not in nogo]


def load(subject, run_id, types=[3012]):
    global Decomposition
    raw, meta, df_phonemes = read_raw(subject, run_id, False)
    channels_to_keep = []
    drop = dropChannel

    for ch in raw.info['chs']:
        if (ch['coil_type'] in types and (not (ch['ch_name'] in drop))):
            channels_to_keep.append(ch['ch_name'])

    raw = raw.pick_channels(channels_to_keep)
    tensor_meg = torch.tensor(raw.get_data())
    if Decomposition is None:
        layout = mne.find_layout(raw.info)
        layout = np.array(
            [layout.pos[layout.names.index(ch)][:2] for ch in raw.ch_names]
        )
        layout[:, 0] = (layout[:, 0] - np.min(layout[:, 0]))
        layout[:, 0] /= np.max(layout[:, 0])
        layout[:, 1] = (layout[:, 1] - np.min(layout[:, 1]))
        layout[:, 1] /= np.max(layout[:, 1])

        distances = cdist(layout, layout)
        distances = np.max(distances) * np.ones(np.shape(distances))
        distances -= distances
        distances -= len(distances) * np.identity(len(distances))
        print('Computing laplacian cholesky decomposition')
        _, Decomposition = np.linalg.eigh(distances)
        torch.save(Decomposition, '/Volumes/KINGSTON/Graph/U.pth')

    df_phonemes = df_phonemes.drop(
        df_phonemes[df_phonemes['delta'] > 30 / 100].index
    )
    df_phonemes = df_phonemes[df_phonemes['phoneme'].isin(choices)]

    del raw
    del meta

    return tensor_meg, equilibrate_phonemes(df_phonemes.reset_index(drop=True))


def generate_samples(subject, types=[3022], off_signal=0.07,
                     size=100, train_ids=0):

    train_tensors, train_phonemes = [], []
    for id in train_ids:
        tensor_meg, df_phonemes = load(subject, id, types=types)
        temp_tensors = [
            tensor_meg[:,
                       (int((df_phonemes.loc[j, 'start'] + off_signal)
                        * 1000)):
                       (int((df_phonemes.loc[j, 'start'] + off_signal)
                        * 1000) + size)]
            for j in range(len(df_phonemes))
        ]
        train_tensors += temp_tensors
        train_phonemes += df_phonemes['phoneme'].tolist()

    train_tensors = torch.stack(train_tensors, dim=0).float()
    train_tensors[:, :, :] *= 10**10
    train_tensors.clamp(-2, 2)
    nan_indices = train_tensors.isnan().nonzero()
    train_tensors[nan_indices] = 0

    indices = [choices.index(value) for value in train_phonemes]
    train_phonemes = F.one_hot(torch.tensor(indices),
                               num_classes=len(choices))

    del (indices, temp_tensors, tensor_meg, df_phonemes)

    train_tensors = train_tensors.float()
    train_phonemes = train_phonemes.float()
    subject = torch.full((train_tensors.size(0),), int(subject))
    return Segment_Batch(train_tensors, train_phonemes, subject)


def Save_data():
    for place in test:
        subject, run = place
        subject = str(subject)
        if run > 9:
            run = [str(run)]
        else:
            run = ['0' + str(run)]
        try:
            data = generate_samples(subject=subject,
                                    types=[3012],
                                    off_signal=0.07,
                                    size=100,
                                    train_ids=run)
            torch.save(data, '/Volumes/KINGSTON/Test_DATA_MATHIS/test_dataSubj'
                       + subject + 'run' + run[0] + '.pth')
        except Exception as e:
            print(e)
            input()


if __name__ == "__main__":
    Save_data()
