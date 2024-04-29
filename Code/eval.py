import torch
import model_components
from dataset import Segment_Batch, vfiles, tfiles, files, Create_Loader
from tqdm import tqdm
import Losses
import numpy as np
import  typing as tp


choices = ['<p:>', 'R', 'a', 't', '@', 'i', 's', 'p', 'k']
device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")

train_freq = np.array([3.47, 12.1759, 14.7065, 11.6211, 16.2848,
                      10.5866, 12.2193, 11.1359, 7.8]) / 100


def eval(loader, model, n, out: tp.Optional[bool] = True):
    criterion = Losses.Classification_Loss()
    model.eval()
    num_correct = [0] * len(choices)
    num_samples = [0] * len(choices)
    class_freq = np.zeros(len(choices))
    total_correct = 0
    total_samples = 0
    running_loss = 0
    with torch.no_grad():
        for data in tqdm(loader):
            data = Segment_Batch(data[0].to(device),
                                 data[1].to(device),
                                 data[2].to(device))
            phonemes = data.batch_phoneme
            outputs = model(data)
            loss = criterion(outputs, phonemes)
            running_loss += loss.item() * phonemes.size(0)

            _, predicted = torch.topk(outputs, n, dim=1)
            for i in range(len(data)):
                class_idx = data.batch_phoneme[i].argmax().item()
                num_samples[class_idx] += 1
                class_freq[class_idx] += 1
                if class_idx in predicted[i]:
                    num_correct[class_idx] += 1
                    total_correct += 1
                total_samples += 1
    probs = [num_correct[i] / (num_samples[i] + 10**(-8))
             for i in range(len(choices))]
    total_prob = total_correct / total_samples
    if out:
        print("loss:", running_loss / total_samples)
        print(f"Total accuracy for top {n} predictions: {total_prob:.4f}")
        print("random accuracy (one shot):",
              np.sum((class_freq / total_samples)**2))
        print("bayes benchmark:", np.dot((class_freq / total_samples),
                                         train_freq))
        for i in range(len(choices)):
            print(f'Class {choices[i]}: Accuracy = {probs[i]:.4f},'
                  f' Frequency = {(class_freq[i] / total_samples):.4f}')
        if n == 1:
            print("one shot gains vs random =",
                  str(total_prob - np.sum((class_freq / total_samples)**2))[:6])
            print("one shot gains vs bayes =",
                  str(total_prob - np.dot((class_freq / total_samples),
                                          train_freq))[:6])
    return total_prob, running_loss / total_samples


if __name__ == "__main__":
    model = model_components.model(
            spatial_dim=144,
            temporal_dim=100,
            n_subjects=58,  # TODO 58 is greater than n_subjects
            num_class=len(choices),
            dropout=0.,
            device=device,
            lstm_norm=True
    ).to(device)
    path = "/Volumes/KINGSTON/model checkpoint/epoch-42.pt"
    model.load_state_dict(torch.load(path))

    valid_loader = Create_Loader('mps', vfiles, 32)
    test_loader = Create_Loader('mps', tfiles, 32)
    # train_loader = Create_Loader(files, 512)

    print('valid:')
    eval(valid_loader, model, 1)
    print('test:')
    eval(test_loader, model, 1)
    # print('train:')
    # eval(train_loader, model, 1)
