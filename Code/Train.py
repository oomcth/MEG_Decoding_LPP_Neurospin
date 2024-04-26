import torch
import dataset
from Losses import Classification_Loss
import model_components
import torch.optim as optim
from tqdm import tqdm
from dataset import Segment_Batch


def count_trainable_params(model):
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))


def train(loader, model, epochs, lr, weight_decay, device):
    criterion = Classification_Loss()
    optimizer = optim.AdamW(model.parameters(),
                            lr=lr,
                            weight_decay=weight_decay)

    train_losses = []
    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0
        for data in tqdm(loader, leave=False):
            data = Segment_Batch(data[0].to(device),
                                 data[1].to(device),
                                 data[2].to(device))
            phonemes = data.batch_phoneme
            model.train()
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, phonemes)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(loader))
        print(f'Epoch {epoch+1}, Train Loss: {train_losses[-1]:.4f}')
    torch.save(train_losses, "trainloss.pth")
    torch.save(model, "model.pth")
    torch.save(optimizer, "optimizer.pth")
    print(train_losses)


if __name__ == "__main__":
    choices = ['<p:>', 'R', 'a', 't', '@', 'i', 's', 'I', 'p', 'k']
    loader = dataset.Create_Loader()
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    assert device != "cpu"

    model = model_components.model(
        spatial_dim=144,
        temporal_dim=100,
        n_subjects=58,  # TODO 58 is greater than n_subjects
        num_class=len(choices),
        dropout=0.,
        device=device,
        lstm_norm=True
    ).to(device)
    print(model)
    print('param count', count_trainable_params(model))
    train(
        loader=loader,
        model=model,
        epochs=100,
        lr=5*10**(-4),
        weight_decay=0.001,
        device=device
    )
