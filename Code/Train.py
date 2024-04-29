import torch
import dataset
from Losses import Classification_Loss
import model_components
import torch.optim as optim
from tqdm import tqdm
from dataset import Segment_Batch
from eval import eval
import typing as tp


def count_trainable_params(model):
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))


def train(loader, model, epochs, lr, weight_decay, device,
          valid_loader: tp.Optional[bool] = None,
          test_loader: tp.Optional[bool] = None):
    criterion = Classification_Loss()
    optimizer = optim.AdamW(model.parameters(),
                            lr=lr,
                            weight_decay=weight_decay)

    train_losses = []
    test_losses = []
    test_acc = []
    valid_losses = []
    valid_acc = []
    for epoch in tqdm(range(epochs)):
        if valid_loader is not None:
            print("valid eval :")
            acc, loss = eval(valid_loader, model, 1, False)
            valid_losses.append(loss), valid_acc.append(acc)
            print("acc:", acc)
            print("loss:", loss)
        if test_loader is not None:
            print("test eval :")
            acc, loss = eval(test_loader, model, 1, False)
            test_losses.append(loss), test_acc.append(acc)
            print("acc:", acc)
            print("loss:", loss)
        model.train()
        running_loss = 0.0
        for data in tqdm(loader):
            data = Segment_Batch(data[0].to(device),
                                 data[1].to(device),
                                 data[2].to(device))
            phonemes = data.batch_phoneme
            print(data.batch_subject)
            input()
            model.train()
            for param in model.parameters():
                param.grad = None
            outputs = model(data)
            loss = criterion(outputs, phonemes)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(loader))
        print(f'Epoch {epoch+1}, Train Loss: {train_losses[-1]:.4f}')
        save_path = "/Volumes/KINGSTON/model checkpoint/epoch-"
        save_path += str(epoch) + ".pt"
        torch.save(model.state_dict(), save_path)
    torch.save(train_losses, "trainloss.pth")
    torch.save(valid_losses, "validloss.pth")
    torch.save(test_losses, "testloss.pth")
    torch.save(valid_acc, "validacc.pth")
    torch.save(test_acc, "testacc.pth")
    torch.save(model.state_dict(), "model.pt")
    torch.save(optimizer, "optimizer.pth")
    print(train_losses)


if __name__ == "__main__":
    choices = ['<p:>', 'R', 'a', 't', '@', 'i', 's', 'p', 'k']
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    loader = dataset.Create_Loader(device)
    valid_loader = dataset.Create_Loader(device, dataset.vfiles, 512)
    test_loader = dataset.Create_Loader(device, dataset.tfiles, 512)
    assert device != "cpu"

    model = model_components.model(
        spatial_dim=144,
        temporal_dim=100,
        n_subjects=58,  # TODO 58 is greater than n_subjects
        num_class=len(choices),
        dropout=0.25,
        device=device,
        lstm_norm=True
    ).to(device)
    print(model)
    train(
        loader=loader,
        model=model,
        epochs=80,
        lr=5*10**(-4),
        weight_decay=0.001,
        device=device,
        valid_loader=valid_loader,
        test_loader=test_loader
    )
