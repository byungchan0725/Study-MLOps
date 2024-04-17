import torch
import matplotlib.pyplot as plt


def save_model(model):
    save_model_path = 'train_result/CNN_MNIST.pt'
    torch.save(model, save_model_path)


def train(model, train_DL, criterion, optimizer, EPOCH, DEVICE):
    model.train()

    loss_history = []
    NoT = len(train_DL.dataset)

    model.train()
    for ep in range(EPOCH):
        rloss = 0
        for x_batch, y_batch in train_DL:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            y_hat = model(x_batch)
            loss = criterion(y_hat, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_b = loss.item() * x_batch.shape[0]
            rloss += loss_b
        loss_e = rloss / NoT
        loss_history += [loss_e]
        print(f"Epoch: {ep + 1}, train loss: {round(loss_e, 3)}")
        print("-" * 20)

    save_model(model)

    return loss_history


def draw_graph(loss_history):
    plt.plot(loss_history)
    plt.show()