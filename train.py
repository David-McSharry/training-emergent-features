import torch
from models import MainNetwork
from loss import Psi_loss
from datetime import datetime
import wandb


def train_supervenient_representation_model(device, trainloader, clip):
    model = MainNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.to(device)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    wandb.init(project="training-emergent-features", id=run_id)
    wandb.watch(model, log="all")

    # TODO: batch will be a tuple of (X_t0, X_t1) * 100 ???
    batch_counter = 0
    for batch in trainloader:
        # size (batch_size, 2, 6)
        batch.to(device)
        batch = batch.to(device)
        print("Batch shape")
        print(batch.shape)
        print(batch)
        print('-------------')

        print("batch[:,0]")
        print(batch[:,0])
        print(batch[:,0].shape)
        print('-------------')

        print("batch[:,1]")
        print(batch[:,1])
        print(batch[:,1].shape)
        print('-------------')


        # we want to minimize the negative of the Psi loss
        # ie maximize Psi
        loss = - Psi_loss(
            batch[:,0],
            batch[:,1],
            model.f_supervenient,
            model.causal_decoupling_critic,
            model.downward_causation_critic,
            device,
            clip=clip,
        )

        # add l2 normalization of the weights to the loss
        l2_lambda = 0.01
        l2_reg = torch.tensor(0.)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        loss += l2_lambda * l2_reg

        print("loss")
        print(loss)
        print('-------------')

        if batch_counter % 100 == 0:
            print("Batch {}, loss {}".format(batch_counter, loss))
        batch_counter += 1
        optimizer.zero_grad()
        loss.backward()
        wandb.log({"Psi loss": loss})
        optimizer.step()

    print("Finished training")
    print(batch_counter)

    return model
