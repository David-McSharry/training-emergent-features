import torch
from models import SupervenientFeatureNetwork, CriticNetworkDecoupled, MainNetwork
from loss import Psi_loss
from mutual_information_estimators import estimate_mutual_information
from datetime import datetime
import wandb


def train_supervenient_representation_model(device, trainloader):
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
        loss = - Psi_loss(batch[:,0], batch[:,1], model.f_supervenient, model.causal_decoupling_critic, model.downward_causation_critic, device)
        if batch_counter % 1000 == 0:
            print("Batch {}, loss {}".format(batch_counter, loss))
        batch_counter += 1
        optimizer.zero_grad()
        loss.backward()
        wandb.log({"Psi loss": loss})
        optimizer.step()

    return model
