# %%
import torch

from trainers import (train_model,
                        train_extra_bit_decoder,
                        train_parity_bit_decoder,
                        train_unsimilar_model,
                        train_extra_bit_decoder,
                        train_parity_bit_decoder,
                        train_correlated_bit_critic
                    )
# packages may have been fucked up and need to be reinstalled :()

batch_size = 1000

dataset = torch.load('datasets/bit_string_dataset_gp=0.99_ge=0.99_n=3e7.pth')
trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


# %%


f = train_model(trainloader)




# %%

# g = train_correlated_bit_critic(trainloader)



# %%


# save model A
# from datetime import datetime


torch.save(f.state_dict(), f'winner_save_v2.pt')


# %%

# model_B = train_unsimilar_model(model_A, trainloader, 0.02, clip = 2)

# from datetime import datetime

# id = datetime.now().strftime("%Y%m%d-%H%M%S")

# torch.save(model_B.state_dict(), f'model_B_{id}.pt')


# %%


# testing model_B_20231107-180733 - the one that output 3 different shit

# # load the model
# model_B = MainNetwork()
# model_B.load_state_dict(torch.load('model_B_20231107-180733.pt'))
# model_B.eval()






# %%
import matplotlib.pyplot as plt
import numpy as np

f = model_B.supervenient_feature_network

with torch.no_grad():
    for batch in trainloader:
        x0 = batch[:100,0]
        x1 = batch[:100,1]

        V1 = f(x1)

        for i in range(100):
            # round Ve


            print(f"{round(float(V1.squeeze()[i]),2)} - {np.sum(np.array(x1[i][:5]))} - {x1[i][-1]}")
        # plot a histogram of the values in V1
        plt.hist(V1.flatten(), bins=9)
        # print the height of the bins
        plt.ylabel('Frequency')
        plt.xlabel('supervenient feature value')
        plt.show()
        break
# %%

extra_bit_decoder = train_extra_bit_decoder(trainloader, f)

parity_bit_decoder = train_parity_bit_decoder(trainloader, f)


# %%

# test the extra_bit decoder

with torch.no_grad():
    for batch in trainloader:
        x0 = batch[:100,0]
        x1 = batch[:100,1]

        V1 = f(x1)

        extra_bits = x1[:,-1].unsqueeze(1)

        print(extra_bits)
        print(torch.round(extra_bit_decoder(V1)))
        # print the accuracy of the decoder vs the extra bits ground truth
        print((torch.round(extra_bit_decoder(V1)) == extra_bits).float().mean())
        break
# %%


# test the parity_bit decoder

with torch.no_grad():
    for batch in trainloader:
        x0 = batch[:100,0]
        x1 = batch[:100,1]

        V1 = f(x1)

        parity_batch = torch.sum(x1[:,:5], dim=1) % 2

        for i in range(15):
            # round Ve


            print(f"{round(float(V1.squeeze()[i]),2)} - {parity_batch[i]}")

        # print(parity_batch)
        # print(torch.round(parity_bit_decoder(V1)))
        # print the accuracy of the decoder vs the extra bits ground truth
        # print((torch.round(parity_bit_decoder(V1)) == parity_batch).float().mean())
        break
    

# %%


