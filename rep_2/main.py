# %%
import torch

from trainers import (train_model,
                        train_extra_bit_decoder,
                        train_parity_bit_decoder,
                        train_extra_bit_decoder,
                        train_parity_bit_decoder,
                        train_unsimilar_model
                    )

batch_size = 500

dataset = torch.load('datasets/bit_string_dataset_gp=0.99_ge=0.99_n=3e7.pth')
trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


# %%


f = train_model(trainloader)




# %%

from models import SupervenientFeatureNetwork
import numpy as np

# load model

model_A = SupervenientFeatureNetwork()
model_A.load_state_dict(torch.load('models/winneRRRRR.pt'))
model_A.eval()


model_B = train_unsimilar_model(model_A, trainloader)


# %%


from models import SupervenientFeatureNetwork
import numpy as np

f1 = SupervenientFeatureNetwork()
f1.load_state_dict(torch.load('models/winneRRRRR.pt'))

f2 = SupervenientFeatureNetwork()
f2.load_state_dict(torch.load('models/learned_parity_bit_f_VMI_only.pt'))

for batch in trainloader:
    x0 = batch[:100,0]
    x1 = batch[:100,1]

    V1 = f1(x1)
    V2 = f2(x1)

    for i in range(100):
        print(f"{round(float(V1.squeeze()[i]),2)} - {round(float(V2.squeeze()[i]),2)}")
        
        

# %%

# load the representations
A_reps = torch.load('datasets/A_reps.pth')
A_reps_dataloarder = torch.utils.data.DataLoader(A_reps, batch_size=batch_size, shuffle=False)


for batch in trainloader:
    print(batch.shape)
    break

for batch in A_reps_dataloarder:
    print(batch.shape)
    break






# %%


# save model A
# from datetime import datetime


torch.save(f.state_dict(), f'weird_model_learns_more_than_a_bit???.pt')


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
import matplotlib.pyplot as plt
import numpy as np

with torch.no_grad():
    for batch in trainloader:
        x0 = batch[:1000,0]
        x1 = batch[:1000,1]

        V1 = f(x1)

        extra_bits = x1[:,-1].unsqueeze(1)

        # print(extra_bits)
        # print(torch.round(extra_bit_decoder(V1)))
        # print the accuracy of the decoder vs the extra bits ground truth
        print((torch.round(extra_bit_decoder(V1)) == extra_bits).float().mean())

        zero_array = []
        one_array = []


        for i in range(1000):
            # round Ve


            print(f"{round(float(V1.squeeze()[i]),2)} - {extra_bits[i]}")

            if extra_bits[i] == 0:
                zero_array.append(V1[i])
            else:
                one_array.append(V1[i])

        # plot a histogram of the values in V1 making zero and one arrays different colors
        plt.hist(np.array(zero_array).flatten(), bins=20, label='zero', color='blue')
        plt.hist(np.array(one_array).flatten(), bins=20, label='one', color='orange')
        # print the height of the bins
        plt.ylabel('Frequency')
        plt.xlabel('f(extra bit)')    
        plt.legend()
        plt.show()
        break
# %%

import matplotlib.pyplot as plt
import numpy as np

# test the parity_bit decoder

with torch.no_grad():
    for batch in trainloader:
        x0 = batch[:100,0]
        x1 = batch[:100,1]

        V1 = f(x1)

        parity_batch = torch.sum(x1[:,:5], dim=1) % 2

        zero_array = []
        one_array = []


        for i in range(10):
            # round Ve


            print(f"{round(float(V1.squeeze()[i]),2)} - {parity_batch[i]}")

            if parity_batch[i] == 0:
                zero_array.append(V1[i])
            else:
                one_array.append(V1[i])

        # convert from tensor to numpy array
                
        print(zero_array)
        
                
        zero_array = np.round(np.array(zero_array),2)
        one_array = np.round(np.array(one_array),2)

        print(zero_array)
        print(one_array)

        # plot a histogram of the values in V1 making zero and one arrays different colors
        plt.hist(zero_array.flatten(), bins=200, label='zero', color='blue')
        plt.hist(one_array.flatten(), bins=200, label='one', color='orange')
        # print the height of the bins
        plt.ylabel('Frequency')
        plt.xlabel('f(parity bit)')
        plt.legend()
        plt.show()






        # print(parity_batch)
        # print(torch.round(parity_bit_decoder(V1)))
        # print the accuracy of the decoder vs the extra bits ground truth
        # print((torch.round(parity_bit_decoder(V1)) == parity_batch).float().mean())
        break
    

# %%


