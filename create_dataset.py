import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class BitStringDataset(Dataset):
    def __init__(self, gamma_parity, gamma_extra, length):
        self.data = self.generate_bit_string(gamma_parity, gamma_extra, length)

    def generate_bit_string(self, gamma_parity, gamma_extra, length):
        bit_strings = np.zeros((length, 6), dtype=np.int64)
        bit_strings[0, :] = np.random.randint(0, 2, 6)
        for t in range(1, length):
            parity = np.sum(bit_strings[t-1, :-1]) % 2
            if np.random.rand() < gamma_parity:
                bit_strings[t, :-1] = np.random.choice([0, 1], size=5)
                sum_parity = bit_strings[t, :-1].sum() % 2
                if sum_parity != parity:
                    bit_strings[t, np.random.choice(5)] ^= 1
            else:
                bit_strings[t, :-1] = np.random.choice([0, 1], size=5)
                sum_parity = bit_strings[t, :-1].sum() % 2
                if sum_parity == parity:
                    bit_strings[t, np.random.choice(5)] ^= 1
            if np.random.rand() < gamma_extra:
                bit_strings[t, -1] = bit_strings[t-1, -1]
            else:
                bit_strings[t, -1] = 1 - bit_strings[t-1, -1]

        adjacent_bits = np.array([bit_strings[i-1:i+1] for i in range(1, length)])
        return torch.tensor(adjacent_bits, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]





if __name__ == "__main__":
    # Create the Dataset
    dataset = BitStringDataset(gamma_parity=0.99, gamma_extra=0.99, length=1000000)


    # save the dataset
    torch.save(dataset.data, 'data/bit_string_dataset.pth')

    # make the dataset into a dataloader
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # for batch in dataloader:
    #     print(batch)
    #     break



    # def compare_parities(timestep1, timestep2):
    #     parity1 = np.sum(timestep1[:-1]) % 2
    #     parity2 = np.sum(timestep2[:-1]) % 2
    #     return parity1 == parity2

    # def compare_extra_bit_parity(timestep1, timestep2):
    #     return timestep1[-1] == timestep2[-1]

    # # Parameters
    # gamma_parity = 0.99
    # gamma_extra = 0.5
    # length = 10000  # Length of the sequence

    # # Generate the dataset
    # bit_strings = dataset.generate_bit_string(gamma_parity, gamma_extra, length + 1)
    # print(bit_strings)
    # print(len(bit_strings))

    # parities_count = 0
    # extra_parity_count = 0
    # # Check that the parity is preserved
    # for bit_string_tuple in bit_strings:
    #     parities_count += (compare_parities(bit_string_tuple[0], bit_string_tuple[1]))
    #     extra_parity_count += (compare_extra_bit_parity(bit_string_tuple[0], bit_string_tuple[1]))        
    # print(parities_count/ (len(bit_strings)-1))
    # print(extra_parity_count/ (len(bit_strings)-1))