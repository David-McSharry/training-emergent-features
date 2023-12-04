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
    # TODO: recreate old datasets eventually for safety
    # gamma_parity = 0.99
    # gamma_extra = 0.99
    # n = '3e5'
    # n_int = int(float(n)) + 1
    # dataset = BitStringDataset(gamma_parity=gamma_parity, gamma_extra=gamma_extra, length=n_int)

    # # save the dataset
    # torch.save(dataset.data, f'rep_2/datasets/bit_string_dataset_gp={gamma_parity}_ge={gamma_extra}_n={n}.pth')

    # ----------------------------

    bit_strings = torch.load(f'rep_2/datasets/bit_string_dataset_gp=0.99_ge=0.99_n=3e6.pth')



    print(len(bit_strings))
    # print(bit_strings.shape)

    # dataloader = DataLoader(bit_strings, batch_size=100, shuffle=True)

    # for batch in dataloader:
    #     bit_strings = batch
    #     break


    # def compare_parities(timestep1, timestep2):
    #     parity1 = torch.sum(timestep1[:-1]) % 2
    #     parity2 = torch.sum(timestep2[:-1]) % 2
    #     return parity1 == parity2

    # def compare_extra_bit_parity(timestep1, timestep2):
    #     return timestep1[-1] == timestep2[-1]

    # equal_parities_count = 0
    # equal_extra_parity_count = 0

    # num_0_0 = 0
    # num_0_1 = 0
    # num_1_0 = 0
    # num_1_1 = 0

    # num_0s_pos_0 = 0
    # num_0s_pos_1 = 0
    # num_1s_pos_0 = 0
    # num_1s_pos_1 = 0


    # for bit_string_pair in bit_strings:
    #     equal_parities_count += (compare_parities(bit_string_pair[0], bit_string_pair[1]))
    #     equal_extra_parity_count += (compare_extra_bit_parity(bit_string_pair[0], bit_string_pair[1])) 
    #     if bit_string_pair[0][-1] == 0 and bit_string_pair[1][-1] == 0:
    #         num_0_0 += 1
    #     elif bit_string_pair[0][-1] == 0 and bit_string_pair[1][-1] == 1:
    #         num_0_1 += 1
    #     elif bit_string_pair[0][-1] == 1 and bit_string_pair[1][-1] == 0:
    #         num_1_0 += 1
    #     elif bit_string_pair[0][-1] == 1 and bit_string_pair[1][-1] == 1:
    #         num_1_1 += 1
        
    #     if bit_string_pair[0][-1] == 0:
    #         num_0s_pos_0 += 1
    #     elif bit_string_pair[0][-1] == 1:
    #         num_1s_pos_0 += 1
    #     if bit_string_pair[1][-1] == 0:
    #         num_0s_pos_1 += 1
    #     elif bit_string_pair[1][-1] == 1:
    #         num_1s_pos_1 += 1
        
    # p_same_extra_bit = equal_extra_parity_count / len(bit_strings)

    # p_0_0 = num_0_0 / len(bit_strings)
    # p_0_1 = num_0_1 / len(bit_strings)
    # p_1_0 = num_1_0 / len(bit_strings)
    # p_1_1 = num_1_1 / len(bit_strings)

    # p_0_pos_0 = num_0s_pos_0 / len(bit_strings)
    # p_0_pos_1 = num_0s_pos_1 / len(bit_strings)
    # p_1_pos_0 = num_1s_pos_0 / len(bit_strings)
    # p_1_pos_1 = num_1s_pos_1 / len(bit_strings)

    # print(f'p_same_extra_bit = {p_same_extra_bit}')
    # print(f'p_0_0 = {p_0_0}')
    # print(f'p_0_1 = {p_0_1}')
    # print(f'p_1_0 = {p_1_0}')
    # print(f'p_1_1 = {p_1_1}')
    # print(f'same extra bit = {p_0_0 + p_1_1}')

    # print(f'p_0_pos_0 = {p_0_pos_0}')
    # print(f'p_0_pos_1 = {p_0_pos_1}')
    # print(f'p_1_pos_0 = {p_1_pos_0}')
    # print(f'p_1_pos_1 = {p_1_pos_1}')

    # MI = p_0_0 * np.log2(p_0_0 / (p_0_pos_0 * p_0_pos_1)) + p_1_0 * np.log2(p_1_0 / (p_1_pos_0 * p_0_pos_1)) + p_0_1 * np.log2(p_0_1 / (p_0_pos_0 * p_1_pos_1)) + p_1_1 * np.log2(p_1_1 / (p_1_pos_1 * p_1_pos_1))
    # print(f'MI = {MI}') 
    # # 0.9192 for two bits with 0.99 correlation
    # # 0 for two bits with 0.5 correlation




