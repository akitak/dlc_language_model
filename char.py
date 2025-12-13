import torch
from torch.utils.data import Dataset

class CharDataset(Dataset):
    """
    Emits batches of characters.

    Adapted from "https://github.com/karpathy/minGPT".
    """

    def __init__(self, block_size, data): # Going to define block_size in notebook above instantiation of CharDataset object when reading data / training model
    # def __init__(self, config, data):

        self.data = data # IMPLEMENTED
        self.block_size = block_size # IMPLEMENTED

        chars = sorted(list(set(self.data))) # get characters from the input data # IMPLEMENTED
        self.stoi = { ch:i for i,ch in enumerate(chars) } # map characters to integer indices
        self.itos = { i:ch for i,ch in enumerate(chars) } # similarly, map integer to indices, necessary for decoding and prediction # IMPLEMENTED
        self.vocab_size = len(chars) # IMPLEMENTED
        self.data_size = len(self.data) # IMPLEMENTED
        
        
        ...

    def get_vocab_size(self):
        return self.vocab_size # IMPLEMENTED

    def __len__(self):
        return self.data_size - self.block_size # IMPLEMENTED # Number of training samples using a sliding window of length block_size #TODO: IMPLEMENT Config

    def __getitem__(self, idx):
        chunk = self.data[idx:idx+self.block_size+1]# grab a chunk of (block_size + 1) characters from the data
        encoded_tensor = torch.tensor([self.stoi[c] for c in chunk], dtype=torch.long) # encode every character to an integer # IMPLEMENTED
        # decode = lambda integers: ''.join([self.itos[integer] for integer in integers]) # decode every character to an integer # IMPLEMENTED
        # return the chunk and the shifted version as tensors
        x = encoded_tensor[:-1] # IMPLEMENTED
        y = encoded_tensor[1:] # IMPLEMENTED
        return x, y # IMPLEMENTED