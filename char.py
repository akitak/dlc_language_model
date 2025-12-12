from torch.utils.data import Dataset

class CharDataset(Dataset):
    """
    Emits batches of characters.

    Adapted from "https://github.com/karpathy/minGPT".
    """

    def __init__(self, config, data):

        chars = sorted(list(set(data))) # get characters from the input data # IMPLEMENTED
        self.stoi = { ch:i for i,ch in enumerate(chars) } # map characters to integer indices
        self.itos = { i:ch for i,ch in enumerate(chars) } # similarly, map integer to indices, necessary for decoding and prediction # IMPLEMENTED
        self.vocab_size = len(chars) # IMPLEMENTED
        
        ...

    def get_vocab_size(self):
        return self.vocab_size # IMPLEMENTED

    def __len__(self):
        return len(self.

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        encode = lambda text: [self.stoi[char] for char in text] # encode every character to an integer # IMPLEMENTED # TODO: save as tensor
        decode = lambda integers: ''.join([self.itos[integer] for integer in integers]) # decode every character to an integer # IMPLEMENTED # TODO: save as tensor
        # return the chunk and the shifted version as tensors
        pass