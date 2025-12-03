from torch.utils.data import Dataset

class CharDataset(Dataset):
    """
    Emits batches of characters.

    Adapted from "https://github.com/karpathy/minGPT".
    """

    def __init__(self, config, data):

        chars = sorted(list(set(data))) # get characters from the input data # IMPLEMENTED
        self.stoi = { ch:i for i,ch in enumerate(chars) } # map characters to integer indices
        self.itos = { i:ch for i,ch in enumerate(chars) } # similarly, map integer to indices, necessary for prediction # IMPLEMENTED
        self.vocab_size = len(chars) # IMPLEMENTED
        
        ...

    def get_vocab_size(self):
        return self.vocab_size # IMPLEMENTED

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        # encode every character to an integer
        # return the chunk and the shifted version as tensors
        pass