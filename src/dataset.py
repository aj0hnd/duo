import glob
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split

class InputPipelineBuilder:
    IMAGE_SIZE = (3, 512, 512)
    LATENT_SIZE = (4, 64, 64)
    def __init__(self, test_size=.1, batch_size=1, shuffle=True):
        super().__init__()
        self.batch_size = batch_size
        self.safe_dir = '../data/nudity/safe/'
        self.unsafe_dir = '../data/nudity/unsafe/'
        
        self.safe_image_path = sorted(glob.glob(self.safe_dir + '*.png'))
        self.unsafe_image_path = sorted(glob.glob(self.unsafe_dir + '*.png'))
        
        self.paired_image_path = list(zip(self.safe_image_path, self.unsafe_image_path))
        train_image_path, test_image_path = train_test_split(self.paired_image_path, test_size=test_size, shuffle=shuffle)
        train_image_path, valid_image_path = train_test_split(train_image_path, test_size=0.05, shuffle=shuffle)
        
        self.image_path = {
            'train': train_image_path,
            'valid': valid_image_path,
            'test': test_image_path
        }
        
    def get_dataloader(self, subset='train', shuffle=False):
        ds = DUODataset(image_path=self.image_path[subset])
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

class DUODataset(Dataset):
    def __init__(self, image_path, safe_prompt='a dressed woman', unsafe_prompt='a naked man'):
        self.image_path = image_path
        self.safe_prompt = safe_prompt
        self.unsafe_prompt = unsafe_prompt
        
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True)
        ])
    def __len__(self):
        return len(self.image_path)
    def __getitem__(self, index):
        safe_image = Image.open(self.image_path[index][0])
        safe_image = self.transform(safe_image)
        unsafe_image = Image.open(self.image_path[index][1])
        unsafe_image = self.transform(unsafe_image)
        
        return {
            'safe_image': safe_image,
            'unsafe_image': unsafe_image,
            'safe_prompt': self.safe_prompt,
            'unsafe_prompt': self.unsafe_prompt
        }       