from torch.utils.data import Dataset

class customds (Dataset): #subclass of dataset so properties are inherited.
    def __init__(self,data,transform=None,aug_factor=0,grad=True): #Data is assumed to be a (x,y) tuple.
        self.spectra=data[0]
        self.attributes=data[1]
        self.transform=transform
        self.aug_factor=aug_factor
        self.total_samples = self.spectra.shape[0]
        self.grad=grad
    def __getitem__(self,index):
        # Calculate the actual index and augmentation factor
        actual_index = index // (self.aug_factor + 1)
        aug_index = index % (self.aug_factor + 1)
        
        # Retrieve the spectra and attributes
        image = self.spectra[actual_index,:]
        probabilities = self.attributes[actual_index,:]
        
        # Apply transformation if specified and augmentation factor is greater than 0
        if self.transform and aug_index > 0:
            image = self.transform(image)
            
        return image.requires_grad_(self.grad),probabilities.requires_grad_(self.grad)
    
    def __len__(self):
        return self.total_samples*(self.aug_factor+1) #We could also use len(probabilities[0]) 