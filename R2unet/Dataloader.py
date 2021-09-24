from torchvision import transforms
from PIL import Image
import torchvision
import torch

def Trainloader(BATCH_SIZE):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([256, 512]),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    target_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([256, 512], interpolation = Image.NEAREST)])
    trainset = torchvision.datasets.Cityscapes(root = './', split='train', target_type = 'semantic',
                                             transform = transform, target_transform = target_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, 
                                              shuffle = True, num_workers = 4)
    return trainloader
    
def Testloader(BATCH_SIZE):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([256, 512]),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    target_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([256, 512], interpolation = Image.NEAREST)])
    testset = torchvision.datasets.Cityscapes(root = './', transform = transform, target_transform = target_transform, split = 'val', target_type = 'semantic' )
    testloader = torch.utils.data.DataLoader(testset, batch_size = BATCH_SIZE, num_workers = 4)
    return testloader