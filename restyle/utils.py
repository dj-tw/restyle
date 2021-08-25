import torchvision.transforms as transforms
import torch
from PIL import Image


def image_to_tensor(img, device):
    # Transform to tensor
    im = transforms.ToTensor()(img)
    # Fake batch dimension required to fit network's input dimensions
    im = im.unsqueeze(0)
    # Move to the right device and convert to float
    return im.to(device, torch.float)


# Reconvert a tensor into PIL image
def tensor_to_image(tensor):
    img = (255 * tensor).cpu().detach().squeeze(0).numpy()
    img = img.clip(0, 255).transpose(1, 2, 0).astype("uint8")
    return Image.fromarray(img)
