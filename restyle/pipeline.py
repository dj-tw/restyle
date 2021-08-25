from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import copy
import os
import numpy as np

# global vars, is there a way to avoid this?
history = []
iterations = 0


def upload_files():
    # if in Google colab, upload files each time
    if os.path.isfile('style.png'):
        print('file exists')
        os.remove('style.png')

    if os.path.isfile('content.png'):
        print('file exists')
        os.remove('content.png')

    from google.colab import files
    uploaded = files.upload()
    return uploaded


def get_params(**kwargs):

    params = {'n_iter': 100,
              'image_width':  128,
              'image_height': 128,
              'content_weight': 1.0,
              'style_weight': 1.0,
              'total_variation_weight': 10.0,
              'input_image': 'hybrid',
              'hybrid_weight_content': 0.90,
              'hybrid_weight_style': 0.0,
              'show_image_interval': 50,
              'content_layers': ['conv_1', 'conv_2', 'conv_4'],
              'style_layers': ['conv_2', 'conv_3', 'conv_4', 'conv_7', 'conv_10', 'conv_8'],
              'content_image_path': 'content.png',
              'style_image_path':  'style.png',
              'output_image_path': 'result.png',
              'combined_image_path': 'combined.png',
              'plot_y_range': (0.5, 10000),
              'random_seed': 420,
              'demand_cuda_on_colab': True}

    # any overrides?
    for k, v in kwargs.items():
        params[k] = v

    print('\nparams')
    print('--------------')
    for k, v in params.items():
        print("%s: %s" % (k, v))
    print('--------------\n')

    return params


def open_image(url):
    if url.startswith("http"):
        return Image.open(BytesIO(requests.get(url).content)).convert('RGB')
    else:
        return Image.open(url).convert('RGB')


def load_content_image(params):
    content_image = open_image(params['content_image_path'])
    original_content_image_size = content_image.size

    print('Original content image size', original_content_image_size)
    print('Saving content image')
    content_image.save(params['content_image_path'])
    print('resizing')
    im_size = (params['image_width'], params['image_height'])
    content_image = content_image.resize(im_size, resample=Image.BICUBIC)
    print('Resized content imaged sixze', content_image.size)
    return content_image, original_content_image_size


def load_style_image(params):
    style_image = open_image(params['style_image_path'])
    print('Style image size', style_image.size)
    print('Saving style image')
    style_image.save(params['style_image_path'])
    print('resizing')
    im_size = (params['image_width'], params['image_height'])
    style_image = style_image.resize(im_size, resample=Image.BICUBIC)
    print('Resized style imaged size',style_image.size)
    return style_image


class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # We 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable.
        self.target = target.detach()
        self.loss = None

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()
        self.loss = None

    def forward(self, input):
        g = self.gram_matrix(input)
        self.loss = F.mse_loss(g, self.target)
        return input

    @staticmethod
    def gram_matrix(input):
        a, b, c, d = input.size()
        # Here:
        # a is the batch size(=1)
        # b is the number of feature maps
        # (c,d) are the dimensions of a feature map

        # We reshape the activation layer into a collection of feature vectors
        features = input.view(a * b, c * d)

        # Compute the gram product
        g = torch.mm(features, features.t())

        # We 'normalize' the values of the gram matrix
        # by dividing by the norm of gram matrix filled with ones.
        return g.div((a * b * c * d) ** 0.5)


def total_variation_loss(x): # Expect a mini batch of dim NxWxH
    image_height = x.shape[1]
    image_width = x.shape[2]
    dx = x[:, :image_height-1, :image_width-1, :] - x[:, 1:, :image_width-1, :]
    dy = x[:, :image_height-1, :image_width-1, :] - x[:, :image_height-1, 1:, :]
    loss = (dx ** 2 + dy ** 2).sum() ** 0.5
    # Return loss normalized by image and batch size
    return loss / (image_width * image_height * x.shape[0])


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # Reshape the mean and std to make them [C x 1 x 1] so that they can
        # directly broadcast to image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # Normalize img
        return (img - self.mean) / self.std


def get_model(device):
    # Neural network used.
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    # Normalization mean and standard deviation.
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    return cnn, cnn_normalization_mean, cnn_normalization_std


def get_model_and_losses(style_img,
                         content_img,
                         cnn,
                         cnn_normalization_mean,
                         cnn_normalization_std,
                         device,
                         params):

    # We make a deep copy of vgg19 in order to not modify the original
    cnn = copy.deepcopy(cnn)

    # Normalization module
    normalization = Normalization(cnn_normalization_mean, cnn_normalization_std).to(device)

    # This list will contain the losses computed by the network.
    content_losses = []
    style_losses = []

    # We rebuild the model as a nn sequential whose first layer is
    # our normalization layer.
    model = nn.Sequential(normalization)

    # We brows the layer of `cnn` and stack them into `model`.
    i = 0  # Incremented every time we see a conv layer.
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        # Check if the layer we just added was in the content layer list.
        # If so, we just stack a Content Loss layer.
        if name in params['content_layers']:
          target = model(content_img).detach()
          content_loss = ContentLoss(target)
          model.add_module("content_loss_{}".format(i), content_loss)
          content_losses.append(content_loss)

        # Check if the layer we just added was in the style layer list.
        # If so, we just stack a Style Loss layer.
        if name in params['style_layers']:
          target_feature = model(style_img).detach()
          style_loss = StyleLoss(target_feature)
          model.add_module("style_loss_{}".format(i), style_loss)
          style_losses.append(style_loss)

    # Now we trim off the layers after the last content and style losses
    # to keep the model as small as possible.
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


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


def show_image(input_img):
    # To visualise it better, instead of clipping values, we rescale
    # them to fit [-1,1], and convert to an image. This is mostly because
    # the visualization given is closer to what the actual values stored
    # in the tensor are.
    if INPUT_IMAGE == 'noise':
      img = transforms.ToPILImage()(input_img[0].cpu())
    else:
      img = tensor_to_image(input_img[0])

    img.show()


def get_input_optimizer(input_img):
  # This line tell LBFGS what parameters we should optimize
  optimizer = optim.LBFGS([input_img.requires_grad_()])
  #optimizer = optim.Adam([input_img.requires_grad_()])
  return optimizer


def show_evolution(tensor, history=[], title=None, y_range=None):
    if y_range is None:
        y_range = (0.5, 1000)

    image = tensor.cpu().clone().squeeze(0)
    image = tensor_to_image(image)
    # Display a big figure
    plt.rcParams['figure.figsize'] = [10, 5]
    plt.figure()
    plt.ion()
    plt.subplot(121)
    plt.imshow(image)
    plt.axis('off')
    if title is not None:
        plt.title(title)
    # Losses
    ax = plt.subplot(122)
    plt.yscale('log')
    plt.title('Losses')
    history = np.array(history).T
    plt.plot(history[0], label='Style')
    plt.plot(history[1], label='Content')
    plt.plot(history[2], label='Variation')
    plt.plot(history[3], label='Sum')
    plt.legend(loc="upper right")
    plt.ylim(y_range)
    # Finaly show the graph
    plt.draw()
    plt.pause(0.001)
    # Display a textual message
    message = 'Iter: {}, Style Loss : {:4f} Content Loss: {:4f} Variation Loss: {:4f} Sum: {:4f}'
    print(message.format(iterations, history[0][-1], history[1][-1], history[2][-1], history[3][-1]))


def show_combined(params):
    combined = Image.new("RGB", (params['image_width'] * 3, params['image_height']))
    x_offset = 0
    for image in map(Image.open, [params['content_image_path'],
                                  params['style_image_path'],
                                  params['output_image_path']]):

        im_size = (params['image_width'], params['image_height'])
        image = image.resize(im_size, resample=Image.BICUBIC)
        combined.paste(image, (x_offset, 0))
        x_offset += params['image_width']

    combined.save(params['combined_image_path'])
    plt.figure()
    plt.ion()
    plt.title('All images')
    plt.imshow(combined)


def get_initial_image(params, content_img, style_img, device):
    torch.manual_seed(params['random_seed'])
    if params['input_image'] == 'noise':
        input_img = torch.randn(content_img.data.size(), device=device)
    elif params['input_image'] == 'content':
        input_img = content_img.clone()
    elif params['input_image'] == 'style':
        input_img = style_img.clone()
    elif params['input_image'] == 'hybrid':
        input_img_noise = torch.randn(content_img.data.size(), device=device)
        input_img_content = content_img.clone()
        input_img_style = style_img.clone()
        w_content = params['hybrid_weight_content']
        w_style = params['hybrid_weight_style']
        w_noise = 1.0 - (w_content + w_style)
        assert 0 <= w_content <= 1
        assert 0 <= w_style <= 1
        assert w_noise >= 0.0
        input_img = w_noise * input_img_noise \
            + w_content * input_img_content \
            + w_style * input_img_style
    else:
        image = open_image(params['input_image']).resize(content_img.data.size())
        input_img = image_to_tensor(image)
        input_img += torch.randn(content_img.data.size(), device=device) * 0.05

    return input_img


def run(params):
    device = check_device(params)
    content_image, original_content_image_size = load_content_image(params)
    style_image = load_style_image(params)

    # get the content image
    content_image, original_content_image_size = load_content_image(params)

    # get the style image
    style_image = load_style_image(params)

    global history
    global iterations

    content_img = image_to_tensor(content_image, device)
    style_img = image_to_tensor(style_image, device)

    input_img = get_initial_image(params, content_img, style_img, device)
    cnn, cnn_normalization_mean, cnn_normalization_std = get_model(device)

    print('Building the style transfer model..')
    start_time = time()

    model, style_losses, content_losses = get_model_and_losses(style_img,
                                                               content_img,
                                                               cnn,
                                                               cnn_normalization_mean,
                                                               cnn_normalization_std,
                                                               device,
                                                               params)

    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    iterations = 0
    history = []
    max_iter = params['n_iter']
    keep_going = True
    while iterations <= max_iter and keep_going:
        # Compute the loss and backpropagate to the input_image.
        # (The LBFGS optimizer only accept work through closures.)
        try:
            def closure():
                global history
                global iterations

                optimizer.zero_grad()

                # Compute the total variation loss
                variation_score = total_variation_loss(input_img) * params['total_variation_weight']
                # Compute the features through the model
                model(input_img)
                # Compute style and content losses
                style_score = sum(sl.loss for sl in style_losses)
                style_score *= params['style_weight'] / len(style_losses)
                content_score = sum(cl.loss for cl in content_losses)
                content_score *= params['content_weight'] / len(content_losses)
                # Our global loss is the sum of the 3 values
                loss = style_score + content_score + variation_score
                # Save the value of loss in order to draw them as a graph
                history += [[style_score.item(), content_score.item(), variation_score.item(), loss.item()]]

                # If the iteration is a multiple of 100, display some informations
                if iterations % params['show_image_interval'] == 0:
                    show_evolution(input_img.data.clone().detach().clamp(0, 1),
                                   history,
                                   title="Iteration %d:" % iterations,
                                   y_range=params['plot_y_range'])

                iterations += 1
                # Backpropagate gradients and leave the optimizer do his job.
                loss.backward()
                return loss

            optimizer.step(closure)
        except (Exception, KeyboardInterrupt):
            # does not work in colab for some reason
            print('Interrupt or exception detected')
            break

    # @title Our beautiful result
    img = tensor_to_image(input_img)
    now_time = time()
    run_time = (now_time - start_time) / 60.0
    print('Runtime: %0.2f minutes' % run_time)

    img_final = img.resize(original_content_image_size, resample=Image.BICUBIC)
    img_final.save(params['output_image_path'])
    print('final image size: ', img_final.size)

    plt.figure()
    plt.ion()
    plt.title('All images')
    plt.title('Final image')
    plt.imshow(img_final)

    show_combined(params)
    print('All done')


def upload_images():
    if 'google.colab' in str(get_ipython()):
        # use Google colab file upload widget
        print('Running on CoLab')
        upload_files()
    else:
        # use local files
        print('Not running on CoLab, using existing content, style files')


def check_device(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: %s' % device)

    if 'google.colab' in str(get_ipython()):
        if device != 'cuda' and params['demand_cuda_on_colab']:
            raise ValueError("Device is cpu. You want an instance with GPU"
                             "while running on colab (or change params['demand_cuda_on_colab']"
                             "to False")
    return device


def pipeline():
    # get parameters, can change any variables with keywords
    params = get_params()

    # upload the images if on Google Colab, otherwise expects
    # to find content.png and style.png in root dir
    upload_images()

    # run the style transfer process
    run(params)
