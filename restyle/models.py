import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


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