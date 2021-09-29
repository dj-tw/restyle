import torchvision.models as tv_models
import torch
from PIL import Image
from io import BytesIO
import requests
from restyle.utils import image_to_tensor
import os


def upload_image_file(file_type, params):
    if 'google.colab' not in str(get_ipython()):
        print('Using local file')
        return

    save_file_name = None
    assert file_type in {'content', 'style'}
    if file_type == 'content':
        save_file_name = params['content_image_path']
    if file_type == 'style':
        save_file_name = params['style_image_path']

    # if in Google colab, upload files each time
    from google.colab import files
    if os.path.isfile(save_file_name):
        print('%s file exists' % save_file_name)
        os.remove(save_file_name)

    uploaded = files.upload()
    file_name = list(uploaded.keys())[0]
    image = open_image(file_name)
    image.save(save_file_name)
    return image


def get_cnn_model(device):
    # Neural network used.
    cnn = tv_models.vgg19(pretrained=True).features.to(device).eval()
    # Normalization mean and standard deviation.
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    return cnn, cnn_normalization_mean, cnn_normalization_std


def open_image(url):
    if url.startswith("http"):
        return Image.open(BytesIO(requests.get(url).content)).convert('RGB')
    else:
        return Image.open(url).convert('RGB')


def get_image_file(params, file_type):
    assert file_type in {'content', 'style'}
    if file_type == 'content':
        file = params['content_image_path']
    else:
        file = params['style_image_path']

    if not os.path.isfile(file):
        # use the provided defaults
        file = "%s/%s" % ('example_data', file)

    return file


def load_content_image(params):
    content_file = get_image_file(params, 'content')
    content_image = open_image(content_file)
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
    style_file = get_image_file(params, 'style')
    style_image = open_image(style_file)
    print('Style image size', style_image.size)
    print('Saving style image')
    style_image.save(params['style_image_path'])
    print('resizing')
    im_size = (params['image_width'], params['image_height'])
    style_image = style_image.resize(im_size, resample=Image.BICUBIC)
    print('Resized style imaged size',style_image.size)
    return style_image


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


def get_image_tensors(params, device):
    # get the content image
    content_image, original_content_image_size = load_content_image(params)

    # get the style image
    style_image = load_style_image(params)

    content_img = image_to_tensor(content_image, device)
    style_img = image_to_tensor(style_image, device)

    return content_img, style_img, original_content_image_size
