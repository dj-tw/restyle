from time import time
import torch
import torch.optim as optim
import os
from IPython import get_ipython
from restyle.file_io import get_initial_image, get_cnn_model, get_image_tensors
from restyle.models import get_model_and_losses, total_variation_loss
from restyle.show_images import show_evolution, show_combined, show_final_results

# global vars, is there a way to avoid this?
history = []
iterations = 0


def upload_files():
    # if in Google colab, upload files each time
    from google.colab import files
    if os.path.isfile('style.png'):
        print('file exists')
        os.remove('style.png')

    if os.path.isfile('content.png'):
        print('file exists')
        os.remove('content.png')

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


def get_input_optimizer(input_img):
    # This line tell LBFGS what parameters we should optimize
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    # optimizer = optim.Adam([input_img.requires_grad_()])
    return optimizer


def upload_images():
    if 'google.colab' in str(get_ipython()):
        # use Google colab file upload widget
        print('Running on CoLab')
        upload_files()
    else:
        # use local files
        print('Not running on CoLab, using existing content, style files')


def device_is_cuda(device):
    return device.__str__() == 'cuda'


def check_device(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: %s' % device)

    if 'google.colab' in str(get_ipython()):
        if not device_is_cuda(device) and params['demand_cuda_on_colab']:
            raise ValueError("Device is cpu. You want an instance with GPU"
                             " while running on colab (or change params['demand_cuda_on_colab']"
                             " to False")
    return device


def run(params):
    global history
    global iterations

    device = check_device(params)

    content_img, style_img, original_content_image_size = get_image_tensors(params, device)

    input_img = get_initial_image(params, content_img, style_img, device)

    cnn, cnn_normalization_mean, cnn_normalization_std = get_cnn_model(device)

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
        # Compute the loss and back-propagate to the input_image.
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
                                   iterations,
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

    now_time = time()
    run_time = (now_time - start_time) / 60.0
    print('Runtime: %0.2f minutes' % run_time)

    # show final results
    show_final_results(params, input_img, original_content_image_size)
    show_combined(params)

    print('All done')


def pipeline():
    # get parameters, can change any variables with keywords
    params = get_params()

    # upload the images if on Google Colab, otherwise expects
    # to find content.png and style.png in root dir
    upload_images()

    # run the style transfer process
    run(params)
