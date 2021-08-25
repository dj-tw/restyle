from time import time
import torch
import torch.optim as optim
from IPython import get_ipython
from restyle.file_io import get_initial_image, get_cnn_model, get_image_tensors
from restyle.models import get_model_and_losses, total_variation_loss
from restyle.show_images import show_evolution, show_combined, show_final_results

# global vars, is there a way to avoid this?
history = []
iterations = 0


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


def get_input_optimizer(input_img):
    # This line tell LBFGS what parameters we should optimize
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    # optimizer = optim.Adam([input_img.requires_grad_()])
    return optimizer


def get_closure(params, input_img, optimizer, model, style_losses, content_losses):
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

    return closure


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
        # The LBFGS optimizer only accept work through closures.
        closure = get_closure(params, input_img, optimizer, model, style_losses, content_losses)

        optimizer.step(closure)

    now_time = time()
    run_time = (now_time - start_time) / 60.0
    print('Runtime: %0.2f minutes' % run_time)

    # show final results
    show_final_results(params, input_img, original_content_image_size)
    show_combined(params)

    print('All done')
