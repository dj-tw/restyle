from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from restyle.utils import tensor_to_image


def show_evolution(tensor, iterations, history=[], title=None, y_range=None):
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


def show_final_results(params, input_img, original_content_image_size):
    # show final results and save
    img = tensor_to_image(input_img)
    img_final = img.resize(original_content_image_size, resample=Image.BICUBIC)
    img_final.save(params['output_image_path'])

    plt.figure()
    plt.ion()
    plt.title('All images')
    plt.title('Final image')
    plt.imshow(img_final)
