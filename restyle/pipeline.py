import os
from IPython import get_ipython
from restyle.run_restyle import run
from restyle.file_io import upload_image_file

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


def upload_images():
    if 'google.colab' in str(get_ipython()):
        # use Google colab file upload widget
        print('Running on Google CoLab')
        upload_files()
    else:
        # use local files
        print('Not running on CoLab, using existing content, style files')


def pipeline():
    # get parameters, can change any variables with keywords
    params = get_params()

    # upload the images if on Google Colab, otherwise expects
    # to find content.png and style.png in root dir
    # upload_images()

    upload_image_file('content', params)
    upload_image_file('style', params)

    # run the style transfer process
    run(params)
