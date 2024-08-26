import time

class config:
    sampling_ratio = 30
    #-------------learning_related--------------------#
    batch_size = 12
    workers = 4
    iter_size = 2
    num_episodes = 20000
    test_episodes = 500
    save_episodes = 20000
    resume_model = '' #'model/8_28_21_16000.pth'
    display = 100
    #-------------rl_related--------------------#
    pi_loss_coeff = 1.0
    v_loss_coeff = 0.25
    beta = 0.1
    c_loss_coeff = 0.5 # 0.005
    switch = 4
    warm_up_episodes = 1000
    episode_len = 3
    gamma = 1
    reward_method = 'abs'
    noise_scale = 0.2 #0.5
    #-------------continuous parameters--------------------#
    actions = {
        'box': 1,
        'bilateral': 2,
        'median': 3,
        'Gaussian': 4,
        'Laplace': 5,
        'Sobel_v1': 6,
        'Sobel_v2': 7,
        'Sobel_h1': 8,
        'Sobel_h2': 9,
        'unsharp': 10,
        'subtraction': 11,
    }
    num_actions = len(actions) + 1

    parameters_scale = {
        'Laplace': 0.2,
        'Sobel_v1': 0.2,
        'Sobel_v2': 0.2,
        'Sobel_h1': 0.2,
        'Sobel_h2':  0.2,
        'unsharp': 1.0,
    }

    #-------------lr_policy--------------------#
    base_lr = 0.001
    # poly
    lr_policy = 'poly'
    policy_parameter = {
      'power': 1,
      'max_iter' : 40000,
    }

    #-------------folder--------------------#
    dataset = 'MICCAI'
    root = 'hemorrhage/'
