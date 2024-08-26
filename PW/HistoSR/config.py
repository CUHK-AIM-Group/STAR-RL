import time

class config:
    sampling_ratio = 30
    #-------------learning_related--------------------#
    batch_size = 12
    workers = 4
    iter_size = 2
    num_episodes = 20000
    test_episodes = 500 #default : 500
    early_break = True # default : False, early break during validation
    save_episodes = 20000 #default : 20000
    resume_model = '' #'model/8_28_21_16000.pth'
    resume_picker = ''
    resume_patchCritic = ''
    display = 100 # default: 100
    #-------------rl_related--------------------#
    pi_loss_coeff = 1.0
    v_loss_coeff = 0.25
    beta = 0.1
    c_loss_coeff = 0.5 # 0.005
    switch = 4
    warm_up_episodes = 1000
    episode_len = 8 # default: 8
    episode_len_patch = 6
    gamma = 0.5 # default: 0.5
    reward_method = 'abs'
    noise_scale = 0.2 #0.5
    #------------agent1-------------#
    switch_iter_agent1 = 4
    ac1_c_loss_coeff = 0.5 
    ac1_v_loss_coeff = 1
    ac1_warm_up_episodes = 1000
    #------------agent2-------------#
    use_RGB_actions = False # default : False
    no_addition_subtraction = False # default : False
    shift_add_sub = 3.0 # default : 3; 6,12,24,48
    useless_actions_list = ['box','bilateral','median'] # default : []
    # whether use HistoSR dataset
    use_HistoSR_dataset = True # default: True 
    data_degradation = 'bicubic' 
    data_root = '/path/to/data'
    histosr_data_train = data_root + '/'+data_degradation+'/train'
    histosr_data_test = data_root + '/' +data_degradation+'/test'
    
    test_list_dir =  data_root + '/Anno/test.txt'

    use_bicubic_upsample = False # default : False 
    
    hidden_feat_ch = 64 # default: 64
    #-----------test---------------#
    cut_edge_test = True # default: False; cut edge for testing
    #-------------continuous parameters--------------------#
    use_large_params = False # default : False
    use_small_params = False # default: False
    use_few_sm_params = False # default: False
    use_few_lg_params = False # default: False
    # assert use_small_params != use_large_params , 'cannot use large and small params at the same time.'

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
        'addition': 12
    }

    actions_RGB = {
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
        'addition': 12,
        'box_R': 13,
        'bilateral_R': 14,
        'median_R': 15,
        'Gaussian_R': 16,
        'Laplace_R': 17,
        'Sobel_v1_R': 18,
        'Sobel_v2_R': 19,
        'Sobel_h1_R': 20,
        'Sobel_h2_R': 21,
        'unsharp_R': 22,
        'subtraction_R': 23,
        'addition_R': 24,
        'box_G': 25,
        'bilateral_G': 26,
        'median_G': 27,
        'Gaussian_G': 28,
        'Laplace_G': 29,
        'Sobel_v1_G': 30,
        'Sobel_v2_G': 31,
        'Sobel_h1_G': 32,
        'Sobel_h2_G': 33,
        'unsharp_G': 34,
        'subtraction_G': 35,
        'addition_G': 36,
        'box_B': 37,
        'bilateral_B': 38,
        'median_B': 39,
        'Gaussian_B': 40,
        'Laplace_B': 41,
        'Sobel_v1_B': 42,
        'Sobel_v2_B': 43,
        'Sobel_h1_B': 44,
        'Sobel_h2_B': 45,
        'unsharp_B': 46,
        'subtraction_B': 47,
        'addition_B': 48
    }

    if no_addition_subtraction : 
        del actions['subtraction']
        del actions['addition']
        del actions_RGB['subtraction']
        del actions_RGB['addition']

    if len(useless_actions_list) > 0:
        for item in useless_actions_list:
            del actions[item]
            if use_RGB_actions:
                del actions_RGB[item]
                del actions_RGB[item+'_R']
                del actions_RGB[item+'_G']
                del actions_RGB[item+'_B']
        # rearrange 
        actions_list = sorted(actions.items(), key=lambda x: x[1], reverse=False)
        tmp_actions = {}
        for idx, item in enumerate(actions_list):
            tmp_actions[item[0]] = idx + 1
        actions = tmp_actions
        if use_RGB_actions:
            actions_list = sorted(actions_RGB.items(), key=lambda x: x[1], reverse=False)
            tmp_actions = {}
            for idx, item in enumerate(actions_list):
                tmp_actions[item[0]] = idx + 1
            actions_RGB = tmp_actions
        print("actions:", actions)
        print("actions_RGB:", actions_RGB)

    if use_RGB_actions:
        actions = actions_RGB

    num_actions = len(actions) + 1
    # each action = RG + RB + GB + RGB
    parameters_scale = {
        'Laplace': 0.2,
        'Sobel_v1': 0.2,
        'Sobel_v2': 0.2,
        'Sobel_h1': 0.2,
        'Sobel_h2':  0.2,
        'unsharp': 1.0,
    }
    if use_large_params:
        parameters_scale = {
            'Laplace': 1.0,
            'Sobel_v1': 1.0,
            'Sobel_v2': 1.0,
            'Sobel_h1': 1.0,
            'Sobel_h2':  1.0,
            'unsharp': 1.0,
        }
    if use_small_params:
        parameters_scale = {
            'Laplace': 0.1,
            'Sobel_v1': 0.1,
            'Sobel_v2': 0.1,
            'Sobel_h1': 0.1,
            'Sobel_h2':  0.1,
            'unsharp': 1.0,
        }
    if use_few_sm_params:
        parameters_scale = {
            'Laplace': 0.2,
            'Sobel_v1': 0.2,
            'Sobel_v2': 0.02,
            'Sobel_h1': 0.02,
            'Sobel_h2':  0.2,
            'unsharp': 1.0,
        }
    if use_few_lg_params:
        parameters_scale = {
            'Laplace': 0.2,
            'Sobel_v1': 0.2,
            'Sobel_v2': 1,
            'Sobel_h1': 1,
            'Sobel_h2':  0.2,
            'unsharp': 1.0,
        }

    #-------------patch picker-----------------#
    image_height = 192 # HR : 192x192 ; LR : 96x96
    image_width = 192
    factor = 2 #16 #2
    patch_height = image_height // factor
    patch_width = image_width // factor
    #-------------lr_policy--------------------#
    base_lr = 0.001
    # poly
    lr_policy = 'poly'
    policy_parameter = {
      'power': 1,
      'max_iter' : 40000,
    }

    #-------------folder--------------------#
    dataset = 'HistoSR'
    root = './dataset/HistoSR/' 
