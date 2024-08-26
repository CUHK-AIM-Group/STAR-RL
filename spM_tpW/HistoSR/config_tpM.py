import time

class config:
    sampling_ratio = 30
    #-------------learning_related--------------------#
    batch_size = 12 # default: 12
    workers = 4
    iter_size = 2
    num_episodes = 20000 # default : 20000
    test_episodes = 500 #default : 500
    save_episodes = 1000 #default : 20000 ; when value is larger than the total episodes, it doesn't save models
    resume_model = './logs/2_14_18_8/models/best_old_agent2.pth' # load pretrained pw
    resume_picker = ''
    resume_patchCritic = ''
    resume_agent1 = './logs/2_14_18_8/models/best_agent1.pth' # load pretrained spM
    display = 10 # default: 100
    display_learning_rate = 1000 
    #-------------rl_related--------------------#
    pi_loss_coeff = 1.0
    v_loss_coeff = 0.25
    beta = 0.1
    c_loss_coeff = 0.5 # 0.005
    switch = 4
    warm_up_episodes = 1000 # default: 1000
    episode_len = 3 # default : 3 ; T2 episode_len for agent2 during training
    episode_len_test = 3 # default : 3 (same as episode_len) ; T2 for test; episode_len for agent2 during testing 
    episode_len_patch = 8 # default : 6 ; T1
    gamma = 0.5 # default: 0.5
    gamma_agent1 = 0.5 # default : 1 
    accumulate_reward_agent1 = False # default: False
    reward_method = 'abs'
    noise_scale = 0.2 #0.5
    #------------agent1-------------#
    switch_iter_agent1 = 4
    ac1_c_loss_coeff = 0.5 
    ac1_v_loss_coeff = 0.25
    ac1_pi_loss_coeff = 1.0
    ac1_entropy_loss_coeff = 0.1 # default : 0.1
    ac1_v_loss_use_l1 = False # default : False

    ac1_warm_up_episodes = 1000 # default: 1000
    use_iou_reward = False # default : False
    mask_based_critic = False # default : False
    no_sigmoid = False # default : False ; set the last layer of critic as ReLU(IN(x))
    predict_shift = False # default : False ; predict the shift for the position
    predict_abs_pos = False # defualt: False; predict the absolute position for the image
    use_gt_critic = False # default : False
    use_discrete_action = True # default : False
    use_pixel_oriented_patch = False # default : False
    use_coordinate_classify_agent = True # default : False
    use_agent2_in_agent1 = True # default : False
    update_old_agent2_model_episode = 2000 # default : 2000, copy the agent2
    valid_old_agent2 = False # default : False ; During validation, we apply the old agent2 instead of current agent2
    T2_in_agent1 = 4 # default: 1 ; During trainig agent1, how many times we apply agent2 for SR image
    load_spectral_sr = False # defalut : False ; Load the SR image processed by spectral part
    spectral_sr_train_root = ''
    spectral_sr_test_root = ''# path for the sr image processed by spectral
    useless_actions_list = ['box','bilateral','median'] # default : []
    # settings for discriminator
    use_discrminator = True # default : False
    n_layers_discr = 4 # default : 5 
    base_lr_discr = 0.0001 # learning rate
    beta1_disc = 0.5 # momentum term of adam
    beta2_disc = 0.999 # momentum term of adam
    lr_policy_disc = 'linear'

    discriminator_in_agent2 = False # default : False ; For new agent 2
    discriminator_in_old_agent2 = False # default : False ; For old agent2
    reward_idx = 0 # default : 0
    add_subtract_shift = 3. # default : 3.
    # whether use HistoSR dataset
    use_HistoSR_dataset = True # default: True 
    data_degradation = 'bicubic' #'blur_bicubic_k_3_sig_1.0' #'bicubic' # 'bicubic'
    data_root = '/path/to/data'
    histosr_data_train = data_root + '/'+data_degradation+'/train'
    histosr_data_test = data_root + '/' +data_degradation+'/test'
    
    test_list_dir = data_root + '/Anno/test.txt'
    # ---------- discriminator ------------#
    use_action_input = True # defualt : False ; if True, use action as input of discriminator
    action_input_type = 'vector' # 'map' or 'vector'
    use_img_patches = False 
    discr_input_nc = 1 if use_action_input else 3 
    last_layer_ch = 1024 #512 for blurbicubic
    
    #-----------reward--------------#
    weight_reward_mae = 1.0 # default: 1.0/ 7.0
    bias_reward_mae = 0.0 # default: 0.0
    weight_reward_iou = 1.0 # default: 1.0
    bias_reward_iou = 0.0 # default: 0.0
    #-----------agent2--------------#
    hidden_feat_ch = 64 # default: 64 
    #------------ agent1------------#
    # use light agent1
    use_sm_agent1 = True # default : False
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
        'addition': 12
    }
    '''actions = {
        'box_RG' : 1,
        'box_RB' : 2,
        'box_GB' : 3,
        'bilateral_RG' : 4,
        'bilateral_RB' : 5,
        'bilateral_GB' : 6,
        'median_RG' : 7,
        'median_RB' : 8,
        'median_GB' : 9,
        'Gaussian_RG' : 10,
        'Gaussian_RB' : 11,
        'Gaussian_GB' : 12,
        'Laplace_RG' : 13,
        'Laplace_RB' : 14,
        'Laplace_GB' : 15,
        'Sobel_v1_RG' : 16,
        'Sobel_v1_RB' : 17,
        'Sobel_v1_GB' : 18,
        'Sobel_v2_RG' : 19,
        'Sobel_v2_RB' : 20,
        'Sobel_v2_GB' : 21,
        'Sobel_h1_RG' : 22,
        'Sobel_h1_RB' : 23,
        'Sobel_h1_GB' : 24,
        'Sobel_h2_RG' : 25,
        'Sobel_h2_RB' : 26,
        'Sobel_h2_GB' : 27,
        'unsharp_RG' : 28,
        'unsharp_RB' : 29,
        'unsharp_GB' : 30,
        'subtraction_RG' : 31,
        'subtraction_RB' : 32,
        'subtraction_GB' : 33,
        'addition_RG' : 34,
        'addition_RB' : 35,
        'addition_GB' : 36,
    }'''
    

    if len(useless_actions_list) > 0:
        for item in useless_actions_list:
            del actions[item]
        # rearrange 
        actions_list = sorted(actions.items(), key=lambda x: x[1], reverse=False)
        tmp_actions = {}
        for idx, item in enumerate(actions_list):
            tmp_actions[item[0]] = idx + 1
        actions = tmp_actions
        print("actions:", actions)


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
    #-------------patch picker-----------------#
    input_ch = 3
    image_height = 192 # HR : 192x192 ; LR : 96x96
    image_width = 192
    factor = 2
    patch_height = image_height // factor
    patch_width = image_width // factor
    #-------------lr_policy--------------------#
    base_lr = 0.001
    base_lr_agent1 = 0.001
    # poly
    lr_policy = 'poly'
    policy_parameter = {
      'power': 1,
      'max_iter' : 40000,
    }

    #-------------folder--------------------#
    dataset = 'HistoSR'
    root = './dataset/HistoSR/' 
