# utility function to return number of output variables given cfg
def get_num_of_joints(cfg):

    if cfg.data.joints_version == '1':
        return 11
    elif cfg.data.joints_version == '1s':
        return 4
    elif cfg.data.joints_version == '2':
        return 15
    elif cfg.data.joints_version == '2c':
        return 6
    elif cfg.data.joints_version == '3':
        return 15
    elif cfg.data.joints_version == '4':
        return 17
    elif cfg.data.joints_version == 'custom':
        return len(cfg.data.joint_names)
    else: # mode == 'plain'
        return 0