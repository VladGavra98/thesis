from envs.phlabenv import CitationEnv

try:
    from envs.lunarlander import LunarLanderWrapper
except:
    print('LunarLanderContinuous not available on this system')

def select_env (environemnt_name : str):
    _name = environemnt_name

    if 'lunar'  in _name.lower():
        wrapper = LunarLanderWrapper()
        return wrapper.env
    
    elif 'ph' in _name.lower():
        phlab_config, phlab_mode = _name.lower().split('_')[-2:]
        return CitationEnv(configuration=phlab_config, mode= phlab_mode )

    else:
        raise ValueError('Unknown environment type')

if __name__ == '__main__':

    name = 'phlab_symmetric'
    tokens = name.split('_')
    print(tokens[-1])