from envs.phlabenv import CitationEnv
from envs.lunarlander import LunarLanderWrapper

def select_env (environemnt_name : str):
    _name = environemnt_name

    if 'lunar'  in _name.lower():
        wrapper = LunarLanderWrapper()
        return wrapper.env
    
    elif 'ph' in _name.lower():
        phlab_config = _name.lower().split('_')[-1]
        return CitationEnv(configuration=phlab_config)

    else:
        raise ValueError('Unknown environment type')

if __name__ == '__main__':

    name = 'phlab_symmetric'
    tokens = name.split('_')
    print(tokens[-1])