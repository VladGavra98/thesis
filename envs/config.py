from envs.phlabenv import CitationEnv
from envs.lunarlander import LunarLanderWrapper

def select_env (environemnt_name : str):
    _name = environemnt_name

    if 'lunar' or 'lander' in _name.lower():
        wrapper = LunarLanderWrapper()
        return wrapper.env
    
    elif 'ph' or 'citation' in _name.lower():
        
        return CitationEnv()

    else:
        raise ValueError('Unknown environment type')

