def get_model(model_name, args):
    name = model_name.lower()
    if name == "l2p":
        from models.l2p import Learner
    elif name == "dualprompt":
        from models.dualprompt import Learner
    elif name == "coda_prompt":
        from models.coda_prompt import Learner
    elif name == 'coso':
        from models.coso import Learner
    else:
        assert 0
    return Learner(args)