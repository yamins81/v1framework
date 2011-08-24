

def compute_instance(instance):
    run_with = instance['run_with']
    tokens = run_with.split('.')
    modulename = '.'.join(tokens[:-1])
    fnname = tokens[-1]
    mod = __import__(modulename,globals(),locals(),[run_with])
    return getattr(mod,fnname)(instance)

