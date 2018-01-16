import sys

funcs = []

def cmd():
    def add_func(func):
        varnames = func.__code__.co_varnames[:func.func_code.co_argcount]
        defaults =  func.__defaults__
        
        m = {}
        m['func'] = func
        m['name'] = func.__name__
        m['doc'] = func.__doc__
        m['args'] = []
        for i, a in enumerate(varnames):
            if(a != 'self'):
                arg = {}
                arg['var'] = a
                default_index = len(varnames) - 1 - i
                if(defaults and 0 <= default_index < len(defaults)):
                    arg['default'] = defaults[default_index]
                else:   
                    arg['default'] = None
                m['args'].append(arg)
        funcs.append(m)
    return add_func
        
def parse_cmd_args():
    cmd_args = sys.argv[1:]
    
    if('--help' in cmd_args or '-h' in cmd_args):
        return False
        
    if(not cmd_args):
        return False
    
    func = next((f for f in funcs if f['name'] == cmd_args[0]), None)
    
    if(not func):
        return False

    cmd_args.pop(0)

    args = []
    kwargs = {}
    
    arg_check = iter(func['args'])
    it = iter(cmd_args)
    for arg in it:
        if(arg.startswith('--')):
            arg = arg[2:]
            
            match = {}
            for i, a in enumerate(func['args']):
                if(a['var'] == arg):
                    match = func['args'].pop(i)
                    break
            
            if(not match):
                return False
                
            if(type(match['default']) is bool):
                kwargs[arg] = True
            else:
                kwargs[arg] = next(it)
        else:
            match = {}
            for i, a in enumerate(func['args']):
                if(not a['default'] and type(a['default']) is not bool):
                    match = func['args'].pop(i)
                    break

            if(not match):
                return False
                    
            args.append(arg)
    
    func['func'](*args, **kwargs)
    return True
            
def help():
    help_string = 'Usage:'
    for func in funcs:
        help_string += "\n  "
        help_string += func['name']
        for arg in func['args']:
            default = arg['default']
            if(default != None):
                if(type(default) is bool):
                    help_string += " --{var}".format(**arg)
                else: 
                    help_string += " --{var}=<{var}>".format(**arg)
            else:
                help_string += " <{var}>".format(**arg)
        
    print help_string
    
def run():
    if(not parse_cmd_args()):
        help()
