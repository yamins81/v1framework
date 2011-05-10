import subprocess
import re
import tempfile
import cPickle
import string
import os

def callfunc(fn,argfile):
    args = cPickle.loads(open(argfile).read())
    
    if isinstance(args,list):
        pos_args = args[0]
        kwargs = args[1]
    elif isinstance(args,dict):
        pos_args = ()
        kwargs = args
    else:
        pos_args = args
        kwargs = {}
        
    fn(*pos_args,**kwargs)
    
SGE_SUBMIT_PATTERN = re.compile("Your job ([\d]+) ")

def qsub(fn,args,queueName='all.q'):

    module_name = fn.__module__
    fnname = fn.__name__
    
    f = tempfile.NamedTemporaryFile(delete=False)
    argfile = f.name
    cPickle.dump(args,f)
    f.close()

    f = tempfile.NamedTemporaryFile(delete=False)
    scriptfile = f.name
    call_script = string.Template(call_script_template).substitute({'MODNAME':module_name,
                                                         'FNNAME':fnname,
                                                         'ARGFILE':argfile})
    f.write(call_script)
    f.close()    

    p = subprocess.Popen('qsub -l qname=' + queueName + ' ' + scriptfile,shell=True,stdout=subprocess.PIPE)
    sts = os.waitpid(p.pid,0)[1]

    if sts == 0:
        output = p.stdout.read()
        jobid = int(SGE_SUBMIT_PATTERN.search(output).groups()[0])
    else:
        raise 

    os.remove(argfile)
    os.remove(scriptfile)
    
    return jobid
    
call_script_template = """#!/bin/bash
#$$ -V
#$$ -cwd

python -c "import $MODNAME, sge_utils; sge_utils.callfunc($MODNAME.$FNNAME,'$ARGFILE')"

"""
