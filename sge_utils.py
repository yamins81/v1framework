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
    
import subprocess
import re

SGE_SUBMIT_PATTERN = re.compile("Your job ([\d]+) ")

def qsub(fn,args,queueName='all.q'):

    module_name = f.__module__
    fnname = f.__name__
    
    f = tempfile.NamedTemporaryFile()
    argfile = f.name
    cPickle.dump(args,f)
    f.close()

    f = tempfile.NamedTemporaryFile()
    scriptfile = f.name
    f.write(call_script)
    f.close()    
        
    p = subprocess.Popen('qsub -l qname=' + queueName + ' ' + scriptfile + ' ' + module_name + ' ' + fnname + ' ' + argfile,shell=True,stdout=subprocess.PIPE)
    sts = os.waitpid(p.pid,0)[1]

    if sts == 0:
        output = p.stdout.read()
        jobid = int(SGE_SUBMIT_PATTERN.search(output).groups()[0])
    else:
        raise 

    os.remove(argfile)
    os.remove(scriptfile)
    
    return jobid
    
call_script = """
#!/bin/bash
#$ -V
#$ -cwd

python -c "import $1, sge_utils; sge_utils.callfunc($1.$2,$3)"

"""