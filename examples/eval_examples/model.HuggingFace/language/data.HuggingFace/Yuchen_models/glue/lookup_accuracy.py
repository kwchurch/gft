
import sys,json,os

def lookup(f):
    try:
        p = os.path.join(f, 'eval_results.json')
        with open(p, 'r') as fd:
            j = json.loads(fd.read())
            for fig in ['eval_accuracy', 'eval_pearson']:
                if fig in j:
                    return j[fig]
    except:
        return 'NA'


for line in sys.stdin:
    f = line.rstrip()
    print(f + '\t' + str(lookup(f)))
    
