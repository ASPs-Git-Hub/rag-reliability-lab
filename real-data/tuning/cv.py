import sys, pickle, os
SP = os.environ["SCRATCH"]  # directory holding squad/dev-v2.0.json and the cached candidates (tune_cands.pkl)
sys.path.insert(0,"/Users/asp/code/freelance/proofs/rag-reliability-lab/real-data")

import run_eval_v2 as R
from rag2 import RAGv2, FEATURES
docs,qs=R.load(SP+"/squad/dev-v2.0.json","tune")
sy=RAGv2(docs)
c=R.candidates(sy,qs)
pickle.dump((qs,c),open(SP+"/tune_cands.pkl","wb"))
print("cached")
