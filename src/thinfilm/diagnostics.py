import numpy as np

def compute_metrics(y,yhat):
    sse=np.sum((yhat-y)**2); sst=np.sum((y-np.mean(y))**2); return 1.-sse/max(sst,1e-16), np.mean((yhat-y)**2), sse
