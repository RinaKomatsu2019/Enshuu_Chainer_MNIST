import chainer
from chainer import Chain
import chainer.functions as F
import chainer.links as L

class myLinear(Chain):
    def __init__(self,k_num):
        self.k_num=k_num
        super(myLinear,self).__init__(
            ln1=L.Linear(None,1024),
            ln2=L.Linear(None,1024),
            ln3=L.Linear(None,self.k_num),
        )
    def __call__(self,x):
        h=F.relu(self.ln1(x))
        h=F.relu(self.ln2(h))
        h=self.ln3(h)

        return h