import os
import numpy as np
import chainer
from chainer import optimizers,serializers
import chainer.functions as F
from model import myLinear

def train_myLinear(n_epoch):
    # Create save model dir
    save_dir='./myModel_Linear'
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)
    # Setup model
    my_model=myLinear(k_num=10)
    optimizer=optimizers.Adam()
    optimizer.setup(my_model)
    # STEP 1-----------------------------------------------------
    # Load MNIST
    train,test=chainer.datasets.get_mnist() # First, maybe take time
    train_num=len(train)
    test_num=len(test)
    # -----------------------------------------------------------
    # Set Batch Size
    batch_size=100
    for epoch in range(0,n_epoch):
        perm=np.random.permutation(train_num)
        proceed=0
        for i in range(0,len(perm),batch_size):
            # STEP 2---------------------------------------------
            input,target=train[perm[i:i+batch_size]] # input is img data(np.float32), target is label(int)
            # Reshape each data
            input=input.reshape(input.shape[0],input.shape[1])
            target=np.array(target,np.int32)
            # increment
            proceed+=input.shape[0]
            # input to model
            output=my_model(input)
            # ---------------------------------------------------
            # STEP3----------------------------------------------
            loss=F.softmax_cross_entropy(output,target)
            accuracy=F.accuracy(output,target)
            # Backward model
            my_model.cleargrads()
            loss.backward()
            optimizer.update()
            # ---------------------------------------------------
            print("epoch:{} {}/{}".format(epoch+1,proceed,train_num))
            print("\t loss:{} accuracy:{}".format(loss.data,accuracy.data))

    # Save trained model
    serializers.save_npz('{}/epoch{}_myLinearmodel.npz'.format(save_dir,n_epoch),my_model)

if __name__ =='__main__':
    train_myLinear(n_epoch=10)

