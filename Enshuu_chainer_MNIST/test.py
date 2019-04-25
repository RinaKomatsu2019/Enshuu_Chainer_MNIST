import os
import numpy as np
import chainer
from chainer import serializers
import chainer.functions as F
from model import myLinear

class test_myModel():
    def __init__(self,epoch):
        self.save_result_src='./myModel_Linear'

        self.my_model=myLinear(k_num=10)
        serializers.load_npz('{}/epoch{}_myLinearmodel.npz'.format(self.save_result_src,epoch),self.my_model)

    def calc_loss_and_accuracy(self):
        sum_loss=0
        sum_accu=0
        # STEP 1-----------------------------------------------------
        # Load MNIST
        train,test=chainer.datasets.get_mnist()
        test_num=len(test)
        # -----------------------------------------------------------
        # Set Batch Size
        batch_size=100
        perm=np.random.permutation(test_num)
        proceed=0
        for i in range(0,len(perm),batch_size):
            # STEP 2---------------------------------------------
            input,target=train[perm[i:i+batch_size]]
            # Reshape each data
            input=input.reshape(input.shape[0],input.shape[1])
            target=np.array(target,np.int32)
            # increment
            proceed+=input.shape[0]
            # input to model
            output=self.my_model(input)
            # ---------------------------------------------------
            # STEP3----------------------------------------------
            loss=F.softmax_cross_entropy(output,target)
            accuracy=F.accuracy(output,target)
            # ---------------------------------------------------
            # Add sum loss and sum accuracy
            sum_loss+=loss.data*input.shape[0]
            sum_accu+=accuracy.data*input.shape[0]
            print("test {}/{}".format(proceed,test_num))
            print("\t loss:{} accuracy:{}".format(loss.data,accuracy.data))

        # Calc loss average and accuracy average
        ave_loss=sum_loss/test_num
        ave_accu=sum_accu/test_num
        print('----------------------------------------')
        print('Test Result')
        print('\t ave loss:{} ave accuracy:{}'.format(ave_loss,ave_accu))

    def visualize_recognition(self):
        import matplotlib.pyplot as plt
        # STEP 1-----------------------------------------------------
        # Load MNIST
        train,test=chainer.datasets.get_mnist()
        test_num=len(test)
        # -----------------------------------------------------------
        # Set Batch Size
        vis_num=9
        perm=np.random.permutation(test_num)
        # STEP 2---------------------------------------------
        input,target=train[perm[0:0+vis_num]]
        # Reshape each data
        input=input.reshape(input.shape[0],input.shape[1])
        target=np.array(target,np.int32)
        # ---------------------------------------------------
        # STEP 3---------------------------------------------
        output=self.my_model(input)
        # ---------------------------------------------------
        plt.figure(1)
        # Visualize Process
        for i in range(0,vis_num):
            # Set to image
            img=np.array(input[i]*255.0,np.uint8)
            img=np.reshape(img,(28,28))
            # Get target answer
            target_answer=target[i]
            # Get output answer
            output_answer=np.argmax(output[i].data)

            # Append image and label to subplot
            plt.subplot(3,3,i+1)
            plt.gray()
            plt.imshow(img)
            plt.xlabel('real answer:{}\n model answer:{}'.format(target_answer,output_answer))
        # Adjust 
        plt.tight_layout()
        # Save image
        plt.savefig('{}/result.png'.format(self.save_result_src))
        # Visualize
        plt.show()



def test(**karg):
    epoch=karg['epoch']
    mode=karg['mode']
    t=test_myModel(epoch)
    if mode=='calc':
        t.calc_loss_and_accuracy()
    elif mode=='vis':
        t.visualize_recognition()
    else:
        print('Invalid Mode!')

if __name__ =='__main__':
    epoch=10
    mode='calc'
    #mode='vis'
    test(epoch=epoch,mode=mode)
