import tensorflow as tf
import numpy as np
import json
import edward as ed
from edward.models import *
import random
import sys
import keras
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Dropout, Reshape
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
import os


class BRNN(tf.nn.rnn_cell.GRUCell):
    def __init__(self, num_units, activation=None, reuse=None, kernel_initializer=None,
                 bias_initializer=None, inputsize=None, dictin=None, trainable=True):
        super(BRNN, self).__init__(num_units, activation, reuse, kernel_initializer, bias_initializer)
        self.scalev=0.005
        self.w=Normal(loc=(tf.glorot_normal_initializer()([inputsize, 2*num_units])), scale=tf.ones([inputsize, 2*num_units]))
        self.b=Normal(loc=tf.glorot_normal_initializer()([2*num_units]), scale=tf.ones([2*num_units]))

        self.postw=Normal(loc=tf.get_variable(shape=[inputsize, 2*num_units], name=str(len(dictin))+'locpostw', trainable=trainable)
                          ,scale=tf.Variable(tf.ones([inputsize, 2*num_units])*self.scalev, name=str(len(dictin))+'scalepostw',
                                             trainable=trainable))
        self.postb=Normal(loc=tf.get_variable(shape=[2*num_units], name=str(len(dictin))+'locpostb', trainable=trainable)
                          , scale=tf.Variable(tf.ones([2*num_units])*self.scalev, name=str(len(dictin))+'scalepostb', trainable=trainable))

        self.w1=Normal(loc=tf.glorot_normal_initializer()([inputsize, num_units]), scale=tf.ones([inputsize, num_units]))
        self.b1=Normal(loc=tf.glorot_normal_initializer()([num_units]), scale=tf.ones([num_units]))
        self.postw1=Normal(loc=tf.get_variable(shape=[inputsize, num_units], name=str(len(dictin))+'locpostw1', trainable=trainable),
                           scale=tf.Variable(tf.ones([inputsize, num_units])*self.scalev, name=str(len(dictin))+'scalepostw1',
                                             trainable=trainable))
        self.postb1=Normal(loc=tf.get_variable(shape=[num_units], name=str(len(dictin))+'locpostb1', trainable=trainable),
                           scale=tf.Variable(tf.ones([num_units])*self.scalev, name=str(len(dictin))+'scalepostb1', trainable=trainable))

        dictin[self.w]=self.postw
        dictin[self.b]=self.b
        dictin[self.w1]=self.postw1
        dictin[self.b1]=self.postb1

    def call(self, inputs, state):
        self._gate_linear=tf.matmul(tf.concat([inputs, state],1), self.w)+self.b
        r,u=tf.split(tf.nn.sigmoid(self._gate_linear), 2, 1)
        r_state=r*state
        self._candidate_linear=tf.matmul(tf.concat([inputs, r_state],1), self.w1)+self.b1
        c=self._activation(self._candidate_linear)
        new_h=u*state+(1-u)*c
        return new_h, new_h
        

class model(object):
    def __init__(self, n_task, lr=1e-4, calsnp=False):
        self.diction={}
        self.scalev=0.001
        self.embedsize=100
        self.lr=lr
        trainable=True
        self.iter=tf.Variable(0, trainable=False)
        self.n_task=n_task
        initializer=tf.orthogonal_initializer()

        self.inputs=tf.placeholder(tf.int32, [None, None], name='inputs')
        self.y=tf.placeholder(tf.int32, [None], name='label')
        self.task=tf.placeholder(tf.int32, [None], name='task')

        self.emb=tf.get_variable(name='emb', shape=[4, self.embedsize], trainable=trainable)
        self.dense1=tf.layers.Dense(256, activation=tf.nn.relu, trainable=trainable)
        self.dense2=tf.layers.Dense(256, activation=tf.nn.relu, trainable=trainable)
        self.dense3=tf.layers.Dense(100, activation=tf.nn.sigmoid, trainable=trainable)
        self.dense4=tf.layers.Dense(3, activation=tf.nn.relu, trainable=trainable)
        self.dense5=tf.layers.Dense(256*2, trainable=trainable)
        
        def addalpha(shape, name, initializer, trainable):            
            w=self.addv(shape, name, initializer, trainable)
            
            w=self.dense1(w)
            w=self.dense2(w)
            w=self.dense3(w)
            return w




        self.z=addalpha([self.n_task,100], "selfz", initializer=tf.constant_initializer(random.random()),
                        trainable=trainable) 
        self.z=tf.concat([self.z, tf.ones([1,100], dtype=tf.float32)],0)
        self.w1=self.addv(name='w1', shape=[10, 100, 1, 512], 
                                initializer=initializer, trainable=trainable)
        self.w2=self.addv(name='w2', shape=[16, 1024, 256], 
                                initializer=initializer, trainable=trainable)
        self.w3=self.addv(name='w3', shape=[16, 256, 256], 
                                initializer=initializer, trainable=trainable)
        self.w4=self.addv(name='w4', shape=[16, 256, 128], 
                                initializer=initializer, trainable=trainable)
        self.w5=self.addv(name='w5', shape=[32, 128, 128], 
                                initializer=initializer, trainable=trainable)
        self.w10=self.addv(name='w10',shape=[2*256, 2000], 
                                initializer=initializer, trainable=trainable)
        self.w11=self.addv(name='w11',shape=[2000,100], 
                                initializer=initializer, trainable=trainable)
        self.w12=self.addv(name='w12',shape=[100,1], 
                                initializer=initializer, trainable=trainable)

        with tf.variable_scope("pos/p"):
            self.rnncell=BRNN(256, inputsize=128+256, dictin=self.diction,trainable=trainable)
        with tf.variable_scope('pos/r'):
            self.rnncellb=BRNN(256, inputsize=128+256,dictin=self.diction, trainable=trainable)
      
        self.optimizer=tf.train.AdamOptimizer(lr)
        self.scaldict={item:0.0001 for item in self.diction}
        self.sess=tf.Session()

    def net(self, inputs, calsnp=False):
        netinput=tf.expand_dims(tf.nn.embedding_lookup(self.emb, inputs), -1)
        output=self.cnnnet(netinput,  calsnp)
        result=tf.nn.relu(tf.matmul(output, self.w10))
        result=tf.nn.relu(tf.matmul(result, self.w11))
        z=tf.nn.embedding_lookup(self.z, self.task)
        if calsnp:
            z=1
        result=result*z
        return result

    def fine_tune(self,fineinput,cnnnet=None, task=None):
        n=tf.shape(fineinput)[0]
        if task<self.n_task:
            print "####################USING SINGLE TASK##########################"
            fineinput_=fineinput*tf.nn.embedding_lookup(self.z, self.task)
        elif task==self.n_task:
            print "####################USING ALL TASK##############################"
            fineinput_ = tf.expand_dims(fineinput, 1) * tf.expand_dims(self.z, 0)
            fineinput_=tf.reshape(fineinput_,[n*(self.n_task+1), 100])
            fineinput_=tf.matmul(fineinput_, self.w12)
            fineinput_=tf.reshape(fineinput_, [n,self.n_task+1])
        else:
            print "####################USING BACKGROUND INFORMATION###############"
            fineinput_=fineinput
        with tf.variable_scope("wwres"):
            p=tf.concat([ fineinput_[:n/2],fineinput_[n/2:], fineinput_[:n/2]-fineinput_[n/2:],
                          fineinput_[n/2:]-fineinput_[:n/2]],1)
            p=tf.layers.batch_normalization(p)
            p=tf.layers.dense(p, 1024, activation=tf.nn.relu)
            p=tf.layers.batch_normalization(p)
            p=tf.layers.dense(p, 512, activation=tf.nn.relu)
            p=tf.layers.batch_normalization(p)
            p=tf.layers.dense(p, 256, activation=tf.nn.relu)
            p=tf.layers.batch_normalization(p)
            p=tf.layers.dense(p, 128, activation=tf.nn.relu)
            p=tf.layers.batch_normalization(p)
            res=tf.layers.dense(p, 1)
            res=tf.reduce_sum(tf.layers.batch_normalization((res)),1)
        return res

    def addv(self, shape, name, initializer, trainable):            
        w=Normal(loc=tf.zeros(shape), scale=tf.ones(shape)*100.0, name=name)
        loc1=tf.get_variable(name='loc'+name, shape=shape, initializer=initializer, trainable=trainable, regularizer=tf.contrib.layers.l2_regularizer(1e-4))
        scale1=tf.Variable(tf.ones(shape=shape)*self.scalev, name=name+'scale', trainable=trainable)
        postw=Normal(loc=loc1,scale=scale1, name='post'+name)
        self.diction[w]=postw
        return w

    def cnnnet(self, inputs,  calsnp=False):
        with tf.variable_scope("cnn", reuse=tf.AUTO_REUSE):
            cnn1=tf.nn.relu(tf.nn.conv2d(inputs, self.w1, [1,1,1,1],padding='VALID'))
            cnn2=tf.nn.max_pool(cnn1, [1,50,1,1],[1,1,1,1], padding='SAME')
            cnn1=tf.concat([cnn1, cnn2],3)

            cnn1=tf.reshape(cnn1, [-1,tf.shape(cnn1)[1], 1024])
            cnn1=tf.nn.relu(tf.nn.conv1d(cnn1, self.w2, 4, padding='VALID'))
            cnn1=tf.nn.relu(tf.nn.conv1d(cnn1, self.w3, 1, padding='VALID'))
            cnn1=tf.nn.relu(tf.nn.conv1d(cnn1, self.w4, 3, padding='VALID'))
    
            cnn1=tf.nn.relu(tf.nn.conv1d(cnn1, self.w5, 1, padding='VALID'))
            cnn1=tf.layers.batch_normalization(cnn1)


        cnn1=tf.reshape(cnn1, [-1, 41, 128])
        o, state=tf.nn.bidirectional_dynamic_rnn(self.rnncell, self.rnncellb, cnn1, dtype=tf.float32)
        at=self.dense5(self.dense4(tf.concat(o,2)))
        at=at*tf.concat(o, 2)
        at=tf.expand_dims(tf.nn.softmax(tf.reduce_sum(at, 2),1),2)
        output=tf.reduce_sum(tf.concat(o, 2)*at,1)#
        self.cnnoutput=output
        return output


    def random_sample(self, n, t, cglist=None, dataset=None):
        res=[]
        if cglist is not None:
            for item in cglist:
                try:
                    res.append(dataset[random.randint(0,len(dataset)-1)])
                except:
                    print item
            return res
        for j in range(t):
            res.append(random.randint(0, n)%n)
        return res


    def train(self, traindata,  iteration, batch_size, traindir, save_step=50, random_neg=True):
        import time
        start=time.time()
        start_c=time.clock()
        result=self.net(self.inputs )
        result=tf.matmul(result, self.w12)
        lossvalue=-result*tf.cast(tf.reshape(self.y, [ -1,1]), tf.float32)
        lossvalue=Bernoulli(logits=lossvalue)
        resultv=tf.zeros(tf.shape(lossvalue), dtype=tf.int32)
        inference=ed.ReparameterizationKLKLqp(self.diction, data={lossvalue:resultv })
        random.shuffle(traindata)
        self.sess=ed.get_session()
        inference.initialize(n_samples=1, optimizer=self.optimizer, n_iter=5000, logdir="./log",
                             kl_scaling=self.scaldict)
        saver=tf.train.Saver(max_to_keep=500)
        self.sess.run(tf.global_variables_initializer())

        if tf.train.get_checkpoint_state(traindir):
            saver.restore(self.sess, tf.train.latest_checkpoint(traindir))
            print "loading from train dir"
        else:
            print "building new model"

        trainneg_=[[item for item in jtem if item[-1]<1] for jtem in traindata]
        negset={}

        trainpos=[[item for item in jtem if item[-1]>0] for jtem in traindata]
        negset=[[item for item in jtem if item[-1]<1] for jtem in traindata]
        sys.stdout.flush()
        for i in range(iteration):
            trainseq=[]
            traintag=[]
            task=[]
            for j in range(self.n_task):
                if random_neg:
                    ilist=[(i*batch_size/2+w)%len(trainpos[j]) for w in range(batch_size/2)]
                    temptrain=[trainpos[j][w] for w in ilist]
                    ilist=[(i*batch_size/2+w)%len(negset[j]) for w in range(batch_size/2)]
                    temptrain+=[negset[j][w] for w in  ilist]
                else:
                    ilist=self.random_sample(len(traindata[j]), batch_size)
                    temptrain=[traindata[j][w] for w in ilist]
                tseq=[item[0] for item in temptrain]
                ttag=[(item[-1]-0.5)*2 for item in temptrain]
                trainseq=trainseq+(tseq)
                traintag=traintag+(ttag)
                task=task+[j for item in tseq]
            info_dict=inference.update({self.inputs:trainseq, self.y:traintag, self.task:task})
            del trainseq
            del traintag

            iterr=self.sess.run(self.iter)
            self.sess.run(tf.assign(self.iter, iterr+1))
            if iterr%save_step==0:
                saver.save(self.sess, os.path.join(traindir,'model.ckpt'), global_step=iterr)
                end=time.time()
                end_c=time.clock()
                print "ITER %d TIME %f SAVING MODEL" %(iterr, end-start)
                sys.stdout.flush()

    def test(self,testdata,load_path,i,batchsize=400, n_samples=100):
        results=self.net(self.inputs, i)
        results=tf.matmul(results, self.w12)#
        sess=self.sess#
        if i==0:
            collection=[item for item in tf.all_variables() if 'ww' not in item.name]
            saver=tf.train.Saver()
            saver.restore(sess, load_path)
        loc_copy=ed.copy(results, self.diction)

        w=0
        n=len(testdata)
        result=[]
        tk=i
        while(w<n):
            temp=testdata[w:w+batchsize]
            seq=[item[0] for item in temp]
            task=[tk for i in seq]
            label=[item[-1] for item in temp]
            feed_dict={self.inputs:seq, self.task:task}
            probs=([sess.run(loc_copy, feed_dict) for i in range(n_samples)])
            loc=np.mean(probs,0).tolist()
            var=np.var(probs,0).tolist()
            res=[[i,j, l] for i,j,l in zip(loc,var, label)]
            result.extend(list(res))
            
            w+=batchsize
        return result

    def calSNP(self,train, testdata, load_path, n_sample1=100, traindir=None, task=None):
        from sklearn.metrics import *
        import time
        startt=time.time()
        startc=time.clock()
        sess=ed.get_session()
        result=self.net(self.inputs, calsnp=True)
        fine_tune=self.fine_tune(result, self.cnnoutput, task)


        fine_tune_=Bernoulli(logits=(fine_tune))
        inference=ed.ReparameterizationKLKLqp(self.diction, data={fine_tune_:tf.reshape(tf.cast(self.y, tf.int32),[-1])})
        inference.initialize(n_samples=1, optimizer=self.optimizer, n_iter=5000,
                             kl_scaling=self.scaldict)
        y_copy=ed.copy(fine_tune,self.diction) 
        collection=[item for item in tf.all_variables() if 'ww' not in item.name]

        w=0
        n=len(testdata)
        result=[]
        pos=[item for item in testdata if item[-1]>0]
        neg=[item for item in testdata if item[-1]<1]
        b=len(neg)/len(pos)
        n1=len(pos)/5
        n2=len(neg)/5
        res=[]
        if task is None:
            task=self.n_task+1
        elif task>self.n_task:
            task=self.n_task+1
        test=testdata
        print "START FINE-TUNING"
        sess.run(tf.global_variables_initializer())
        saver=tf.train.Saver(collection)
        saver.restore(sess, load_path)
        random.shuffle(train)
        for k in range(len(train)*15/150+1):
            klist=self.random_sample(len(train),150)
            temptrain=[train[w] for w in klist]
            trainseq=[]
            trainseq=[item[0] for item in temptrain]+[item[1] for item in temptrain]
            traintag=[max(item[-1],0) for item in temptrain]
            ttask=[task for item in trainseq]
            info_dict=inference.update({self.inputs:trainseq, self.y:traintag, self.task:ttask})

            trainseq=[item[1] for item in temptrain]+[item[0] for item in temptrain]
            traintag=[max(item[-1],0) for item in temptrain]
            ttask=[task for item in trainseq]
            info_dict=inference.update({self.inputs:trainseq, self.y:traintag, self.task:ttask})
            saver=tf.train.Saver()
            if traindir is not None and k%200==0:
                iterr=sess.run(self.iter)
                sess.run(tf.assign(self.iter, iterr+1+i*5+1))
                saver.save(sess, traindir+'model.ckpt', global_step=iterr)
        tempn=len(test)
        tw=0
        while (tw<tempn):
            temp=test[tw:tw+150]
            inputs=[]
            inputs=[item[0] for item in temp]+[item[1] for item in temp]
            label=[item[-1] for item in temp]
            ttask=[task for item in inputs]
            pred=np.array([sess.run(y_copy, {self.inputs:inputs, self.task:ttask}) for i in range(n_sample1)])

            mean=np.mean(pred,0).tolist()
            var=np.var(pred,0).tolist()
            res.extend([[v1,v2,v3] for v1,v2,v3 in zip(mean, var, label)]  )
            del pred
            del mean
            del var
            tw+=150
        endt=time.time()
        endc=time.clock()
        r=[(item[0]) for item in res]
        l=[item[-1] for item in res]
        fpr, tpr, t=roc_curve(l,r,pos_label=1)
        print "RUNNING TIME %f, AUC %f" %(endt-startt, auc(fpr, tpr))
        sys.stdout.flush()
        return res
