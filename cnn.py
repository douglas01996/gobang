from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
import gzip
import os
import sys
import time
from random import randint
import copy
class ZIYINET(object):
    def __init__(self,game,n_board=8,n_hidden_1=2*2*64,n_hidden_2=64,batch_size=20,keep_prob=1):
        self.sess=tf.Session()
        self.game=game
        self.n_board=n_board
        self.n_input=n_board*n_board
        self.n_hidden_1=n_hidden_1
        self.n_hidden_2=n_hidden_2
        self.batch_size=batch_size
        self.keep_prob=keep_prob
        self.build_model()
        self.board_size=n_board*n_board

    def build_model(self):    
        self.x=tf.placeholder(tf.float32,[None,self.n_input])
        self.y=tf.placeholder(tf.float32,[None,1])
        self.W = {
            'wc1':tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.01)),
            'wc2':tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.01)),
            'wfc1':tf.Variable(tf.truncated_normal([self.n_hidden_1,self.n_hidden_2],stddev=0.01)),
            'out':tf.Variable(tf.truncated_normal([self.n_hidden_2,1],stddev=0.01))
        }

        self.biases = {
            'bc1':tf.Variable(tf.truncated_normal([32],stddev=0.01)),
            'bc2':tf.Variable(tf.truncated_normal([64],stddev=0.01)),
            'bfc1':tf.Variable(tf.truncated_normal([self.n_hidden_2],stddev=0.01)),
            'out':tf.Variable(tf.truncated_normal([1],stddev=0.01))
        }


        def conv2d(x,W,b,strides=1):
            x=tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding='SAME')
            x=tf.nn.bias_add(x,b)
            return tf.nn.relu(x)

        def maxpool2d(x,k=2):
            return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')

        self.input=tf.reshape(self.x,shape=[-1,self.n_board,self.n_board,1])
        
        self.conv1=conv2d(self.input,self.W['wc1'],self.biases['bc1'])
        self.conv1=maxpool2d(self.conv1,k=2)

        self.conv2=conv2d(self.conv1,self.W['wc2'],self.biases['bc2'])
        self.conv2=maxpool2d(self.conv2,k=2)

        self.fc1=tf.reshape(self.conv2,[-1,self.W['wfc1'].get_shape().as_list()[0]])
        self.fc1=tf.add(tf.matmul(self.fc1,self.W['wfc1']),self.biases['bfc1'])
        self.fc1=tf.nn.relu(self.fc1)
        if self.keep_prob<1:
            self.fc1=tf.nn.dropout(self.fc1,self.keep_prob)

        self.pred_v=tf.add(tf.matmul(self.fc1,self.W['out']),self.biases['out'])
    

    def test(self):
        self.sess=tf.Session()
        with self.sess:
            ckpt = tf.train.get_checkpoint_state('.')
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            while(1):
                self.game.new_game()
                self.game.printBoard()
                print
                while(1):
                    nextmoves=self.game.getListOfBlankTiles()
                    countmove=len(nextmoves)
                    if self.game.player==2:
                        userplay=int(input("Enter move :"))
                        self.game.board[userplay-1]=2
                    else:
                        #rule
                        rule1_moves,rule2_moves=self.game.rule()
                        cond=False
                        for i in rule1_moves:
                            self.game.board[i]=self.game.player
                            cond=True
                            break
                        if cond==False:
                            for i in rule2_moves:
                                self.game.board[i]=self.game.player
                                cond=True
                                break
                        if cond==False:
                            max_v=-100
                            max_idx=0
                            for i in nextmoves:
                                x=self.game.board
                                x[i]=1
                                temp=np.array(x)
                                temp.shape=(self.board_size,1)
                                temp=temp.transpose()
                                v=self.sess.run([self.pred_v],feed_dict={self.x:temp})
                                if v[0] > max_v:
                                    max_v=v[0]
                                    max_idx=i
                                x[i]=0
                            self.game.board[max_idx]=1
                    self.game.printBoard()
                    print
                    won=self.game.hasWinner(self.game.board)
                    if won==1:
                        print('end!')
                        break
                    elif won==-1:
                        print('draw!')
                        break
                    else:
                        self.game.player=self.game.switchPlayer(self.game.player)

            
                
    def train(self,max_iter=10000,alpha=0.01,learning_rate=0.001,start_epsilon=1.0,final_epsilon=0.05):
        self.saver=tf.train.Saver()
        with self.sess:

            self.max_iter=max_iter
            self.alpha=alpha
            self.learning_rate=learning_rate
            
            self.step=tf.Variable(0,trainable=False)

            self.loss=tf.reduce_sum(tf.square(self.pred_v-self.y))
            self.optim=tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            


            steps=range(0,0+self.max_iter)
            
            init=tf.initialize_all_variables() 
            self.sess.run(init)
            init_x,init_y=self.game.init_cnn()
            for x,y in zip(init_x,init_y):
                temp=np.array(x)
                temp.shape=(self.board_size,1)
                temp=temp.transpose()
                temp2=np.array(y)
                temp2.shape=(1,1)
                self.sess.run([self.optim],feed_dict={self.x:temp,self.y:temp2})
                
            self.game.new_game()
            prev_x=0
            for step in steps:
                print(step)
                nextmoves=self.game.getListOfBlankTiles()
                countmove=len(nextmoves)
                exploring =False
                if self.game.player==2:
                    #rule
                    rule1_moves,rule2_moves=self.game.rule()
                    cond=False
                    for i in rule1_moves:
                        self.game.board[i]=self.game.player
                        cond=True
                        break
                    if cond==False:
                        for i in rule2_moves:
                            self.game.board[i]=self.game.player
                            cond=True
                            break
                    #random
                    if cond==False:
                        userPlay=nextmoves[randint(0,countmove-1)]
                        self.game.board[userPlay]=2
                else:
                    #rule
                    rule1_moves,rule2_moves=self.game.rule()
                    cond=False
                    for i in rule1_moves:
                        self.game.board[i]=1
                        if not self.game.firstPlay:
                            temp=np.array(self.game.board)
                            temp.shape=(self.board_size,1)
                            temp=temp.transpose()
                            y=self.game.updateV(self.game.board,prev_x)
                            temp2=np.array(y)
                            temp2.shape=(1,1)
                            self.sess.run([self.optim],feed_dict={self.x:temp,self.y:np.array(temp2)})
                        prev_x=copy.deepcopy(self.game.board)
                        self.game.firstPlay=False
                        cond=True
                        break
                    
                    if cond==False:
                        for i in rule2_moves:
                            self.game.board[i]=1
                            if not self.game.firstPlay:
                                temp=np.array(self.game.board)
                                temp.shape=(self.board_size,1)
                                temp=temp.transpose()
                                y=self.game.updateV(self.game.board,prev_x)
                                temp2=np.array(y)
                                temp2.shape=(1,1)
                                self.sess.run([self.optim],feed_dict={self.x:temp,self.y:np.array(temp2)})
                            prev_x=copy.deepcopy(self.game.board)
                            self.game.firstPlay=False
                            cond=True
                    
                    if cond==False:
                        ex=randint(1,100)/100.0
                        if ex<=self.game.exploreRate:
                            userPlay=nextmoves[randint(0,countmove-1)]
                            exploring=True
                            self.game.board[userPlay]=1
                            if not self.game.firstPlay:
                                temp=np.array(self.game.board)
                                temp.shape=(self.board_size,1)
                                temp=temp.transpose()
                                y=self.game.updateV(self.game.board,prev_x)
                                temp2=np.array(y)
                                temp2.shape=(1,1)
                                self.sess.run([self.optim],feed_dict={self.x:temp,self.y:np.array(temp2)})
                            prev_x=copy.deepcopy(self.game.board)
                        else:
                            #normal
                            max_v=-100
                            max_idx=0
                            for i in nextmoves:
                                self.game.board[i]=1
                                temp=np.array(self.game.board)
                                temp.shape=(self.board_size,1)
                                temp=temp.transpose()
                                v=self.sess.run([self.pred_v],feed_dict={self.x:temp})
                                if v[0] > max_v:
                                    max_v=v[0]
                                    max_idx=i
                                self.game.board[i]=0
                            self.game.board[max_idx]=1
                            if not self.game.firstPlay:
                                y=self.game.updateV(self.game.board,prev_x)
                                temp=np.array(self.game.board)
                                temp.shape=(self.board_size,1)
                                temp=temp.transpose()
                                temp2=np.array(y)
                                temp2.shape=(1,1)
                                self.sess.run([self.optim],feed_dict={self.x:temp,self.y:np.array(temp2)})
                            prev_x=copy.deepcopy(self.game.board)
                            self.game.firstPlay=False
                    #win?
                won=self.game.hasWinner(self.game.board)
                if won==1:
                    if self.game.player==2:
                        y=self.game.updateV(self.game.board,prev_x)
                        temp=np.array(self.game.board)
                        temp.shape=(self.board_size,1)
                        temp=temp.transpose()
                        temp2=np.array(y)
                        temp2.shape=(1,1)
                        self.sess.run([self.optim],feed_dict={self.x:temp,self.y:temp2})
                    self.game.new_game()
                elif won==-1:
                    self.game.new_game()
                else:
                    self.game.player=self.game.switchPlayer(self.game.player)   
            self.saver.save(self.sess, './model.ckpt')  


