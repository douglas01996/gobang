import os
from random import randint



class gobang(object):
    def __init__(self,win_cond=4,board_size=8,playTimes=0,alpha=0.01):
        self.win_cond=win_cond
        self.playTimes=playTimes
        self.alpha=alpha
        self.board_size=board_size
        self.n_size=board_size*board_size
        self.tiles=[0,1,2]
        self.states=[]
        self.V=[]
        self.board=[0]*self.n_size

    def initBoard(self):
        for i in range(0,self.n_size):
            self.board[i] = 0


    def determineValue(self,_board,player): 
        won = self.hasWinner(_board)
  
        # win
        if won==1:
            if player==1:
                return 1.0
            else:
                return 0.0
        # draw
        elif won==-1:
            return 0.0
        else:
            return 0.5

    def switchPlayer(self,player):
        if player == 1:
            return 2
        else:
            return 1

    def printBoard(self):
        _board=self.board
        size = len(_board)
        for index in range(0, size):
            if _board[index] == 1:
                print( '  X',end=' ')
            elif _board[index] == 2:
                print( '  O',end=' ')
            else:
                print( '%3d' %(index+1),end=' ')
            if 0 == ((index + 1) %self.board_size):
                print()
  
    def hasWinner(self,_board):
        for player in range(1, 3):
            tile = self.tiles[player]
    
    # check horizontal
            for i in range(0, self.board_size):
                temp_sum=0
                for j in range(0,self.board_size):
                    if _board[i*self.board_size+j]==tile:
                        temp_sum=temp_sum+1
                        if temp_sum > self.win_cond:
                            return 1
                    else:
                        temp_sum=0
           
            # check vertical
            for i in range(0, self.board_size):
                temp_sum=0
                for j in range(0,self.board_size):
                    if _board[j*self.board_size+i]==tile:
                        temp_sum=temp_sum+1
                        if temp_sum > self.win_cond:
                            return 1
                    else:
                        temp_sum=0
           
    # check backward diagonal
            for i in range(0, self.board_size-self.win_cond):
                temp_sum=0
                for j in range(0,self.board_size-i):
                    if _board[(i+j)*self.board_size+j]==tile:
                        temp_sum=temp_sum+1
                        if temp_sum > self.win_cond:
                            return 1
                    else:
                        temp_sum=0
            for i in range(1, self.board_size-self.win_cond):
                temp_sum=0
                for j in range(0,self.board_size-i):
                    if _board[j*self.board_size+i+j]==tile:
                        temp_sum=temp_sum+1
                        if temp_sum > self.win_cond:
                            return 1
                    else:
                        temp_sum=0
         
    # check forward diagonal
            for i in range(0, self.board_size-self.win_cond):
                temp_sum=0
                for j in range(0,self.board_size-i):
                    if _board[(i+j)*self.board_size+self.board_size-j]==tile:
                        temp_sum=temp_sum+1
                        if temp_sum > self.win_cond:
                            return 1
                    else:
                        temp_sum=0
            for i in range(1, self.board_size-self.win_cond):
                temp_sum=0
                for j in range(0,self.board_size-i):
                    if _board[j*self.board_size+self.board_size-i-j]==tile:
                        temp_sum=temp_sum+1
                        if temp_sum > self.win_cond:
                            return 1
                    else:
                        temp_sum=0
  
          # check for draw
        if 0 in _board:
            return 0
        else:
    # -1 is for draw match
            return -1

    def updateBoard(self,_board, player, index):
        if _board[index] == 0:
            _board[index] = player
            return True
        return False
  

    def getListOfBlankTiles(self):
        blanks = []
        for i in range(0, self.board_size*self.board_size):
            if self.board[i] == 0:
                blanks.append(i)
        return blanks


# State-Value Function V(s)
# V(s) = V(s) + alpha [ V(s') - V(s) ]
# s  = current state
# s' = next state
# alpha = learning rate
    def updateEstimateValueOfS(self,sPrime, s):
        self.V[s] = self.V[s] + self.alpha*(V[sPrime] - V[s])

    def updateV(self,x,prev_x):
        if x not in self.states:
            self.states.append(x)
            self.V.append(self.determineValue(x,1))
        if prev_x not in self.states:
            self.states.append(prev_x)
            self.V.append(self.determineValue(prev_x,1))
        sPrime=self.states.index(x)
        s=self.states.index(prev_x)
        self.V[s] = self.V[s] + self.alpha*(self.V[sPrime] - self.V[s])
        return self.V[s]
        

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# PROGRAM STARTS HERE
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# step-size parameter (rate of learning)
    def init_cnn(self):
        self.initBoard()
        self.count=0
        self.count_max=1000
        self.init_cnn_recur(1)
        self.count=0
        self.count_max=1000
        self.init_cnn_recur(2)
        return self.states,self.V

    def init_cnn_recur(self,player):
        won=self.hasWinner(self.board)
        self.count=self.count+1
        if self.count>self.count_max:
            return
        if won==1 or won==-1:
            return
        else:
            for i in range(0,self.board_size*self.board_size):
                if self.board[i]==0:
                    self.board[i]=player
                    if self.board[:] not in self.states:
                        self.states.append(self.board[:])
                        self.V.append(self.determineValue(self.board,player))
                    self.init_cnn_recur(self.switchPlayer(player))
                    self.board[i]=0

    def rule(self):
        nextmoves=self.getListOfBlankTiles()
        rule1_moves=[]
        rule2_moves=[]
        for i in nextmoves:
            #win
            self.board[i]=self.player
            won=self.hasWinner(self.board)
            if won==1:
                rule1_moves.append(i)
                break
            self.board[i]=0
            #gonna lose
            self.player=self.switchPlayer(self.player)
            self.board[i]=self.player
            won=self.hasWinner(self.board)
            if won==1:
                rule2_moves.append(i)
            self.player=self.switchPlayer(self.player)
            self.board[i]=0
        return rule1_moves,rule2_moves
            
                    
    def new_game(self):
        self.initBoard()
        self.exploreRate = 0.1
        self.player=1
        self.preIndex=0
        self.maxIndex=0
        self.player = self.switchPlayer(self.player)
        self.firstPlay = True       # flag for first play of a game episode
