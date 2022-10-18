
#ライブラリ

import streamlit as st

st.title('Othello')
st.caption('自分の手を打ったら決定を押してください。')



#ライブラリ

import numpy as np
import random
import sys
from scipy.spatial import distance
import time

#定数宣言

# マスの状態
EMPTY = 0 # 空きマス
WHITE = -1 # 白石
BLACK = 1 # 黒石
WALL = 2 # 壁

# ボードのサイズ
BOARD_SIZE = 8

# 方向(2進数)
NONE = 0
LEFT = 2**0 # =1
UPPER_LEFT = 2**1 # =2
UPPER = 2**2 # =4
UPPER_RIGHT = 2**3 # =8
RIGHT = 2**4 # =16
LOWER_RIGHT = 2**5 # =32
LOWER = 2**6 # =64
LOWER_LEFT = 2**7 # =128

# 手の表現
IN_ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
IN_NUMBER = ['1', '2', '3', '4', '5', '6', '7', '8']

# 手数の上限
MAX_TURNS = 60

# 人間の色
if len(sys.argv) == 2:
    HUMAN_COLOR = sys.argv[1]
else:
    HUMAN_COLOR = 'B'

#変数定義
input = 64
output = 64
neurons = 100
layers = 2

sigmoid_a = 1

#突然変異率
mutation_rate = 0.005

#勝率カウント
WIN_RATE = [0,0,0]   #[black,white,draw]

#試行回数
epoc = 1000
now_epoc = 1

#genomのパス
path = 'C:/Users/hamuh/Desktop/genomdata.txt'

# 手の表現
IN_ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
IN_NUMBER = ['1', '2', '3', '4', '5', '6', '7', '8']


class Board:

    def __init__(self):

        # 全マスを空きマスに設定
        self.RawBoard = np.zeros((BOARD_SIZE + 2, BOARD_SIZE + 2), dtype=int)

        # 壁の設定
        self.RawBoard[0, :] = WALL
        self.RawBoard[:, 0] = WALL
        self.RawBoard[BOARD_SIZE + 1, :] = WALL
        self.RawBoard[:, BOARD_SIZE + 1] = WALL

        # 初期配置
        self.RawBoard[4, 4] = WHITE
        self.RawBoard[5, 5] = WHITE
        self.RawBoard[4, 5] = BLACK
        self.RawBoard[5, 4] = BLACK

        # 手番
        self.Turns = 0

        # 現在の手番の色
        self.CurrentColor = BLACK

        # 置ける場所と石が返る方向
        self.MovablePos = np.zeros((BOARD_SIZE + 2, BOARD_SIZE + 2), dtype=int)
        self.MovableDir = np.zeros((BOARD_SIZE + 2, BOARD_SIZE + 2), dtype=int)

        # MovablePosとMovableDirを初期化
        self.initMovable()

        # ユーザの石の色をhumanColorに格納
        if HUMAN_COLOR == 'B':
            self.humanColor = BLACK
        elif HUMAN_COLOR == 'W':
            self.humanColor = WHITE
        else:
            print('引数にBかWを指定してください')
            sys.exit()

    """
    どの方向に石が裏返るかをチェック
    """
    def checkMobility(self, x, y, color):

        # 注目しているマスの裏返せる方向の情報が入る
        dir = 0

        # 既に石がある場合はダメ
        if(self.RawBoard[x, y] != EMPTY):
            return dir

        ## 左
        if(self.RawBoard[x - 1, y] == - color): # 直上に相手の石があるか

            x_tmp = x - 2
            y_tmp = y

            # 相手の石が続いているだけループ
            while self.RawBoard[x_tmp, y_tmp] == - color:
                x_tmp -= 1

            # 相手の石を挟んで自分の石があればdirを更新
            if self.RawBoard[x_tmp, y_tmp] == color:
                dir = dir | LEFT

        ## 左上
        if(self.RawBoard[x - 1, y - 1] == - color): # 直上に相手の石があるか

            x_tmp = x - 2
            y_tmp = y - 2

            # 相手の石が続いているだけループ
            while self.RawBoard[x_tmp, y_tmp] == - color:
                x_tmp -= 1
                y_tmp -= 1

            # 相手の石を挟んで自分の石があればdirを更新
            if self.RawBoard[x_tmp, y_tmp] == color:
                dir = dir | UPPER_LEFT

        ## 上
        if(self.RawBoard[x, y - 1] == - color): # 直上に相手の石があるか

            x_tmp = x
            y_tmp = y - 2

            # 相手の石が続いているだけループ
            while self.RawBoard[x_tmp, y_tmp] == - color:
                y_tmp -= 1

            # 相手の石を挟んで自分の石があればdirを更新
            if self.RawBoard[x_tmp, y_tmp] == color:
                dir = dir | UPPER

        ## 右上
        if(self.RawBoard[x + 1, y - 1] == - color): # 直上に相手の石があるか

            x_tmp = x + 2
            y_tmp = y - 2

            # 相手の石が続いているだけループ
            while self.RawBoard[x_tmp, y_tmp] == - color:
                x_tmp += 1
                y_tmp -= 1

            # 相手の石を挟んで自分の石があればdirを更新
            if self.RawBoard[x_tmp, y_tmp] == color:
                dir = dir | UPPER_RIGHT

        ## 右
        if(self.RawBoard[x + 1, y] == - color): # 直上に相手の石があるか

            x_tmp = x + 2
            y_tmp = y

            # 相手の石が続いているだけループ
            while self.RawBoard[x_tmp, y_tmp] == - color:
                x_tmp += 1

            # 相手の石を挟んで自分の石があればdirを更新
            if self.RawBoard[x_tmp, y_tmp] == color:
                dir = dir | RIGHT

        ## 右下
        if(self.RawBoard[x + 1, y + 1] == - color): # 直上に相手の石があるか

            x_tmp = x + 2
            y_tmp = y + 2

            # 相手の石が続いているだけループ
            while self.RawBoard[x_tmp, y_tmp] == - color:
                x_tmp += 1
                y_tmp += 1

            # 相手の石を挟んで自分の石があればdirを更新
            if self.RawBoard[x_tmp, y_tmp] == color:
                dir = dir | LOWER_RIGHT

        ## 下
        if(self.RawBoard[x, y + 1] == - color): # 直上に相手の石があるか

            x_tmp = x
            y_tmp = y + 2

            # 相手の石が続いているだけループ
            while self.RawBoard[x_tmp, y_tmp] == - color:
                y_tmp += 1

            # 相手の石を挟んで自分の石があればdirを更新
            if self.RawBoard[x_tmp, y_tmp] == color:
                dir = dir | LOWER

        ## 左下
        if(self.RawBoard[x - 1, y + 1] == - color): # 直上に相手の石があるか

            x_tmp = x - 2
            y_tmp = y + 2

            # 相手の石が続いているだけループ
            while self.RawBoard[x_tmp, y_tmp] == - color:
                x_tmp -= 1
                y_tmp += 1

            # 相手の石を挟んで自分の石があればdirを更新
            if self.RawBoard[x_tmp, y_tmp] == color:
                dir = dir | LOWER_LEFT

        return dir


    """
    石を置くことによる盤面の変化をボードに反映
    """
    def flipDiscs(self, x, y):

        # 石を置く
        self.RawBoard[x, y] = self.CurrentColor

        # 石を裏返す
        # MovableDirの(y, x)座標をdirに代入
        dir = self.MovableDir[x, y]

        ## 左
        if dir & LEFT: # AND演算子

            x_tmp = x - 1

            # 相手の石がある限りループが回る
            while self.RawBoard[x_tmp, y] == - self.CurrentColor:

                # 相手の石があるマスを自分の石の色に塗り替えている
                self.RawBoard[x_tmp, y] = self.CurrentColor

                # さらに1マス左に進めてループを回す
                x_tmp -= 1

        ## 左上
        if dir & UPPER_LEFT: # AND演算子

            x_tmp = x - 1
            y_tmp = y - 1

            # 相手の石がある限りループが回る
            while self.RawBoard[x_tmp, y_tmp] == - self.CurrentColor:

                # 相手の石があるマスを自分の石の色に塗り替えている
                self.RawBoard[x_tmp, y_tmp] = self.CurrentColor

                # さらに1マス左上に進めてループを回す
                x_tmp -= 1
                y_tmp -= 1

        ## 上
        if dir & UPPER: # AND演算子

            y_tmp = y - 1

            # 相手の石がある限りループが回る
            while self.RawBoard[x, y_tmp] == - self.CurrentColor:

                # 相手の石があるマスを自分の石の色に塗り替えている
                self.RawBoard[x, y_tmp] = self.CurrentColor

                # さらに1マス上に進めてループを回す
                y_tmp -= 1

        ## 右上
        if dir & UPPER_RIGHT: # AND演算子

            x_tmp = x + 1
            y_tmp = y - 1

            # 相手の石がある限りループが回る
            while self.RawBoard[x_tmp, y_tmp] == - self.CurrentColor:

                # 相手の石があるマスを自分の石の色に塗り替えている
                self.RawBoard[x_tmp, y_tmp] = self.CurrentColor

                # さらに1マス右上に進めてループを回す
                x_tmp += 1
                y_tmp -= 1

        ## 右
        if dir & RIGHT: # AND演算子

            x_tmp = x + 1

            # 相手の石がある限りループが回る
            while self.RawBoard[x_tmp, y] == - self.CurrentColor:

                # 相手の石があるマスを自分の石の色に塗り替えている
                self.RawBoard[x_tmp, y] = self.CurrentColor

                # さらに1マス右に進めてループを回す
                x_tmp += 1

        ## 右下
        if dir & LOWER_RIGHT: # AND演算子

            x_tmp = x + 1
            y_tmp = y + 1

            # 相手の石がある限りループが回る
            while self.RawBoard[x_tmp, y_tmp] == - self.CurrentColor:

                # 相手の石があるマスを自分の石の色に塗り替えている
                self.RawBoard[x_tmp, y_tmp] = self.CurrentColor

                # さらに1マス右下に進めてループを回す
                x_tmp += 1
                y_tmp += 1

        ## 下
        if dir & LOWER: # AND演算子

            y_tmp = y + 1

            # 相手の石がある限りループが回る
            while self.RawBoard[x, y_tmp] == - self.CurrentColor:

                # 相手の石があるマスを自分の石の色に塗り替えている
                self.RawBoard[x, y_tmp] = self.CurrentColor

                # さらに1マス下に進めてループを回す
                y_tmp += 1

        ## 左下
        if dir & LOWER_LEFT: # AND演算子

            x_tmp = x - 1
            y_tmp = y + 1

            # 相手の石がある限りループが回る
            while self.RawBoard[x_tmp, y_tmp] == - self.CurrentColor:

                # 相手の石があるマスを自分の石の色に塗り替えている
                self.RawBoard[x_tmp, y_tmp] = self.CurrentColor

                # さらに1マス左下に進めてループを回す
                x_tmp -= 1
                y_tmp += 1


    """
    石を置く
    """
    def move(self, x, y):

        # 置く位置が正しいかどうかをチェック
        if x < 1 or BOARD_SIZE < x:
            return False
        if y < 1 or BOARD_SIZE < y:
            return False
        if self.MovablePos[x, y] == 0:
            return False

        # 石を裏返す
        self.flipDiscs(x, y)

        # 手番を進める
        self.Turns += 1

        # 手番を交代する
        self.CurrentColor = - self.CurrentColor

        # MovablePosとMovableDirの更新
        self.initMovable()

        return True


    """
    MovablePosとMovableDirの更新
    """
    def initMovable(self):

        # MovablePosの初期化（すべてFalseにする）
        self.MovablePos[:, :] = False

        # すべてのマス（壁を除く）に対してループ
        for x in range(1, BOARD_SIZE + 1):
            for y in range(1, BOARD_SIZE + 1):

                # checkMobility関数の実行
                dir = self.checkMobility(x, y, self.CurrentColor)

                # 各マスのMovableDirにそれぞれのdirを代入
                self.MovableDir[x, y] = dir

                # dirが0でないならMovablePosにTrueを代入
                if dir != 0:
                    self.MovablePos[x, y] = True


    """
    終局判定
    """
    def isGameOver(self):

        # 60手に達していたらゲーム終了
        if self.Turns >= MAX_TURNS:
            return True

        # (現在の手番)打てる手がある場合はゲームを終了しない
        if self.MovablePos[:, :].any():
            return False

        # (相手の手番)打てる手がある場合はゲームを終了しない
        for x in range(1, BOARD_SIZE + 1):
            for y in range(1, BOARD_SIZE + 1):

                # 置ける場所が1つでもある場合はゲーム終了ではない
                if self.checkMobility(x, y, - self.CurrentColor) != 0:
                    return False

        # ここまでたどり着いたらゲームは終わっている
        return True


    """
    パスの判定
    """
    def skip(self):

        """
        # すべての要素が0のときだけパス(1つでも0以外があるとFalse)
        if any(MovablePos[:, :]):
            return False

        # ゲームが終了しているときはパスできない
        if isGameOver():
            return False
        """

        # ここまで来たらパスなので手番を変える
        self.CurrentColor = - self.CurrentColor

        # MovablePosとMovableDirの更新
        self.initMovable()

        return True


    """
    オセロ盤面の表示
    """
    def display(self):

        # 横軸
        print(' a b c d e f g h')
        # 縦軸方向へのマスのループ
        new_text = [' a b c d e f g h','\n']
        for y in range(1, 9):
            # 縦軸
            print(y, end="")
            text = []
            # 横軸方向へのマスのループ
            for x in range(1, 9):
                # マスの種類(数値)をgridに代入
                grid = self.RawBoard[x, y]
                """
                # マスの種類によって表示を変化
                if grid == EMPTY: # 空きマス
                    print('□', end="")
                elif grid == WHITE: # 白石
                    print('●', end="")
                elif grid == BLACK: # 黒石
                    print('〇', end="")
                """
                # マスの種類によって表示を変化
                if grid == EMPTY: # 空きマス
                    print('* ', end="")
                    text.append('* ')
                elif grid == WHITE: # 白石
                    print('w ', end="")
                    text.append('w ')
                elif grid == BLACK: # 黒石
                    print('b ', end="")
                    text.append('b ')
            text.append(str(y))
            new_text.append(''.join(text))
            new_text.append('\n')
            # 最後に改行
            print()
        new_text = ''.join(new_text)
        with st.empty():
          st.text(new_text)


    def make_data(self):
        board_data = []
        # 縦軸方向へのマスのループ
        for y in range(1, 9):
            # 横軸方向へのマスのループ
            for x in range(1, 9):
                # マスの種類(数値)をgridに代入
                grid = self.RawBoard[x, y]
                # マスの種類によって表示を変化
                if grid == EMPTY: # 空きマス
                    board_data.append(0)
                elif grid == WHITE: # 白石
                    board_data.append(-1)
                elif grid == BLACK: # 黒石
                    board_data.append(1)
        #print(board_data)
        return board_data
    """
    入力された手の形式をチェック
    """
    def checkIN(self, IN):

        # INが空でないかをチェック
        if not IN:
            return False

        # INの1文字目と2文字目がそれぞれa~h,1~8の範囲内であるかをチェック
        if IN[0] in IN_ALPHABET:
            if IN[1] in IN_NUMBER:
                return True

        return False

    """
    ランダムに手を打つCPU
    """
    def randomInput(self):

        # マス判定(skip)をして置けるマスが無い場合はFalseを返す
        if board.skip == True:
            return False

        # 置けるマス(MovablePos=1)のインデックスをgridsに格納
        grids = np.where(self.MovablePos == 1)

        # 候補からランダムに手を選ぶ
        randam_chosen_index = random.randrange(len(grids[0]))
        x_grid = grids[0][randam_chosen_index]
        y_grid = grids[1][randam_chosen_index]

        # オセロの正式な座標表現で返す
        return IN_ALPHABET[x_grid - 1] + IN_NUMBER[y_grid - 1]


# NN変数作成 ex) w1,w2,b1,b2...
for i in range(layers+1):
  exec('w{} = {}'.format(i+1,[]))

for i in range(layers+1):
  exec('b{} = {}'.format(i+1,[]))


# 活性化関数
def sigmoid(x):
  return 1 / (1 + np.exp(-x * sigmoid_a))

# 順伝達
# input: 入力, w1: 入力層から中間層への重み, w2: 中間層から出力層への重み
def forward(input,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11):
  layer1 = sigmoid(np.dot(input, w1) + b1)
  layer2 = sigmoid(np.dot(layer1, w2) + b2)
  layer3 = sigmoid(np.dot(layer2, w3) + b3)
  layer4 = sigmoid(np.dot(layer3, w4) + b4)
  layer5 = sigmoid(np.dot(layer4, w5) + b5)
  layer6 = sigmoid(np.dot(layer5, w6) + b6)
  layer7 = sigmoid(np.dot(layer6, w7) + b7)
  layer8 = sigmoid(np.dot(layer7, w8) + b8)
  layer9 = sigmoid(np.dot(layer8, w9) + b9)
  layer10 = sigmoid(np.dot(layer9, w10) + b10)
  output = sigmoid(np.dot(layer10, w11) + b11)
  return output

# 初期個体作成
def creat(x,y):
  genom = []
  for n in range(y):
    a = []
    for i in range(x):
      a.append(random.uniform(-1.0,1.0))
    genom.append(a)
  return genom

# genomを重みとバイアスに変換
def convert(genom):
  w1 = np.array( genom[0][:(input*neurons)] )
  w2 = np.array( genom[0][(input*neurons):(input*neurons) + (neurons*neurons)])
  w3 = np.array( genom[0][(input*neurons) + (neurons*neurons)*1:(input*neurons) + (neurons*neurons)*2])
  w4 = np.array( genom[0][(input*neurons) + (neurons*neurons)*2:(input*neurons) + (neurons*neurons)*3])
  w5 = np.array( genom[0][(input*neurons) + (neurons*neurons)*3:(input*neurons) + (neurons*neurons)*4])
  w6 = np.array( genom[0][(input*neurons) + (neurons*neurons)*4:(input*neurons) + (neurons*neurons)*5])
  w7 = np.array( genom[0][(input*neurons) + (neurons*neurons)*5:(input*neurons) + (neurons*neurons)*6])
  w8 = np.array( genom[0][(input*neurons) + (neurons*neurons)*6:(input*neurons) + (neurons*neurons)*7])
  w9 = np.array( genom[0][(input*neurons) + (neurons*neurons)*7:(input*neurons) + (neurons*neurons)*8])
  w10 = np.array( genom[0][(input*neurons) + (neurons*neurons)*8:(input*neurons) + (neurons*neurons)*9])
  w11 = np.array( genom[0][(input*neurons) + (neurons*neurons)*9:(input*neurons) + (neurons*neurons)*9 + (neurons*output)])

  FromB = (input*neurons) + (neurons*neurons)*10 + (neurons*output)

  b1 = np.array( genom[0][FromB:FromB + neurons])
  b2 = np.array( genom[0][FromB + neurons*1:FromB + neurons*2])
  b3 = np.array( genom[0][FromB + neurons*2:FromB + neurons*3])
  b4 = np.array( genom[0][FromB + neurons*3:FromB + neurons*4])
  b5 = np.array( genom[0][FromB + neurons*4:FromB + neurons*5])
  b6 = np.array( genom[0][FromB + neurons*5:FromB + neurons*6])
  b7 = np.array( genom[0][FromB + neurons*6:FromB + neurons*7])
  b8 = np.array( genom[0][FromB + neurons*7:FromB + neurons*8])
  b9 = np.array( genom[0][FromB + neurons*8:FromB + neurons*9])
  b10 = np.array( genom[0][FromB + neurons*9:FromB + neurons*10])
  b11 = np.array( genom[0][FromB + neurons*10:FromB + neurons*10 + output])

  w1 = w1.reshape(input,neurons)
  w2 = w2.reshape(neurons,neurons)
  w3 = w3.reshape(neurons,neurons)
  w4 = w4.reshape(neurons,neurons)
  w5 = w5.reshape(neurons,neurons)
  w6 = w6.reshape(neurons,neurons)
  w7 = w7.reshape(neurons,neurons)
  w8 = w8.reshape(neurons,neurons)
  w9 = w9.reshape(neurons,neurons)
  w10 = w10.reshape(neurons,neurons)
  w11 = w11.reshape(neurons,output)

  return w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11

# 個体の遺伝子に従って移動
def AI_move(A,genom,board):
  w1 = convert(genom)[0]
  w2 = convert(genom)[1]
  w3 = convert(genom)[2]
  w4 = convert(genom)[3]
  w5 = convert(genom)[4]
  w6 = convert(genom)[5]
  w7 = convert(genom)[6]
  w8 = convert(genom)[7]
  w9 = convert(genom)[8]
  w10 = convert(genom)[9]
  w11 = convert(genom)[10]
  b1 = convert(genom)[11]
  b2 = convert(genom)[12]
  b3 = convert(genom)[13]
  b4 = convert(genom)[14]
  b5 = convert(genom)[15]
  b6 = convert(genom)[16]
  b7 = convert(genom)[17]
  b8 = convert(genom)[18]
  b9 = convert(genom)[19]
  b10 = convert(genom)[20]
  b11 = convert(genom)[21]

  z1 = forward(A,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11)
  max_index = np.argmax(z1)

  AI = max_index//8,max_index%8
  moveable = MoveableData(board)
  result = nearestValue(moveable,AI)
  return IN_ALPHABET[result[1]] + IN_NUMBER[result[0]]

# 二つの個体を交配　一様交配
def crossover(succed_genom):
  new_genom = []
  parents_genom1 = random.sample(succed_genom,2)[0]
  parents_genom2 = random.sample(succed_genom,2)[1]
  for i in range(len(parents_genom1)):
    a = random.randint(0,1)
    if a == 0:
      new_genom.append(parents_genom1[i])
    else:
      new_genom.append(parents_genom2[i])
    return new_genom

def MoveableData(board):
  Moveabledata = []
  moveAble = np.where(board.MovablePos == 1)
  for i in range(len(moveAble[0])):
    Moveabledata.append(tuple(reversed((int(moveAble[0][i]) - 1,int(moveAble[1][i]) -1))))
  return Moveabledata

def nearestValue(data,value):
  nearestValue = data[0]
  for i in range(len(data)):
    if abs(distance.euclidean(data[i], value)) < abs(distance.euclidean(nearestValue[0], value)):
      nearestValue = data[i]
    elif abs(distance.euclidean(data[i], value)) == abs(distance.euclidean(nearestValue[0], value)):
      coin = random.random()
      if coin < 0.5:
        nearestValue = data[i]
  return nearestValue

def mutation(genom):
  for i in range(len(genom[0])):
    coin = random.random()
    if coin < mutation_rate:
      genom[0][i] = random.uniform(-1.0,1.0)



#勝率カウント
WIN_RATE = [0,0,0]   #[black,white,draw]

# ボートインスタンスの作成
#board = Board()
if 'board' not in st.session_state:
    st.session_state.board = Board()
# 勝率計測
board = st.session_state.board

# 盤面の表示
board.display()

# 終局判定
if board.isGameOver():
    # ゲーム終了後の表示
  board.display()
  print('おわり')
  st.text('おわり')
  print()

  ## 各色の数
  count_black = np.count_nonzero(board.RawBoard[:, :] == BLACK)
  count_white = np.count_nonzero(board.RawBoard[:, :] == WHITE)

  print('黒:', count_black)
  print('白:', count_white)
  st.text('黒:', count_black)
  st.text('白:', count_white)
  ## 勝敗
  dif = count_black - count_white
  if dif > 0:
      print('黒の勝ち')
      st.text('黒の勝ち')
      WIN_RATE[0] += 1
  elif dif < 0:
      print('白の勝ち')
      st.text('白の勝ち')
      WIN_RATE[1] += 1
  else:
      print('引き分け')
      st.text('引き分け')
      WIN_RATE[2] += 1

  #盤面リセット
  board.__init__()
  st.session_state.board = board.__init__()


# 手番の表示
if board.CurrentColor == BLACK:
  where = st.text_input('例：c6')
  print('黒の番です:', end = "")
  st.text('黒の手番です')
else:
  print('白の番です:', end = "")
  aa = where = st.text_input('例：c6')
  where = board.randomInput()
  st.text('白の手番です：{}'.format(where))
submit_btn = False
submit_btn = st.button('決定')

# 対戦を終了
if where == "e":
    print('おつかれ')
    st.text('おつかれ')
else:
    # 入力手をチェック
    if board.checkIN(where):
      x = IN_ALPHABET.index(where[0]) + 1
      y = IN_NUMBER.index(where[1]) + 1
      # 手を打つ
      if not board.move(x, y):
        print('そこには置けません')
        #st.text('そこには置けません')
        # パス
        if not board.MovablePos[:, :].any():
          board.CurrentColor = - board.CurrentColor
          board.initMovable()
          print('パスしました')
          st.text('パスしました')
          print()
    else:
      print('正しい形式(例：f5)で入力してください')
      st.text('正しい形式(例：f5)で入力してください')

if submit_btn:
    for i in range (2):
        if board.CurrentColor == board.humanColor:
          # 人
          IN = where
          print(IN)
        else: # ランダム
          IN = where
          print(IN)
        print()
