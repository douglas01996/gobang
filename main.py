import tensorflow as tf
from gobang import gobang
from cnn import ZIYINET

def main(_):
    game=gobang(alpha=0.05)
    cnn=ZIYINET(game)
    cnn.train()
    cnn.test()

if __name__=='__main__':
    tf.app.run()
