import zmq
import random
import sys
import time


class Publisher:

    def __init__(self, port):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind("tcp://127.0.0.101:%s" % port)

    def send(self, topic, message_data):

        # print "sent : %s %s" % (topic, message_data)
        self.socket.send("%s %s" % (topic, message_data))

    def close(self):
        self.socket.close()
        self.context.term()
