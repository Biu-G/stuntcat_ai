import pygame
import tensorflow as tf
import cv2
import sys
import random
import numpy as np
from collections import deque
import os
GAME = "stuntcat"
ACTIONS = 256
GAMMA = 0.99
#OBSERVE = 100000.
OBSERVE = 100.
EXPLORE = 200000.
#EXPLORE = 2000000.
FINAL_EPSILON = 0.0001
#INITIAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.4
REPLAY_MEMORY = 50000
BATCH = 32
FRAME_PER_ACTION = 1
#observe must >= 4!(stack process)

class Gambling:
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)
    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)
    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")
    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1],
                              padding = "SAME")
    def createNetwork(self):
        W_conv1 = self.weight_variable([8, 8, 4, 32])
        b_conv1 = self.bias_variable([32])

        W_conv2 = self.weight_variable([4, 4, 32, 64])
        b_conv2 = self.bias_variable([64])

        W_conv3 = self.weight_variable([3, 3, 64, 64])
        b_conv3 = self.bias_variable([64])

        W_fc1 = self.weight_variable([1600, 768])
        b_fc1 = self.bias_variable([768])

        W_fc2 = self.weight_variable([768, ACTIONS])
        b_fc2 = self.bias_variable([ACTIONS])

        # input layer
        self.s = tf.placeholder("float", [None, 80, 80, 4])

        # hidden layers
        h_conv1 = tf.nn.relu(self.conv2d(self.s, W_conv1, 4) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, 2) + b_conv2)
        # h_pool2 = max_pool_2x2(h_conv2)

        h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 1) + b_conv3)
        # h_pool3 = max_pool_2x2(h_conv3)

        # h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
        h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        # readout layer
        readout = tf.matmul(h_fc1, W_fc2) + b_fc2

        return self.s, readout

    def __init__(self):
        self.readout_t = [0 for _ in range(ACTIONS)]
        self.sess = tf.InteractiveSession()
        self.t = 0
        self.epsilon = INITIAL_EPSILON
        self.s, self.readout = self.createNetwork()
        self.a = tf.placeholder("float", [None, ACTIONS])
        self.y = tf.placeholder("float", [None])
        readout_action = tf.reduce_sum(tf.multiply(self.readout, self.a), reduction_indices = 1)
        cost = tf.reduce_mean(tf.square(self.y - readout_action))
        self.train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)
        self.D = deque()
        self.stack_first = False

    def stacker(self, image_data):
        self.t += 1
        if self.epsilon > FINAL_EPSILON and self.t > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
        if self.stack_first == True:
            a_t = np.zeros(ACTIONS)
            action_index = 0
            if self.t % FRAME_PER_ACTION == 0:
                if random.random() <= self.epsilon:
                    print("----------Random Action----------")
                    action_index = random.randrange(ACTIONS)
                    a_t[random.randrange(ACTIONS)] = 1
                else:
                    self.readout_t = self.readout.eval(feed_dict={self.s: [self.s_t]})[0]
                    action_index = np.argmax(self.readout_t)
                    a_t[action_index] = 1
            else:
                a_t[0] = 1  # do nothing
            return a_t
        self.stack_first = True
        #virtual 4's
        do_nothing = np.zeros(ACTIONS)
        #assume arg max is 0->do nothing
        do_nothing[0] = 1
        x_t = cv2.cvtColor(cv2.resize(image_data, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
        self.s_t = np.stack((x_t, x_t, x_t, x_t), axis=2) #new:: self.s_t 4 frame!
        #print("TYPE" + type(self.s_t))
        self.saver = tf.train.Saver()
        self.sess.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")
        return do_nothing


    def fight(self, image_data, reward):
        #not the first time
        if self.stack_first == True:
            # x_t1_colored, r_t = game_state.frame_step(a_t)
            x_t1 = cv2.cvtColor(cv2.resize(image_data, (80, 80)), cv2.COLOR_BGR2GRAY)
            ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
            x_t1 = np.reshape(x_t1, (80, 80, 1))
            # s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
            s_t1 = np.append(x_t1, self.s_t[:, :, :3], axis=2)

            # store the transition in D
            self.D.append((self.s_t, self.action, reward, s_t1))
            if len(self.D) > REPLAY_MEMORY:
                self.D.popleft()

            # only train if done observing
            if self.t > OBSERVE:
                # sample a minibatch to train on
                minibatch = random.sample(self.D, BATCH)

                # get the batch variables
                s_j_batch = [d[0] for d in minibatch]
                a_batch = [d[1] for d in minibatch]
                r_batch = [d[2] for d in minibatch]
                s_j1_batch = [d[3] for d in minibatch]

                y_batch = []
                readout_j1_batch = self.readout.eval(feed_dict={self.s: s_j1_batch})
                for i in range(0, len(minibatch)):
                    # if terminal, only equals reward
                    if r_batch[i] < 0:
                        y_batch.append(r_batch[i])
                    else:
                        y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

                # perform gradient step
                self.train_step.run(feed_dict={
                    self.y: y_batch,
                    self.a: a_batch,
                    self.s: s_j_batch}
                )

            # update the old values
            self.s_t = s_t1

            # save progress every 10000 iterations
            if self.t % 10000 == 0:
                self.saver.save(self.sess, 'saved_networks/' + GAME + '-dqn', global_step=self.t)

            # print info
            state = ""
            if self.t <= OBSERVE:
                state = "observe"
            elif self.t > OBSERVE and self.t <= OBSERVE + EXPLORE:
                state = "explore"
            else:
                state = "train"

            print("TIMESTEP", self.t, "/ STATE", state, \
                  "/ EPSILON", self.epsilon, "/ ACTION", self.actors, "/ REWARD", reward, \
                  "/ Q_MAX %e" % np.max(self.readout_t))
            # write info to files
            '''
            if t % 10000 <= 100:
                a_file.write(",".join([str(x) for x in readout_t]) + '\n')
                h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
                cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
            '''

        #last round above, next round below
        self.action = self.stacker(image_data)
        #return what?
        act = 0
        for i in range(ACTIONS):
            if self.action[i] > 0:
                act = i
                break
        res = []
        if (act & (1<<0)) > 0:
            res.append(pygame.event.Event(pygame.KEYDOWN, {"key":pygame.K_LEFT, "mod":0, "unicode":u' '}))
        if (act & (1<<1)) > 0:
            res.append(pygame.event.Event(pygame.KEYDOWN, {"key":pygame.K_RIGHT, "mod":0, "unicode":u' '}))
        if (act & (1<<2)) > 0:
            res.append(pygame.event.Event(pygame.KEYDOWN, {"key":pygame.K_UP, "mod":0, "unicode":u' '}))
        if (act & (1<<3)) > 0:
            res.append(pygame.event.Event(pygame.KEYUP, {"key":pygame.K_LEFT, "mod":0, "unicode":u' '}))
        if (act & (1<<4)) > 0:
            res.append(pygame.event.Event(pygame.KEYUP, {"key":pygame.K_RIGHT, "mod":0, "unicode":u' '}))
        if (act & (1 << 5)) > 0:
            res.append(pygame.event.Event(pygame.KEYUP, {"key": pygame.K_UP, "mod": 0, "unicode": u' '}))
        if (act & (1<<6)) > 0:
            res.append(pygame.event.Event(pygame.KEYDOWN, {"key":pygame.K_a, "mod":0, "unicode":u' '}))
        if (act & (1<<7)) > 0:
            res.append(pygame.event.Event(pygame.KEYDOWN, {"key":pygame.K_d, "mod":0, "unicode":u' '}))
        self.actors = act
        return res

    def say(self):
        print("halo")