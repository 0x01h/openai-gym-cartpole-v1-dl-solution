#!/usr/bin/python3

import gym
import numpy as np
import tensorflow as tf

prev_obs = []
x_train = []
y_train = []
moves = []
rewards = []
dones = []
scores = []
obs_memory = []
action_memory = []
total_episode = 10000
trained_episode = 100
score_threshold = 70
total_score_deb = 0
num_episodes = 0
env = gym.make('CartPole-v1')
env.reset()

# print(vars(env))

for _ in range(total_episode):
    first_move = True
    action_memory = []
    obs_memory = []
    num_episodes += 1
    score = 0
    env.reset()

    while (True):
        print('Training data is preparing ' + str(num_episodes + 1) + '/' + str(total_episode), end='.\r')

        if (not first_move):
            prev_obs = observation

        action = np.random.randint(2)
        observation, reward, done, info = env.step(action)
        
        score += reward

        if (not first_move):
            obs_memory.append(prev_obs)
            action_memory.append(action)

        moves.append(action)
        rewards.append(reward)
        dones.append(done)
        first_move = False

        if (done):
            if (score > score_threshold and len(action_memory) > score_threshold):
                for i in range(score_threshold):
                    x_train.append(obs_memory[i])
                    y_train.append(action_memory[i])

            scores.append(score)
            total_score_deb += score
            break

    # print(reward)
    # print(done)
    # print(info['is_success']) # 1.0 or 0.0
    # env.render()

env.close()
print()
print('Average score before trained: ' + str(total_score_deb/total_episode) + ' in ' + str(total_episode) + ' game.')
scores.sort(reverse=True)
# print(scores)

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

# print(x_train)
# print(y_train)

# print(x_train.shape)
# print(y_train.shape)

print(str(y_train.shape[0]) + ' successful actions regarding to observations will be given to the model.')

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.3),
  tf.keras.layers.Dense(256, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(256, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(128, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.3),
  tf.keras.layers.Dense(2, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

"""test_loss, test_acc = model.evaluate(x_train, y_train)
predictions = model.predict(x_train[0:1])"""

def play_trained():
    total_score = 0
    num_episodes = 0
    scores = []

    for _ in range(trained_episode):
        env.reset()
        score = 0
        num_episodes += 1
        first_move = True

        while(True):
            print('Trained model is playing ' + str(num_episodes) + '/' + str(trained_episode), end='.\r')

            if first_move:
                action = np.random.randint(2)
                first_move = False
            else:
                predictions = model.predict(np.asarray([observation]))
                action = np.argmax(predictions)

            observation, reward, done, info = env.step(action)

            if (done):
                score += reward
                scores.append(score)
                total_score += score
                score = 0
                env.reset()
                env.render()
                break
            else:
                score += reward

    return scores, total_score

trained_scores, total_score = play_trained()
trained_scores.sort(reverse=True)
print()
print('Average score after trained: ' + str(total_score/trained_episode) + ' in ' + str(trained_episode) + ' game.')
print(trained_scores)