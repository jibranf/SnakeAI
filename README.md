# SnakeAI

![snake](https://user-images.githubusercontent.com/116676609/197864373-af6f51d1-b4cf-451e-ab9a-94bf14683589.png)

- This project creates a neural network and trains it to play the classic game Snake.
- This project is written in Python, and makes use of the following libraries:
  - pyTorch
  - pygame
  - numpy
  - enum
  - iPython
  - MatplotLib
- I chose to use these libraries because they helped to create the neural network, its learning behaviors, the game interface, and the output windows for the current game iteration and the high score plot
- The approach I took to implement the AI was to use a feed-forward multi-layered neural net that takes in 11 inputs/states and responds with 3 possible actions
- To train the neural net I utilized Deep Q Learning, which is a form of RL that uses states, actions, and reward to determine the quality of some action. This is done through using the Bellman Equation
  - New Q(s,a) = Q(s,a) + LR(R(s,a) + DR * max(Q(s'a')) - Q(s, a))
  - LR = Learning Rate, DR = discount rate, s = state, a = action, s' = next state, a' = next action, R = reward
  - This is a recursive equation, and the learning rate essentially dicates how much newly learned behaviors will impact previous behaviors 
  - To get the next state and next action, the model predicts these values and uses them in the equation

# How to Install and Run the AI
- To install, download each of the .py files to a directory
-  To run the AI, run agent.py. Doing so will have two windows pop up, one that shows the current game being played, and the other being a plot of the scores over number of games played.
-  Once it is running, the model will save itself after each game played in a directory called "model", and the model will be a file 'model.pth'

