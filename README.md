# RL_Monkey

In this project, a reinforcement learning agent learns to navigate its environment through use of Q-learning with an epsilon-greedy policy.

![SwingyMonkey](https://user-images.githubusercontent.com/55005116/183547209-a88f4a3b-7de1-4b87-8d89-6f637489752b.png)

The game being played is called SwingyMonkey, which can be played with the pygame module. To play the game, install pygame and then run the file SwingyMonkey.py. To see one version of Q-learning, run stub_1.py or stub_2.py (and choose whether or not to see visuals of the monkey training by commenting or uncommenting that line of code at the top of either file). stub_2.py generally performs better.


![monkey1](https://user-images.githubusercontent.com/55005116/183546818-c48d4d5a-e8f4-4cda-890f-80f38cd38729.png)
![monkey2](https://user-images.githubusercontent.com/55005116/183546820-00737aa3-5f18-4444-b4f3-4749832fbe54.png)

The results for monkey 1 and monkey 2 (trained in stub_1.py and stub_2.py, respectively) can be observed above. Monkey 1 is trained through Q-learning, and it appears to me that low values for epsilon, the learning rate alpha, and the discount rate gamma work well.

In an attempt to improve monkey 1, I created monkey 2. This second monkey is given a more informative state value which includes not only information about its height and distance to the next tree but also information about its current velocity and the effects of gravity (gravity can change each round of the game).

Looking at the two graphs, it seems clear to me that monkey 2 is able to perform better than monkey 1 given this additional information, as monkey 2 is able to reach a peak score of over 1400 and many scores around 500 or higher while monkey 1 only cracks a score of 500 one single time.
