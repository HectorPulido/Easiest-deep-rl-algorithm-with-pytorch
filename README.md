# Easiest deep reinforcement learning algorithm with pytorch

Here it is, the simplest deep reinforcement learning algorithm ever written in pytorch.

## How it works
1. Pick your model and create your agent
2. Play a lot of sessions
3. Select the most successful sessions in terms of reward
4. Trair your agent with that sessions
5. Repeat

## Why Does It Work ?
This is a fancy genetic algorithm, the model can learn from the best training data of the sessions. The problem with this is that the unsuccessful sessions will be discarted, so our agent will not learn from it's failures. 

## Perfomance
This algorithm has a poor performance, a lot of sessions are just discarted, but it works well at the end.

### Average reward vs iterations
![graph](/resources/average_reward.png)

### How it looks like
![animation](/resources/cartpole.gif)

## License
This project is under MIT License, use it as you want.

## More interesting projects
I have a lot of fun projects, check this:

### Blockchain
- https://github.com/HectorPulido/Amazon-QLDB-Login-Example
- https://github.com/HectorPulido/Decentralized-Twitter-with-blockchain-as-base

### Machine learning
- https://github.com/HectorPulido/Machine-learning-Framework-Csharp
- https://github.com/HectorPulido/Evolutionary-Neural-Networks-on-unity-for-bots
- https://github.com/HectorPulido/Imitation-learning-in-unity
- https://github.com/HectorPulido/Chatbot-seq2seq-C-

### You also can follow me in social networks
- Twitter: https://twitter.com/Hector_Pulido_
- Youtube: http://youtube.com/c/hectorandrespulidopalmar
- Twitch: https://www.twitch.tv/hector_pulido_

