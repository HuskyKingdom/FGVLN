


model = PPO(CustomActorCriticPolicy, "CartPole-v1", verbose=1)
model.learn(5000)