# Projekt CarRacingV3 i gymnasium

Vi Löste modellen med två olika agenter:
- Deep Q Network
- Proximal Policy Optimization

Tränade över 500000 timesteps.

## DQN

![DQN racing](/resources/DQN.gif)

## PPO

![PPO racing ](/resources/PPO.gif)

## Jämförelse
![comparision](/resources/model_comparison_500k.png)
![performance](/resources/model_performance_500k.png.png)

<details>
<summary>
Content based output
</summary>

```bash
=== Evaluating DQN ===
Episode 1: Reward = 482.52, Steps = 1000
Episode 2: Reward = 763.01, Steps = 1000
Episode 3: Reward = 929.90, Steps = 701
Episode 4: Reward = 893.33, Steps = 1000
Episode 5: Reward = 682.33, Steps = 1000

=== Evaluating PPO ===
Episode 1: Reward = 803.47, Steps = 1000
Episode 2: Reward = 62.94, Steps = 1000
Episode 3: Reward = 848.63, Steps = 1000
Episode 4: Reward = 817.76, Steps = 1000
Episode 5: Reward = 838.57, Steps = 1000

=== RESULTS ===
DQN  - Mean Reward: 750.22 ± 160.75, Mean Length: 940.2
PPO  - Mean Reward: 674.27 ± 306.07, Mean Length: 1000.0
```

</details>

### Resultat 

- DQN:  Mean Reward: 750.22 ± 160.75, Mean Length: 940.2
- PPO:  Mean Reward: 674.27 ± 306.07, Mean Length: 1000.0

### Tolkning

- PPO när ofta max längd
- DQN får ofta högre belöning


DQN agenten verkar komma i mål, alltså visitera alla "tiles" oftare än PPO agenten. Man ser detta i antalet steps per episod var DQN har flera episoder under 1000 steps (max steps) medan PPO bara har en (testat över 50 timesteps). Mean rewarden är också högre för DQN agenten. Med detta kan vi konstatera att DQN agenten blev bättre på 500000 timesteps.
