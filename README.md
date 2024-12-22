
<div id="user-content-toc">
  <ul align="center" style="list-style: none;">
    <summary>
      <h1> HumEnv: Humanoid Environment for Reinforcement Learning</h1>
    </summary>
  </ul>
</div>

# Overview

HumEnv is an environment based on SMPL humanoid which aims for reproducible studies of humanoid control. It is designed to facilitate algorithmic research on reinforcement learning (RL), goal-based RL, unsupervised RL, and imitation learning. It consists of a basic environment interface, as well as an optional benchmark to evaluate agents on different tasks.

## Features

 * An environment that enables simulation of a realistic humanoid on a range of proprioceptive tasks
 * A MuJoCo-based humanoid robot definition tuned for more realistic behaviors (friction, joint actuation, and movement range) 
 * 9 configurable reward classes to enable learning basic humanoid skills, including locomotion, spinning, jumping, crawling, and more
 * Benchmarking code to evaluate RL agents on three classes of tasks: reward-based, goal-reaching and motion tracking
 * Various initialisation options: a static "T-pose", random fall, frame from MoCap data, and their combinations
 * Full compatibility with Gymnasium 

[469838886_592650273138757_9015533655681330954_n.pdf](https://github.com/user-attachments/files/18220909/469838886_592650273138757_9015533655681330954_n.pdf)
