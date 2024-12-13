import gymnasium as gym
from algorithms.ppo import ppo, play


def main():
    ppo()
    play("human")
    
    
if __name__ == "__main__":
    main()