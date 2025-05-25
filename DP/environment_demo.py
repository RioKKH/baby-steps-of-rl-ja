import random
from environment import Environment


class Agent:
    def __init__(self, env):
        self.actions = env.actions

    def policy(self, state):
        """
        今回はランダムに行動するだけ
        """
        return random.choice(self.actions)


def main():
    # Make grid environment.
    grid = [[0, 0, 0, 1], [0, 9, 0, -1], [0, 0, 0, 0]]
    env = Environment(grid)
    agent = Agent(env)

    # Try 10 game.
    for i in range(10):
        # Initialize position of agent.
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # policyに従って行動を選択
            action = agent.policy(state)
            print(f"{i}, {action}")
            # 環境に行動を与えて次の状態と報酬を得る
            next_state, reward, done = env.step(action)
            # 今回の状態の変化に応じた報酬を足しこむ
            total_reward += reward
            # 状態を更新
            state = next_state

        print("Episode {}: Agent gets {} reward.".format(i, total_reward))


if __name__ == "__main__":
    main()
