import argparse

from src.kindred.qlearning import learn, learn_async


def run():
    parser = argparse.ArgumentParser(
        description='Run synchronous/asynchronous Q learning for the Grid World environment.'
    )

    parser.add_argument('-as', '--async', help='Use async if set to True.', action='store_true')
    parser.add_argument('-ne', '--episodes', type=int, help='Number of episodes', default=1000)
    parser.add_argument('-e', '--epsilon', type=float, help='Epsilon value for greedy policy.', default=0.5)
    parser.add_argument('-a', '--alpha', type=float, help='Learning rate value.', default=0.3)
    parser.add_argument('-g', '--gamma', type=float, help='Discount factor value.', default=0.95)
    parser.add_argument('-na', '--agents', type=int, help='Number of agents.', default=5)
    parser.add_argument('-i', '--iasync', type=int, help='I async update value.', default=5)
    parser.add_argument('-t', '--tmax', type=int, help='Maximum value for T.', default=20000)
    parser.add_argument('-s', '--size', type=int, help='Size of grid (rows * cols).', default=54)

    args = parser.parse_args()

    Q = None
    if args.async:
        Q = learn_async(
            num_agents=args.agents,
            I_async_update=args.iasync,
            T_max=args.tmax,
            size=args.size,
            epsilon=args.epsilon,
            alpha=args.alpha,
            gamma=args.gamma,
        )
    else:
        _, Q = learn(
            num_episodes=args.episodes,
            epsilon=args.epsilon,
            alpha=args.alpha,
            gamma=args.gamma,
        )

    return Q

if __name__ == '__main__':
    run()
