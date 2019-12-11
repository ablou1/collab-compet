from noise import GaussianNoise


class AgentParameters():
    """Contains the parameters to apply to the agent"""

    def __init__(self, action_size, gamma=0.99, tau_param=(0.01, 0.001, 0.999), batch_size=1024, learn_param=(20, 20), noise_factor=0.4, noise_reducer_param=(1, 0.01, 0.9995), model_update=(1, 1), model_param=(256, 512), lr_actor_critic=(1e-4, 1e-4)):
        """Initialize agent parameters.
        Params
        ======
            gamma (float): discount factor
            tau_param (float, float, float): for soft update of target parameters (factor, min_factor, decay_rate)
            batch_size (int): minibatch size
            learn_param (int, int): tuple that contains the number of iterations without training and the number of training done each time
            noise_factor: factor apply to the Gaussian noise
            noise_reducer_param(float, float, float): for noise applied in act method (factor, min_factor, decay_rate)
            model_update(int, int): tuple that contains the number of iterations without training and the number of training done each time
            model_param(int, int): tuple of hidden layer size for the actor and critic network
            lr_actor_critic (float, float): tuple of learning rates of the actor and of the critic
        """
        self.gamma = gamma
        self.tau_param = tau_param
        self.batch_size = batch_size
        self.learn_param = learn_param
        self.noise = GaussianNoise(action_size, noise_factor)
        self.noise_reducer_param = noise_reducer_param
        self.model_param = model_param
        self.lr_actor_critic = lr_actor_critic
        self.model_update = model_update
