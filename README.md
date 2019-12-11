# p3-collab-compet
Deep Reinforcement Learning - Project dedicated to train two agents control rackets to bounce a ball over a net.

This project is part of the **Deep Reinforcement learning NanoDegree - Udacity**. It's the third project names p3-collab-compet

# The Environment

(source: Udacity Deep Reinforcement Learning NanoDegree)

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

> - After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
> - This yields a single score for each episode.

The environment is considered solved, when **the average (over 100 episodes) of those scores is at least +0.5**.

# Python environment

(source: Udacity Deep Reinforcement Learning NanoDegree)

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6 (named **drlnd** or the name of your choice).
   - Linux or Mac:
     > Conda create --name drlnd python=3.6  
     > source activate drlnd
   - Windows:
     > conda create --name drlnd python=3.6  
	 > activate drlnd

2. If not already done, clone the current repository and navigate to the root folder. Then install required dependencies.
	> git clone https://github.com/ablou1/continuous-control.git  
	> cd continuous-control  
	> pip install -r requirements.txt

3. Install Pytorch & Cuda
	> conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

4. Clone the udacity/deep-reinforcement-learning repository (outside the current project) and navigate to the python/ folder. Then, install several dependencies.
	> git clone https://github.com/udacity/deep-reinforcement-learning.git  
	> cd deep-reinforcement-learning/python  
	> pip install .

5. Create an [IPython kernel](https://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the drlnd environment.
	> python -m ipykernel install --user --name drlnd --display-name "drlnd"

6. Before running code in a notebook, change the kernel to match the drlnd environment by using the drop-down Kernel menu.

![Kernel](kernel2.PNG)


# Download the Environment
To use this repository, you do not need to install Unity. You can download the environment from one of the links below. You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

Then, place the file in the root of this repository, and unzip (or decompress) the file.

(For Windows users) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(For AWS) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the "headless" version of the environment. You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the Linux operating system above.)

# Train an agent

Two choices are available to train an agent inside this repository.
- Either use the **train_agent.py** file
- Or use the **Training Analysis.ipynb** notebook

## train_agent.py

The train_agent.py file is dedicated to run a single train for a specified agent.

1. Update the file_name of the UnityEnvironment in order to match the location of the Unity environment that you downloaded.
- Mac: "path/to/Tennis.app"
- Windows (x86): "path/to/Tennis_Windows_x86/Tennis.exe"
- Windows (x86_64): "path/to/Tennis_Windows_x86_64/Tennis.exe"
- Linux (x86): "path/to/Tennis_Linux/Tennis.x86"
- Linux (x86_64): "path/to/Tennis_Linux/Tennis.x86_64"
- Linux (x86, headless): "path/to/Tennis_Linux_NoVis/Tennis.x86"
- Linux (x86_64, headless): "path/to/Tennis_Linux_NoVis/Tennis.x86_64"

	For instance, if you are using a Mac, then you downloaded Banana.app. If this file is in the same folder as the notebook, then the line below should appear as follows:
	> env = UnityEnvironment(file_name="Tennis.app")

*example with windows (x86_64) :*

![example with windows (x86_64)](LoadEnvironment.PNG)

2. Adjust the agent parameter depending on the test you want to execute.
- Either on the AgentParameters constructor
- Or by modifying the training method parameters

![parameters](AdjustParameters.PNG)

3. Run the training process by executing the following command :
	> python train_agent.py

	It automatically create two checkpoint files for each network with the following name format ({simu_name}_actor_checkpoint.pth and {simu_name}_critic_checkpoint.pth).

## Training Analysis.ipynb
The **Training Analysis.ipynb** notebook is dedicated to compare different parameterization to train the agent.

You can define the number of episodes allocated for each training process.

In order to test different parameters, you just have to indicate the values you want to test. Take care of the calculation time. After each training, a file save the result in order to display graph at the end of the notebook. The checkpoint files are also saved in this part.

Inside the AGENT_ATTRIBUTES_TO_TEST dictionary, you have to indicate the values you want to test that are not the default ones. Take care to use AgentParameters constructor attributes names for the dictionnary keys.

Inside the BASE_ATTRIBUTES_VALUES, you have to indicate the default values of each parameter tested. See the example below:

![parameters](Parameters.PNG)

### Results analysis
This part shows graphs comparing different attributes values for a single parameter.

### checkpoint.pth format
The checkpoint.pth file contains a dictionary with the following informations:
> - 'state_size': the state size of the environment during the training,
> - 'action_size': the action size of the environment during the training,
> - 'fc1_units': the size of the first hidden layer used in the models.
> - 'fc2_units': the size of the second hidden layer used in the models.
> - 'state_dict': the state_dict of the network trained for the agent who succeed.

It exist one checkpoint file per model (one for the actor and one for the critic)