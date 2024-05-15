# robot-optimal-control-

Learning and Optimization for Robot Control
As final project choose between three options: 
Project A : Learning a Value function with a Neural Network, and using it as terminal cost in an optimal control problem. 
this time you have to write all the code (almost) from scratch.
Common Tools
1 Solving Optimal Control Problems (OCPs)
projects described above (and detailed in the following) begin by solving a series of OCPs. As an interesting alternative to using the single-shooting formulation that we have seen during our lab sessions, we suggest students to use CasADi. CasADi is a software library for modeling and solving OCPs, coming with Python bindings. CasADi can be easily installed using pip:
pip install casadi
Alternative, you can use a modified version of the docker image of this course:
docker pull andreadelprete/orc23:casadi
An example of a script using CasADi to solve a simple OCP is available here casadi example.py. 

When using CasADi, the system dynamics must be explicitly written down in the code. While the dynamics of a single pendulum is trivial, the dynamics of a double pendulum is rather complex. For this reason, we provide in the project folder a python script containing a function to evaluation the dynamics of a double pendulum.

2 Training Neural Networks 
All projects require training neural networks. A template file containing code for creating and training a neural network using the Tensor Flow/Keras library is available here. Students are free to use other libraries for training neural networks, such as PyTorch or JAX/FLAX. 
3 Parallel Computation 
To make your code run faster you can try to parallelize it, so that it can exploit multiple CPU cores. To do that, you can use the Python library multiprocessing. 
Project A — Learning a Terminal Cost The aim of this project is to learn an approximate Value function that can then be used as terminal cost in an MPC formulation. The idea is similar to the one explored in the second assignment. First, we have to solve many OCP’s starting from different initial states (either chosen randomly or on a grid). For every solved OCP, we should store the initial state x0 and the corresponding optimal cost J(x0) in a buffer. Then, we should train a neural network to predict the optimal cost J given the initial state x0. Once such a network has been trained, we must use it as a terminal cost inside an OCP with the same formulation, but with a shorter horizon (e.g. half). We should be able to empirically show that the introduction of the terminal cost compensates the decrease of the horizon length. For this test use  to a double pendulum.

to include the neural network inside the optimal control problem. The challenge comes from an incompatibility of tensorflow data types with casadi. This problem can be solved in two ways.

One option is to re-implement the neural network function using casadi data types and operators. As an example, this is how a student of mine implemented in casadi a network using ReLU activation functions:

def nn_decisionfunction(params, x):
    out = x
    it = 0

    for param in params:
        param = SX(param.tolist())

        if it % 2 == 0:
            out = param @ out # linear layer
        else:
            out = param + out # add bias
            out = fmax(0., out) # apply ReLU function

        it += 1

    return out
   
Another option is to use pytorch instead of tensorflow, and then use the l4casadi library (https://pypi.org/project/l4casadi/) that allows you to include directly a pytorch network inside casadi.

Suggestions 
When preparing the final report, make sure to account for the following tips. 
1	Mathematically describe the optimal control problem formulation.
2. Describe the structure of the neural network (e.g., number of layers, number of neurons per
layer, activation functions).
3. Report the problem constraints.
4. Include plots of some trajectories (position, velocity, torque) obtained controlling the system
with the optimal policy.
5. Report the value of the hyper-parameters of the algorithm (e.g., learning rate, mini-batch size,
number of data points).
6. Report the training time.
7. Include plots (color maps) of the Value and policy function whenever possible.
8. Include videos of the robots controlled with the policy found.
9. Include a discussion of potential improvements you did not have time to implement/test
