# Solving Combinatorial Optimization Problems with Reinforcement Learning methods

In this project we study how Reinforcement Learning (RL) can be applied to combinatorial and graph optimization problems. 
We closely follow choosen sources and investigate the RL setup for neural network architectures, model parameters and study several RL algorithms. 
We provide the results for Bin Packing problem, News Vendor, Minimum Vertex Cover (MVC) and Maximum Cut (MAXCUT).
Results are quite encouraging, we think that Reinforcement Learning technics can be widely used in solving combinatorial optimization problems.

## How to run the code

Our project requires intensive computing, so we needed GPU support and decided to use Amazon AWS SageMaker.
Here are some instrcutions, how you can run code, using AWS.

1. Login to your AWS account.
1. Search for `SageMaker`
1. Click `Notebook instances`
1. Enter notebook name of your choice &rarr; choose instance type `ml.t2.medium` 
1. Create a new IAM role if required. Also attach `AmazonEC2ContainerRegistryPowerUser` policy to the role. See [guide](https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies_manage-attach-detach.html#add-policies-console) for adding permissions to an IAM Role.
1. Under `Git repositories`, click `Additional repository` &rarr; Select `Clone a Public repo` and enter `https://github.com/d-kirichik/ML_project.git`
1. Click `Create Notebook` and wait for the instance to come up.
1. Once the instance is ready, select it and click `Open Jupyter` 
1. Above step will load this repository and then you can follow the python notebooks for each problem.

The main launcher allows to launch different problems, but you should change specific variables, comments in the launcher notebook will guide you.
Code for MVC and MAXCUT problems does not require launcher and can be launched in colab.
