# Neural Networks Memorization

The goal is to study when and how training data gets memorized
## Installation

1. `git clone --recurse-submodules https://github.com/RoyRin/neural_nets_memorization` 
   1. Note This library uses a git submodule.
2. `pip3 install poetry`  (poetry is the package manager used for this library)
3. `poetry shell`  - source into a virtual environment (create a new one if it doesn't exist)
4. `poetry install` - install the dependencies for this package

## Memorization Experiment organization:

Data Collection Process:

1. Step 1: Plan Creation and Organization
    1. Create the plan for the memorization estimator
    2. split the plan into chunks (different files)
    3. upload the different chunks to S3

2. Step 3: EC2 Instance Creation (done by AWS administrator)
    1. create a number of instance in the cloud
        # todo: need to adjust the size of the instances
        # todo: need to write the instances in a nice format

3. Step 4: Code Running
    1. get the instance IPs
    2. run the same code on each instance, except pointing to a different chunk. 
        the code should first write the results to the local instance, then copy them to the s3 bucket

4. Step 5: Data Aggregation
    1. collect the results off the s3 instances
    2. aggregate the results

## memorization experiment plan:

Goal: need to train ~2000 models. Each one takes ~6 minutes on a GPU, so, my plan is to:

Spin up a GContainerE cluster with gpus. 

You could design it against minikube. Build it in docker. Get it working. 
Spin up a small cluster with one node, make sure it works and then scale your cluster up

steps:
1. containerize a single training
2. create some kind of scheduler using GCE cluster or Minikube

## Deployments:
1. Moved the cloud deployments infrastructure code to : https://github.com/RoyRin/cloud_compute_utils
## Hot Link
* [Main Colab](https://colab.research.google.com/drive/1PglaG8GXSUsNeMIMGDiLIBXztTyAR9NY)
* [Secondary colab](https://colab.research.google.com/drive/1CgqHvzUlA29okP8wiNES-SZib_dYvYFL)
* [Google Drive Directory](https://drive.google.com/drive/u/0/folders/17O2phSWfgsyUL86LWuvB_ajFQzGCcD97)
* [Ongoing Research Gdrive Document](https://docs.google.com/document/d/10CDG_bGwTWhkE1Uvgq76aWhzE5Tic1KiNg6AuDPk440/edit)
* [Notion](https://www.notion.so/Papers-124bdd6b43aa470481daee3b760d7cdb)




---------
notes to self:
1. once we have a function that we think describes the trend of influence
  we should convolve all the influences-over-time with that function, to measure how good of a fit this function is

speed up for computing influences:
beep boop baap -
1. use numba. prange
2. use multiprocessing -  I think I need to do : https://docs.python.org/3/library/multiprocessing.shared_memory.html
3. use cuda - https://curiouscoding.nl/phd/2021/03/24/numba-cuda-speedup/

#------------------------------------------------------------------------------------------
data that we want:
1. influence + memorization over time
2. a distribution/histogram of influence
2.a.     generate a heat map for memorization over time
3. we need a single metric for memorization (i.e. average, median, etc)
    - given a metric we should look at generalization gap correlated with
----
look at memorization versus training loss**
^ this is going to be a sick plot.
      training loss is actually a function of how close the learned plot. It should be monotonic.

#Words to say:
thoughts: if you only train for a little bit, you are effectively underparametrized.
based on : that early on you are intuition: an underparametrized model doesnt memorize

stages of learning:
as you train, you pick up more and more spectral modes
at a certain point, you reach a point where you are effectively overparametrized
this is probably the point where memorization occurs.
###
##So, we should look at rate of memorization given the "Strength of the model"
###

#note : we also want to do these same trainings, on synthetic datasets.

NOTE :
numba DOES NOT HAVE FLOAT16