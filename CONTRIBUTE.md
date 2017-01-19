# Contributions 

If you have a method that fits within this framework it is quite simple to add to the current codebase, which takes care of some convenience issues like output form, dealing with labels, etc. The basic structure of the code is to have a parent class that defines the expectation step while a child class defines the dataset-specific optimization. 

## MT\_baseclass 

This class implements a parent class that has methods which define update operations for a Gaussian distribution given samples as explained in the paper. It needs to be called from an inheriting class that has the **fit\_model** method. To extend this class simply inherit and alter the **fit\_prior** function. 

## MT\_linear

This class implements the likelihood computation and is used as the external interface to the code (for usage, see *testscript.m*). It passes on the appropriate arguments to its parent class and *performs all data safety steps as well as pre-processing*. In order to access the output of the parent function it accesses the property obj.prior. To extend the functionality it is sufficient to re-implement the following functions:

1. fit\_model
2. fit\_new\_task
3. loss
4. predict

## MT\_logistic

This is an example of an extension of MT\_linear (see Fiebig et al. 2016) which shows how new algorithms can be implemented with relatively little code. 
