# Transfer Learning in Brain-Computer Interfaces

[![Latest Version](http://img.shields.io/pypi/v/Markdown.svg)](http://pypi.python.org/pypi/Markdown)
[![Downloads](http://img.shields.io/pypi/dm/Markdown.svg)](https://pypi.python.org/pypi/Markdown#downloads)

Object-oriented multi-task learning framework for Bayesian hierarchial models written in MATLAB. Implements the regression based approaches from Jayaram et al. [1] and the logistic approach by Fiebig et al. [2], but should be easily extendable.

# Installation

The folder simply needs to be put into the MATLAB path

# Usage

The file testscript.m has sample data and runs through classification with the various approaches. Generally, however, the order of things is as follows:

1. Instantiate the model with appropriate size parameters and switches

```
model = MT_linear([flags]);
moel = MT_FD_model({'linear','logistic', [flags])
```

Note that there are two possible models (for details see [1]): the linear and bilinear. The linear model requires datasets of the form (features x labels) while the bilinear model (accessible through MT\_FD\_model) requires datsets of the form (electrodes x features x labels). 



2. Train the prior (and optionally classify using the prior mean)

```
model.fit_prior(X_cell, y_cell)
# prior mean classification
y_hat = model.prior_predict(new_X)
```

3. Train the subject specific model. The output struct includes a classification function as well as various explanatory parameters.

```
updated = model.fit_new_task(Xtrain, ytrain)
y_hat = updated.predict(new_X)
```

For more help information and information on flags, please check the documentation for the functions MT\_baseclass and MT\_linear


# Functionality

## Base class

MT_baseclass implements a class that can be inherited which sketches out the general form of the two algorithms: Dataset specific models are updated in parallel (when possible) and then a distribution is generated from the data-specific models, repeated in an alternating fashion. So, your basic E-M approach. Any class can inherit from this in order to create more algorithms of this nature

## Dataset-specific models

MT_linear is the basic approach that considers the multi-task problem in a linear regression setting [1]. To create new methods, the easiest way is to inherit from the class and simply change the method that fits the classifier given a prior and a dataset (see MT\_logistic for an example). 

# Additions

There is an enormous space of possibilties for how this framework can be extended and improved. If you are interested in adding a method please let me know (vjayaram@tue.mpg.de) and see the included CONTRIBUTE.md file for some help on how your method could be fit into this framework.


# Feedback
Please feel free (indeed, urged) to let me know through the issues feature whether something is not working and I will be happy to fix them as soon as I can. If preferrable, feel free also to send mail to vjayaram@tue.mpg.de 

# Python
For the python Python version please check out our related [page](https://github.com/bibliolytic/pyMTL). 


# Citations:

[1] Jayaram, Vinay, et al. "Transfer learning in brain-computer interfaces." IEEE Computational Intelligence Magazine 11.1 (2016): 20-31.

[2] Karl-Heins Fiebig, Vinay Jayaram, Jan Peters, and Moritz Grosse-Wentrup. Multi-
task logistic regression in brain-computer interfaces. In Proceedings of the 2016 IEEE
International Conference on Systems, Man, and Cybernetics, 2016.

