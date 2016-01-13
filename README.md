# Transfer Learning in Brain-Computer Interfaces
Multi-task learning framework (see Jayaram et al. 2016, to be publilshed Feb. 2016) written in MATLAB. 

# Functionality
The functions MT and MT_FD implement the core functionality as described in the paper (that is, they take as input the data and labels, and return as output the learned priors and the multi-task decision boundaries). Given the appropriate options they also can use given priors to give an updated decision boundary for a new subject/session. However to actually apply these decision boundaries it is necessary to use the other functions. For details on usage please check the individual help files, but for your convenience a wrapper function (multitask2015) has been written that should simplify things. 

# Examples
Test data will be bundled with the repo shortly

# Feedback
Please feel free (indeed, urged) to let me know through the issues feature whether something is not working and I will be happy to fix them as soon as I can. If preferrable, feel free also to send mail to vjayaram@tuebingen.mpg.de 

# Python
Python version coming shortly... (note that the test script references data on the MPI-IS server and so is not usable by outside parties)
