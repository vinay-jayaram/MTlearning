% X must be a cell array of 3D matrices with dimensions (channels, bandpower/time
% domain features, trials) and y must be a cell array of equal size with elements 
  % of size (trials,1). Test data will be included as of Friday 16th

% solution with uninformed spatial prior (note that accuracies are *not*
% transfer learning accuracies but rather multitask accuracies. For
% transfer learning accuracies you must update the decision rule with data
% from the newest subject)

MT=multitask2015(X(1:end-1),y(1:end-1),'verbose',1,'out_acc',1);


% solution with ridge-regression prior
MT_rr=multitask2015(X(1:end-1),y(1:end-1),'rr_init',1,'verbose',1);

% train on half the data
MT_s7=multitask2015(X{end}(:,:,11:20),y{end}(11:20),'prior',MT_rr);

% test the other half
acc=getbinacc(X{end}(:,:,1:10),y{end}(1:10),MT_s7.weight.mat, MT_s7.alpha.mat);

% cross-validated accuracy for all
[priors,accs]=MTdataset(X,y,{'verbose',1});

% For other options please look up documentation for each function
% (available once the function is in your path by using 'help
% <functionname>'
