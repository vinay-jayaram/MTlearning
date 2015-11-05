fband=[1 5 8 14 20 30 40 50];
X={};
y={};

for i = 1:7
    for j = 2
        z=mbci;
        loadstr=sprintf('/agbs/bcigroup/Studies/z009_ALSCogBCI/data/LEK2/LEK2%.3d/LEK2S%.3dR0%d.dat',i,i,j);
        z.load('file',{loadstr},'protocol','LEKBCITrainingProtocol');
        z.rmchans('chans',125:130);
        z.cav;
        z.bandpower('freqs',fband);
        X=cat(2,X,{cat(3,z.spectrum{end}.features{1},z.spectrum{end}.features{2})});
        y=cat(2,y,{cat(1,-ones(size(z.spectrum{end}.features{1},3),1),ones(size(z.spectrum{end}.features{2},3),1))});
    end
end

% solution with uninformed spatial prior (note that accuracies are *not*
% transfer learning accuracies but rather multitask accuracies. For
% transfer learning accuracies you must update the decision rule with data
% from the newest subject)

MT=multitask2015(X(1:end-1),y(1:end-1),'verbose',1,'out_acc',1);


% solution with ridge-regression prior
MT_rr=multitask2015(X(1:end-1),y(1:end-1),'rr_init',1,'verbose',1);

% cross-validated accuracy for all
[priors,accs]=MTdataset(X,y,{'verbose',1});

% For other options please look up documentation for each function
% (available once the function is in your path by using 'help
% <functionname>'