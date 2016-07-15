clear;
clc;
fprintf('Script started...\n')

% X must be a cell array of 3D matrices with dimensions (trials, channels, bandpower/time
% domain features) and y must be a cell array of equal size with elements 
% of size (trials,1).
data = load('MotorImageryData.mat');
s = 1; % Index of subject-specific dataset to use
% Extract task-datasets for MTL omitting the subject-spefific one (haha "spefific" sounds way too funny, I will keep that typo xD)
T_X = {}; % MTL task datasets (cell array of 3D matrices)
T_y = {}; % MTL task datasets (cell array of vectors)
count = 1;
for i = 1:size(data.X, 1)-5
    if i ~= s
        % Extract FD features in the form
        % (trials, spatial features, spectral/temporal features)
        T_X{count} = permute(double(squeeze(data.X3d(i, :, :, :))), [3, 1, 2]);
        % Parse labels from {-1, 1} to {0, 1}
        T_y{count} = (double(data.Y(i, :))'+1)*0.5;
        count = count+1;
    end
end
% Extract individual subject specific dataset in the form
% (trials, spatial features, spectral/temporal features)
X_s = permute(double(squeeze(data.X3d(s, :, :, :))), [3, 1, 2]);
y_s = (double(data.Y(s, :))'+1)*0.5;


% % Instantiate model
% k = 128; % Spatial feature dimensionality
% d = 12; % Spectral feature dimensionality
% model = MT_FD_LogReg(d, k);
% 
% % Train Gaussian prior from task datasets with MTL and print prior accuracy
% % on unseen subject-specific dataset
% model.fit_prior(T_X, T_y, 5);
% acc = sum((model.predict(X_s) > 0.5) == y_s) / double(length(y_s));
% fprintf('Prior accuracy: %.2f\n', acc*100);
% 
% % Adapt model with trained prior to subject-specific dataset and print
% % accuracy on the training task data
% err = model.fit_task(X_s, y_s);
% acc = 1 - err;
% fprintf('Adapted accuracy: %.2f\n', acc*100);


fprintf('Script finished!\n')
% PS: I'm still laughing about the "spefific" typo - I should really stop reading it out loud

%% Vinay's little bit for now

X2d = {};
for i = 1:4
    X2d{i} = reshape(T_X{i}(:,5:6,:),300,[]);
    X2d{i} = X2d{i}';
end
test = MT_linear;
prior_newcode = test.fit_prior(X2d, T_y);