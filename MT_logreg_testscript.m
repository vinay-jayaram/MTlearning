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
T_X2 = {};
T_y = {}; % MTL task datasets (cell array of vectors)
count = 1;
for i = 1:size(data.X, 1)-5
    if i ~= s
        % Extract FD features in the form
        % (trials, spatial features, spectral/temporal features)
        T_X{count} = double(squeeze(data.X3d(i, :, :, :)));
        T_X2d{count} = reshape(T_X{count},size(T_X{count},1)*size(T_X{count},2),[]);
        T_X2d{count}(1409:end,:) = [];
        % Parse labels from {-1, 1} to {0, 1}
        T_y{count} = (double(data.Y(i, :))'+1)*0.5;
        count = count+1;
    end
end

% Extract individual subject specific dataset in the form
% (trials, spatial features, spectral/temporal features)
X_s = double(squeeze(data.X3d(s, :, :, :)));
X2d_s = reshape(X_s, size(X_s,1)*size(X_s,2),[]);
X2d_s(1409:end,:) = [];
y_s = (double(data.Y(s, :))'+1)*0.5;


% % Instantiate model
 FD = MT_FD_linear('tr_adjust',1);
 linear = MT_linear('dim_reduce',0);
disp('Confirm prior computation switches: FD');
 FD.printswitches;
disp('Confirm prior computation switches: linear');
 linear.printswitches;
 
 

% % Train Gaussian prior from task datasets with MTL and print prior accuracy
% % on unseen subject-specific dataset
FD.fit_prior(T_X, T_y);
FD_prior = FD.getprior;
acc = mean(FD.predict({FD_prior.weight.mu, FD_prior.alpha.mu},X_s, FD.labels) == y_s);
fprintf('FD prior accuracy: %.2f\n', acc*100);

linear.fit_prior(T_X2d, T_y);
linear_prior = linear.getprior;
acc = mean(linear.predict(linear_prior.mu,X2d_s, linear.labels) == y_s);
fprintf('linear prior accuracy: %.2f\n', acc*100);

% % Adapt model with trained prior to subject-specific dataset and print
% % accuracy on the training task data
out = FD.fit_new_task(X_s, y_s,'ml',1);
fprintf('New FD task training accuracy: %.2f\n', mean(y_s == out.predict(X_s))*100);
out = linear.fit_new_task(X2d_s, y_s,'ml',1);
fprintf('New task training accuracy: %.2f\n', mean(y_s == out.predict(X2d_s))*100);


fprintf('Script finished!\n')
% PS: I'm still laughing about the "spefific" typo - I should really stop reading it out loud
