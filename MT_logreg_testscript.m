clear
clc
fprintf('Script started...\n')

% X must be a cell array of 3D matrices with dimensions (trials, channels, bandpower/time
% domain features) and y must be a cell array of equal size with elements 
% of size (trials,1).
data = load('MotorImageryData.mat');
s = 4; % Index of subject-specific dataset to use
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

% Instantiate models
n_its = 5;

linreg = MT_linear(1408, 'dim_reduce', 0, 'n_its', n_its);
disp('Confirm prior computation switches: linear');
linreg.printswitches;

logreg = MT_logistic(1408, 'dim_reduce', 0, 'n_its', n_its);
disp('Confirm prior computation switches: logistic');
logreg.printswitches;

FD_linreg = MT_FD_model(12, 128, 'linear', 'n_its', n_its, 'tr_adjust', 1);
disp('Confirm prior computation switches: FD linear');
FD_linreg.printswitches;

FD_logreg = MT_FD_model(12, 128, 'logistic', 'n_its', n_its, 'tr_adjust', 0);
disp('Confirm prior computation switches: FD logistic');
FD_logreg.printswitches;

fprintf('\n###\n')
fprintf('Training linreg prior...\n')
linreg.fit_prior(T_X2d, T_y);
fprintf('Training logreg prior...\n')
logreg.fit_prior(T_X2d, T_y);
fprintf('Training FD linreg prior...\n')
FD_linreg.fit_prior(T_X, T_y);
fprintf('Training FD logreg prior...\n')
FD_logreg.fit_prior(T_X, T_y);

acc = mean(linreg.prior_predict(X2d_s) == y_s);
fprintf('Linreg prior accuracy: %.2f\n', acc*100);
acc = mean(logreg.prior_predict(X2d_s) == y_s);
fprintf('Logreg prior accuracy: %.2f\n', acc*100);

% note: this is unclear notation for FD. Make a new function to make it
% more obvious what is happening here. 
acc = mean(FD_linreg.prior_predict(X_s) == y_s);
fprintf('Linreg FD prior accuracy: %.2f\n', acc*100);
acc = mean(FD_logreg.prior_predict(X_s) == y_s);
fprintf('Logreg FD prior accuracy: %.2f\n', acc*100);

out = linreg.fit_new_task(X2d_s, y_s, 'ml', 1);
acc = mean(out.predict(X2d_s) == y_s);
fprintf('New task training accuracy (linreg): %.2f\n', acc*100);
out = logreg.fit_new_task(X2d_s, y_s, 'ml', 1);
acc = mean(out.predict(X2d_s) == y_s);
fprintf('New task training accuracy (logreg): %.2f\n', acc*100);
out = FD_linreg.fit_new_task(X_s, y_s, 'ml', 1);
acc = mean(out.predict(X_s) == y_s);
fprintf('New task training accuracy (FD linreg): %.2f\n', acc*100);
out = FD_logreg.fit_new_task(X_s, y_s, 'ml', 1);
acc = mean(out.predict(X_s) == y_s);
fprintf('New task training accuracy (FD logreg): %.2f\n', acc*100);

fprintf('Script finished!\n');
