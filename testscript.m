%clear
clc
fprintf('Script started...\n')

% load fast features
load MTtestdata;

% Instantiate models
n_its = 5;

linreg = MT_linear(size(T_X2d{1},1), 'dim_reduce', 0, 'n_its', 100);
disp('Confirm prior computation switches: linear');
linreg.printswitches;
logreg = MT_logistic(size(T_X2d{1},1), 'dim_reduce',0, 'n_its',n_its);
disp('Confirm prior computation switches: logistic');
logreg.printswitches;

FD_linreg = MT_FD_model(size(T_X{1},2), size(T_X{1},1), 'linear', 'n_its', n_its, 'tr_adjust', 1);
disp('Confirm prior computation switches: FD linear');
FD_linreg.printswitches;

FD_logreg = MT_FD_model(size(T_X{1},2), size(T_X{1},1), 'logistic', 'n_its', n_its, 'tr_adjust', 0);
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


% note: these are overfitted: they update based on the whole data and then
% classify the training data from the new subject. In actual use this would
% be only updated with a fraction of the subject-specific data.

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
