%clear
clc
fprintf('Script started...\n')

% load fast features
load MTtestdata;

% Instantiate models
n_its = 5;

order = {'linear','logistic','l1'};

model{1} = MT_linear('dim_reduce', 0, 'n_its', 100);
disp('Confirm prior computation switches: linear');
model{1}.printswitches;
model{2} = MT_logistic('dim_reduce',0, 'n_its',n_its);
disp('Confirm prior computation switches: logistic');
model{2}.printswitches;

model{3} = MT_FL('n_its',100);
disp('Confirm prior computation switches: l1');
model{3}.printswitches;

FD_linreg = MT_FD_model('linear', 'n_its', n_its, 'tr_adjust', 1);
disp('Confirm prior computation switches: FD linear');
FD_linreg.printswitches;

FD_logreg = MT_FD_model('logistic', 'n_its', n_its, 'tr_adjust', 0);
disp('Confirm prior computation switches: FD logistic');
FD_logreg.printswitches;

fprintf('\n###\n')
for i = 1:3
fprintf(['Training ' order{i} ' prior...\n'])
model{i}.fit_prior(T_X2d, T_y);
acc = mean(model{i}.prior_predict(X2d_s) == y_s);
fprintf('%s prior accuracy: %.2f\n', order{i}, acc*100);
out = model{i}.fit_new_task(X2d_s, y_s, 'ml', 1);
acc = mean(out.predict(X2d_s) == y_s);
fprintf('New task training accuracy (%s): %.2f\n', order{i}, acc*100);
end



fprintf('Training FD linreg prior...\n')
FD_linreg.fit_prior(T_X, T_y);
fprintf('Training FD logreg prior...\n')
FD_logreg.fit_prior(T_X, T_y);

% note: these are overfitted: they update based on the whole data and then
% classify the training data from the new subject. In actual use this would
% be only updated with a fraction of the subject-specific data.

acc = mean(FD_linreg.prior_predict(X_s) == y_s);
fprintf('Linreg FD prior accuracy: %.2f\n', acc*100);
acc = mean(FD_logreg.prior_predict(X_s) == y_s);
fprintf('Logreg FD prior accuracy: %.2f\n', acc*100);

out = FD_linreg.fit_new_task(X_s, y_s, 'ml', 1);
acc = mean(out.predict(X_s) == y_s);
fprintf('New task training accuracy (FD linreg): %.2f\n', acc*100);
out = FD_logreg.fit_new_task(X_s, y_s, 'ml', 1);
acc = mean(out.predict(X_s) == y_s);
fprintf('New task training accuracy (FD logreg): %.2f\n', acc*100);

fprintf('Script finished!\n');
