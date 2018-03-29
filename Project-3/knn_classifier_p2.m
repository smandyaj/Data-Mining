file_path = 'D:\ASU\Spring-2017_Course\DM\Assignments+Projects\Assignment-3\VidTIMIT(1)\VidTIMIT\';
xtrain_filename = strcat(file_path ,'X_train.mat');
ytrain_filename = strcat(file_path,'y_train.mat'); 
xtest_filename = strcat(file_path ,'X_test.mat');
ytest_filename = strcat(file_path,'y_test.mat');

xtr = load(xtrain_filename);
xtrain = xtr.X_train;
ytr = load(ytrain_filename);
ytrain = transpose(ytr.y_train);
xte = load(xtest_filename);
xtest = xte.X_test;
yte = load(ytest_filename);
ytest = transpose(yte.y_test);


trained_model = fitcknn(xtrain,ytrain,'NumNeighbors',5,'Distance','euclidean');

% predict
%test = xtrain(60,:);
output = predict(trained_model,xtest);
binary = output==ytest;

[total_records, ~] = size(ytest);

accuracy = sum(binary(:) == 1) * 100/total_records;

fprintf('Accuracy of KNN : %f ', accuracy);


