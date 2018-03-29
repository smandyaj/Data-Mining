file_path='D:\ASU\Spring-2017_Course\DM\Assignments+Projects\Assignment-3\Human Activity Recognition(1)\Human Activity Recognition\';
xtrain_filename = strcat(file_path,'X_train.txt');
ytrain_filename = strcat(file_path,'y_train.txt'); 
xtest_filename = strcat(file_path,'X_test.txt');
ytest_filename = strcat(file_path,'y_test.txt');

delimiter = ' ';
% training data
xtrain = importdata(xtrain_filename,delimiter);
ytrain = importdata(ytrain_filename);

% testing data 
xtest = importdata(xtest_filename,delimiter);
ytest = importdata(ytest_filename);

trained_model = fitcknn(xtrain,ytrain,'NumNeighbors',5,'Distance','euclidean');

% predict
%test = xtrain(60,:);
output = predict(trained_model,xtest);
binary = output==ytest;

[total_records, ~] = size(ytest);
accuracy = sum(binary(:) == 1) * 100/total_records;

fprintf('Accuracy of KNN : %f ', accuracy);

