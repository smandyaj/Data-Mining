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

unique_labels = unique(ytrain);
numInst = size(xtrain,1);
numLabels = max(unique_labels);

model = cell(numLabels,1);
for k=1:numLabels
    indx = eq(ytrain,unique_labels(k));
    model{k} = fitcsvm(xtrain,indx,'ClassNames',[false true],'KernelFunction','polynomial','PolynomialOrder',2);
end

N = size(xtest,1);
Scores = zeros(N,numLabels);

for j=1:numLabels
[~,score] = predict(model{j},xtest);
Scores(:,j) = score(:,2);
end;

[~,maxScore] = max(Scores,[],2);


binary = maxScore==ytest;

[total_records, ~] = size(ytest);
accuracy = sum(binary(:) == 1) * 100/total_records;

fprintf('Accuracy of SVM : %f ', accuracy);