function[ accuracy,prediction] = svm_p2()
load('X_test');
load('X_train');
load('y_test');
load('y_train');
unique_labels = unique(y_train);
numInst = size(X_train,1);
numLabels = max(unique_labels);

model = cell(numLabels,1);
for k=1:numLabels
    indx = eq(y_train,unique_labels(k));
    model{k} = fitcsvm(X_train,indx,'ClassNames',[false true],'KernelFunction','polynomial','PolynomialOrder',2);
end

N = size(X_test,1);
Scores = zeros(N,numLabels);

for j=1:numLabels
    [~,score] = predict(model{j},X_test);
    Scores(:,j) = score(:,2);
end;

[~,maxScore] = max(Scores,[],2);

binary = maxScore==y_test;

[total_records, ~] = size(y_test);
accuracy = sum(binary(:) == 1) * 100/total_records;

%fprintf('Accuracy of SVM : %f ', accuracy);
prediction = maxScore;
end