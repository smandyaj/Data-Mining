function[ accuracy,prediction] = knn_p2()
%tic    %Code Run Time
load X_train;
%size(M);
load y_train;
%size(N);
Model = fitcknn(X_train, y_train, 'NumNeighbors', 7, 'Distance', 'euclidean');   %Fitting Model With 5NN

rloss = resubLoss(Model);  %Testing Quality of Model

load X_test;
load y_test;

Prediction = predict(Model, X_test);
%size(Prediction);

%Accuracy of the 5NN Classifier
Accuracy_7NN = length(find(y_test == Prediction)) / length(y_test) * 100;

%fprintf('The accuracy of KNN with 5 nearest neighbors is %f \n', Accuracy_5NN);
accuracy = Accuracy_7NN;
prediction = Prediction;
%toc;

end