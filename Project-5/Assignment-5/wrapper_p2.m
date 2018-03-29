% wrapper for svm,kn and nn
load('y_test');
tic;
[svmA,svmP] = svm_p2();
[knnA,knnP] = knn_p2();
[nnA,nnP] = nn_p2();

combinedPred = [svmP knnP nnP];
ensemblePred = mode(combinedPred,2);
correct = ensemblePred==y_test;
total = size(correct,1);
accuracy = sum(correct(:) == 1) * 100/ total;
toc;
fprintf('Accuracy of SVM : %f \n', svmA);
fprintf('Accuracy of KNN : %f \n', knnA);
fprintf('Accuracy of NN : %f \n', nnA);
fprintf('Accuracy of Ensemble : %f ', accuracy);