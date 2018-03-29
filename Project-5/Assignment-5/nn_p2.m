function[ accuracy,prediction] = nn_p2()
load X_train
load X_test
load y_train
load y_test

net.trainParam.showWindow = false;
Y = zeros(10, 500);

for i = 1:10
    %num = length(find(y_train == i));
    pos = find(y_train == i);
    Y(i, pos) = 1;
end
network = feedforwardnet(25);
%network = configure(network, X_train', y_train);
network = train(network, X_train', Y);

Pred = zeros(1, 3251);
% 
y = network(X_test');

Y = vec2ind(y);

accuracy = length(find(y_test == Y'))/ length(y_test) * 100;
%fprintf('The accuracy of the Neural Network is: %f \n', accuracy);
prediction = Y';
end
