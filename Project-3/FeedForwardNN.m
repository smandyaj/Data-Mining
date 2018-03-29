file_path = 'D:\ASU\Spring-2017_Course\DM\Assignments+Projects\Assignment-3\VidTIMIT(1)\VidTIMIT\';
xtrain_filename = strcat(file_path ,'X_train.mat');
ytrain_filename = strcat(file_path,'y_train.mat'); 
xtest_filename = strcat(file_path ,'X_test.mat');
ytest_filename = strcat(file_path,'y_test.mat');

xtr = load(xtrain_filename);
xtrain = xtr.X_train;
ytr = load(ytrain_filename);
ytrain = ytr.y_train;
xte = load(xtest_filename);
xtest = xte.X_test;
yte = load(ytest_filename);
ytest = yte.y_test;

target=zeros(25,3500);

for i=1:3500 
    temp=ytrain(i);
    target(temp,i)=1;
end

net = feedforwardnet(25);
[net,tr] = train(net, xtrain.', target);
test_target=net(xtest.');
int_test = vec2ind(test_target); 
count=0;
for j=1:1000
    if(int_test(j)~=ytest(j))
        count=count+1;
    end
end
accuracy=((1000-count)/1000)*100;
fprintf('Accuracy of ANN : %f',accuracy);



