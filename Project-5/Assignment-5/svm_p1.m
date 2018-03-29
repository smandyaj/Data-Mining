load('X_test');
load('X_train');
load('y_test');
load('y_train');
numLabels = 6;

% kernel 1 = polynomial
% kernel 2 = gaussian

for kernel = 1 : 2
    model = cell(numLabels,1);
    for k=1:numLabels
        if kernel == 1
            model{k} = fitcsvm(X_train,y_train(:,k),'ClassNames',[false true],'KernelFunction','polynomial','PolynomialOrder',2);
        else
            model{k} = fitcsvm(X_train,y_train(:,k),'ClassNames',[false true],'KernelFunction','gaussian','KernelScale','auto');
        end
    end
    
    N = size(X_test,1);

    for j=1:numLabels
        [label,score] = predict(model{j},X_test);
        Scores(:,j) = score(:,2);
    end;
    
    Scores(Scores>0)=1;
    Scores(Scores<0)=0;
    total = size(y_test,1);
    
    similarityScore = zeros(total,1);
    acc = 0;
    for i=1:total
        intersection = 0;
        union = 0;
        for j = 1:6
            if(Scores(i,j) == 1 && y_test(i,j) == 1)
                intersection = intersection+1;
            end
            if(Scores(i,j) ~= 0 || y_test(i,j) ~= 0)
                union = union +1;
            end
        end
        acc = acc + intersection/union;
        
        similarityScore(i,1) = acc;
    end
    
    correct = length(find(similarityScore > 0 ));
    
    accuracy = (acc/total)*100;
    
    if ( kernel == 1 )
        fprintf('Accuracy of SVM using Polynomial kernel: %f \n', accuracy);
    else
        fprintf('Accuracy of SVM using Guassian kernel: %f ', accuracy);
    end
end