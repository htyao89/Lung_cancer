clc;
clear;

data = csvread('./LC 365-POS.csv');
label = data(1,2:312);
feature_index = data(3:end,1);
data = data(3:end,2:312);
data = data';

addpath('./liblinear-2.20/matlab/');
fprintf('train process ... ... \n');
option=['-c 5 -q'];

if ~exist('./code_data')
     mkdir('./code_data/')
end

if ~exist('./code_data/weights.mat')
    all_weights=[];
    all_acc=[];
    all_score=[];
    for r=1:2000
        r
        train_data=[];
        train_label=[];
        test_data=[];
        test_label=[];

        neg_idx = find(label==0);
        len = length(neg_idx);
        rand_idx = randperm(len);
        neg_idx = neg_idx(rand_idx);
        for i=1:length(neg_idx)
            if mod(i,4)==0
                test_data=[test_data;data(neg_idx(i),:)];
                test_label =[test_label;label(neg_idx(i))];
            else
                train_data=[train_data;data(neg_idx(i),:)];
                train_label =[train_label;label(neg_idx(i))];        
             end
        end

        pos_idx = find(label==1);
        len=length(pos_idx);
        rand_idx = randperm(len);
        pos_idx = pos_idx(rand_idx);
        for i=1:length(pos_idx)
            if mod(i,4)==0
                test_data=[test_data;data(pos_idx(i),:)];
                test_label =[test_label;label(pos_idx(i))];
            else
                train_data=[train_data;data(pos_idx(i),:)];
                train_label =[train_label;label(pos_idx(i))];        
             end
        end  
        model = train(train_label,sparse(normr(train_data(:,:))),option);
        [p1,c1,d1] = predict(train_label,sparse(normr(train_data(:,:))),model);
        stats1=confusionmatStats(train_label,p1);
        [p2,c2,d2] = predict(test_label,sparse(normr(test_data(:,:))),model);
        stats2=confusionmatStats(test_label,p2);
        all_score=[all_score;stats1.specificity(2),stats1.sensitivity(2),c1(1)/100,stats2.specificity(2),stats2.sensitivity(2),c2(1)/100];
        svm_weight = model.w;
        svm_weight = svm_weight.^2;
        all_acc=[all_acc;c1(1),c2(1)];
        all_weights=[all_weights;svm_weight];
        auc(r) = roc_curve(d2,test_label,0);
    end
    weights = mean(all_weights);
    save ./code_data/weights.mat weights all_acc all_score auc all_weights
else
    load ./code_data/weights.mat
end

[muhat,sigmahat,muci,sigmaci]=normfit(all_score(:,1:6));

[sorted_weight,sorted_idx ]=sort(weights,'descend');
feature_index=feature_index(sorted_idx);
acc=[];
acc_val=[];
parpool('local',6)
for r=1:500
    train_data=[];
    train_label=[];
    test_data=[];
    test_label=[];

    neg_idx = find(label==0);
    len = length(neg_idx);
    rand_idx = randperm(len);
    neg_idx = neg_idx(rand_idx);
    for i=1:length(neg_idx)
        if mod(i,4)==0
            test_data=[test_data;data(neg_idx(i),:)];
            test_label =[test_label;label(neg_idx(i))];
        else
            train_data=[train_data;data(neg_idx(i),:)];
            train_label =[train_label;label(neg_idx(i))];        
         end
    end
    
    pos_idx = find(label==1);
    len=length(pos_idx);
    rand_idx = randperm(len);
    pos_idx = pos_idx(rand_idx);
    for i=1:length(pos_idx)
        if mod(i,4)==0
            test_data=[test_data;data(pos_idx(i),:)];
            test_label =[test_label;label(pos_idx(i))];
        else
            train_data=[train_data;data(pos_idx(i),:)];
            train_label =[train_label;label(pos_idx(i))];        
         end
    end
    parfor i=1:length(sorted_idx)
        tmp_train_data = train_data(:,sorted_idx(1:i));
        tmp_test_data = test_data(:,sorted_idx(1:i));
        addpath('./liblinear-2.20/matlab/');
        fprintf('train process ... ... \n');
        option=['-c 5 -q'];
        model = train(train_label,sparse(normr(tmp_train_data)),option);
        [p1,c1,d1] = predict(train_label,sparse(normr(tmp_train_data)),model);
        [p2,c2,d2] = predict(test_label,sparse(normr(tmp_test_data)),model);
        acc(r,i)=c2(1);
    end
end
mean_acc=mean(acc);
plot(mean_acc);


