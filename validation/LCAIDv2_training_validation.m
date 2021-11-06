clc;
clear;

data = csvread('./Exploratory cohort.csv');
label = data(1,:);
label = label';
data = data(2:end,:);
data = data';
data(45:46,:)=[];label(45:46)=[]; % remove two outlets

data1 = csvread('./Training Cohort.csv');
label1 = data1(1,:);
label1 = label1';
data1 = data1(2:end,:);
data1 = data1';

data2 = csvread('./Independent validation cohort.csv');
label2 = data2(1,:);
label2 = label2';
data2 = data2(2:end,:);
data2 = data2';

data3 = csvread('./Prospective validation Cohort.csv');
label3 = data3(1,:);
label3 = label3';
data3 = data3(2:end,1:end);
data3 = data3';

data4 = csvread('./Screening Cohort.csv');
label4 = data4(1,:);
label4 = label4';
data4 = data4(2:end,1:end);
data4 = data4';



data = sqrt(data);
data1 = sqrt(data1);

diff = mean(data)./mean(data1); 
data1 = data1.*diff;

train_data=[data;data1];
train_label=[label;label1];

pos_idx = find(train_label==0);
neg_idx = find(train_label==1);


data2 = sqrt(data2);
diff = mean(train_data(:,:))./mean(data2);
data2 = data2.*diff;


data3 = sqrt(data3);
diff = mean(train_data(:,:))./mean(data3);
data3 = data3.*diff;


data4 = sqrt(data4);
diff = mean(train_data(pos_idx,:))./mean(data4);
data4 = data4.*diff;


addpath('./liblinear-2.20/matlab/');
fprintf('train process ... ... \n');
option=['-c 2 -q'];


test_data = data2;
test_label = label2;
test_data1 = data3;
test_label1 = label3;
test_data2 = data4;
test_label2 = label4;
train_data = [train_data];
train_label = [train_label];

src_train_data=train_data;
data_mean = mean(train_data);
data_std = std(train_data);
train_data = (train_data-repmat(data_mean,size(train_data,1),1))./repmat(data_std,size(train_data,1),1);
test_data = (test_data-repmat(data_mean,size(test_data,1),1))./repmat(data_std,size(test_data,1),1);
test_data1 = (test_data1-repmat(data_mean,size(test_data1,1),1))./repmat(data_std,size(test_data1,1),1);
test_data2 = (test_data2-repmat(data_mean,size(test_data2,1),1))./repmat(data_std,size(test_data2,1),1);


model = train(train_label,sparse(((train_data))),option);
[train_predict,train_acc,train_score] = predict(train_label,sparse(train_data),model);
[test_predict,test_acc,test_score] = predict(test_label,sparse(test_data),model);
[test_predict1,test_acc1,test_score1] = predict(test_label1,sparse(test_data1),model);
[test_predict2,test_acc2,test_score2] = predict(test_label2,sparse(test_data2),model);


train_stats=confusionmatStats(train_label,train_predict);
test_stats=confusionmatStats(test_label,test_predict);
test_stats1=confusionmatStats(test_label1,test_predict1);
test_stats2=confusionmatStats(test_label2,test_predict2);


fprintf('train: spec:%f; sen; %f; acc: %f\n',train_stats.specificity(2),train_stats.sensitivity(2),train_acc(1)/100);
fprintf('test: spec:%f; sen; %f; acc: %f\n',test_stats.specificity(2),test_stats.sensitivity(2),test_acc(1)/100);
fprintf('test1: spec:%f; sen; %f; acc: %f\n',test_stats1.specificity(2),test_stats1.sensitivity(2),test_acc1(1)/100);
fprintf('test2: spec:%f; sen; %f; acc: %f\n',test_stats2.specificity(2),test_stats2.sensitivity(2),test_acc2(1)/100);


% plot auc

figure;
train_auc = roc_curve(-train_score,train_label,1);
figure;
test_auc = roc_curve(-test_score,test_label,0);
figure;
test_auc1 = roc_curve(-test_score1,test_label1,0);
%figure;
%test_auc2 = roc_curve(-test_score2,test_label2,0);


% plot pr
prec_rec(-train_score,train_label,'plotROC',0,'plotPR',1);
prec_rec(-test_score,test_label,'plotROC',0,'plotPR',1);
prec_rec(-test_score1,test_label1,'plotROC',0,'plotPR',1);

pos_idx = find(test_label2==1);
len_pos = length(pos_idx);
for i=1:1010
    curr_idx = randi(len_pos);
    sel_idx = pos_idx(curr_idx);
    test_score2=[test_score2;test_score2(sel_idx)];
    test_label2=[test_label2;test_label2(sel_idx)];
end
figure;
test_auc2 = roc_curve(-test_score2,test_label2,0);
prec_rec(-test_score2,test_label2,'plotROC',0,'plotPR',1);




return;

