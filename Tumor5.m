clc
clear
close all
feature = xlsread('TumorDataset.csv');


label=feature(:,1);

class1 = label(1:24,:);
class11 = label(190:213,:);
class14 = label(241:264,:);
clas22 = label(316:339,:);
label = [class1;class11;class14;clas22];
%Data spliting----------20-------80---------------------------------------


test1 = feature(1:5,:);
train1 = feature(6:24,:);
test2 = feature(190:194,:);
train2 = feature(195:213,:);
test3 = feature(241:245,:);
train3 = feature(246:264,:);
test4 = feature(316:320,:);
train4 = feature(321:339,:);
test = [test1;test2;test3;test4];
Xtrain = [train1;train2;train3;train4];
Xtrain(:,5)=0;
Xtrain(1,4)=0;
Xtrain(74,4)=0;
Xtrain(50,4)=0;
Xtrain(47,4)=0;
Xtrain(23,4)=0;
Xtrain(5,13)=0;
Xtrain(12,16)=0;

ltest1 =ones(5,1)*1;
ltrain1 = ones(19,1)*1;

ltest2 = ones(5,1)*2;
ltrain2 = ones(19,1)*2;

ltest3 = ones(5,1)*3;
ltrain3 = ones(19,1)*3;

ltest4 = ones(5,1)*4;
ltrain4 = ones(19,1)*4;
ltest = [ltest1;ltest2;ltest3;ltest4];
ltrain = [ltrain1;ltrain2;ltrain3;ltrain4];


%KNN-----------------------------------------------------------------------

KNNMdl = fitcknn(Xtrain , ltrain , 'NumNeighbors' , 3 , 'Standardize' , 1);
resubLoss(KNNMdl )*100
cv_1 = crossval(KNNMdl );
performance_KNN = (1-kfoldLoss(cv_1))*100

%SVM-----------------------------------------------------------------------

SVMMdl = fitcecoc(Xtrain,ltrain);
 
%% Beyzian-------------------------------------------------------------------

 %NBMdl = fitcnb(Xtrain,ltrain);
 
 %Decision Tree------------------------------------------------------------
 
 Mdltree = fitctree(Xtrain,ltrain);
 
 %Spliting with crossval---------------------------------------------------

 CVKNNMdl=crossval(KNNMdl);
 
 CVMdl=crossval(SVMMdl);
 
% CVNBMdl=crossval(NBMdl);
 
 %Class Error--------------------------------------------------------------
 
 classError_KNN = kfoldLoss(CVKNNMdl);
 
 classError_SVM = kfoldLoss(CVMdl);
 
% classError_NB = kfoldLoss(CVNBMdl);

 %Label Predict------------------------------------------------------------
 
Label = predict(KNNMdl,test);

Labelsvm = predict(SVMMdl,test);

%LabelNB = predict(NBMdl,test);

LabelTree = predict(Mdltree,test);

%One Hot-------------------------------------------------------------------

one_hot_labels1 = zeros(4,20);
one_hot_ltest1 = zeros(4,20);
one_hot_labels2 = zeros(4,20);
one_hot_ltest2 = zeros(4,20);
one_hot_labels3 = zeros(4,20);
one_hot_ltest3 = zeros(4,20);
one_hot_labels4 = zeros(4,20);
one_hot_ltest4 = zeros(4,20);

for i = 1:20
     one_hot_labels1(Label(i,1),i) = 1;
     one_hot_ltest1(ltest(i,1),i) = 1;
     one_hot_labels2(Labelsvm(i,1),i) = 1;
     one_hot_ltest2(ltest(i,1),i) = 1;
%     one_hot_labels3(LabelNB(i,1),i) = 1;
     one_hot_ltest3(ltest(i,1),i) = 1;
    
     one_hot_labels4(LabelTree(i,1),i) = 1;
     one_hot_ltest4(ltest(i,1),i) = 1;

end

 figure
 plotconfusion(one_hot_ltest1,one_hot_labels1,'KNN');
 figure
 plotconfusion(one_hot_ltest2,one_hot_labels2,'SVM');
 %figure
% plotconfusion(one_hot_ltest3,one_hot_labels3);
 figure
 plotconfusion(one_hot_ltest4,one_hot_labels4,'Tree');
 

 figure
 plotroc(one_hot_ltest1, one_hot_labels1,'KNN');
 figure
 plotroc(one_hot_ltest2, one_hot_labels2,'SVM');
 %figure
% plotroc(one_hot_ltest3, one_hot_labels3);
 figure
 plotroc(one_hot_ltest4, one_hot_labels4,'Tree');