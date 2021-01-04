digitDatasetPath = fullfile('Iris_Imgs');
categories={'setosa','versicolor','virginica'};
imds= imageDatastore(fullfile(digitDatasetPath,categories), ...
   'IncludeSubfolders',true,'LabelSource','foldernames');
countEachLabel(imds)
setosa = find(imds.Labels == 'setosa', 1);
versicolor = find(imds.Labels == 'versicolor', 1);
virginica = find(imds.Labels == 'virginica', 1);
convnet = alexnet;
layers = convnet.Layers(1:end-9)
%
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomize');
numTrainImages = numel(imdsTrain.Labels);
inputSize = convnet.Layers(1).InputSize
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
finalLayers= [
    fullyConnectedLayer(500); 
    reluLayer
    fullyConnectedLayer(200); 
    reluLayer
    fullyConnectedLayer(3);  
    softmaxLayer
    classificationLayer
    ];
All_Layers= [layers ; finalLayers  ];
opts= trainingOptions('sgdm', ...
    'ExecutionEnvironment', 'parallel', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 40, ...  
    'MiniBatchSize', 75,...
    'Plots','training-progress',...
    'Verbose', false);
Net= trainNetwork(augimdsTrain, All_Layers, opts);
YPred = classify(Net,augimdsValidation);  
YValidation = imdsValidation.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation)
%plotconfusion(YPred,YValidation)
cm_test= confusionmat(YPred,YValidation);
disp(cm_test)
num_labels=3
for i = 1:num_labels
    TP = cm_test(i);
    FP = sum(cm_test(:, i), 1) - TP;
    FN = sum(cm_test(i, :), 2) - TP;
    TN = sum(cm_test(:)) - TP - FP - FN;

    Accuracy = (TP+TN)./(TP+FP+TN+FN);

    TPR = TP./(TP + FN);%tp/actual positive  RECALL SENSITIVITY
    if isnan(TPR)
        TPR = 0;
    end
    TPR
    PPV = TP./ (TP + FP); % tp / predicted positive PRECISION
    if isnan(PPV)
        PPV = 0;
    end
    PPV
    TNR = TN./ (TN+FP); %tn/ actual negative  SPECIFICITY
    if isnan(TNR)
        TNR = 0;
    end
    TNR
    FPR = FP./ (TN+FP);
    if isnan(FPR)
        FPR = 0;
    end
    FPR
    FScore = (2*(PPV * TPR)) / (PPV+TPR);

    if isnan(FScore)
        FScore = 0;
    end
    FScore 
    
end
analyzeNetwork(Net)
trainLabels = imdsTrain.Labels
testLabels = imdsValidation.Labels
[predictedLabels,scores]=classify(Net,augimdsValidation);
[X,Y,T,AUC] = perfcurve(testLabels ,abs(scores(:,1)),'setosa');
figure(1)
plot(X,Y,'LineWidth',3)
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC for Classification Setosa');
grid on
[X,Y,T,AUC] = perfcurve(testLabels ,abs(scores(:,2)),'versicolor');
figure(2)
plot(X,Y,'LineWidth',3)
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC for Classification for versicolor ');
grid on
[X,Y,T,AUC] = perfcurve(testLabels ,abs(scores(:,3)),'virginica');
figure(3)
plot(X,Y,'LineWidth',3)
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC for Classification for virginica ');
grid on
figure(4)
plotconfusion(YPred,YValidation)
