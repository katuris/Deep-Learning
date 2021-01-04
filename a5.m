digitDatasetPath = fullfile('Iris_Imgs')
categories={'setosa','versicolor','virginica'}
imds= imageDatastore(fullfile(digitDatasetPath,categories), ...
   'IncludeSubfolders',true,'LabelSource','foldernames');
numTrainFiles = 30;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');
img = readimage(imds,1);
size(img)
inputLayer=imageInputLayer([200,200,3])
filterSize= [6 6];         
numFilters= 32;
middleLayers=[
    convolution2dLayer(filterSize,numFilters,'Padding',2);
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(filterSize/ 2, numFilters, 'Padding', 2);
    batchNormalizationLayer
    reluLayer();
    maxPooling2dLayer(2, 'Stride',2);
    convolution2dLayer(filterSize/ 2,   2*numFilters,  'Padding', 2);
    batchNormalizationLayer
    reluLayer();
    maxPooling2dLayer(2, 'Stride',2);
    ]
finalLayers= [
    fullyConnectedLayer(64); 
    reluLayer;
    fullyConnectedLayer(3);  
    softmaxLayer
    classificationLayer
    ];
All_Layers= [  inputLayer;middleLayers;finalLayers  ];
%All_Layers(2).Weights = 0.0001 * randn([filterSize numChannels numFilters]);
opts= trainingOptions('sgdm', ...
    'ExecutionEnvironment', 'parallel', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 20, ...  
    'MiniBatchSize', 5,...
    'Plots','training-progress',...
    'Verbose', false);
Net= trainNetwork(imdsTrain, All_Layers, opts);
YPred = classify(Net, imdsValidation);  
YValidation = imdsValidation.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation)
plotconfusion(YPred,YValidation)
analyzeNetwork(Net)