clear;
setup;
% Set up the train/val datasets
%dbTrain= dbTokyoTimeMachine('train');
%dbVal= dbTokyoTimeMachine('val');
doPitts250k= false;
dbTrain= dbPitts(doPitts250k, 'train');
dbVal= dbPitts(doPitts250k, 'val');
%dbTest= dbPitts(doPitts250k, 'test');
gpuDevice(8);%no 1
%
%
% --- Train the VGG-16 network + APA module, tuning down to conv5_1
%Trainweakly1:adopting the learning rate decay as described in APANet,For
%Pitts 30k-train 'learningRate' is 0.001 and Tokyo TM is 0.0005.
%Trainweakly1:adopting the learning rate decay strategy as described in APANet,
sessionID= trainWeakly1(dbTrain, dbVal, ...
    'netID', 'caffe', 'layerName', 'conv5', 'backPropToLayer', 'conv2', ...
    'method', 'RMACcaffe2468_l2nanwavg', ...
    'learningRate', 0.001, 'nEpoch', 2, ...
    'sessionID', 'pitts_caffe_conv5_RMACvd2468_l2nanwavg_1', 'startEpoch', 1 ,...
    'doDraw', true);

% Get the best network
% This can be done even if training is not finished, it will find the best network so far
[~, bestNet]= pickBestNet(sessionID);
% Either use the above network as the image representation extractor (do: finalNet= bestNet), or do whitening (recommended):
finalNet= addPCA(bestNet, dbTrain, 'doWhite', true, 'pcaDim', 256,'dowhite',11);%doWhite = 11 for PCA PW,1 for PCAW, else PCA



