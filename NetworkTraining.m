%load('full_network1.mat', 'net'); %Load the saved neural network
imageDir = fullfile('C:\Users\Joel\Desktop\SPRING 2020\EE 496\Neural Network Training (Human)\Augmented_imagesCombinedResized'); %Location of training images
labelDir = fullfile('C:\Users\Joel\Desktop\SPRING 2020\EE 496\Neural Network Training (Human)\Augmented_trainingCombinedResized'); %Location of labeled training images
imds = imageDatastore(imageDir); %Function should be given the training images variable above
pixelLabelIds = [1, 2]; %Label 1 is Background, label 2 is Embryo
classNames = ["Background","Embryo"]; %Labels used
pxds = pixelLabelDatastore(labelDir,classNames,pixelLabelIds);
imageSize = [496 496]; %Must be a multiple of 2^D; make sure it matches your images/labels
numClasses = 2; %Embryo, background
lgraph = unetLayers(imageSize, numClasses, 'EncoderDepth', 4);
options = trainingOptions('adam','InitialLearnRate',1e-4,'MaxEpochs', 20, ...
   'MiniBatchSize', 2, 'Shuffle','every-epoch','CheckpointPath','C:\Users\Joel\Desktop\SPRING 2020\EE 496\Checkpoint', ...
    'Plots','training-progress','ExecutionEnvironment','parallel');
ds = pixelLabelImageDatastore(imds,pxds);
[net,info] = trainNetwork(ds,lgraph,options); 
humanEmbryoNetwork = net;
save humanEmbryoNetwork; %Save network
gpuDevice(1); %Resets GPU CUDA device to free up memory

imageDir = fullfile('C:\Users\Joel\Desktop\SPRING 2020\EE 496\Neural Network Training (Human)\training_images'); %Location of testing images
labelDir = fullfile('C:\Users\Joel\Desktop\SPRING 2020\EE 496\Neural Network Training (Human)\training_labels'); %Location of testing labeled images
imds = imageDatastore(imageDir);  %Function should be given the training images variable above
pxdsResults = semanticseg(imds, net, 'MiniBatchSize', 2, 'Verbose', false); 
    %Calculate the predicted embryo area; try and match MiniBatchSize to the training
pxdsTruth = pixelLabelDatastore(labelDir,classNames,pixelLabelIds); %Store the labeled data info
ssm = evaluateSemanticSegmentation(pxdsResults,pxdsTruth); %Calculate metrics