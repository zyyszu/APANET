function net= addPCA(net, dbTrain, varargin)
    opts= struct(...
        'pcaDim', 256, ...
        'doWhite', 11, ...
        'nTrainCap', 10000, ...
        'useGPU', true, ...
        'numThreads', 12, ...
        'batchSize', 10 ...
        );
    opts= vl_argparse(opts, varargin);
    
    if ~isfield(net.meta, 'sessionID') || ~isfield(net.meta, 'epoch')
        error('net should have a sessionID and epoch properties, I''m assuming you didn''t use the provided functions to create the networks? (loadNet, addLayers, pickBestNet)');
    end
    
    paths= localPaths();
    
    trainDbFeatFn= sprintf('%s%s__ep%d_traindescspitts10w_for_pca.bin', paths.outPrefix, net.meta.sessionID, net.meta.epoch);
    pcaFn= sprintf('%s%s__ep%d_pitts10wpca.mat', paths.outPrefix, net.meta.sessionID, net.meta.epoch);
    
    D= relja_netOutputDim(net);
    
    if ~exist(pcaFn, 'file')
        relja_displayDateTime();
        
        if ~exist(trainDbFeatFn, 'file')
            relja_display('%s Computing training vectors for PCA', net.meta.sessionID);
            
            imageFns= dbTrain.dbImageFns;
            nTrain= length(imageFns);
            
            if nTrain>opts.nTrainCap
                rng(43);
                imageFns= imageFns(randsample(nTrain, opts.nTrainCap));
            end
            
            serialAllFeats(net, dbTrain.dbPath, imageFns, trainDbFeatFn, ...
                'useGPU', opts.useGPU, 'numThreads', opts.numThreads, 'batchSize', opts.batchSize);
            clear nTrain;
        end
        
        relja_display('%s Computing PCA', net.meta.sessionID);
        
        dbFeat= fread( fopen(trainDbFeatFn, 'rb'), [D, inf], 'float32=>single');
        
        nTrain= size(dbFeat, 2);
        assert( opts.pcaDim < D+1 );
        
        if nTrain>opts.nTrainCap
            rng(43);
            dbFeat= dbFeat(:, randsample(nTrain, opts.nTrainCap));
        end
        
        [U, lams, mu, Utmu]= relja_PCA(dbFeat, opts.pcaDim);
        clear dbFeat;
        
        save(pcaFn, 'U', 'lams', 'mu', 'Utmu');
        clear U lams mu Utmu;
    end
    
    pcaParams= load(pcaFn, 'U', 'lams', 'mu', 'Utmu');
    assert(opts.pcaDim<=size(pcaParams.U, 2));
    U= pcaParams.U(:, 1:opts.pcaDim);
    lams= pcaParams.lams(1:opts.pcaDim);
    
    if opts.doWhite ==11
        U= U*diag(1./sqrt(sqrt(lams+1e-9)));   %lams variance
        pcaStr= 'WPCA';
    elseif opts.doWhite ==1
        U= U*diag(1./sqrt(sqrt(lams+1e-9).^0.5));   %PCA PW-alpha=0.5
        pcaStr= 'PWPCA'; 
    else
        pcaStr= 'PCA';
    end
    
    Utmu= U'*pcaParams.mu;
    U = single(U);
    Utmu= single(Utmu);
    Utmu = single(zeros(size(Utmu)));    %didn't perform zero centering
    
    % --- Add the PCA (+whitening) layer to the network:
    
    net.layers{end+1}= struct('type', 'conv', 'name', pcaStr, ...
        'weights', {{ reshape(U, [1,1,D,opts.pcaDim]), -Utmu }}, ...
        'stride', 1, 'pad', 0, 'opts', {{}}, 'precious', false);
    
    % final normalization
    net.layers{end+1}= layerWholeL2Normalize('finalL2');
    net.meta.sessionID= sprintf('%s_%s', net.meta.sessionID, pcaStr);
    
    % account for future changes in MatConvNet
    net = relja_simplenn_tidy(net);
end
