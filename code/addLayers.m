function net= addLayers(net, opts, dbTrain)
    
    
    
    methodOpts= strsplit(opts.method, '_');
    [~, sz]= relja_netOutputDim(net);
    D= sz(3);


    if ismember('RMACcaffe246', methodOpts)
        methodOpts= removeOpt(methodOpts, 'RMACcaffe246');
        net.layers{end+1}= layerRMAC_caffe246('RMAC_caffe246');
    elseif ismember('RMACcaffe2468', methodOpts)
        methodOpts= removeOpt(methodOpts, 'RMACcaffe2468');
        net.layers{end+1}= layerRMAC_caffe2468('RMAC_caffe2468');
    elseif ismember('RMACcaffe123', methodOpts)
        methodOpts= removeOpt(methodOpts, 'RMACcaffe123');
        net.layers{end+1}= layerRMAC_caffe123('RMAC_caffe123');
    elseif ismember('RMACcaffe1234', methodOpts)
        methodOpts= removeOpt(methodOpts, 'RMACcaffe1234');
        net.layers{end+1}= layerRMAC_caffe1236('RMAC_caffe1234');
    elseif ismember('RMACcaffe', methodOpts)
        methodOpts= removeOpt(methodOpts, 'RMACcaffe');
        net.layers{end+1}= layerRMAC_caffe('RMAC_caffe');
    elseif ismember('RMACvd123', methodOpts)
        methodOpts= removeOpt(methodOpts, 'RMACvd123');
        net.layers{end+1}= layerRMAC_vd123('RMAC_vd123');
    elseif ismember('RMACvd2468', methodOpts)
        methodOpts= removeOpt(methodOpts, 'RMACvd2468');
        net.layers{end+1}= layerRMAC_vd2468('RMAC_vd2468');
    end
    %% prel2
    if ismember('preL2', methodOpts)
        % normalize feature-wise
        net.layers{end+1}= struct('type', 'normalize', 'name', 'preL2', ...
            'param', [2*D, 1e-12, 1, 0.5], 'precious', 0);
        methodOpts= removeOpt(methodOpts, 'preL2');
        doPreL2= true;
    else
        doPreL2= false;
    end
    %% aggregation
    if ismember('max', methodOpts)
        methodOpts= removeOpt(methodOpts, 'max');
        net.layers{end+1}= layerTotalMaxPool('max:core');
    elseif ismember('avg', methodOpts)
        methodOpts= removeOpt(methodOpts, 'avg');
        net.layers{end+1}= layerTotalAvgPool('avg:core');

    elseif ismember('l2wavg', methodOpts)
        methodOpts= removeOpt(methodOpts, 'l2wavg');
        net.layers{end+1}= layerL2Attention('l2attention');
        sc1 = sqrt(3/(1*1*(1+D)/2));  %identify the scale,numin+out
        weights{1} = (rand(1,1,D,1,'single')*2-1)*sc1;
        weights{2}= single(zeros(1,1));
        net.layers{end}= net.layers{end}.constructor({weights{1}, weights{2}},D);
        net.layers{end+1}= layerTotalAvgPool('avg:core');
    elseif ismember('l2nanwavg', methodOpts)
        methodOpts= removeOpt(methodOpts, 'l2nanwavg');
        net.layers{end+1}= layerL2nanAttention('l2nanattention');
        sc1 = sqrt(3/(1*1*(1+D)/2));  %identify the scale,numin+out
        weights{1} = (rand(1,1,D,1,'single')*2-1)*sc1;
        weights{2}= single(zeros(1,1));
        sc2 = sqrt(3/(1*1*(D+D)/2));
        weights{3} = (rand(1,1,D,D,'single')*2-1)*sc2;
        weights{4}= single(zeros(1,D));
        net.layers{end}= net.layers{end}.constructor({weights{1}, weights{2},weights{3}, weights{4}},D);
        net.layers{end+1}= layerTotalAvgPool('avg:core');

    elseif any( ismember( {'vlad', 'vladv2'}, methodOpts) )
        
        if doPreL2
            L2str= '_preL2';
        else
            L2str= '';
        end
        
        whichDesc= sprintf('%s_%s%s', opts.netID, opts.layerName, L2str);
        
        k= 64;
        paths= localPaths();
        trainDescFn= sprintf('%s%s_%s_traindescs.mat', paths.initData, dbTrain.name, whichDesc);
        clstFn= sprintf('%s%s_%s_k%03d_clst.mat', paths.initData, dbTrain.name, whichDesc, k);
        
        clsts= getClusters(net, opts, clstFn, k, dbTrain, trainDescFn);
        
        load( trainDescFn, 'trainDescs');
        load( clstFn, 'clsts');
        net.meta.sessionID= sprintf('%s_%s', net.meta.sessionID, dbTrain.name);
   
        % --- VLAD layer
        
        if ismember('vladv2', methodOpts)
            methodOpts= removeOpt(methodOpts, 'vladv2');
            
            % set alpha for sparsity
            [~, dsSq]= yael_nn(clsts, trainDescs, 2); clear trainDescs;
            alpha= -log(0.01)/mean( dsSq(2,:)-dsSq(1,:) ); clear dsSq;
            
            net.layers{end+1}= layerVLADv2('vlad:core');
            net.layers{end}= net.layers{end}.constructor({alpha*2*clsts, -alpha*sum(clsts.^2,1), -clsts});
            
        elseif ismember('vlad', methodOpts)
            % see comments on vladv2 vs vlad in the README_more.md
            methodOpts= removeOpt(methodOpts, 'vlad');
            % set alpha for sparsity
            clstsAssign= relja_l2normalize_col(clsts);
            dots= sort(clstsAssign'*trainDescs, 1, 'descend'); clear trainDescs;  
            alpha= -log(0.01)/mean( dots(1,:) - dots(2,:) ); clear dots;
            
            net.layers{end+1}= layerVLAD('vlad:core');
            net.layers{end}= net.layers{end}.constructor({alpha*clstsAssign, clsts});    
            
        %else
            %error('Unsupported method "%s"', opts.method);
        end
        
        if ismember('intra', methodOpts)
            % --- intra-normalization
            net.layers{end+1}= struct('type', 'normalize', 'name', 'vlad:intranorm', ...
                'param', [2*D, 1e-12, 1, 0.5], 'precious', 0);
            methodOpts= removeOpt(methodOpts, 'intra');
        end
        
    %else
        %error('Unsupported method "%s"', opts.method);
    end
   
    % --- final normalization
    net.layers{end+1}= layerWholeL2Normalize('postL2');
    
    
    
    % --- check if all options are used
    if ~isempty(methodOpts)
        error('Unsupported options (method=%s): %s', opts.method, strjoin(methodOpts, ', '));
    end
    
    net.meta.sessionID= sprintf('%s_%s', net.meta.sessionID, opts.method);
    net.meta.epoch= 0;
    
end



function opts= removeOpt(opts, optName)
    opts(ismember(opts, optName))= [];
end
