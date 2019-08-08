classdef layerL2Attention
    %single attention block in the APANet paper
    properties
        type= 'custom'
        name= 'L2attention'
        weights
        param 
        momentum
        learningRate = [10,10];
        weightDecay
        precious= false
    end
    
    methods
        
        function l= layerL2Attention(name)
            if nargin>0, l.name= name; end
        end
 
        function l= constructor(l, weights,D)
            l.weights= {weights{1}, weights{2}};
            l.param = [2*D, 1e-12, 1, 0.5];
        end
    end
    
    
    methods (Static)
        
        function res1= forward(l, res0, res1)
            norm = vl_nnnormalize(res0.x, l.param) ;
            mask1= vl_nnconv(norm, l.weights{1},l.weights{2});
            %mask1 = vl_nnrelu(mask1);
            mask = vl_nntanh(mask1);
            res1.x = bsxfun(@times, res0.x, mask);
        end
        
        
        function res0= backward(l, res0, res1)
            norm = vl_nnnormalize(res0.x, l.param) ;
            mask1= vl_nnconv(norm, l.weights{1},l.weights{2});
            mask = vl_nntanh(mask1);  %forward again
            dzdy= bsxfun(@times, res1.dzdx, res0.x);  %28*38*256*12
            dzdx= bsxfun(@times, res1.dzdx, mask);
            dzdy = sum(dzdy,3);  
           
            dzdy = vl_nntanh(mask1,dzdy);
            [res0.dzdx, res0.dzdw{1}, res0.dzdw{2}] = vl_nnconv(norm, l.weights{1}, l.weights{2}, dzdy);  %dzdy should be H-W-1
         
            res0.dzdx = vl_nnnormalize(res0.x, l.param, res0.dzdx) ;
            res0.dzdx = res0.dzdx + dzdx;
        end
        
        
    end
    
end