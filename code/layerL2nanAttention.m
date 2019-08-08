classdef layerL2nanAttention
    %Cascaded attention block
    properties
        type= 'custom'
        name= 'L2nanattention'
        weights
        param 
        momentum
        learningRate = [10,10,10,10];
        weightDecay
        precious= false
    end
    
    methods
        
        function l= layerL2nanAttention(name)
            if nargin>0, l.name= name; end
        end
     
        function l= constructor(l, weights,D)
            l.weights= {weights{1}, weights{2},weights{3}, weights{4}};
            l.param = [2*D, 1e-12, 1, 0.5];
        end
    end
    
    
    methods (Static)
        
        function res1= forward(l, res0, res1)
            norm = vl_nnnormalize(res0.x, l.param) ;
            mask1= vl_nnconv(norm, l.weights{1},l.weights{2});
            res1x = bsxfun(@times, res0.x, mask1);
            res2x= vl_nnpool(res1x, [size(res1x,1), size(res1x,2)], ...
                'method', 'avg');
            res3x = vl_nnnormalize(res2x, l.param) ;
            para1= vl_nnconv(res3x, l.weights{3},l.weights{4});
            para = vl_nntanh(para1);
            
            mask2 = bsxfun(@times,norm,para);
            mask3 = sum(mask2,3);
            res1.x = bsxfun(@times, res0.x, mask3);
        end
        
        
        function res0= backward(l, res0, res1)
            norm = vl_nnnormalize(res0.x, l.param) ;
            mask1= vl_nnconv(norm, l.weights{1},l.weights{2});
            res1x = bsxfun(@times, res0.x, mask1);
            res2x= vl_nnpool(res1x, [size(res1x,1), size(res1x,2)], ...
                'method', 'avg');
            res3x = vl_nnnormalize(res2x, l.param) ;
            para1= vl_nnconv(res3x, l.weights{3},l.weights{4});
            para = vl_nntanh(para1);
            mask2 = bsxfun(@times,norm,para);  
            mask3 = sum(mask2,3);  %forward first
   
            dzdy1= bsxfun(@times, res1.dzdx, res0.x);  
            dzdx= bsxfun(@times, res1.dzdx, mask3);
         
            dzdy1 = sum(dzdy1,3); %1*200*1
            dzdx1 = bsxfun(@times,dzdy1,para);
            dzdy2 = bsxfun(@times,dzdy1,norm);
            dzdy2 = sum(dzdy2,2);
            dzdy2 = sum(dzdy2,1);
           
            dzdy2 = vl_nntanh(para1,dzdy2);
            [res0dx1, res0.dzdw{3}, res0.dzdw{4}] = vl_nnconv(res3x, l.weights{3}, l.weights{4}, dzdy2);  %dzdy should be H-W-1
          
            res0dx1 = vl_nnnormalize(res2x,l.param,res0dx1);
            res0dx1 = vl_nnpool(res1x, [size(res1x,1), size(res1x,2)],res0dx1, ...
                'method', 'avg'); 
            
            dzdx3 = bsxfun(@times,res0dx1,mask1);
            res0dx2 = bsxfun(@times,res0dx1,res0.x);
            res0dx2 = sum(res0dx2,3);
            [res0dx2, res0.dzdw{1}, res0.dzdw{2}] = vl_nnconv(norm, l.weights{1}, l.weights{2},res0dx2);
            dzdx1 = dzdx1+res0dx2;
            res0.dzdx = vl_nnnormalize(res0.x, l.param, dzdx1) ;
            res0.dzdx = res0.dzdx + dzdx+dzdx3;
        end
        
        
    end
    
end