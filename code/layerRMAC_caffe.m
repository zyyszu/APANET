classdef layerRMAC_caffe
    
    properties
        type= 'custom'
        name= 'RMAC_caffe'
        precious= false
    end
    
    methods
        
        function l= layerRMAC_caffe(name)
            if nargin>0, l.name= name; end
        end
        
    end
    
    methods (Static)
        
        function res1= forward(l, res0, res1)

             if size(res0.x,1) == 38 &&size(res0.x,2) ==28
                resx1= vl_nnpool(res0.x, [28, 28], ...
                    'method', 'max','stride',[10,1]);  %1*2
                resx2= vl_nnpool(res0.x, [19, 19], ...
                    'method', 'max','stride',[10,10],'pad',[1,0,1,0]);  %1*2
                resx3= vl_nnpool(res0.x, [14,14], ...
                    'method', 'max','stride',[8,7],'pad',[0,0,0,0]);  %3*3
             elseif size(res0.x,1) == 21 && size(res0.x,2) ==38
                resx1= vl_nnpool(res0.x, [21, 21], ...
                    'method', 'max','stride',[1,17]);  %1*2
                resx2= vl_nnpool(res0.x, [14, 14], ...
                    'method', 'max','stride',[7,12],'pad',[0,0,0,0]);  %1*2
                resx3= vl_nnpool(res0.x, [11,11], ...
                    'method', 'max','stride',[5,9],'pad',[0,0,0,0]); 
             elseif size(res0.x,1) == 38 && size(res0.x,2) ==21
                resx1= vl_nnpool(res0.x, [21, 21], ...
                    'method', 'max','stride',[17,1]);  %1*2
                resx2= vl_nnpool(res0.x, [14, 14], ...
                    'method', 'max','stride',[12,7],'pad',[0,0,0,0]);  %1*2
                resx3= vl_nnpool(res0.x, [11,11], ...
                    'method', 'max','stride',[9,5],'pad',[0,0,0,0]); 
             else
                resx1= vl_nnpool(res0.x, [28, 28], ...
                    'method', 'max','stride',[1,10]);  %1*2
                resx2= vl_nnpool(res0.x, [19, 19], ...
                    'method', 'max','stride',[10,10],'pad',[1,0,1,0]);  %1*2
                resx3= vl_nnpool(res0.x, [14,14], ...
                    'method', 'max','stride',[7,8],'pad',[0,0,0,0]);
             end

            resx1 = reshape(resx1,[2,size(resx1,3),size(resx1,4)]); %1*x*256
            resx2 = reshape(resx2,[6,size(resx2,3),size(resx2,4)]);
            resx3 = reshape(resx3,[12,size(resx3,3),size(resx3,4)]);
            res1.x = [resx1;resx2;resx3];
            res1.x = reshape(res1.x ,[1,20,size(res1.x,2),size(res1.x,3)]);
        end

        function res0= backward(l, res0, res1)
            dy = reshape(res1.dzdx ,[size(res1.dzdx,1)*size(res1.dzdx,2),size(res1.dzdx,3),size(res1.dzdx,4)]);
            dy1 = dy(1:2,:,:);
            dy2 = dy(3:8,:,:);
            dy3 = dy(9:20,:,:);
            
            dzdy1 = reshape(dy1,[1,2,size(dy1,2),size(dy1,3)]);
            dzdy2 = reshape(dy2,[2,3,size(dy2,2),size(dy1,3)]);
            dzdy3 = reshape(dy3,[3,4,size(dy3,2),size(dy1,3)]);

            dzdx1= vl_nnpool(res0.x, [28,28],dzdy1, ...
                'method', 'max','stride',[1,10],'pad',[0,0,0,0]); %1*2
            dzdx2= vl_nnpool(res0.x, [19,19],dzdy2, ...
                'method', 'max','stride',[10,10],'pad',[1,0,1,0]);  %1*2
            dzdx3= vl_nnpool(res0.x, [14,14], dzdy3,...
                'method', 'max','stride',[7,8],'pad',[0,0,0,0]);  %3*3

            res0.dzdx = dzdx1+dzdx2+dzdx3;
        end

    end
    
end