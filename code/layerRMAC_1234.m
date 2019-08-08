classdef layerRMAC_1234
    properties
        type= 'custom'
        name= 'RMAC_1234'
        precious= false
    end
    
    methods
        
        function l= layerRMAC_1234(name)
            if nargin>0, l.name= name; end
        end
        
    end
    
    methods (Static)
        
        function res1= forward(l, res0, res1)
            resx1 = SPPPool(res0.x,1,true);
            resx2 = SPPPool(res0.x,2,true);
            resx3 = SPPPool(res0.x,3,true);
            resx4 = SPPPool(res0.x,4,true);
            
            resx1 = reshape(resx1,[1,size(resx1,3),size(resx1,4)]); %1*x*256
            resx2 = reshape(resx2,[4,size(resx2,3),size(resx2,4)]);
            resx3 = reshape(resx3,[9,size(resx3,3),size(resx3,4)]);
            resx4 = reshape(resx4,[16,size(resx4,3),size(resx4,4)]);
            res1.x = cat(1,resx1,resx2,resx3,resx4);
            res1.x = reshape(res1.x ,[1,30,size(res1.x,2),size(res1.x,3)]);
        end

        function res0= backward(l, res0, res1)
            dy = reshape(res1.dzdx ,[size(res1.dzdx,1)*size(res1.dzdx,2),size(res1.dzdx,3),size(res1.dzdx,4)]);
            dy1 = dy(1,:,:);
            dy2 = dy(2:5,:,:);
            dy3 = dy(6:14,:,:);
            dy4 = dy(15:30,:,:);
            dzdy1 = reshape(dy1,[1,1,size(dy1,2),size(dy1,3)]);
            dzdy2 = reshape(dy2,[2,2,size(dy2,2),size(dy1,3)]);
            dzdy3 = reshape(dy3,[3,3,size(dy3,2),size(dy1,3)]);
            dzdy4 = reshape(dy4,[4,4,size(dy4,2),size(dy1,3)]);
            
            dzdx1 = SPPPool(res0.x,1,true,dzdy1);
            dzdx2 = SPPPool(res0.x,2,true,dzdy2);
            dzdx3 = SPPPool(res0.x,3,true,dzdy3);
            dzdx4 = SPPPool(res0.x,4,true,dzdy4);
            res0.dzdx = dzdx1+dzdx2+dzdx3+dzdx4;
        end

    end
    
end
