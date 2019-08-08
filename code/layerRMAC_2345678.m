classdef layerRMAC_2345678
    
    properties
        type= 'custom'
        name= 'RMAC_2345678'
        precious= false
    end
    
    methods
        
        function l= layerRMAC_2345678(name)
            if nargin>0, l.name= name; end
        end
        
    end
    
    methods (Static)
        
        function res1= forward(l, res0, res1)
            resx1 = SPPPool(res0.x,2,true);
            resx2 = SPPPool(res0.x,3,true);
            resx3 = SPPPool(res0.x,4,true);
            resx4 = SPPPool(res0.x,5,true);
            resx5 = SPPPool(res0.x,6,true);
            resx6 = SPPPool(res0.x,7,true);
            resx7 = SPPPool(res0.x,8,true);

            resx1 = reshape(resx1,[4,size(resx1,3),size(resx1,4)]); %1*x*256
            resx2 = reshape(resx2,[9,size(resx2,3),size(resx2,4)]);
            resx3 = reshape(resx3,[16,size(resx3,3),size(resx3,4)]);
            resx4 = reshape(resx4,[25,size(resx4,3),size(resx4,4)]);
            resx5 = reshape(resx5,[36,size(resx5,3),size(resx5,4)]); %1*x*256
            resx6 = reshape(resx6,[49,size(resx6,3),size(resx6,4)]);
            resx7 = reshape(resx7,[64,size(resx7,3),size(resx7,4)]);
           
            res1.x = cat(1,resx1,resx2,resx3,resx4,resx5,resx6,resx7);
            res1.x = reshape(res1.x ,[1,203,size(res1.x,2),size(res1.x,3)]);
        end

        function res0= backward(l, res0, res1)
            
            dy = reshape(res1.dzdx ,[size(res1.dzdx,1)*size(res1.dzdx,2),size(res1.dzdx,3),size(res1.dzdx,4)]);
            dy1 = dy(1:4,:,:);
            dy2 = dy(5:13,:,:);
            dy3 = dy(14:29,:,:);
            dy4 = dy(30:54,:,:);
            dy5 = dy(55:90,:,:);
            dy6 = dy(91:139,:,:);
            dy7 = dy(140:end,:,:);

            dzdy1 = reshape(dy1,[2,2,size(dy1,2),size(dy1,3)]);
            dzdy2 = reshape(dy2,[3,3,size(dy2,2),size(dy1,3)]);
            dzdy3 = reshape(dy3,[4,4,size(dy3,2),size(dy1,3)]);
            dzdy4 = reshape(dy4,[5,5,size(dy4,2),size(dy1,3)]);
            dzdy5 = reshape(dy5,[6,6,size(dy5,2),size(dy1,3)]);
            dzdy6 = reshape(dy6,[7,7,size(dy6,2),size(dy1,3)]);
            dzdy7 = reshape(dy7,[8,8,size(dy7,2),size(dy1,3)]);
            
            dzdx1 = SPPPool(res0.x,2,true,dzdy1);
            dzdx2 = SPPPool(res0.x,3,true,dzdy2);
            dzdx3 = SPPPool(res0.x,4,true,dzdy3);
            dzdx4 = SPPPool(res0.x,5,true,dzdy4);
            dzdx5 = SPPPool(res0.x,6,true,dzdy5);
            dzdx6 = SPPPool(res0.x,7,true,dzdy6);
            dzdx7 = SPPPool(res0.x,8,true,dzdy7);

            res0.dzdx = dzdx1+dzdx2+dzdx3+dzdx4+dzdx5+dzdx6+dzdx7;
        end

    end
    
end
