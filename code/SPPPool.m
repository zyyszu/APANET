function resx1= SPPPool(resx, num_bins,overlap,dzdy)
%Spatial pyramid pooling to get region features
%input:
% resx:CNN feature maps
% num_bins:size of spatial grid is num_bins*num_bins
% overlap = 1--- adjacent regions is 50 percent overlap, 0--- non overlap
%output:
%resx1£ºregion features with size of num_bins*num_bins
    if nargin < 4
        nback = 1;   %nback: backward or forward
    else
        nback = 0;
    end
    if overlap
            
        [kernel_h,kernel_w,pad_h1,pad_h2,pad_w1,pad_w2,stride_h,stride_w]= GetOPoolingParam(resx, num_bins);
       %% the code below is useful for image retrieval task where the image sizes are diverse
        %{
        if 7<= size(resx,1)&&size(resx,1) <=10&&num_bins ==7
            
            stride_h = 1;
            kernel_h = size(resx,1)-6;
            pad_h1 =0;
            pad_h2 = 0;
        elseif size(resx,1) ==6&&num_bins ==7
            stride_h = 1;
            kernel_h = 1;
            pad_h1 =0;
            pad_h2 = 0;
        end
        %
        if 7 < size(resx,1) && size(resx,1) <=12&&num_bins ==8
            
            stride_h = 1;
            kernel_h = size(resx,1)-7;
            pad_h1 =0;
            pad_h2 = 0;
        elseif  7 < size(resx,2) &&size(resx,2) <=12&&num_bins ==8
            stride_w = 1;
            kernel_w = size(resx,2)-7;
            pad_w1 =0;
            pad_w2 = 0;
        end
        if 5 < size(resx,1) &&size(resx,1) <=7&&num_bins ==8
            
            stride_h = 1;
            kernel_h = 3;
            pad_h1 =floor((10-size(resx,1))/2);
            pad_h2 = ceil((10-size(resx,1))/2);
        elseif 5 < size(resx,2) &&size(resx,2) <=7&&num_bins ==8
            stride_w = 1;
            kernel_w = 3;
            pad_w1 =floor((10-size(resx,2))/2);
            pad_w2 = ceil((10-size(resx,2))/2);
        elseif size(resx,1) ==5&&num_bins ==8
            
            stride_h = 1;
            kernel_h = 4;
            pad_h1 =floor((11-size(resx,1))/2);
            pad_h2 = ceil((11-size(resx,1))/2);
        
        elseif size(resx,2) ==5&&num_bins ==8
            stride_w = 1;
            kernel_w = 4;
            pad_w1 =floor((11-size(resx,2))/2);
            pad_w2 = ceil((11-size(resx,2))/2);
        end
        if size(resx,1) <=8&&num_bins ==6
            
            stride_h = 1;
            kernel_h = size(resx,1)-5;
            pad_h1 =0;
            pad_h2 = 0;
        elseif size(resx,2) <=8&&num_bins ==6
            stride_w = 1;
            kernel_w = size(resx,2)-5;
            pad_w1 =0;
            pad_w2 = 0;
        end
        if size(resx,1) <=5&&num_bins ==6
            
            stride_h = 1;
            kernel_h = 3;
            pad_h1 =floor((8-size(resx,1))/2);
            pad_h2 = ceil((8-size(resx,1))/2);
        elseif size(resx,2) <=5&&num_bins ==6
            stride_w = 1;
            kernel_w = 3;
            pad_w1 =floor((8-size(resx,2))/2);
            pad_w2 = ceil((8-size(resx,2))/2);
        end
        if size(resx,1) <=22&&num_bins ==9
            
            stride_h = 2;
            kernel_h = size(resx,1)-16;
            pad_h1 =0;
            pad_h2 = 0;
        elseif size(resx,2) <=22&&num_bins ==9
            stride_w = 2;
            kernel_w = size(resx,2)-16;
            pad_w1 = 0;
            pad_w2 = 0;
        end
       %}
    else
        [kernel_h,kernel_w,pad_h1,pad_h2,pad_w1,pad_w2,stride_h,stride_w]= GetPoolingParam(resx, num_bins) ;
    end
    if nback
        resx1= vl_nnpool(resx, [kernel_h,kernel_w], ...
                'method', 'max','stride',[stride_h,stride_w],'pad',[pad_h1,pad_h2,pad_w1,pad_w2]);  %1*2
    else
        resx1= vl_nnpool(resx, [kernel_h,kernel_w],dzdy, ...
                'method', 'max','stride',[stride_h,stride_w],'pad',[pad_h1,pad_h2,pad_w1,pad_w2]);  %1*2
    end
end


function [kernel_h,kernel_w,pad_h1,pad_h2,pad_w1,pad_w2,stride_h,stride_w]= GetOPoolingParam(resx, num_bins)

    H = size(resx,1);
    W = size(resx,2);
    kernel_h = ceil(2*H/(num_bins+1));
    kernel_w = ceil(2*W/(num_bins+1));
    stride_h = ceil(H/(num_bins+1));
    stride_w = ceil(W/(num_bins+1));
    remainder_h = kernel_h + (num_bins-1)*stride_h - H;
    pad_h1 = floor(remainder_h/2);
    pad_h2 = ceil(remainder_h/2);
    remainder_w = kernel_w + (num_bins-1)*stride_w - W;
    pad_w1 = floor(remainder_w/2)  ;
    pad_w2 = ceil(remainder_w/2)  ;    
end

function [kernel_h,kernel_w,pad_h1,pad_h2,pad_w1,pad_w2,stride_h,stride_w]= GetPoolingParam(resx, num_bins)
    H = size(resx,1);
    W = size(resx,2);
    kernel_h = ceil(H/num_bins);
    kernel_w = ceil(W/num_bins);
    stride_h = kernel_h;
    stride_w = kernel_w;
    remainder_h = kernel_h*num_bins - H;
    pad_h1 = floor(remainder_h/2);
    pad_h2 = ceil(remainder_h/2);
    remainder_w = kernel_w*num_bins - W;
    pad_w1 = floor(remainder_w/2)  ;
    pad_w2 = ceil(remainder_w/2)  ;      
end