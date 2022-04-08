function A1=local_align2(imgs,box_size,xb,yb)
H=box_size*yb;
W=box_size*xb;
C=size(imgs,3);
N=size(imgs,4);

box_size1=box_size;
box_size2=box_size1/2;

A1=zeros(H,W,2,N);
A2=zeros(box_size1,box_size1,3);

xLimits = [1 box_size1];
yLimits = [1 box_size1];
AlignedView = imref2d([box_size1 box_size1], xLimits, yLimits);

for y1=1:box_size1:H
    for x1=1:box_size1:W
        imgs_crop1=imgs(y1:y1+box_size1-1,x1:x1+box_size1-1,:,:);
        [img_align, Tvec1]=RegisterDFT(imgs_crop1,1);
        for y2=1:box_size2:box_size1
            for x2=1:box_size2:box_size1
                imgs_crop2=img_align(y2:y2+box_size2-1,x2:x2+box_size2-1,:,:);
                [~, Tvec2]=RegisterDFT(imgs_crop2,4);
                for n=1:N
                    shift2=Tvec2{n}.T;
                    xshift2=shift2(3,2);
                    yshift2=shift2(3,1);
                    A2(y2:y2+box_size2-1,x2:x2+box_size2-1,1,n)=xshift2;
                    A2(y2:y2+box_size2-1,x2:x2+box_size2-1,2,n)=yshift2;
                end
            end
        end
        for n=1:N
            inv_shift1=Tvec1{n}.invert;
            A2_warped = imwarp(A2(:,:,:,n), inv_shift1, 'cubic', 'FillValues', 0,'OutputView', AlignedView);
            
            shift1=Tvec1{n}.T;
            xshift1=shift1(3,2);
            yshift1=shift1(3,1);
            
            A1(y1:y1-1+box_size1,x1:x1-1+box_size1,1,n)=A2_warped(:,:,1)+xshift1;
            A1(y1:y1-1+box_size1,x1:x1-1+box_size1,2,n)=A2_warped(:,:,2)+yshift1;
        end
    end
end
end