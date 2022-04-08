function A=local_align(imgs,box_size,xb,yb)
H=box_size*yb;
W=box_size*xb;
C=size(imgs,3);
N=size(imgs,4);

A=zeros(H,W,2,N);

for y1=1:box_size:H
    for x1=1:box_size:W
        imgs_crop1=imgs(y1:y1+box_size-1,x1:x1+box_size-1,:,:);
        [~, Tvec1]=RegisterDFT(imgs_crop1,4);
        for n=1:N
            shift1=Tvec1{n}.T;
            xshift1=shift1(3,2);
            yshift1=shift1(3,1);
            A(y1:y1-1+box_size,x1:x1-1+box_size,1,n)=xshift1;
            A(y1:y1-1+box_size,x1:x1-1+box_size,2,n)=yshift1;
        end
    end
end
end