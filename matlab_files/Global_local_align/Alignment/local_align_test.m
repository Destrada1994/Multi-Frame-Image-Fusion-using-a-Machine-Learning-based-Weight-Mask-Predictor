function [imgs_out,A]=local_align_test(imgs,box_size,stride)
H=size(imgs,1);
W=size(imgs,2);
C=size(imgs,3);
N=size(imgs,4);

center_size=box_size-(2*stride);

A=zeros(H,W,2,N);
imgs_out=zeros(H,W,3,N);

for y1=1:stride:(H-box_size)
    for x1=1:stride:(H-box_size)
        imgs_crop1=imgs(y1:y1+box_size-1,x1:x1+box_size-1,:,:);
        [imgs_align, Tvec1]=RegisterImageSeq(imgs_crop1);
        for n=1:N
            shift1=Tvec1{n}.T;
            xshift1=shift1(3,2);
            yshift1=shift1(3,1);

            if x1==1
                X_corr=x1:x1+stride+center_size-1;
                X_corr2=1:stride+center_size;
            else
                X_corr=x1+stride:x1+stride+center_size-1;
                X_corr2=stride+1:stride+center_size;
            end
            if y1==1
                Y_corr=y1:y1+stride+center_size-1;
                Y_corr2=1:stride+center_size;
            else
                Y_corr=y1+stride:y1+stride+center_size-1;
                Y_corr2=stride+1:stride+center_size;
            end

            imgs_out(Y_corr,X_corr,:,n)=imgs_align(Y_corr2,X_corr2,:,n);
            A(Y_corr,X_corr,1,n)=xshift1;
            A(Y_corr,X_corr,2,n)=yshift1;
        end
    end
    imgs_crop1=imgs(y1:y1+box_size-1,W-box_size:W,:,:);
    [imgs_align, Tvec1]=RegisterImageSeq(imgs_crop1);
    for n=1:N
        shift1=Tvec1{n}.T;
        xshift1=shift1(3,2);
        yshift1=shift1(3,1);

        X_corr=W-box_size+stride+1:W;
        X_corr2=1+stride:(2*stride)+center_size;

        if y1==1
            Y_corr=y1:y1+stride+center_size-1;
            Y_corr2=1:stride+center_size;
        else
            Y_corr=y1+stride:y1+stride+center_size-1;
            Y_corr2=stride+1:stride+center_size;
        end
        imgs_out(Y_corr,X_corr,:,n)=imgs_align(Y_corr2,X_corr2,:,n);
        A(Y_corr,X_corr,1,n)=xshift1;
        A(Y_corr,X_corr,2,n)=yshift1;
    end
end


for x1=1:stride:(H-box_size)
    imgs_crop1=imgs(H-box_size:H,x1:x1+box_size-1,:,:);
    [imgs_align, Tvec1]=RegisterImageSeq(imgs_crop1);
    for n=1:N
        shift1=Tvec1{n}.T;
        xshift1=shift1(3,2);
        yshift1=shift1(3,1);

        if x1==1
            X_corr=x1:x1+stride+center_size-1;
            X_corr2=1:stride+center_size;
        else
            X_corr=x1+stride:x1+stride+center_size-1;
            X_corr2=stride+1:stride+center_size;
        end

        Y_corr=H-box_size+stride+1:H;
        Y_corr2=1+stride:(2*stride)+center_size;


        imgs_out(Y_corr,X_corr,:,n)=imgs_align(Y_corr2,X_corr2,:,n);
        A(Y_corr,X_corr,1,n)=xshift1;
        A(Y_corr,X_corr,2,n)=yshift1;
    end
end

imgs_crop1=imgs(H-box_size:H,W-box_size:W,:,:);
[imgs_align, Tvec1]=RegisterImageSeq(imgs_crop1);
for n=1:N
    shift1=Tvec1{n}.T;
    xshift1=shift1(3,2);
    yshift1=shift1(3,1);

    X_corr=W-box_size+stride+1:W;
    X_corr2=1+stride:(2*stride)+center_size;
    Y_corr=H-box_size+stride+1:H;
    Y_corr2=1+stride:(2*stride)+center_size;


    imgs_out(Y_corr,X_corr,:,n)=imgs_align(Y_corr2,X_corr2,:,n);
    A(Y_corr,X_corr,1,n)=xshift1;
    A(Y_corr,X_corr,2,n)=yshift1;
end



end