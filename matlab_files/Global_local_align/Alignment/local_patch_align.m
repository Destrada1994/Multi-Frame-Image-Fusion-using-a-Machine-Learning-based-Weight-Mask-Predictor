function [stack_out,Tvec]=local_patch_align(stack,bfn,stride)
H=size(stack,1);
W=size(stack,2);
C=size(stack,3);
N=size(stack,4);

box_size=3*stride;

Tvec=zeros(H,W,2,N);
stack_out=zeros(H,W,C,N);

for x=1:stride:(W-box_size)
    patch=stack(H-box_size+1:H,x:x+box_size-1,:,:);
    [~,patch_Tvec]=Register_DFT(patch,bfn,1);
    patch_align=UpSample_Shift(patch,patch_Tvec,1,1);
    for n=1:N
        shift=patch_Tvec{n}.T;
        xshift=shift(3,2);
        yshift=shift(3,1);

        if x==1
            X_corr=1:stride*2;
            X_corr2=1:stride*2;
        else
            X_corr=x+stride:x+stride*2-1;
            X_corr2=stride+1:stride*2;
        end

        Y_corr=H-box_size+stride+1:H;
        Y_corr2=1+stride:box_size;


        stack_out(Y_corr,X_corr,:,n)=patch_align(Y_corr2,X_corr2,:,n);
        Tvec(Y_corr,X_corr,1,n)=xshift;
        Tvec(Y_corr,X_corr,2,n)=yshift;
    end

end

patch=stack(H-box_size+1:H,W-box_size+1:W,:,:);
[~, patch_Tvec]=Register_DFT(patch,bfn,1);
patch_align=UpSample_Shift(patch,patch_Tvec,1,1);
for n=1:N
    shift=patch_Tvec{n}.T;
    xshift=shift(3,2);
    yshift=shift(3,1);

    X_corr=W-box_size+stride+1:W;
    X_corr2=1+stride:box_size;
    Y_corr=H-box_size+stride+1:H;
    Y_corr2=1+stride:box_size;


    stack_out(Y_corr,X_corr,:,n)=patch_align(Y_corr2,X_corr2,:,n);
    Tvec(Y_corr,X_corr,1,n)=xshift;
    Tvec(Y_corr,X_corr,2,n)=yshift;
end

for y=1:stride:H-box_size
    patch=stack(y:y+box_size-1,W-box_size+1:W,:,:);
    [~,patch_Tvec]=Register_DFT(patch,bfn,1);
    patch_align=UpSample_Shift(patch,patch_Tvec,1,1);
    for n=1:N
        shift=patch_Tvec{n}.T;
        xshift=shift(3,2);
        yshift=shift(3,1);

        X_corr=W-box_size+stride+1:W;
        X_corr2=1+stride:box_size;

        if y==1
            Y_corr=1:stride*2;
            Y_corr2=1:stride*2;
        else
            Y_corr=y+stride:y+(stride*2)-1;
            Y_corr2=stride+1:stride*2;
        end
        stack_out(Y_corr,X_corr,:,n)=patch_align(Y_corr2,X_corr2,:,n);
        Tvec(Y_corr,X_corr,1,n)=xshift;
        Tvec(Y_corr,X_corr,2,n)=yshift;
    end
end

for y=1:stride:H-box_size
    for x=1:stride:W-box_size
        patch=stack(y:y-1+box_size,x:x-1+box_size,:,:);
        [~,patch_Tvec]=Register_DFT(patch,bfn,1);
        patch_align=UpSample_Shift(patch,patch_Tvec,1,1);
        for n=1:N
            shift=patch_Tvec{n}.T;
            xshift=shift(3,2);
            yshift=shift(3,1);

            if x==1
                X_corr=1:stride*2;
                X_corr2=1:stride*2;
            else
                X_corr=x+stride:x+(stride*2)-1;
                X_corr2=stride+1:stride*2;
            end
            if y==1
                Y_corr=1:stride*2;
                Y_corr2=1:stride*2;
            else
                Y_corr=y+stride:y+(stride*2)-1;
                Y_corr2=stride+1:stride*2;
            end
            stack_out(Y_corr,X_corr,:,n)=patch_align(Y_corr2,X_corr2,:,n);
            Tvec(Y_corr,X_corr,1,n)=xshift;
            Tvec(Y_corr,X_corr,2,n)=yshift;
        end
    end
    
end






end