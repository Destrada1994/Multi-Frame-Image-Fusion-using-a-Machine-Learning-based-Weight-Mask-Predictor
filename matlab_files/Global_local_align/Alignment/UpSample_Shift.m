function LR_Up = UpSample_Shift(LR, Tvec, resFactor,keepdim)
if resFactor ~= 1
    for i=1:size(LR,4)
        LR_up(:,:,:,i)=imresize(LR(:,:,:,i),resFactor,'bilinear');
    end
else
    LR_up=LR;
end

H=size(LR_up,1);
W=size(LR_up,2);
C=size(LR_up,3);
N=size(LR_up,4);

for i=1:N
    shift=Tvec{i}.T;
    D(i,:)=round([shift(3,1),shift(3,2)].*resFactor);
end

maxY=max(D(:,1));
minY=min(D(:,1));
maxX=max(D(:,2));
minX=min(D(:,2));

addX=maxX-minX+1;
addY=maxY-minY+1;

LR_Up=zeros(H+addY,W+addX,C,N);
Map=zeros(H+addY,W+addX,C);
for i=1:N
    cy=D(i,1)-minY+1;
    cx=D(i,2)-minX+1;
    LR_Up(cy:cy+H-1,cx:cx+W-1,:,i)= LR_up(:,:,:,i);
    Map(cy:cy+H-1,cx:cx+W-1,i)= ones(H,W);        
end

if keepdim==1
    LR_Up=LR_Up(abs(minY)+1:end-maxY-1,abs(minX)+1:end-maxX-1,:,:);
    Map=Map(abs(minY)+1:end-maxY-1,abs(minX)+1:end-maxX-1,:,:);
end



