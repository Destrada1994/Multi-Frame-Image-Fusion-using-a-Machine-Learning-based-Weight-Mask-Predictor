function img_out = sim_turbulence(img_in,param)

rows = size(img_in,1);
cols = size(img_in,2);
channels = size(img_in,3);

D=param.D; %Aperture diameter in meters
d=param.d; %Focal length in meters
pixel_sep=param.pixel_sep; %pixel seperation of sensor
lambda=param.lambda; %Wavelength
N=param.N; %Size of PSF filter
L = param.L; %Distance from target
Cn2 = param.Cn2; %refractivity structure index parameter
N_psf_gen=param.N_psf_gen; %Number of PSF to generate

fN=d/D;  %F-number
NA=1/(2*fN);
df=1/(pixel_sep*N);
fNA=NA/lambda;
pr=fNA/df;
ND=round(pr*2)+mod(round(pr*2),2)+1;

[x,y]=meshgrid(-ND/2+1/2:ND/2-1/2,-ND/2+1/2:ND/2-1/2);
[~,r]=cart2pol(x,y);

W=zeros(ND,ND);
W(r<ND/2-1) = 1;

% w     = fftshift(abs(ifft2(W, N, N)).^2);
% figure(1)
% imagesc(w)


mytable = [
    0 0 1;
    1 1 2;
    1 1 3;
    2 0 4;
    2 2 5;
    2 2 6;
    3 1 7;
    3 1 8;
    3 3 9;
    3 3 10;
    4 0 11;
    4 2 12;
    4 2 13;
    4 4 14;
    4 4 15;
    5 1 16;
    5 1 17;
    5 3 18;
    5 3 19;
    5 3 20;
    5 5 21;
    6 0 22;
    6 2 23;
    6 2 24;
    6 4 25;
    6 4 26;
    6 6 27;
    6 6 28;
    7 1 29;
    7 1 30;
    7 3 31;
    7 3 32;
    7 5 33;
    7 5 34;
    7 7 35;
    7 7 36];

n = mytable(:,1);
m = mytable(:,2);

C = zeros(36,36);
for i=1:36
    for j=1:36
        ni = n(i); nj = n(j); % n and m are given in
        mi = m(i); mj = m(j); % Tab 1 of [Noll 1976]
        if (mod(i-j,2)~=0)||(mi~=mj)
            C(i,j) = 0;
        else
            den = gamma((ni-nj+17/3)/2)*...
                gamma((nj-ni+17/3)/2)*...
                gamma((ni+nj+23/3)/2);
            num = 0.0072*(-1)^((ni+nj-2*mi)/2)*...
                sqrt((ni+1)*(nj+1))*pi^(8/3)*...
                gamma(14/3)*gamma((ni+nj-5/3)/2);
            C(i,j) = num/den;
        end
    end
end

C     = C(4:36,4:36);
[U,S] = eig(C);
R     = real(U*sqrt(S));

f1      = @(z) (z/L).^(5/3);
r0      = ((0.423*(2*pi/lambda)^2)*Cn2*integral(f1, 0, L))^(-3/5);
delta0  = L*lambda/(2*D);
kappa = sqrt( (D/r0)^(5/3));

for i=1:N_psf_gen
    b = randn(size(C,1),1);
    a = kappa*R*b;
    [ph, ~] = ZernikeCalc(4:36, a, ND, 'STANDARD');
    U      = exp(1i*2*pi*ph/2).*W;
    uu     = fftshift(abs(ifft2(U, N, N)).^2);
    outPSF(:,:,i) = uu/sum(uu(:));
end

[rowsH, colsH,T] = size(outPSF);
K       = sqrt(T);
blocklength = rowsH + rows/K;
[x,y] = meshgrid(-blocklength/2+1:blocklength/2, -blocklength/2+1:blocklength/2);
[~,r] = cart2pol(x,y);
weight = exp(-r.^2/(2*(blocklength/4)^2));

num = zeros(rows+rowsH);
den = zeros(rows+rowsH);
for C = 1:channels
    img=squeeze(img_in(:,:,C,:));
    img_pad = padarray(img, [(rowsH-1)/2+1, (colsH-1)/2+1], 'symmetric');

    for i=1:K
        for j=1:K
            idx   = (i-1)*K+j;
            idx1  = round((i-1)*rows/K+(rowsH-1)/2+[-(rowsH-1)/2+1:rows/K+(rowsH-1)/2+1]);
            idx2  = round((j-1)*rows/K+(rowsH-1)/2+[-(rowsH-1)/2+1:rows/K+(rowsH-1)/2+1]);
            block = img_pad(idx1, idx2);
            tmp   = imfilter(block, outPSF(:,:,idx), 'symmetric');
            num(idx1, idx2) = num(idx1, idx2) + weight.*tmp;
            den(idx1, idx2) = den(idx1, idx2) + weight;
        end
    end
    out      = num./den;
    img_blur(:,:,C) = out(round(rowsH/2+1):round(rows+rowsH/2), round(colsH/2+1):round(cols+colsH/2));

end

c1      = 2*((24/5)*gamma(6/5))^(5/6);
c2      = 4*c1/pi*(gamma(11/6))^2;

N     = 2*rows;
smax  = delta0/D*N;

sset  = linspace(0,delta0/D*N,N);
f     = @(k) k^(-14/3)*besselj(0,2*sset*k)*besselj(2,k)^2;
I0    = integral(f, 1e-8, 1e3, 'ArrayValued', true);
g     = @(k) k^(-14/3)*besselj(2,2*sset*k)*besselj(2,k)^2;
I2    = integral(g, 1e-8, 1e3, 'ArrayValued', true);   

[x,y] = meshgrid(1:N,1:N);
s     = round(sqrt((x-N/2).^2 + (y-N/2).^2));
s     = min(max(s,1),N);
C     = (I0(s) + I2(s))/I0(1);
C(N/2,N/2)= 1;

kappa2 = I0(1)*c2*(D/r0)^(5/3)/(2^(5/3))*(2*lambda/(pi*D))^2*2*pi;
C     = C*kappa2;

Cfft = fft2(C);
S_half = sqrt(Cfft);
S_half(S_half<0.002*max(S_half(:))) = 0;

MVx    = real(ifft2(S_half.*randn(2*rows,2*cols)))*sqrt(2)*2*rows*(L/delta0);
MVx    = MVx(rows/2+1:2*rows-rows/2, 1:rows);
MVy    = real(ifft2(S_half.*randn(2*rows,2*cols)))*sqrt(2)*2*rows*(L/delta0);
MVy    = MVy(1:cols,cols/2+1:2*cols-cols/2);

for c=1:channels
    img_out(:,:,c) = MotionCompensate(img_blur(:,:,c),MVx,MVy,0.5);
end