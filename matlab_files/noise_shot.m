function img_out = noise_shot(img,scale,rand_opts)
% img - input image is double in range [0,1]

if rand_opts.noise_shot==1
    scale_rand=rand_opts.noise_scale_min+(rand(1)*(rand_opts.noise_scale_max-rand_opts.noise_scale_min));
    img_scaled=img/scale_rand;
    img_noise=imnoise(img_scaled,'poisson');
    img_out=img_noise*scale_rand;
else
    img_scaled=img/scale;
    img_noise=imnoise(img_scaled,'poisson');
    img_out=img_noise*scale;

end