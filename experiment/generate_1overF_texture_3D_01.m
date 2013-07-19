
%  Generate 1/f^2 noise texture and save as .bmp image for fish
%  appreciation
% Written by MBA June 21 2013

% generate_1overF_texture_01.m
clear all

sizx = 200;    % for example
sizy = 150;
siz_t = 200;
XX = [1:sizx (sizx-1):-1:1];% XX = repmat(XX,(2*siz-1),1);
YY = [1:sizy (sizy-1):-1:1];

TT=[1:siz_t (siz_t-1):-1:1];

D3 = zeros(2*sizx-1,2*sizy-1,2*siz_t-1);
scale_t = .02;
for x = 1:2*sizx-1
    x
    for y = 1:2*sizy-1
        t = 1:2*siz_t-1;
        D3(x,y,:) = XX(x)^3 + YY(y)^3 + TT(t).^3/scale_t;
%        D3(x,y,:) = XX(x)^3 + YY(y)^3 + TT(t).^3/scale_t;
    end
end
%D3=XX.^2+YY.^2;

MF=randn(2*sizx,2*sizy,2*siz_t);
NF=zeros(2*sizx,2*sizy,2*siz_t);
NF(2:end,2:end,2:end)=1./sqrt(D3);
%S=real(ifft2(MF.*NF));
S=real(ifftn(MF.*NF));
%    MF = MF(:,:,1);
%    NF = NF(:,:,2);
%    S = real(ifftn(MF.*NF));

Snorm = S-mean(S(:));
Snorm = Snorm./ std(Snorm(:));

Snorm = Snorm + 2;
Snorm = Snorm .* (Snorm > 0) .* (Snorm <= 4) + 4*(Snorm > 4);
Snorm = Snorm / 4 * 254 + 1;

Snorm_extended = zeros(size(Snorm,1)*3,size(Snorm,2)*3,size(Snorm,3));
for i = 1:size(Snorm,3)
    Snorm_extended(:,:,i) = repmat(squeeze(Snorm(:,:,i)),3,3);
end

%%
for i = 1:size(Snorm,3)
    surf((squeeze(Snorm(:,:,i)')));
        shading interp
    pbaspect([1 1 .05]);
    colormap gray
    pause(.016)
end
%%
fInds = {};
pad = numel(num2str(2*siz_t-1));
for i = 0:(2*siz_t-1)
        fInds{i+1} = [repmat('0',1,pad-numel(num2str(i))) num2str(i)];
end

figure;
for i = 0:(2*siz_t-1)
    imout = ((Snorm_extended(:,:,i+1)'.^5/256^5));
    imagesc(imout);
    colormap gray
    pause(.016)
    imwrite(imout, ['C:\Dropbox\Davis\im\frames\' fInds{i+1} '.png'], 'png');
end



