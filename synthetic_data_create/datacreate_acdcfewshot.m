clear;

files = dir('G:\publicDataset\acdc\acdcmy\099_038_050_100_058_021_049\vol\*.mha');
%files = dir('F:\remotedev\synforosmis\voldata\*.mha');
num =length(files);

for i = 1:num
    Name = files(i).name;
    %name = 'patient021_frame13';
    %Name = [name, '.mha']
    gtName = [Name(1:18),'_gt','.mha'];
    V_ref =  mha_read_volume(['G:\publicDataset\acdc\acdcmy\099_038_050_100_058_021_049\vol\',Name]);
    %V_ref =  mha_read_volume(['F:\remotedev\synforosmis\voldata\',Name]);
    seg1 = mha_read_volume(['G:\publicDataset\acdc\acdcmy\099_038_050_100_058_021_049\seg\',gtName]);
    %seg1 = mha_read_volume(['F:\remotedev\synforosmis\anndata\',Name]);
    a = size(V_ref);
    for j = 1:20
        spacing=[25 25 4];
        O_grid=make_init_grid(spacing,size(V_ref));
        %X_d = 30*(rand(9,11,12,3)-0.5);
        %X_d = 30*(rand(10,12,8,3)-0.5);
        X_d = 30*(rand(size(O_grid))-0.5);
        %acdc太扁了，第三维没法做形变，做了就会有黑洞，所以第三维的形变幅度重新赋值取0
        %X_d(:,:,:,2) = 0;
        X_d(:,:,:,3) = 0;
        O_trans=reshape(X_d,size(O_grid))+O_grid;
        [trans_volume,trans_field] = bspline_transform(O_trans,V_ref,spacing,3);
        trans_volume = int16(trans_volume);

        [x,y,z] = ndgrid(1:a(1),1:a(2),1:a(3));
        x_prime = x + trans_field(:,:,:,1);
        y_prime = y + trans_field(:,:,:,2);
        z_prime = z + trans_field(:,:,:,3);
        seg3 = interpn(x,y,z,seg1,x_prime,y_prime,z_prime,'nearest');

        tempname = [strrep(Name,'.mha','_'),num2str(j),'.mha'];
        writemetaimagefilevf(['G:\publicDataset\acdc\acdcmy\synout_10percent\vol\',tempname],trans_volume,[1,1,1],[0,0,0]);

        writemetaimagefilevf(['G:\publicDataset\acdc\acdcmy\synout_10percent\seg\',tempname],seg3,[1,1,1],[0,0,0]);
    
    
    
    end
end
%
