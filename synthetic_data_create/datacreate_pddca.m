%%
clear;

%这是spacing60，控制点12的，正常一点

files = dir('I:\PDDCA_iso\theone\vol\*.mha');

num =length(files);

for i = 1:1
    Name = files(i).name;
    V_ref =  mha_read_volume(['I:\PDDCA_iso\theone\vol\',Name]);
    seg1 = mha_read_volume('I:\PDDCA_iso\theone\seg\0522c0727.mha');
    
    for j = 1:36
        spacing=[15 15 15];
        X_d = 20*(rand(12,7,10,3)-0.5);
        O_grid=make_init_grid(spacing,[128,46,96]);

        O_trans=reshape(X_d,[12 7 10 3])+O_grid;

        [trans_volume,trans_field] = bspline_transform(O_trans,V_ref,spacing,3);
        trans_volume = int16(trans_volume);

        [x,y,z] = ndgrid(1:128,1:46,1:96);
        x_prime = x + trans_field(:,:,:,1);
        y_prime = y + trans_field(:,:,:,2);
        z_prime = z + trans_field(:,:,:,3);
        seg3 = interpn(x,y,z,seg1,x_prime,y_prime,z_prime,'nearest');

        tempname = [strrep(Name,'.mha','_'),num2str(j),'.mha'];
        writemetaimagefilevf(['I:\PDDCA_iso\deform_output\vol\',tempname],trans_volume,[1,1,1],[0,0,0]);
        save(['I:\PDDCA_iso\deform_output\field\',strrep(tempname,'mha','mat')],'trans_field');
        writemetaimagefilevf(['I:\PDDCA_iso\deform_output\seg\',tempname],seg3,[1,1,1],[0,0,0]);

    
    
    
    end
end
%
