mat_files = dir("/home/lab-user/datasets/project_data/outsider_data/*mat");
target_test_path = "/home/lab-user/datasets/project_data/outsider_data/Test";
Fs=48000;
for cur_file = 1:length(mat_files)
    path = mat_files(cur_file).name;
    fullpath = sprintf("/home/lab-user/datasets/project_data/outsider_data/%s", path)
    loaded_mat = load(fullpath);
        for i = 1:size(loaded_mat.Data.A, 2)
            loaded_teoae_A = loaded_mat.Data.A(:, i); %first have our reading
            cut_teoae_A = loaded_teoae_A(183:912);
            mat_filename = sprintf('%s/%s %d A.mat', target_test_path, mat_files(cur_file).name, i)
            save(mat_filename, "cut_teoae_A");
            
        end
        for i = 1:size(loaded_mat.Data.B, 2)
            loaded_teoae_B = loaded_mat.Data.B(:, i);
            cut_teoae_B = loaded_teoae_B(183:912);
            mat_filename_B = sprintf('%s/%s %d B.mat', target_test_path, mat_files(cur_file).name, i)
            save(mat_filename_B, "cut_teoae_B");
        end
end