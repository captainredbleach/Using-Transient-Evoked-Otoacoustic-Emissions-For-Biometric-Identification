mat_files = dir("/home/lab-user/datasets/project_data/in_fitting_data (copy)/*mat");
target_train_path = "/home/lab-user/datasets/project_data/avg/train";
target_test_path = "/home/lab-user/datasets/project_data/avg/test";
target_val_path = "/home/lab-user/datasets/project_data/avg/val";
Fs=48000;
for cur_file = 1:length(mat_files)
    path = mat_files(cur_file).name;
    fullpath = sprintf("/home/lab-user/datasets/project_data/in_fitting_data (copy)/%s", path);
    loaded_mat = load(fullpath);
    for i = 1:5:181
        loaded_teoae_A = loaded_mat.Data.A(:, i:5); %first have our reading
        loaded_teoae_B = loaded_mat.Data.B(:, i:5); 
        cut_teoae_A = loaded_teoae_A(183:912);
        cut_teoae_B = loaded_teoae_B(183:912);
        avg_A = mean(cut_teoae_A,2);
        avg_B = mean(cut_teoae_B,2);
        mat_filename = sprintf('%s/%s %d A.mat', target_train_path, mat_files(cur_file).name, i)
        mat_filename_B = sprintf('%s/%s %d B.mat', target_train_path, mat_files(cur_file).name, i)
        save(mat_filename, "cut_teoae_A");
        save(mat_filename_B, "cut_teoae_B");
    end
    for i = 181:5:231 
        loaded_teoae_A = loaded_mat.Data.A(:, i); %first have our reading
        loaded_teoae_B = loaded_mat.Data.B(:, i); 
        cut_teoae_A = loaded_teoae_A(183:912);
        cut_teoae_B = loaded_teoae_B(183:912);
        mat_filename = sprintf('%s/%s %d A.mat', target_val_path, mat_files(cur_file).name, i)
        mat_filename_B = sprintf('%s/%s %d B.mat', target_val_path, mat_files(cur_file).name, i)
        save(mat_filename, "cut_teoae_A");
        save(mat_filename_B, "cut_teoae_B");
    end
    for i = 231:5:256
        loaded_teoae_A = loaded_mat.Data.A(:, i); %first have our reading
        loaded_teoae_B = loaded_mat.Data.B(:, i); 
        cut_teoae_A = loaded_teoae_A(183:912);
        cut_teoae_B = loaded_teoae_B(183:912);
        mat_filename = sprintf('%s/%s %d A.mat', target_test_path, mat_files(cur_file).name, i)
        mat_filename_B = sprintf('%s/%s %d B.mat', target_test_path, mat_files(cur_file).name, i)
        save(mat_filename, "cut_teoae_A");
        save(mat_filename_B, "cut_teoae_B");
    end
end