mat_files = dir("/home/lab-user/datasets/project_data/in_fitting_data/*mat");
target_train_path = "/home/lab-user/datasets/project_data/frequency-domain/train";
target_test_path = "/home/lab-user/datasets/project_data/frequency-domain/test";
target_val_path = "/home/lab-user/datasets/project_data/frequency-domain/val";
Fs=48000;
for cur_file = 1:length(mat_files)
    path = mat_files(cur_file).name;
    fullpath = sprintf("/home/lab-user/datasets/project_data/in_fitting_data/%s", path)
    loaded_mat = load(fullpath);
    for i = 1:179
        loaded_teoae_A = loaded_mat.Data.A(:, i); %first have our reading
        loaded_teoae_B = loaded_mat.Data.B(:, i); 
        cut_teoae_A = loaded_teoae_A(183:912);
        cut_teoae_B = loaded_teoae_B(183:912);
        mat_filename = sprintf('%s/%s %d A.mat', target_train_path, mat_files(cur_file).name, i)
        mat_filename_B = sprintf('%s/%s %d B.mat', target_train_path, mat_files(cur_file).name, i)
        cwt_a = abs(cwt(loaded_teoae_A));
        cwt_b = abs(cwt(loaded_teoae_B));
        save(mat_filename, "cwt_a");
        save(mat_filename_B, "cwt_b");
    end
    for i = 180:230
        loaded_teoae_A = loaded_mat.Data.A(:, i); %first have our reading
        loaded_teoae_B = loaded_mat.Data.B(:, i); 
        cut_teoae_A = loaded_teoae_A(183:912);
        cut_teoae_B = loaded_teoae_B(183:912);
        mat_filename = sprintf('%s/%s %d A.mat', target_val_path, mat_files(cur_file).name, i)
        mat_filename_B = sprintf('%s/%s %d B.mat', target_val_path, mat_files(cur_file).name, i)
        cwt_a = abs(cwt(loaded_teoae_A));
        cwt_b = abs(cwt(loaded_teoae_B));
        save(mat_filename, "cwt_a");
        save(mat_filename_B, "cwt_b");
    end
    for i = 231:256
        loaded_teoae_A = loaded_mat.Data.A(:, i); %first have our reading
        loaded_teoae_B = loaded_mat.Data.B(:, i); 
        cut_teoae_A = loaded_teoae_A(183:912);
        cut_teoae_B = loaded_teoae_B(183:912);
        mat_filename = sprintf('%s/%s %d A.mat', target_test_path, mat_files(cur_file).name, i)
        mat_filename_B = sprintf('%s/%s %d B.mat', target_test_path, mat_files(cur_file).name, i)
        cwt_a = abs(cwt(loaded_teoae_A));
        cwt_b = abs(cwt(loaded_teoae_B));
        save(mat_filename, "cwt_a");
        save(mat_filename_B, "cwt_b");
    end
end