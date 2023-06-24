files = dir("data_split/*TE*");
Fs=48000;
dt = 1/Fs;
for idx = 1:length(files)
    loaded_mat = load(files(idx).name);
    for i = 1:length(loaded_mat.Data.A)
        loaded_teoae = loaded_mat.Data.A(:, i); %first have our reading
        cut_teoae = loaded_teoae(170:912); % then take a slice
        t = 0:dt:numel(cut_teoae)*dt-dt;
        [sst,f] = cwt(cut_teoae, Fs);
        pcolor(t,f,abs(sst))
        shading interp
        xlabel('Seconds')
        ylabel('Frequency (Hz)')
        ylim([500 7000])
        plot_title = sprintf("%s %d A", files(idx).name, i)
        title(plot_title)
        filename = sprintf('plots/%s %d A.fig', files(idx).name, i)
        savefig(filename)
        mat_filename = sprintf('processed data/%s %d A.mat', files(idx).name, i)
        abs_sst = abs(sst);
        save(mat_filename, "abs_sst");
    end
    for i = 1:255
        loaded_teoae = loaded_mat.Data.B(:, i); %first have our reading
        cut_teoae = loaded_teoae(170:912); % then take a slice
        t = 0:dt:numel(cut_teoae)*dt-dt;
        [sst,f] = cwt(cut_teoae, Fs);
        pcolor(t,f,abs(sst))
        shading interp
        xlabel('Seconds')
        ylabel('Frequency (Hz)')
        ylim([500 7000])
        plot_title = sprintf("%s %d B", files(idx).name, i)
        title(plot_title)
        filename = sprintf('plots/%s %d B.fig', files(idx).name, i)
        savefig(filename)
        mat_filename = sprintf('processed data/%s %d B.mat', files(idx).name, i)
        abs_sst = abs(sst);
        save(mat_filename, "abs_sst")
    end
  
end
