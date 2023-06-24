files = dir("data/*TE*");
Fs=48000;
dt = 1/Fs;
for idx = 1:length(files)
    loaded_mat = load(files(idx).name);
    for i = 182:length(loaded_mat.Data.A):912
        processed_teoae = loaded_mat.Data.A(:, i);
        t = 0:dt:numel(processed_teoae)*dt-dt;
        [sst,f] = cwt(processed_teoae,'bump', Fs);
    end
    pcolor(t,f,abs(sst))
    shading interp
    xlabel('Seconds')
    ylabel('Frequency (Hz)')
    ylim([500 7000])
    title(files(idx).name)
    filename = sprintf('../cwt_plots/%s.fig', files(idx).name)
    savefig(filename)
    mat_filename = sprintf('../cwt_data/%s', files(idx).name)
    save(mat_filename, "sst")
end