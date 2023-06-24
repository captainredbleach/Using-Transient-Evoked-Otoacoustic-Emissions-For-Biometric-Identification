files = dir("data/*TE*");
Fs=48000;
dt = 1/Fs;
for idx = 1:length(files)
    loaded_mat = load(files(idx).name);
    processed_teoae = loaded_mat.output_teoae;
    t = 0:dt:numel(processed_teoae)*dt-dt;
    [sst,f] = wsst(processed_teoae,Fs,'amor');
    pcolor(t,f,abs(sst))
    shading interp
    xlabel('Seconds')
    ylabel('Frequency (Hz)')
    ylim([500 7000])
    title(files(idx).name)
    filename = sprintf('../plots_wsst/%s.fig', files(idx).name)
    savefig(filename)
    mat_filename = sprintf('../wsst_data/%s', files(idx).name)
    save(mat_filename, "sst")
end