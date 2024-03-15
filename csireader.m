clear
%% csireader.m
%
% read and plot CSI from UDPs created using the nexmon CSI extractor (nexmon.org/csi)
% modify the configuration section to your needs
% make sure you run >mex unpack_float.c before reading values from bcm4358 or bcm4366c0 for the first time
%
% the example.pcap file contains 4(core 0-1, nss 0-1) packets captured on a bcm4358
%
%�޸�����������FILENAME,csi_arr_max_length,fid1��fid2��fid3
%% configuration
CHIP = '4358';          % wifi chip (possible values 4339, 4358, 43455c0, 4366c0)
BW = 20;                % bandwidth
% PATH = 'data/3DHAR/20231230/';         % data path
PATH = 'data/voice/20240312/';         % data path
FILENAME = 'voice_sil_1';% capture file
FILE = strcat(PATH,FILENAME,'.pcap');
FILE_end = strcat(PATH,FILENAME);
NPKTS_MAX = 200000;       % max number of UDPs to process
NUM_SUB = 50;               %���������ز�����

%% read file
HOFFSET = 16;           % header offset
NFFT = BW*3.2;          % fft size
p = readpcap();
p.open(FILE);
n = min(length(p.all()),NPKTS_MAX);
p.from_start();
csi_buff = complex(zeros(n,NFFT),0);
csi = complex(zeros(n,NUM_SUB),0);
csi_ampt = zeros(n,NUM_SUB);
csi_phase = zeros(n,NUM_SUB);
timestamp=zeros(1,n);
k = 1;

while (k <= n)
    f = p.next();
    if isempty(f)
        disp('no more frames');
        disp(FILE);
        disp(k);
        break;
    end
    
    if f.header.orig_len-(HOFFSET-1)*4 ~= NFFT*4
        disp('skipped frame with incorrect size');
        disp(k);
        continue;
    end

    payload = f.payload;
    
%     ���payload��ά���Ƿ��������ģ�ֵӦΪ79
    if size(payload,1)<HOFFSET+NFFT-1
        continue;
    end
    
    H = payload(HOFFSET:HOFFSET+NFFT-1);
    if (strcmp(CHIP,'4339') || strcmp(CHIP,'43455c0'))
        Hout = typecast(H, 'int16');
    elseif (strcmp(CHIP,'4358'))
        Hout = unpack_float(int32(0), int32(NFFT), H);
    elseif (strcmp(CHIP,'4366c0'))
        Hout = unpack_float(int32(1), int32(NFFT), H);
    else
        disp('invalid CHIP');
        break;
    end

    Hout = reshape(Hout,2,[]).';
    cmplx = double(Hout(1:NFFT,1))+1j*double(Hout(1:NFFT,2));
    csi_buff(k,:) = cmplx.';
    
%% ��������64�����ز�    
%     csi(k,:) = csi_buff(k,:);
%     csi_ampt(k,:) = abs(csi(k,:));
%     csi_phase(k,:) = angle((csi(k,:)));

%% ����52�����ز�
% �ڶ�csi_buff�����渵��Ҷ�仯��ȥ��8����Ϊ�����������ز�-32��-31��-30��-29��0��29��30��31����ȥ����Ƶ�����ز�-21��-7��7��21
%     csi_1 = csi_buff(k,2:29);
% %     csi_2 = csi_buff(k,9:21);
% %     csi_3 = csi_buff(k,23:29);
%     csi_4 = csi_buff(k,37:64);
% %     csi_5 = csi_buff(k,45:57);
% %     csi_6 = csi_buff(k,59:64);
%     csi(k,:) = [csi_1,csi_4];
%     csi_ampt(k,:) = abs(csi(k,:));
    
%% ����52,����ȥ��8����������ĸ�Ƶ�ʷָ��  
    csi_1 = csi_buff(k,3:7);
    csi_2 = csi_buff(k,9:21);% 5
    csi_3 = csi_buff(k,23:29);% 18
%     csi(k,:) = [csi_1,csi_2,csi_3];
    csi_4 = csi_buff(k,38:43);
    csi_5 = csi_buff(k,45:57);%31
    csi_6 = csi_buff(k,59:64);%44
    csi(k,:) = [csi_1,csi_2,csi_3,csi_4,csi_5,csi_6];  
    csi_ampt(k,:) = abs(fftshift(csi(k,:)));
    csi_phase(k,:) = angle((csi(k,:)));


    
    %����ʱ���
    if (k==1)
        temp_sec=f.header.ts_sec;
        temp_usec=f.header.ts_usec;
    elseif (temp_usec > f.header.ts_usec)
        timestamp(k) = double(f.header.ts_sec - temp_sec) - double(temp_usec - f.header.ts_usec)/1000000;
    else
        timestamp(k) = double(f.header.ts_sec - temp_sec) + double(f.header.ts_usec - temp_usec)/1000000;
    end

    k = k + 1;
end
csi_ampt(:,51) = timestamp;

% csi_buff = fftshift(csi,2);
% csi_temp1 = abs(csi_buff(:,6:32));
% csi_temp2 = abs(csi_buff(:,35:61));
%csi_ampt2 = [csi_temp1,csi_temp2];
% csi_ampt_temp = abs(fftshift(csi,2));
% csi_pha_temp = angle(fftshift(csi,2));
% csi_ampt2 = [csi_ampt_temp(:,5:32),csi_ampt_temp(:,34:61)];
% csi_pha2 = [csi_pha_temp(:,5:32),csi_pha_temp(:,34:61)];
%plot
%plotcsi(csi_buff, NFFT, false);

% writematrix(csi_ampt,FILE_end);

%�ر��ļ�
File_close = fclose(p.fid);

