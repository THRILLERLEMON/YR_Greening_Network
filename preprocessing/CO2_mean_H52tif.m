clc
clear all
%% ������Ϣ
% �����ļ�����Ŀ¼
maindir = 'F:\Learning\Yanjiusheng1\��������\����CO2���ݴ���\SWIRL3CO2_mix';
% ����ļ���
outpt = 'F:\Learning\Yanjiusheng1\��������\����CO2���ݴ���\SWIRL3_H5_read';
% ������/����ֵ
nrows = 72;
ncols = 144;
bv = -9999;
%% �����õ�ÿһ��h5�ļ�
subdir =  dir( maindir );
for i = 3 : length( subdir )
    % ��õ�ǰ����i�µ�h5�ļ�
    temp_dir1 = fullfile( subdir(i).folder, subdir( i ).name);
    temp_dir2 = dir(temp_dir1);
    temp_dir3 = dir(fullfile(temp_dir2(3).folder, temp_dir2(3).name));
    temp_dir4 = fullfile(temp_dir3(3).folder, temp_dir3(3).name);
    % 
    name = temp_dir4;
    basename = subdir(i).name;
    % h5disp(name)���h5���ݽṹ
    %% ��ȡ��Ʒ���ƣ��������ƣ�����������
    product_Name = h5read(name, '/Global/metadata/productName');
    satellite_Name = h5read(name, '/Global/metadata/satelliteName');
    sensor_Name = h5read(name, '/Global/metadata/sensorName');

    lat = h5read(name, '/Data/geolocation/latitude');
    lon = h5read(name, '/Data/geolocation/longitude');
    lons = [-1.7875000e+02,1.7875000e+02];
    lats = [-88.7500000,88.7500000];
    %% ��ȡ��������
    XCO2Average = h5read(name, '/Data/latticeInformation/XCO2Average');
    XCO2Maximum = h5read(name, '/Data/latticeInformation/XCO2Maximum');
    XCO2Median = h5read(name, '/Data/latticeInformation/XCO2Median');
    XCO2Minimum = h5read(name, '/Data/latticeInformation/XCO2Minimum');
    XCO2Mode = h5read(name, '/Data/latticeInformation/XCO2Mode');
    XCO2StandardDeviation = h5read(name, '/Data/latticeInformation/XCO2StandardDeviation');
    numObservationPoints = h5read(name, '/Data/latticeInformation/numObservationPoints');
    %% h5תtif
    hds = 'XCO2Average';
    Rmat = makerefmat('RasterSize',[nrows,ncols],...
        'Latlim',[lats(1) lats(2)], 'Lonlim',[lons(1) lons(2)],...
        'ColumnsStartFrom','north');
    geotiffwrite([outpt,'\',hds,'_',basename,'.tif'],XCO2Average',Rmat)
    fprintf('%s has converted\n',basename);

end
