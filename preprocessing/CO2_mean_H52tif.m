clc
clear all
%% 基本信息
% 输入文件夹主目录
maindir = 'F:\Learning\Yanjiusheng1\复杂网络\大气CO2数据处理\SWIRL3CO2_mix';
% 输出文件夹
outpt = 'F:\Learning\Yanjiusheng1\复杂网络\大气CO2数据处理\SWIRL3_H5_read';
% 行列数/背景值
nrows = 72;
ncols = 144;
bv = -9999;
%% 遍历得到每一个h5文件
subdir =  dir( maindir );
for i = 3 : length( subdir )
    % 获得当前索引i下的h5文件
    temp_dir1 = fullfile( subdir(i).folder, subdir( i ).name);
    temp_dir2 = dir(temp_dir1);
    temp_dir3 = dir(fullfile(temp_dir2(3).folder, temp_dir2(3).name));
    temp_dir4 = fullfile(temp_dir3(3).folder, temp_dir3(3).name);
    % 
    name = temp_dir4;
    basename = subdir(i).name;
    % h5disp(name)输出h5数据结构
    %% 获取产品名称，卫星名称，传感器名称
    product_Name = h5read(name, '/Global/metadata/productName');
    satellite_Name = h5read(name, '/Global/metadata/satelliteName');
    sensor_Name = h5read(name, '/Global/metadata/sensorName');

    lat = h5read(name, '/Data/geolocation/latitude');
    lon = h5read(name, '/Data/geolocation/longitude');
    lons = [-1.7875000e+02,1.7875000e+02];
    lats = [-88.7500000,88.7500000];
    %% 获取各个参数
    XCO2Average = h5read(name, '/Data/latticeInformation/XCO2Average');
    XCO2Maximum = h5read(name, '/Data/latticeInformation/XCO2Maximum');
    XCO2Median = h5read(name, '/Data/latticeInformation/XCO2Median');
    XCO2Minimum = h5read(name, '/Data/latticeInformation/XCO2Minimum');
    XCO2Mode = h5read(name, '/Data/latticeInformation/XCO2Mode');
    XCO2StandardDeviation = h5read(name, '/Data/latticeInformation/XCO2StandardDeviation');
    numObservationPoints = h5read(name, '/Data/latticeInformation/numObservationPoints');
    %% h5转tif
    hds = 'XCO2Average';
    Rmat = makerefmat('RasterSize',[nrows,ncols],...
        'Latlim',[lats(1) lats(2)], 'Lonlim',[lons(1) lons(2)],...
        'ColumnsStartFrom','north');
    geotiffwrite([outpt,'\',hds,'_',basename,'.tif'],XCO2Average',Rmat)
    fprintf('%s has converted\n',basename);

end
