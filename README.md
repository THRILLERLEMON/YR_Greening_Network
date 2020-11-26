# YR_Greening_Network

>This repository is the code of Thriller柠檬

## **THRILLER柠檬**

## **thrillerlemon@outlook.com**
## **thrillerlemon@snnu.edu.cn**

### 说明

|文件/文件夹名称|内容|创建日期|备注|
|:-:|:-:|:-:|:-:|
|<u>**README**<u>|代码库ReadMe|20201014|无|
|<u>**data**<u>|数据文件夹|20201111|无|
|<u>**GeoAgent**<u>|自然地理单元的shp文件|20201114|无|
|<u>**SampleData**<u>|示例数据，测试数据|20201120|无|
|<u>**docs**<u>|代码库说明文档|20201014|无|
|<u>**preprocessing**<u>|数据预处理模块，包括GeoAgent提取、数据提取和数据修复|20201117|无|
|**_GeoAgent_01Prepare_InfoImg_forIdentify_**|制备识别GeoAgent的数据|20201111|GEE代码|
|**_GeoAgent_02_Out_Identify_GeoAgent_**|识别GeoAgent并输出|20201111|GEE代码|
|**_NDeposition-NC_to_tif_**|提取出1981-2020年两种N化合物的沉降数据|20201111|matlab代码|
|**_NDeposition-Add2tif21_**|两种N化合物的沉降数据加和合成|20201111|matlab代码|
|**_repair_sta_data_**|插值修复统计Excel数据|20201118|Ronganlly合作|
|<u>**yrnetwork**<u>|核心模块，包括核心的类和方法|20201014|无|
|**_coupled_network_**|耦合网络构建方法和展示|20201122|存档|
|**_coupled_network_causal_**|耦合因果网络构建方法和展示|20201125|无|
|**_luc_network_**|构建和展示土地利用/覆被转移网络和伴随网络（比如叶面积转移网络）|20201101|无|
|**_setting_**|工程设置，代码中用到的路径默认参数等设置|20201014|无|
|**_useful_class_**|代码中用到的外部引用的类或者方法|20201014|无|
|<u>**main**<u>|代码进行主程序|20201014|目前无效|
|<u>**requirements**<u>|开发依赖说明|20201014|无|

### To Do

**✅1、土地利用/覆被转移网络构建和展示———Done！**  
**✅2、地学智能体识别---Done！**  
**📌3、自然变量提取**  
>变量确认  
>数据提取  

**📌4、社会经济变量提取**  
>变量确认  
>数据修复  

**📌5、耦合网络**  


**目前Ronganlly和Shaylee任务**  
>📌1、根据docs/UsedDataVar.xlsx在GEE中提取变量，具体要求和格式与THRILLER交流【Shaylee】  
```
伪代码 
方案一
[下面以一个月、年均为平均的数据为例，注意降水等再月和年上是加和的,需要改]  
加载GeoAgent和对应数据  
GeoAgentYearly=GeoAgent.map(addInfoYearly)  
Export GeoAgentYearly as csv  
GeoAgentMonthly=GeoAgent.map(addInfoMonthly)  
Export GeoAgentMonthly as csv  

var addInfoYearly=function(agent)  
{  
    for 年 in N年:  
        var thisYearImageCollection = 对数据集进行时间筛选  
        var mean_thisYear = thisYearImageCollection.mean()  
        var 每个GA中每年的平均值 = mean_thisYear.reduceregion({agent,......})  
        agent=agent.set(ee.String(年) , 每个GA中每年的平均值.get(...))  
}  
var addInfoMonthly=function(agent)  
{  
    for 年 in N年:  
        for 月 in 12个月  
            var thisMonthIC=对数据集进行时间筛选  
            var mean_thisMonth=thisMonthIC.mean()  
            var 每个GA中每个月的平均值=mean_thisMonth.reduceregion({agent,......})  
            agent=agent.set(ee.String(年+月) , 每个GA中每个月的平均值.get(...))  
}
方案二
var yearsI=[]
for 年 in N年:  
    var thisYearImageCollection = 对数据集进行时间筛选  
    var mean_thisYear = thisYearImageCollection.mean()  
    yearsI=yearsI.append(mean_thisYear)
var yearsIC=ee.ImageCollection(yearsI)
var GAs中每年的平均值=yearsIC.reduceregions({Geoagents,......})  
export GAs中每年的平均值

var monthsI=[]
for 年 in N年:  
    for 月 in 12个月  
        var thisMonthIC=对数据集进行时间筛选  
        var mean_thisMonth=thisMonthIC.mean()
        monthsI=monthsI.append(mean_thisMonth)
var monthsIC=ee.ImageCollection(monthsI)
var GAs中每月的平均值=monthsIC.reduceregions({Geoagents,......})  
export GAs中每月的平均值
```
>📌2、根据docs/UsedDataVar.xlsx中的N沉降（需要用matlab提取）和CO2数据（需要下载和提取）需要特殊处理【Ronganlly】  
>📌3、对docs/UsedStaData.xlsx中的数据进行插值处理（1）异常值排查（2）数据填补【Ronganlly：只需要提供方法代码preprocessing/repair_sta_data】  
>📌4、需要对UsedInvestment开头的文件进行整理，概览论文，然后把Excel中的数据整理成政府对黄河流域投资数据【Shaylee】  
