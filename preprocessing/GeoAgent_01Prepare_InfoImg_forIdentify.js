/**** Start of imports. If edited, may not auto-convert in the playground. ****/
var YRLP = ee.FeatureCollection("users/thrillerlemon/FatLP_HR_merge_Buffer");
/***** End of imports. If edited, may not auto-convert in the playground. *****/

var studyBounds = YRLP
// var studyBounds = ee.Geometry.Polygon(coords,proj ,false )

Map.centerObject(studyBounds,5);
Map.addLayer(studyBounds, {}, 'studyBounds');


var main=function(str)
{
  //***2.Set Image here***
  //Yearly LAI mean,variance,sensSlope
  var imagelist=ee.List([]);  
  for(var yearnumber=1986;yearnumber<2019;yearnumber++)
  {
    var ThisYearLAI=ee.Image('users/thrillerlemon/Annual_NOAA_LAI/NOAA_LAI_' + yearnumber);
    imagelist=imagelist.add(ThisYearLAI);
  }
  var LAIcollection = ee.ImageCollection(imagelist)
  
  var meanYearsLAI=LAIcollection.reduce(ee.Reducer.mean())
  var variYearsLAI=LAIcollection.reduce(ee.Reducer.variance())
  var slopYearsLAI=ee.Image('users/thrillerlemon/Annual_NOAA_LAI/NOAA_LAI_NOAA_LAI_SenSlope')
  var lucc=ee.Image('users/thrillerlemon/NEW_landsat_LC/landcover_2018_tempslid')
  var dem=ee.Image('CGIAR/SRTM90_V4')
  
  var infoImg = dem
            .addBands(lucc)
            .addBands(slopYearsLAI)
            .addBands(variYearsLAI)
            .addBands(meanYearsLAI)
            ;
  saveimage(infoImg,'infoImg','YR_Network/'+str+'_ImgforGAIdentify');
}; 



var saveimage=function(image,imagename,imageid)
{
  Export.image.toAsset({ 
  image:image,
  description:imagename,
  assetId: imageid,
  scale:1000,
  region: studyBounds.geometry().bounds(),
  maxPixels: 999999999999,
  });
  print('already save to Asset '+imagename);
};


main('YR_Greening');