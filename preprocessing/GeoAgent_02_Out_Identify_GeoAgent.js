/**** Start of imports. If edited, may not auto-convert in the playground. ****/
var YRLP = ee.FeatureCollection("users/thrillerlemon/FatLP_HR_merge_Buffer");
/***** End of imports. If edited, may not auto-convert in the playground. *****/


//***Need to refix in offline to delete small polygon***
//***Need to refix in offline to delete small polygon***
//***Need to refix in offline to delete small polygon***



//***1.Set the Study Area***
var studyBounds = YRLP
Map.centerObject(studyBounds,6);
//Set the scale of raster
var rasterScale=5000; 

var main=function(str) 
{ 
  
  //***2.Set Image here***
  var infoImg = ee.Image('users/thrillerlemon/YR_Network/'+str+'_ImgforGAIdentify');
  print(infoImg);
  Map.addLayer(infoImg,{},'infoImg',false);
  
  //***3.Set the seeds here***
  var hexSeeds = ee.Algorithms.Image.Segmentation.seedGrid(15,'hex').clip(studyBounds);
  Map.addLayer(hexSeeds.reproject({crs:infoImg.select(0).projection(),scale:rasterScale}), {palette: "red"},'ORIhexSeeds');
  
  
  //***4.Run SNIC to get Geo Agent***
  
  // Run SNIC on the regular square grid points.
  var gridSNIC = ee.Algorithms.Image.Segmentation.SNIC({
    image: infoImg.clip(studyBounds), 
    compactness: 2000,
    connectivity: 4,
    neighborhoodSize:128,
    seeds: hexSeeds
  }).reproject({crs:infoImg.select(0).projection(),scale:rasterScale})
  var clustersGridSNIC = gridSNIC.select("clusters")
  Map.addLayer(clustersGridSNIC.randomVisualizer(), {}, "clustersGridSNIC");
  
  // Map.addLayer(hexSeeds, {opacity: 0.8}, 'newGridSeedsV');

  var clustersPoly = clustersGridSNIC.toInt32().addBands(ee.Image(1).rename('one')).select('clusters','one').reduceToVectors({
    crs: 'EPSG:4326',
    scale: 1000,
    geometryType: 'polygon',
    geometry: studyBounds,
    maxPixels: 50000000000,
    eightConnected: false,
    labelProperty: 'clusters',
    reducer: ee.Reducer.first()
  });
  print('clustersPoly',clustersPoly)
  Map.addLayer(clustersPoly,{},'clustersPoly')
  //***5.Save the GA by shp***
  Export.table.toDrive({
  collection: clustersPoly,
  description:'YR_Network_GeoAgentfromGEE',
  fileFormat: 'SHP'
  });
};

main('YR_Greening');