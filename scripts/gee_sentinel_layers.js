// Google Earth Engine starter script for Syria UNICEF POC
// Purpose: Generate Sentinel-2 NDVI, NDWI, NDMI layers for Homs-Damascus corridor.
// Replace AOI geometry with exact project polygon before export.

var aoi = ee.Geometry.Rectangle([36.25, 33.45, 37.30, 34.80]);
var start = '2025-05-01';
var end = '2025-09-30';

function maskS2clouds(image) {
  var qa = image.select('QA60');
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  return image.updateMask(mask).divide(10000);
}

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi)
  .filterDate(start, end)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 35))
  .map(maskS2clouds)
  .median()
  .clip(aoi);

var ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI');
var ndwi = s2.normalizedDifference(['B3', 'B8']).rename('NDWI');
var ndmi = s2.normalizedDifference(['B8', 'B11']).rename('NDMI');
var stack = ndvi.addBands(ndwi).addBands(ndmi);

Map.centerObject(aoi, 8);
Map.addLayer(ndvi, {min: 0, max: 0.7, palette: ['brown', 'yellow', 'green']}, 'NDVI vegetation');
Map.addLayer(ndwi, {min: -0.4, max: 0.4, palette: ['brown', 'white', 'blue']}, 'NDWI water');
Map.addLayer(ndmi, {min: -0.4, max: 0.5, palette: ['red', 'white', 'blue']}, 'NDMI moisture');

Export.image.toDrive({
  image: stack,
  description: 'syria_homs_damascus_ndvi_ndwi_ndmi',
  folder: 'unicef_syria_poc',
  fileNamePrefix: 'syria_homs_damascus_sentinel_indices',
  region: aoi,
  scale: 10,
  maxPixels: 1e13
});
