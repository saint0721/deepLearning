import sys
import os
from qgis.core import QgsApplication, QgsProject, QgsRasterLayer

# Headless 모드: 디스플레이 없이 실행
os.environ["QT_QPA_PLATFORM"] = 'offscreen'

# QGIS 경로 설정
QgsApplication.setPrefixPath('/usr', True)

# QGIS 어플리케이션 생성
qgs = QgsApplication([], False)
qgs.initQgis()

# .tif파일을 Raster Layer 열기
raster_path = "./2023-gangcheon-bo.tif"
raster_layer = QgsRasterLayer(raster_path, 'gangcheon_raster')

if not raster_layer.isValid():
    print("레이어가 올바르지 않습니다.")
else:
    QgsProject.instance().addMapLayer(raster_layer)
    print("레이어가 정상적으로 추가되었습니다.")
    
qgs.exitQgis()