from collections import namedtuple

ObjectData = namedtuple("ObjectData", "name sourceType connectionString width height frontLidarRange rearLidarRange xSignal ySignal headingSignal properties alerts updateInterval")


class ConfigurationWidgetsBuilder:
    def __init__(self, persistency, mapView, objectsView):
        self.__persistency = persistency
        self.__mapView = mapView
        self.__objectsView = objectsView

    def build(self):
        self.__buildMapWidget()
        self.__buildObjectsWidget()

    def __buildMapWidget(self):
        mapData = self.__persistency.mapData()
        self.__mapView.filePicker.lineEdit.setText(mapData[0])
        self.__mapView.xSpinBox.spinbox.setValue(mapData[1])
        self.__mapView.ySpinBox.spinbox.setValue(mapData[2])
        self.__mapView.widthSpinBox.spinbox.setValue(mapData[3])
        self.__mapView.heightSpinBox.spinbox.setValue(mapData[4])

    def __buildObjectsWidget(self):
        objects = self.__persistency.objectsList()
        for objectToShow in objects:
            self.__buildObject(objectToShow)

    def __buildObject(self, objectToShow):
        properties = list()
        for propertyNode in objectToShow['properties']:
            propertyTuple = propertyNode,'/'.join(objectToShow['properties'][propertyNode])
            properties.append(propertyTuple)

        alerts = list()
        for alertNode in objectToShow['alerts']:
            alertTuple = alertNode,'/'.join(objectToShow['alerts'][alertNode])
            alerts.append(alertTuple)

        objectData = ObjectData(
            name=objectToShow['name'],
            sourceType=objectToShow['sourceType'],
            connectionString=objectToShow['connectionString'],
            width=objectToShow['width'],
            height=objectToShow['height'],
            xSignal='/'.join(objectToShow['xSignal']),
            ySignal='/'.join(objectToShow['ySignal']),
            headingSignal='/'.join(objectToShow['rotationSignal']),
            alerts=alerts,
            properties=properties,
            updateInterval=objectToShow['updateInterval'],
            frontLidarRange=objectToShow['frontLidarRange'],
            rearLidarRange=objectToShow['rearLidarRange']
        )
        self.__objectsView.onObjectAdded(objectData)

