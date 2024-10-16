from collections import namedtuple

RegisterObjectEvent = namedtuple("RegisterObjectEvent", 'objectId name type properties width height frontLidarRange rearLidarRange')
UpdateObjectPositionEvent = namedtuple("UpdateObjectPositionEvent", 'objectId x y')
UpdateObjectRotationEvent = namedtuple("UpdateObjectRotationEvent", 'objectId rotation')
UpdateObjectPropertiesEvent = namedtuple("UpdateObjectPropertiesEvent", 'objectId properties')
UpdateObjectAlertsEvent = namedtuple("UpdateObjectAlertsEvent", 'objectId alerts')
UnregisterObjectEvent = namedtuple("UnregisterObjectEvent", 'objectId')
RefreshObjectEvent = namedtuple("RefreshObjectEvent", 'objectId')
ShutdownEvent = namedtuple("ShutdownEvent", "unused")