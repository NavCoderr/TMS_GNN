def getTestFrame(orderId = 23):
    testFrame = "60    70    91    0    19    136    0    1    7    178    1    1    5    0    0    0    0    0    0    0    10    40    0    1    0    70    7    220    2    13    2    23    21    29    58    147    84    176    0    1    0    {}    0    20    0    48    7    231    3    2    5    12    11    10    0    0    0    0    93    70    62    0    0    0".format(orderId)

    splitted = testFrame.split(' ')
    splitted = list(filter(lambda token: len(token) > 0, splitted))
    print(len(splitted))

    binaryStream = b""
    for token in splitted:
        intToken = int(token)
        if intToken > 255:
            raise Exception("Invalid token")
        binaryStream += intToken.to_bytes(1, 'big')

    return binaryStream
