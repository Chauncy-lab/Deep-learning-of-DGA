def GetHostname(seed):
    aHelperTable = []
    aHexHostname = []
    aHostStrings = None
    aShuffle = []
    for i in range(0, 15):
        aShuffle[aHelperTable[i * 2]] = chr(seed & 1)
        seed >>= 1

    iHost1 = 0
    iHost2 = 0
    for i in range(0, 7):
        iHost1 = 2 * iHost1 | aShuffle[i]
        iHost2 = 2 * iHost2 | aShuffle[i + 7]

    iHost2 = (2 * iHost2 | aShuffle[14]) + 128
    offsetHost1 = aHexHostname[iHost1 * 2] + aHexHostname[iHost1 * 2 + 1] << 0x08
    offsetHost2 = aHexHostname[iHost2 * 2] + aHexHostname[iHost2 * 2 + 1] << 0x08
    host1 = ""
    host2 = ""
    offsetHost1 += 1
    b = aHostStrings[offsetHost1]
    while b != 0:
        host1 += b

    offsetHost2 += 1
    b = aHostStrings[offsetHost2]
    while (b != 0):
        host2 += b

    return host1 + host2 + ".net"
