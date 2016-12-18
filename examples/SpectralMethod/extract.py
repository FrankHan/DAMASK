
# extract data
import damask
import numpy as np

geom = "20grains8x8x8"
load = "1"
refFile = './postProc/%s_%s.txt'%(geom,load)

table = damask.ASCIItable(refFile,readonly=True)
table.head_read()

thresholdKey_strain = 'Mises(ln(V))'
#thresholdKey = 'totalshear'
thresholdKey_stress = 'Mises(Cauchy)'

table.data_readArray(['%i_Cauchy'%(i+1) for i in range(9)]+['%i_ln(V)'%(i+1) for i in range(9)]+[thresholdKey_strain]+[thresholdKey_stress])

np.savetxt("./postProc/%s_%s_ext.txt"%(geom,load),table.data)

