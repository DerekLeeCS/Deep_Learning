import numpy as np
import matplotlib.pyplot as plt

x = np.arange( 0, 0.8, 0.05 ) 
y = [0.508, 0.51, 0.508, 0.538, 0.704, 0.776, 0.82, 0.844, 0.85, 0.852, 0.862, 0.86, 0.858, 0.856, 0.858, 0.86]


plt.figure( figsize = (10,5) )
plt.plot( x, y, '-o' )
plt.xlabel( r'$\alpha$' ) 
plt.ylabel( 'Accuracy', rotation = 0, ha = 'right' ) 
plt.title( "Accuracy vs. Alpha" ) 
plt.show()