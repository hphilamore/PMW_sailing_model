from __future__ import division                 #to avoid integer devision problem
import scipy
import pylab

#just for fun making further development easier and with joy
pi     = scipy.pi
dot    = scipy.dot
sin    = scipy.sin
cos    = scipy.cos
ar     = scipy.array
rand   = scipy.rand
arange = scipy.arange
plot   = pylab.plot
show   = pylab.show
axis   = pylab.axis
grid   = pylab.grid
title  = pylab.title
rad    = lambda ang: ang*pi/180                 #lovely lambda: degree to radian

#the function
def Rotate2D(pts,cnt,ang=pi/4):
    '''pts = {} Rotates points(nx2) about center cnt(2) by angle ang(1) in radian'''
    return dot(pts-cnt,ar([[cos(ang),sin(ang)],[-sin(ang),cos(ang)]]))+cnt

    return dot(pts-cnt,
    		ar([[cos(ang),sin(ang)],
    			[-sin(ang),cos(ang)]])
    		)+cnt

#the code for test
pts = ar([[0,0],[1,0],[1,1],[0.5,1.5],[0,1]])
plot(*pts.T,lw=5,color='k')                     #points (poly) to be rotated
for ang in arange(0,2*pi,pi/8):
    ots = Rotate2D(pts,ar([0.5,0.5]),ang)       #the results
    plot(*ots.T)
axis('image')
grid(True)
title('Rotate2D about a point')
show()