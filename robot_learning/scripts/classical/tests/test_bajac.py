""" 
mostly a dump for testing BA jacobian.
"""

import numpy as np
import sympy
from sympy import pprint
from sympy import Symbol, Matrix, MatrixSymbol, simplify

# parameters
x = Symbol('x')
y = Symbol('y')
h = Symbol('h')
lx,ly,lz = [Symbol(e) for e in ['lx','ly','lz']]

c = sympy.cos(h)
s = sympy.sin(h)

fx,fy,cx,cy = [Symbol(e) for e in ['fx','fy','cx','cy']]
K = Matrix([
    [fx,0,cx],
    [0,fy,cy],
    [0,0,1]])

rxx = sympy.MatrixSymbol('R', 3, 3)
txx = sympy.MatrixSymbol('t', 3, 1)

R_b2c = Matrix(rxx)
t_b2c = Matrix(txx)
P_b2c = R_b2c.row_join(t_b2c)

R_c2b = R_b2c.T
t_c2b = R_b2c.T * t_b2c
T_c2b = R_c2b.row_join(t_c2b).col_join(Matrix([[0,0,0,1]]))

lmk  = Matrix([[lx],[ly],[lz],[1]])

T = Matrix([
    [c,s,0,-y*s-x*c],
    [-s,c,0,x*s-y*c],
    [0,0,1,0],
    [0,0,0,1]
    ])

# d (pt) / d(params) = d(pt) / d(pt_h) * d(pt_h) / d(params)

# pt = (x,y), pt_h = (a,b,c) = (x*s,y*s,s)
# pt = (x/s,y/s), pt_h = (x,y,s)

# d(pt) / d(pt_h) = 
# dx/da, dx/db, dx/dc,
# dy/da, dy/db, dy/dc]
# = [[1/s, 0, -x/s^2],
#    [0, 1/s, -y/s^2]]

pt_h = Matrix(K*P_b2c*T*T_c2b*lmk) # homogeneous 
hx, hy, hs = pt_h[:,0]

pt   = Matrix(pt_h[:2]) / Matrix(pt_h[2:])

J  = pt.jacobian([x,y,h,lx,ly,lz]) # straight jacobian ("right" answer)
print J.shape
#print J[:,0]

pt2 = Matrix(T * T_c2b * lmk) # T and lmk are the x-y params
J2  = K * P_b2c * pt2.jacobian([x,y,h,lx,ly,lz])
#print J2[:,0]
print 'J2', J2.shape

Jl = Matrix(
        [[1/hs, 0, -hx/(hs*hs)],
        [0, 1/hs, -hy/(hs*hs)]])
Jx = Jl * J2 # constructed multi-part jacobian with two parts

dJ = (J - Jx)

# scalar
syms = [x,y,h,fx,fy,cx,cy,lx,ly,lz]
vals = [1000*np.random.normal() for e in syms]
# matrix
msyms = [rxx, txx]
mvals  = [Matrix(np.random.normal(size=e.shape)) for e in msyms]

syms += msyms
vals += mvals

s = dJ.subs({k:v for (k,v) in zip(syms, vals)}).doit()
#s.simplify()
print s

#print dJ

J3 = pt2.jacobian([x,y,h,lx,ly,lz])
# J3[2,:] = 0
J3.simplify()
print J3.shape
#print J3
#pprint(J3)


#print J3
#print J.shape
#print 'J0'
#print J
#J.simplify()
#print 'J1'
#print J
