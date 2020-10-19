# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:23:50 2019

@author: ist
"""

import numpy as np
import matplotlib.pyplot as plt
import math


###Quad1
def quad1(x_o=-9.0,a=1.0,eta=0.1,threshold=0.01,maxiter=1000,anim=1):
    it = 0
    
    x1a = np.linspace(-10, 10, 20)
    
    y = (a*x1a**2)/2
    
    plt.clf()
    plt.plot(x1a,y)
    
    ###Gradient Method####
    while it != maxiter:
        fold = (a*(x_o**2))/2
        grad = a*x_o
        
        x_old = x_o
        
        x_o = x_o-eta*grad
        
        try:
            f = (a*(x_o**2))/2 
            
            if (f < threshold or fold < threshold):  
                
                break
            else:
                f = (a*(x_o**2))/2 
                if anim:
                    plt.plot([x_old, x_o],[fold, f],'r.-')
                    plt.pause(0.2)
                it += 1
        except:
            print('Diverged')
            break
        
    if it == maxiter:
        #print('Did not converge in %d steps, f = %f' %(it+1,f))
        plt.show()
        return it+1
    else:
        #print('Converged in %d steps, f = %f' %(it+1,f))
        plt.show()
        return it+1
    
etha_list=[0.001,0.01,0.03,0.1,0.3,1,3]
a_list=[0.5,1,2,5]
for etha in etha_list:
    for a in a_list:
        
        it=quad1(eta=etha, a=a, anim=0) 
        print('etha=', etha, 'a=', a, 'it=', it)
        

print('****STARTING PART 1****')
x_min = quad1(anim=1)
print('The estimated value is %f' %(x_min))

print('****STARTING PART 2****')
'''
###Quad2
def quad2(x_o=[-9.0,9.0],a=2.0,eta=0.1,threshold=0.01,maxiter=1000,alpha=0,anim = 1):
    it = 0
    x1 = np.linspace(-10,10,21)
    
    x2 = np.linspace(-10,10,21)
    
    [X,Y] = np.meshgrid(x1,x2)
    
    Y = (a*X**2+Y**2)/2
    
    plt.clf()
    plt.contour(Y,10)
    plt.xticks([0,5,10,15,20],[-10, -5, 0, 5, 10])
    plt.yticks([0,5,10,15,20],[-10, -5, 0, 5, 10])
    ax = plt.gca()
    ax.set_aspect('equal','box')
    
    plt.tight_layout()
    
    
    f = (a*x_o[0]**2+x_o[1]**2)/2
    
    varx = np.array([0,0])
    ###Gradient Method####
    while it != maxiter:
        fold = f
        
        grad = np.array([a*x_o[0], x_o[1]])
        
        varx = alpha*varx+(1-alpha)*grad
        x_old = np.asarray(x_o)

        x_o = np.asarray(x_o-eta*varx)
    
        try:
            f = (a*x_o[0]**2+x_o[1]**2)/2
            if (f < threshold or fold < threshold):
                break
            else:
                if anim:
                    plt.plot([x_old[0]+10, x_o[0]+10],[x_old[1]+10,x_o[1]+10],'r.-')
                    plt.pause(0.2)                    
                it += 1
        except:
            print('Diverged!')
            plt.show()
            break
        
    if it == maxiter:
        print('Did not converge in %d steps, f = %f' %(it,f))
        plt.show()
        return x_o
    else:
        print('Converged in %d steps, f = %f' %(it+1,f))
        plt.show()
        return x_o
    

x_min = quad2(anim = 1)
print('The estimated value is %s' %(x_min))

print('****STARTING PART 2 WITH MOMENTUM****')
x_min = quad2(a=20.0,eta=1,anim = 1,alpha=0.9)
print('The estimated value is %s' %(x_min))

def rosen(x_o=[-1.5,1.0],a=20.0,eta=0.001,threshold=0.001,maxiter=1000,alpha=0.0,anim = 1,up = 1,down = 1,reduce = 1):
    it = 0
    x1 = np.linspace(-2,2,201)
    
    x2 = np.linspace(-1,3,201)
    
    [X,Y] = np.meshgrid(x1,x2)
    
    Y = (1-X)**2 + a*(Y-X**2)**2
    
    v = np.linspace(math.floor(a/80)+3,Y.max(),math.floor(a))

    plt.clf()      
    plt.contour(Y,v)
    plt.xticks([0,50,100,150,200],[-2, -1, 0, 1, 2])
    plt.yticks([0,50,100,150,200],[-1, 0, 1, 2, 3])
    ax = plt.gca()
    ax.set_aspect('equal','box')
    
    plt.tight_layout()
    
    plt.plot(150,100,'b.')
    
    f = (1-x_o[0])**2+a*(x_o[1]-x_o[0]**2)**2
    fold = f
    minf = f

    gradold = np.array([0,0])
    
    eta1 = eta
    eta2 = eta
    
    varx = np.array([0.0,0.0])
    ###Gradient Method####
    while it != maxiter:
    
        grad = np.array([-2.0*(1-x_o[0])-4.0*a*(x_o[1]-x_o[0]**2)*x_o[0], 2.0*a*(x_o[1]-x_o[0]**2)])
        
        x_old = np.asarray(x_o)
 
        if (f>minf and reduce < 1):
            x_o[0] = minx1
            x_o[1] = minx2
            
            grad[0] = mingrad1
            grad[1]= mingrad2
            
            varx = np.array([0.0,0.0])
            
            eta1 = eta1*reduce
            eta2 = eta2*reduce
            
            gradold[0] = 0
            gradold[1] = 0
            
            fold = f
            f = minf
        else:
            minf = f
            
            minx1 = x_o[0]
            minx2 = x_o[1]
            
            mingrad1 = grad[0]
            mingrad2 = grad[1]
            
            if grad[0]*gradold[0] >0:
                eta1 = eta1*up
            else:
                eta1= eta1*down
            
            if grad[1]*gradold[1] >0:
                eta2 = eta2*up
            else:
                eta2 = eta2*down
                
            varx[0] = alpha*varx[0]-(1-alpha)*grad[0]
            varx[1] = alpha*varx[1]-(1-alpha)*grad[1]
            
            x_o[0] = x_o[0] + eta1*varx[0]
            x_o[1] = x_o[1] + eta2*varx[1]
            
            gradold = grad
            fold = f
 
        try:
            f = (x_o[0]-1)**2 + a*(x_o[1]-x_o[0]**2)**2
            if (f < threshold or fold < threshold):
                break
            else:
                if anim:
                    plt.plot([50*x_old[0]+100, 50*x_o[0]+100],[50*x_old[1]+50,50*x_o[1]+50],'r.-')
                    plt.xticks([0,50,100,150,200],[-2, -1, 0, 1, 2])
                    plt.yticks([0,50,100,150,200],[-1, 0, 1, 2, 3])
                    ax = plt.gca()
                    ax.set_aspect('equal','box')
                    plt.tight_layout()
                    plt.pause(0.1)
                it += 1
        except:
            print('Diverged!')
            plt.show()
            break
        
    if it == maxiter:
        print('Did not converge in %d steps, f = %f' %(it,f))
        plt.show()
        return x_o
    else:
        print('Converged in %d steps, f = %f' %(it+1,f))
        plt.show()
        return x_o
    
print('****STARTING Rosenbrock****')
x_min = rosen(x_o=[-1.5,1.0],a=20.0,eta=0.001,threshold=.001,maxiter=1000,alpha=0,anim = 1,up = 1,down = 1,reduce = 1)
print('The estimated value is %s' %(x_min))
'''