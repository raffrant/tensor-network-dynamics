import numpy as np
import matplotlib.pyplot as plt
import functools as ft
import itertools
import scipy as sp
import cmath
from scipy.optimize import *
def initial(n):
    basisstate=[]
    for i in range(n):
      #  if array[i]==0:
            basisstate.append([[1/np.sqrt(2)],[1/np.sqrt(2)]])
       # elif array[i]==1:
       #     basisstate.append([[0],[1]])
       # else:
       #     pass
    y=(ft.reduce(np.kron,basisstate))  
    return np.array(y).flatten()

def svdfinal(M,N):
    m=M.reshape(2,2**(N-1))
    u,s,v=np.linalg.svd(m,full_matrices=True)
    v=v[0:len(s),:] 
    u=u.dot(np.diag(s)/np.linalg.norm(s))
    allmatrices=[]
    allmatrices.append(u)
   
    if N>2:
     for i in range(N-3):
      u,s,v=svdmiddle(v)
      allmatrices.append(u)
    else:
        pass
    u,s,v=np.linalg.svd(v)
    v=v[0:len(s),:] 
    u=u.dot(np.diag(s)/np.linalg.norm(s))
    allmatrices.append(u)
    allmatrices.append(v)
  #  allmatrices.append(u.dot(v))
    psi=np.linalg.multi_dot(allmatrices)
        
    return allmatrices#,psi.flatten()#,allmatrices[1],allmatrices[2],allmatrices[3]

def svdmiddle(v):
    u1,s1,v1=np.linalg.svd(v)
    v1=v1[0:len(s1),:] 
    u1=u1.dot(np.diag(s1)/np.linalg.norm(s1))
    return u1,s1,v1

def cphaseforeveryedge(N,con,tar,theta):
    seq1=[]
    seq2=[]
    seq3=[]
    seq4=[]
    sz=np.array([[1,0],[0,-1]],dtype=complex)
    for i in range(N):
        if i!=con and i!=tar:
         seq1.append(np.eye(2))
         seq2.append(np.eye(2))
         seq3.append(np.eye(2))
         seq4.append(np.eye(2))

        if i==con:
            seq1.append(np.eye(2))
            seq2.append(sz)
            seq3.append(np.eye(2))
            seq4.append(sz)
        if i==tar:
            seq1.append(np.eye(2))
            seq2.append(np.eye(2))
            seq3.append(sz)
            seq4.append(sz)
    return (ft.reduce(np.kron,seq3)+ft.reduce(np.kron,seq2))*(1/4-np.exp(1j*theta)/4)+(3/4 + np.exp(1j*theta)/4)*ft.reduce(np.kron,seq1)+(-1/4 + np.exp(1j*theta)/4)*ft.reduce(np.kron,seq4)

#print(cphaseforeveryedge(2,0,1,0.2))
cz1=cphaseforeveryedge(2,0,1,np.pi)#.dot(np.ones(2**n)/np.linalg.norm(np.ones(2**n)))#*cphaseforeveryedge(3,1,2,np.pi)#*cphaseforeveryedge(3,1,2,np.pi)
#print(cz1)
def w(n,theta):
    b=[np.array([[1],[0]]),np.array([[0],[1]])]
    lst = list(itertools.product([0, 1], repeat=n))
    ball=[]
    for i in range(len(lst)):
          ball.append(ft.reduce(np.kron,[b[lst[i][j]] for j in range(n)]).flatten())
    new=[]      
    for i in range(len(ball)):
        new.append(np.outer(ball[i],ball[i]))
    new1=np.zeros((2**n,2**n),dtype=complex)    
    for i in range(len(new)):
      phase=1  
      for j in range(n):  
        phase*=np.exp(1j*(-1)**(lst[i][j]+1)*theta/2)
      new1+=phase*new[i]  
    return new1


def fidelitymatrices(u1,u2,num):
    m=np.matmul(u1,u2.conjugate().T)
    y=(1/(num*(num+1)))*(np.trace(np.matmul(m,m.conjugate().T))+abs(np.trace(m))**2)
        #np.trace((u1.dot(u2.conjugate().T)).dot((u1.dot(u2.conjugate().T)).conjugate().T))+\
        #                 abs(np.trace((u1.dot(u2.conjugate().T))))**2)
    return y
def generalrot(nx,ny,th):
    sxoper=np.array([[0,1],[1,0]],dtype=complex)
    syoper=np.array([[0,-1j],[1j,0]],dtype=complex)
    szoper=np.array([[1,0],[0,-1]],dtype=complex)
    u=sp.linalg.expm(-1j*(th/2)*(np.sin(nx)*np.cos(ny)*sxoper+np.sin(nx)*np.sin(ny)*syoper+np.cos(nx)*szoper))
    return u 
def allgeneral(nxall,nyall,thall,s):
    uall=[]
    for i in range(s):
      uall.append(generalrot(nxall[i],nyall[i],thall[i]))
      
    return ft.reduce(np.kron,uall)    

def aftermeas(w1,th,fi,theta,n):
    
    return ((np.exp(-1j*n*theta/2)*np.cos(th)*np.eye(2**n)+np.exp(1j*fi)*np.sin(th)*w1))
#n=4
#aaa=aftermeas(w(n,0.2), np.pi/4, 0.9, 0.2, n).dot(initial(n))

#print(svdfinal(aaa, n))
def constraint1(x):
    return -x[-1]+np.pi
def constraint2(x):
    return x[-1]+np.pi
def constraint3(x):
    return x[-2]+np.pi/4
def constraint4(x):
    return -x[-2]+np.pi/4
con1 = {'type': 'ineq', 'fun': constraint1}
con2 = {'type': 'ineq', 'fun': constraint2}
con3 = {'type': 'ineq', 'fun': constraint3}
con4 = {'type': 'ineq', 'fun': constraint4}
cons=[con1,con2,con3,con4]
aafid4=[]
aafid5=[]
thetaall=[]
fiall=[]
ii=0
n=4
for theta in np.linspace(0,np.pi,100):
 def a(x):

    n=4
    m=4
    #print(n)
    cphall=cphaseforeveryedge(4, 1, 2, theta).dot(cphaseforeveryedge(4, 2,3, theta).dot(cphaseforeveryedge(4, 0, 3, theta)))
    instate=ft.reduce(np.kron,[[[1/np.sqrt(2)],[0],[0],[1/np.sqrt(2)]],initial(2)]).flatten()
   # print(instate)
    tellmps=(allgeneral(x[0:n],x[n:n+n],x[n+n:n+n+n],n).dot(cphall.dot(aftermeas(w(n,theta), x[n+n+n], x[n+n+n+1], theta, n).dot(instate))))
    #print(tellmps,np.pi-0.3*np.pi)
    cz1=cphaseforeveryedge(m,0,1,np.pi)
    cz2=cphaseforeveryedge(m,1,2,np.pi)
    cz3=cphaseforeveryedge(m,2,3,np.pi)
   # cz4=cphaseforeveryedge(m,3,4,np.pi)
    final=cz3.dot(cz2.dot(cz1.dot(initial(m))))#cz4.dot(cz3.dot(cz2.dot))
    #print(final)
    #psi=tellmps[0]
    #for i in range(1,n-1):
    #    psi=np.matmul(psi,tellmps[i])
     #   print(i)
    #y,s,v=np.linalg.svd(psi)#aftermeas(w(n,theta), x[0], x[1], theta, n).dot(initial(n)).reshape(2,2**(n-1)))

    #s=s/np.linalg.norm(s)
    #print(s)
    #en=0
    #for i in range(len(s)):
    #    en-=abs(s[i])**2*np.log2(abs(s[i])**2)
    return 1-abs(np.vdot(tellmps,final))**2

 yall=minimize(a, x0=np.random.normal(loc=0.2*np.pi,scale=0.05*np.pi,size=3*n+2),constraints=cons)
 yall1=(minimize(a, x0=np.random.normal(loc=0.2*np.pi,scale=0.05*np.pi,size=3*n+2)))
 aafid4.append(yall.fun)
 aafid5.append(yall1.fun)
 thetaall.append(yall.x[-2])
 fiall.append(yall.x[-1])
 print(yall.fun,yall.x[-1],yall.x[-2],ii)
 ii+=1

#plt.plot(np.linspace(0,np.pi,100),aafid)
#plt.plot(np.linspace(0,np.pi,100),aafid2)
plt.plot(np.linspace(0,np.pi,100),aafid4)
#plt.plot(np.linspace(0,np.pi,100),thetaall)
#plt.plot(np.linspace(0,np.pi,100),fiall)
plt.plot(np.linspace(0,np.pi,100),aafid5)
plt.show()
