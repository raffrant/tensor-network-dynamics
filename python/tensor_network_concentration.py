import numpy as np
import qutip
import functools as ft
from scipy.linalg import svdvals
from sympy import Symbol,Matrix,eye,sin,cos
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.optimize import minimize
def initial(array):
    basisstate=[]
    tensorized=[]
    for i in range(np.shape(array)[0]):
        if array[i]==0:
            basisstate.append([[1],[0]])
        elif array[i]==1:
            basisstate.append([[0],[1]])
        else:
            pass
    tensorized.append(ft.reduce(np.kron,basisstate))    
    return np.array((tensorized[0]).flatten())

def initialplus(n):
    basisstate=[]
    tensorized=[]
    for i in range(n):
            basisstate.append([[1/np.sqrt(2)],[1/np.sqrt(2)]])
    
    tensorized.append(ft.reduce(np.kron,basisstate))    
    return np.array((tensorized[0]).flatten())

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
    psi=np.linalg.multi_dot(allmatrices)

  #  allmatrices.append(u.dot(v))
    #for i in range(len(allmatrices)):
    #    if i==len(allmatrices)-1:
    #allmatrices[len(allmatrices)-1]=allmatrices[len(allmatrices)-1][:,0:len(s)]
            
    return allmatrices,psi.flatten()#,allmatrices[1],allmatrices[2],allmatrices[3]

def svdmiddle(v):
    u1,s1,v1=np.linalg.svd(v)
    v1=v1[0:len(s1),:] 
    u1=u1.dot(np.diag(s1)/np.linalg.norm(s1))
    return u1,s1,v1

def contract(m1,m2):
    return np.einsum('ij,kl->il',m1,m2)

def multipleindices(m1,m2):
    return np.einsum('ij,kl->ijkl',m1,m2)

def applygate(u,mtel):
    return np.einsum('ijkl,il->jk',u,mtel)

def calculateoperators(u,psi):
    y=applygate(u, psi)
    y1=np.einsum('ik,ki->',psi.conj(),y)
    return y1

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
'''
ao=initial(np.zeros(7))
a1,full=(svdfinal(ao, 7))


i1i2=multipleindices(a1[0], a1[1])
i1i2=np.einsum('ijkl->ijk',i1i2)
i3i4=multipleindices(a1[2], a1[3])
i3i4=np.einsum('ijkl->ijk',i3i4)
i5i6=multipleindices(a1[4], a1[5])
i5i6=np.einsum('ijkl->ijk',i5i6)


ss=qutip.rand_ket(2**4)

aa=np.array(ss)
aa=aa.reshape(2,2,2,2)
print(aa)

aa=np.einsum('ijkl->ikjl',aa)
print(aa)
'''

plus=(qutip.basis(2,0)+qutip.basis(2,1))/np.sqrt(2)
aa=qutip.tensor(plus,plus,plus)
bb=qutip.tensor(plus,plus,plus,plus)
cc=qutip.tensor(plus,plus,plus)
dd=qutip.tensor(plus,plus,plus)
d=2
i1i2=np.array(aa).reshape(d,d,d)
i5i6=np.array(bb).reshape(d,d,d,d)
i7=np.array(cc).reshape(d,d,d)
i3i4=np.array(dd).reshape(d,d,d)

onecontr=(np.einsum('ijk,klmn->ijlmn',i1i2,i5i6))
twocontr=(np.einsum(onecontr,[0,1,2,3,4],i7,[4,5,6],[0,1,2,3,5,6]))
threecontr=(np.einsum(twocontr,[0,1,2,3,4,5],i3i4,[5,6,7],[0,1,2,3,4,6,7]))
threecontr=(np.einsum(threecontr,[0,1,2,3,4,5,6],[0,1,2,3,4,5]))
threecontr=threecontr/np.linalg.norm(threecontr.reshape(2**6,1))
sz=np.array([[1,0],[0,-1]])
sx=np.array([[0,1],[1,0]])
sy=np.array([[0,-1j],[1j,0]])
cz=cphaseforeveryedge(2,0,1,theta).reshape(2,2,2,2)
#sz1=e.reshape(2,2,2,2)
#i1i2=np.einsum('ij,j'), m2)
#print(threecontr)
def generalrot(nx,ny,th):
    sxoper=np.array([[0,1],[1,0]],dtype=complex)
    syoper=np.array([[0,-1j],[1j,0]],dtype=complex)
    szoper=np.array([[1,0],[0,-1]],dtype=complex)
    u=expm(-1j*(th/2)*(np.sin(nx)*np.cos(ny)*sxoper+np.sin(nx)*np.sin(ny)*syoper+np.cos(nx)*szoper))
    return u 
def measugeneral(nx,ny):
    sxoper=np.array([[0,1],[1,0]],dtype=complex)
    syoper=np.array([[0,-1j],[1j,0]],dtype=complex)
    szoper=np.array([[1,0],[0,-1]],dtype=complex)
    u=(np.eye(2)+np.sin(nx)*np.cos(ny)*sxoper+np.sin(nx)*np.sin(ny)*syoper+np.cos(nx)*szoper)/2
    return u 
def allgeneral(nxall,nyall,thall,s):
    uall=[]
    for i in range(s):
      uall.append(generalrot(nxall[i],nyall[i],thall[i]))
      
    return ft.reduce(np.kron,uall)    


def measugeneralsympy(nx,ny,th):
    sxoper=Matrix([[0,1],[1,0]])
    syoper=Matrix([[0,-1j],[1j,0]])
    szoper=Matrix([[1,0],[0,-1]])
    u=Matrix((eye(2)+sin(nx)*cos(ny)*sxoper+sin(nx)*sin(ny)*syoper+cos(nx)*szoper)/2)
    return u 
nx=Symbol('th1')
ny=Symbol('fi1')
th=Symbol('theta')
print(measugeneralsympy(nx, ny, th))

def ham(u,arx,i,j,qub):
    ya=np.arange(qub)
    for k in range(len(ya)):
        if ya[k]==i:
            ya[k]=i
        if ya[k]==j:
            ya[k]=j    
    return (np.einsum(u,[i,qub,qub+1,j],arx,range(qub),ya))

def hamsinglue(u,arx,i,qub):
    ya=np.arange(qub)
    for k in range(len(ya)):
        if ya[k]==i:
            ya[k]=i
    return (np.einsum(u,[i,qub],arx,range(qub),ya))

def swap1(u1,arx2,qub):
    return np.einsum(u1,range(qub),arx2)

def sing(final,a1,n,q):
   if a1==None: 
       y1=final 
   else:
       y1=swap1(final,a1,q)
   z= svdvals(y1.reshape(2**n,2**(q-n)))
   z=z/np.linalg.norm(z)
   #print(z)
   en=0
   #numsing=0
   for i in range(len(z)):
       en-=abs(z[i])**2*np.log2(abs(z[i])**2)
   #for i in range(len(z)):
   #    if z[i]>0.02:
   #        numsing+=1
   return z,en#,numsing  
#print(threecontr.reshape(2**7,1))  
for i in range(5):
    if i==0:
        
      finalsz1=ham(cz,threecontr,i,i+1,6)#(np.einsum(sz1,[0,7,8,1],threecontr,range(7),ya))
    else:
      finalsz1=ham(cz,finalsz1,i,i+1,6)#(np.einsum(sz1,[0,7,8,1],threecontr,range(7),ya))
finalsz1=ham(cz,finalsz1,0,5,6)#(np.einsum(sz1,[0,7,8,1],threecontr,range(7),ya))
        
#finalsz1=ham(cz,threecontr,1,1)#(np.einsum(sz1,[0,7,8,1],threecontr,range(7),ya))

#print(finalsz1.reshape(2**7,1))#.reshape(2**7,1))
#finalsz1=swap1(finalsz1, [0,1,2,3,4,5], 6)
finalsz1=ham(cz,finalsz1,2,5,6)#(np.einsum(sz1,[0,7,8,1],threecontr,range(7),ya))
finalsz1=ham(cz,finalsz1,3,5,6)#(np.einsum(sz1,[0,7,8,1],threecontr,range(7),ya))
finalsz1=ham(cz,finalsz1,4,5,6)#(np.einsum(sz1,[0,7,8,1],threecontr,range(7),ya))
#finalsz1=swap1(finalsz1, [0,1,2,3,4,5], 6)
#finalsz1=swap1(finalsz1, [0,1,4,3,2,5], 6)
#finalsz1=ham(cz,finalsz1,4,5,6)#(np.einsum(sz1,[0,7,8,1],threecontr,range(7),ya))
#finalsz1=swap1(finalsz1, [0,1,2,3,4,5], 6)
#finalsz1=ham(cz,finalsz1,4,5,6)#(np.einsum(sz1,[0,7,8,1],threecontr,range(7),ya))


cph1=cphaseforeveryedge(6, 0, 1,theta)
for i in range(1,6):
    print(i)
    cph1*=cphaseforeveryedge(6, i, i+1,theta)
cph1*=cphaseforeveryedge(6, 0, 5,theta)
cph1*=cphaseforeveryedge(6, 2, 5,theta)  
cph1*=cphaseforeveryedge(6, 3, 5,theta)  
cph1*=cphaseforeveryedge(6, 4, 5,theta)   
#print(list(range(1,7)))    
finalcz7qub=np.matmul(cph1,initialplus(6))

#cph2=cphaseforeveryedge(4, 0, 2,np.pi)
#cph2*=cphaseforeveryedge(4, 0, 3,np.pi)
#cph2*=cphaseforeveryedge(4, 1, 2,np.pi)
#cph2*=cphaseforeveryedge(4, 1, 3,np.pi)
#cph2*=cphaseforeveryedge(4, 2, 3,np.pi)

cph2=cphaseforeveryedge(4, 0, 1,np.pi)
cph2*=cphaseforeveryedge(4, 1, 3,np.pi)
cph2*=cphaseforeveryedge(4, 2, 3,np.pi)


#for i in range(1,4):
    #print(i)
    #cph2*=cphaseforeveryedge(4, i, i+1,np.pi)
#cph1*=cphaseforeveryedge(6, 0, 5,theta)
#cph1*=cphaseforeveryedge(6, 2, 5,theta)  
#cph1*=cphaseforeveryedge(6, 3, 5,theta)  
#cph1*=cphaseforeveryedge(6, 4, 5,theta)   
#print(list(range(1,7)))    
cluster4=np.matmul(cph2,initialplus(4))
#print(finalsz1.reshape(2**6,1))
#print(sing(finalsz1,[0,1,5,2,3,4,6],3,7))
#print(finalsz1.reshape(2**6,1))
#print(abs(np.vdot(finalcz7qub,finalsz1.reshape(2**6,1))**2))  
def constraint1(x):
    return -x[0]+np.pi/4
def constraint2(x):
    return x[0]+np.pi/4
def constraint5(x):
    return -x[2]+np.pi/4
def constraint6(x):
    return x[2]+np.pi/4
def constraint7(x):
    return x[3]
def constraint8(x):
    return -x[3]+2*np.pi
def constraint3(x):
    return x[1]
def constraint4(x):
    return -x[1]+2*np.pi
con1 = {'type': 'ineq', 'fun': constraint1}
con2 = {'type': 'ineq', 'fun': constraint2}
con3 = {'type': 'ineq', 'fun': constraint3}
con4 = {'type': 'ineq', 'fun': constraint4}
con5 = {'type': 'ineq', 'fun': constraint5}
con6 = {'type': 'ineq', 'fun': constraint6}
con7 = {'type': 'ineq', 'fun': constraint7}
con8 = {'type': 'ineq', 'fun': constraint8}
cons=[con1,con2,con3,con4,con5,con6,con7,con8]
aafid4=[]
thetaall1=[]
thetaall2=[]
fiall=[]
for theta in np.linspace(0,np.pi,100):
 cz=cphaseforeveryedge(2,0,1,theta).reshape(2,2,2,2)   
 for i in range(5):
    if i==0:
        
      finalsz1=ham(cz,threecontr,i,i+1,6)#(np.einsum(sz1,[0,7,8,1],threecontr,range(7),ya))
    else:
      finalsz1=ham(cz,finalsz1,i,i+1,6)#(np.einsum(sz1,[0,7,8,1],threecontr,range(7),ya))
 finalsz1=ham(cz,finalsz1,0,5,6) 
 finalsz1=ham(cz,finalsz1,2,5,6)#(np.einsum(sz1,[0,7,8,1],threecontr,range(7),ya))
 finalsz1=ham(cz,finalsz1,3,5,6)#(np.einsum(sz1,[0,7,8,1],threecontr,range(7),ya))
 finalsz1=ham(cz,finalsz1,4,5,6)
 def a(x):
    global finalsz1
    n=4
    meas1=measugeneral(x[0], x[1]) 
    meas2=measugeneral(x[2],x[3])#np.pi/4,np.pi-theta)#,np.eye(2))
    finalsz1=hamsinglue(meas1,finalsz1,1,6)
    new=np.einsum(finalsz1,[0,1,2,3,4,5],[0,2,3,4,5])
    new=new/np.linalg.norm(new.reshape(2**5,1))
    new=hamsinglue(meas2,new,4,5)    
    new=np.einsum(new,[0,1,2,3,4],[0,1,2,3])
    new=new/np.linalg.norm(new.reshape(2**4,1))    
    #print(new.reshape(2**4,1))
    tel=allgeneral(x[4:n+4],x[n+4:n+n+4],x[n+n+4:n+n+n+4],n).dot(new.reshape(2**n,1))
    #print(new.reshape(2**4,1))
    allstate,en=sing(new,None,1,4)
    allstate1,en1=sing(new,None,2,4)
    allstate2,en2=sing(new,None,3,4)
    
    
    return 1-abs(np.vdot(tel,cluster4))**2#1-abs(en)-abs(en1)-abs(en2)
 arx=np.random.normal(loc=theta,scale=0.25*np.pi,size=12)
 meas=np.random.normal(loc=0,scale=0.0*np.pi,size=4)
 yall1=minimize(a, x0=np.concatenate((meas, arx), axis=0),constraints=cons)   
 aafid4.append(yall1.fun)
 thetaall1.append(yall1.x[12])
 thetaall2.append(yall1.x[14])
 fiall.append(yall1.x[13])
 print(yall1.fun,theta/np.pi)

#plt.plot(np.linspace(0,np.pi,100),thetaall1)
#plt.plot(np.linspace(0,np.pi,100),thetaall2)
#plt.plot(np.linspace(0,np.pi,100),fiall)
plt.plot(np.linspace(0,1,100),aafid4)
plt.yscale('log')
plt.show()

