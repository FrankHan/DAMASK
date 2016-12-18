#! /usr/bin/python
# Yld2000-2d + homogeneous anisotropic hardening model(HAH) fit
# F. Barlat et.al. An alternative to kinematic hardening in classical plasticity
# F. Barlat et.al. Extension of homogeneous anisotropic hardening model to cross-loading with latent effects
# F. Barlat et.al. Enhancements of homogenous anisotropic hardening model and application to mild and dual-phase steels
#@author Zhou Hui hzhou@ethz.ch

import numpy as np
import scipy.optimize as opt
#import scipy.integrate as integrate
import scipy.interpolate as interp
import math
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import collections


def Yld2000_phi(alpha,a,S):
    # plane stress
    # a=8 for fcc 6 for bcc
    (alpha1,alpha2,alpha3,alpha4,alpha5,alpha6,alpha7,alpha8)=alpha
    sxx,syy,szz,syz,sxz,sxy = S

    C1_11 = alpha1
    C1_22 = alpha2
    C1_66 = alpha7
    C2_11 = 1.0/3.0*(4*alpha5 -   alpha3)
    C2_12 = 1.0/3.0*(2*alpha6 - 2*alpha4)
    C2_21 = 1.0/3.0*(2*alpha3 - 2*alpha5)
    C2_22 = 1.0/3.0*(4*alpha4 -   alpha6)
    C2_66 = alpha8
    
    # X',X''
    X1_xx = C1_11*sxx
    X1_yy = C1_22*syy
    X1_xy = C1_66*sxy
    X2_xx = C2_11*sxx + C2_12*syy
    X2_yy = C2_21*sxx + C2_22*syy
    X2_xy = C2_66*sxy
    # X1',X2'
    X1_11 = 1.0/2*(X1_xx+X1_yy+np.sqrt((X1_xx-X1_yy)**2+4*X1_xy**2))
    X1_22 = 1.0/2*(X1_xx+X1_yy-np.sqrt((X1_xx-X1_yy)**2+4*X1_xy**2))
    # X1'',X2''
    X2_11 = 1.0/2*(X2_xx+X2_yy+np.sqrt((X2_xx-X2_yy)**2+4*X2_xy**2))
    X2_22 = 1.0/2*(X2_xx+X2_yy-np.sqrt((X2_xx-X2_yy)**2+4*X2_xy**2))
    # phi',phi'',phi
    Phi1 = abs(X1_11-X1_22)**a
    Phi2 = abs(2*X2_22+X2_11)**a+abs(2*X2_11+X2_22)**a
    Phi = Phi1 + Phi2
##        print 'a_eff:',(Phi/2.)**(1./a)
    return (Phi/2.0)**(1.0/a)

def HAH_phi(phi_s,C,a,g1,g2,gL,gS,S,H,q,kp): 
    # phi_s: stable yield function
    # Yld2000_phi(C,a,S)
    
    H_h = H/np.sqrt(8.0/3.0*np.dot(H,H))
    S_h = S/np.sqrt(8.0/3.0*np.dot(S,S))
    Sc = 8.0/3.0*np.dot(H_h,S)*H_h
    So = S - Sc
    SH = np.dot(H_h,S)
    # kp approx 4   
    phi = np.sqrt(phi_s(C,a,Sc+1.0/gL*So)**2 + phi_s(C,a,kp*(1-gS)/gL*So)**2)
    f1 = (1.0/(g1**q)-1)**(1.0/q)
    f2 = (1.0/(g2**q)-1)**(1.0/q)
    Phi = (phi**q + f1**q*abs(SH-abs(SH))**q + f2**q*abs(SH+abs(SH))**q)**(1.0/q)
    return Phi

def HAH_dx(eps,x,params):
    S,k1,k2,k3,k4,k5,L,kL,Sc,kS,k,R,kR,kRp,A,B,C,D = params
    
    g1,g2,g3,g4,gL,gS,gR,h1,h2,h3,h4,h5,h6 = x
    
    S_h = S/np.sqrt(8.0/3.0*np.dot(S,S))
    S_h1,S_h2,S_h3,S_h4,S_h5,S_h6 = S_h

    H_h = np.array([h1,h2,h3,h4,h5,h6])
    H_h = H_h/np.sqrt(8.0/3.0*np.dot(H_h,H_h))
    h1,h2,h3,h4,h5,h6 = H_h
    
    # Hockett-Sherby
    def sig_f(eps,A,B,C,D):
        sig = A + B*(1 - np.exp(-C*eps**D))
        return sig
    
    sig0 = sig_f(0,A,B,C,D)
    sig = sig_f(eps,A,B,C,D)

    cosx = 8.0/3.0*np.dot(H_h,S_h)

    #print("cosx:",cosx)
    if np.dot(S,H_h)>=0:
        dg1 = k2*(k3*sig0/sig - g1)
        dg2 = k1*(g3 - g2)/g2
        dg3 = 0 
        dg4 = k5*(k4 - g4)
        dh1 = k*(abs(cosx)**(1.0/R)+gR)*(S_h1 - cosx*h1)
        dh2 = k*(abs(cosx)**(1.0/R)+gR)*(S_h2 - cosx*h2)
        dh3 = k*(abs(cosx)**(1.0/R)+gR)*(S_h3 - cosx*h3)
        dh4 = k*(abs(cosx)**(1.0/R)+gR)*(S_h4 - cosx*h4)
        dh5 = k*(abs(cosx)**(1.0/R)+gR)*(S_h5 - cosx*h5)
        dh6 = k*(abs(cosx)**(1.0/R)+gR)*(S_h6 - cosx*h6)
        
    else:
        dg1 = k1*(g4 - g1)/g1
        dg2 = k2*(k3*sig0/sig - g2)
        dg3 = k5*(k4 - g3)
        dg4 = 0
        dh1 = -k*(abs(cosx)**(1.0/R)+gR)*(S_h1 - cosx*h1)
        dh2 = -k*(abs(cosx)**(1.0/R)+gR)*(S_h2 - cosx*h2)
        dh3 = -k*(abs(cosx)**(1.0/R)+gR)*(S_h3 - cosx*h3)
        dh4 = -k*(abs(cosx)**(1.0/R)+gR)*(S_h4 - cosx*h4)
        dh5 = -k*(abs(cosx)**(1.0/R)+gR)*(S_h5 - cosx*h5)
        dh6 = -k*(abs(cosx)**(1.0/R)+gR)*(S_h6 - cosx*h6)
    # R approx 5
##    dh1 = np.sign(cosx)*k*(abs(cosx)**(1.0/R)+gR)*(S_h1 - cosx*h1)
##    dh2 = np.sign(cosx)*k*(abs(cosx)**(1.0/R)+gR)*(S_h2 - cosx*h2)
##    dh3 = np.sign(cosx)*k*(abs(cosx)**(1.0/R)+gR)*(S_h3 - cosx*h3)
##    dh4 = np.sign(cosx)*k*(abs(cosx)**(1.0/R)+gR)*(S_h4 - cosx*h4)
##    dh5 = np.sign(cosx)*k*(abs(cosx)**(1.0/R)+gR)*(S_h5 - cosx*h5)
##    dh6 = np.sign(cosx)*k*(abs(cosx)**(1.0/R)+gR)*(S_h6 - cosx*h6)

    # kRp approx 0.2
    # kR approx 15
    dgR = kR*(kRp*(1-cosx**2)-gR)
    
    # L,kL
    dgL = kL*((sig-sig0)/sig*(np.sqrt(L*(1-cosx**2)+cosx**2)-1)+1-gL)
    dgS = kS*(1+(Sc-1)*cosx**2-gS)
    #print("dg1,dg2,dg3,dg4:",dg1,dg2,dg3,dg4)
    #print("dh1,dh2,dh3,dh4,dh5,dh6:",dh1,dh2,dh3,dh4,dh5,dh6)
    return np.array([dg1,dg2,dg3,dg4,dgL,dgS,dgR,dh1,dh2,dh3,dh4,dh5,dh6])

def HAH_x(eps,S,alpha,a,k1,k2,k3,k4,k5,L,kL,Sc,kS,k,R,kR,kRp,A,B,C,D,q,kp):
    for i,v in enumerate(eps):
        if v>=2e-3:
            #print("%d S0: "%(i),S[i])
            break
    S0 = S[i]
    eps0 = eps[i]
    h1,h2,h3,h4,h5,h6 = S0/np.sqrt(8.0/3.0*np.dot(S0,S0))

    deps = np.diff(eps)
    
    g1 = 1.0
    g2 = 1.0
    g3 = 1.0
    g4 = 1.0
    gL = 1.0
    gS = 1.0
    gR = 0.0
    
    x0 = [g1,g2,g3,g4,gL,gS,gR,h1,h2,h3,h4,h5,h6]

    

    x = np.array(x0)
    #res = np.zeros((len(deps),3))
    Sig = []
    Sig.append(Yld2000_phi(alpha,a,S[0]))
    for i,dt in enumerate(deps):
        x = x + HAH_dx(eps[i], x, (S[i],k1,k2,k3,k4,k5,L,kL,Sc,kS,k,R,kR,kRp,A,B,C,D))*abs(dt)
        g1,g2,g3,g4,gL,gS,gR,h1,h2,h3,h4,h5,h6 = x
        H = [h1,h2,h3,h4,h5,h6]
        sig = HAH_phi(Yld2000_phi,alpha,a,g1,g2,gL,gS,S[i],H,q,kp)
        Sig.append(sig)
    return np.asarray(Sig)

def HAH_state(eps,S,alpha,a,k1,k2,k3,k4,k5,L,kL,Sc,kS,k,R,kR,kRp,A,B,C,D,q,kp):
    for i,v in enumerate(eps):
        if v>=2e-3:
            #print("%d S0: "%(i),S[i])
            break
    S0 = S[i]
    eps0 = eps[i]
    h1,h2,h3,h4,h5,h6 = S0/np.sqrt(8.0/3.0*np.dot(S0,S0))

    deps = np.diff(eps)
    
    g1 = 1.0
    g2 = 1.0
    g3 = 1.0
    g4 = 1.0
    gL = 1.0
    gS = 1.0
    gR = 0.0
    
    x0 = [g1,g2,g3,g4,gL,gS,gR,h1,h2,h3,h4,h5,h6]

    

    x = np.array(x0)
    for i,dt in enumerate(deps):
        x = x + HAH_dx(eps[i], x, (S[i],k1,k2,k3,k4,k5,L,kL,Sc,kS,k,R,kR,kRp,A,B,C,D))*abs(dt)
        #g1,g2,g3,g4,gL,gS,gR,h1,h2,h3,h4,h5,h6 = x
        #H = [h1,h2,h3,h4,h5,h6]

    return np.asarray(x)

def HAH_obj(params,eps_t,S_t,alpha,a,eps_rt,sig_rt):
    k1,k2,k3,k4,k5,k, L,kL,Sc,kS, = params # R,kR,kRp,A,B,C,D,q,kp = params
    obj = np.zeros(len(eps_t[0]))
    #obj = 0
    # Hockett-Sherby
    A = 120.03
    B = 250.99
    C = 3.3
    D = 0.79
    # suggested value
    q = 2
    kp = 4
    R = 5
    kR = 15
    kRp = 0.2
    # k4 control permenant softening
    k4 = 1.0
    k5 = 0.0
##    k1 = 2.97125145e+02 #250 240
##    k2 = 1.51199617e+01 #10
##    k3 = 1.18601485e-01
##    L = 1.0
##    kL = 0.0
##    Sc = 1.0
##    kS = 0.0
    
    
    for i,eps in enumerate(eps_t):
        S = S_t[i]
        
        sig_r = sig_rt[i]
        eps_r = eps_rt[i]
        ref = interp.interp1d(eps_r,sig_r)
        # eps in eps_r
        sig_ref = ref(eps)
        
        sig = HAH_x(eps,S,alpha,a,k1,k2,k3,k4,k5,L,kL,Sc,kS,k,R,kR,kRp,A,B,C,D,q,kp)
        obj = sig - sig_ref
        #obj = obj + sum(np.square(sig*1.0/sig_ref-1))
        #obj = obj + np.sum(np.square(sig-sig_ref))
        
    return obj

def HAH_fit(params0,eps_t,S_t,alpha,a,eps_r,sig_r,bounds=None):
    res = opt.leastsq(HAH_obj,  x0 = params0, args=(eps_t,S_t,alpha,a,eps_r,sig_r),  ) # ,bounds=bounds,
    return res

def main():
    fn0 = "F:\MasterThesis3\DAMASK\examples\SpectralMethod\postProc\AL_0.05_0\\20grains8x8x8_16_ext3.txt"
    fn1 = "F:\MasterThesis3\DAMASK\examples\SpectralMethod\postProc\AL_0.05_36\\20grains8x8x8_1_ext3.txt"
    fn2 = "F:\MasterThesis3\DAMASK\examples\SpectralMethod\postProc\AL_0.05_45\\20grains8x8x8_2_ext3.txt"
    fn3 = "F:\MasterThesis3\DAMASK\examples\SpectralMethod\postProc\AL_0.05_72\\20grains8x8x8_3_ext3.txt"
    fn4 = "F:\MasterThesis3\DAMASK\examples\SpectralMethod\postProc\AL_0.05_90\\20grains8x8x8_4_ext3.txt"
    fn5 = "F:\MasterThesis3\DAMASK\examples\SpectralMethod\postProc\AL_0.05_108\\20grains8x8x8_5_ext3.txt"
    fn6 = "F:\MasterThesis3\DAMASK\examples\SpectralMethod\postProc\AL_0.05_135\\20grains8x8x8_6_ext3.txt"
    fn7 = "F:\MasterThesis3\DAMASK\examples\SpectralMethod\postProc\AL_0.05_144\\20grains8x8x8_7_ext3.txt"
    fn8 = "F:\MasterThesis3\DAMASK\examples\SpectralMethod\postProc\AL_0.05_180\\20grains8x8x8_8_ext3.txt"
    fn9 = "F:\MasterThesis3\DAMASK\examples\SpectralMethod\postProc\AL_0.05_216\\20grains8x8x8_9_ext3.txt"
    fn10 = "F:\MasterThesis3\DAMASK\examples\SpectralMethod\postProc\AL_0.05_225\\20grains8x8x8_10_ext3.txt"
    fn11 = "F:\MasterThesis3\DAMASK\examples\SpectralMethod\postProc\AL_0.05_252\\20grains8x8x8_11_ext3.txt"
    fn12 = "F:\MasterThesis3\DAMASK\examples\SpectralMethod\postProc\AL_0.05_270\\20grains8x8x8_12_ext3.txt"
    fn13 = "F:\MasterThesis3\DAMASK\examples\SpectralMethod\postProc\AL_0.05_288\\20grains8x8x8_13_ext3.txt"
    fn14 = "F:\MasterThesis3\DAMASK\examples\SpectralMethod\postProc\AL_0.05_315\\20grains8x8x8_14_ext3.txt"
    fn15 = "F:\MasterThesis3\DAMASK\examples\SpectralMethod\postProc\AL_0.05_324\\20grains8x8x8_15_ext3.txt"
    fn16 = "F:\MasterThesis3\DAMASK\examples\SpectralMethod\postProc\AL_0.05_t0\\20grains8x8x8_17_ext3.txt"
    fn17 = "F:\MasterThesis3\DAMASK\examples\SpectralMethod\postProc\AL_0.05_c0\\20grains8x8x8_18_ext3.txt"
    fn18 = "F:\MasterThesis3\DAMASK\examples\SpectralMethod\postProc\AL_0.05_t90\\20grains8x8x8_19_ext3.txt"
    fn19 = "F:\MasterThesis3\DAMASK\examples\SpectralMethod\postProc\AL_0.05_c90\\20grains8x8x8_20_ext3.txt"
    fn20 = "F:\MasterThesis3\DAMASK\examples\SpectralMethod\postProc\AL_tensionX_r\\20grains8x8x8_tensionX_0_ext.txt"
    fn21 = "F:\MasterThesis3\DAMASK\examples\SpectralMethod\postProc\AL_tensionX_c\\20grains8x8x8_tc_ext3less.txt"
    
    # 0 1 2 3 4 5 6 7 8 9 10
    
    fn = [fn0,  fn1, fn2, fn3, fn4, fn5, fn6, fn7, fn8, fn9, fn10, fn11, fn12, fn13, fn14, fn15, fn16, fn17, fn18, fn19, fn21, fn20,]
    data = {}
    for i,f in enumerate(fn):
        d = np.loadtxt(f)
        for j,v in enumerate(d):
            d[j][1:11] = v[1:11]/1e6 # Pa->MPa
        data[i] = d
        
    # Yld2000
    alpha = [ 1.14851097,  0.70809294,  0.87830038,  1.00319655,  0.98680459,   0.95239513,  1.     ,  1.   ]
    a = 8
    # Hockett Sherby
    A = 120.03
    B = 250.99
    C = 3.3
    D = 0.79
    
    # initial guess
    k1 = 2.97125145e+02 #250 240
    k2 = 1.51199617e+01 #10
    k3 = 1.18601485e-01 #0.10
    k4 = 1.0 #0.9 #1.0
    k5 = 0.0 #10.0
    k = 2.03467686e+01 # 30
    kS = 5.27882379e+01 #2.65999157e+01 #100.0 2.65999157e+01
    Sc = 1.07344094e+00 #1.23677137e+00 #1.35 1.23677137e+00

    q = 2
    kp = 4
    R = 5
    kR = 15
    kRp = 0.2

    kL = 1.12440583e+02 #5.35853538e+02#7.28794437e+01#300#0.0 #300
    L = 1.22586461e+00 #1.22650532e+00#1.38586389e+00#1.6 #1.0

    od = collections.OrderedDict(sorted(data.items()))
    eps_t = []
    Sig_t = []
    sig_rt = []
    S_t = []
    for k,v in od.items():
        eps = v[:,0]
        Sigmas = v[:,2:11]
        S = np.zeros((len(eps),6))
        for i,sigma in enumerate(Sigmas):
            sigm = 1.0/3.0*(sigma[0]+sigma[4]+sigma[8])
            # Voigt notation
            S[i][0] = sigma[0] - sigm
            S[i][1] = sigma[4] - sigm
            S[i][2] = sigma[8] - sigm
            S[i][3] = sigma[5]
            S[i][4] = sigma[2] 
            S[i][5] = sigma[1]
        sig_r = v[:,1]
        
        eps_t.append(eps)
        Sig_t.append(Sigmas)
        S_t.append(S)
        sig_rt.append(sig_r)
        
##    k1,k2,k3,k4,k5,k, = [  3.15215325e+02,   3.21058525e+01,   2.73366701e-01,
##         9.80386209e-01,   2.10440447e+01,   -1.59020294e+01]
##    k1,k2,k3,k4,k5,k, L,kL,Sc,kS = [ 348.03107055,   10.68050607,    0.69108352,    0.9895288 ,
##         11.07134851,   20.39396936,    1.        ,    0.        ,
##         -1.62146644,  100.6612326 ]
    
    params0 = [k1,k2,k3,k4,k5,k, L,kL,Sc,kS,] # R,kR,kRp,A,B,C,D,q,kp]
    i = 14
    bounds = [(20,3e2),(10,1e2),(0,1),(0.8,1),  (1,10),(1,150), (1,2),(1e2,1e3),(0.4,1),(1,1e2),]#(5,5),(15,15),(0.2,0.2),(0,1e3),(0,1e3),(0,1e2),(0,1),(2,2),(4,4)]
    res = HAH_fit(params0,[eps_t[i]],[S_t[i]],alpha,a,[eps_t[i]],[sig_rt[i]],bounds=None,)
    #res = HAH_fit(params0,np.asarray(eps_t[0:16:4]),np.asarray(S_t[0:16:4]),alpha,a,np.asarray(eps_t[0:16:4]),np.asarray(sig_rt[0:16:4]),bounds=None,)
    print(res)
    
    k1,k2,k3,k4,k5,k, L,kL,Sc,kS, = res[0] #[  2.97125145e+02,   1.51199617e+01,   1.18601485e-01,
         #1.00000000e+00,   0.00000000e+00,   2.32649882e+00,
         #1.25172518e+00,   2.43919657e+02,   1.12868746e+00,
         #1.28584785e+02]
        #[  2.97125145e+02,   1.51199617e+01,   1.18601485e-01,
         #1.00000000e+00,   0.00000000e+00,   0.0,#-1.59030226e+01,
         #1.00000000e+00,   0.00000000e+00,   1.00000000e+00,
         #0.00000000e+00]  #  R,kR,kRp,A,B,C,D,q,kp = res[0]
    
    def sig_f(eps,A,B,C,D):
        sig = A + B*(1 - np.exp(-C*eps**D))
        return sig
##    for i in [0,2,4,6,8,10,12,14]:
##        plt.plot(eps_t[i],sig_rt[i],)
##    for i in [0,4,8,12]:
##    plt.plot(eps_t[i],sig_rt[i],'.')
##    plt.plot(eps_t[i],HAH_x(eps_t[i],S_t[i],alpha,a,k1,k2,k3,k4,k5,L,kL,Sc,kS,k,R,kR,kRp,A,B,C,D,q,kp))
##    #plt.plot(eps_t[-1],sig_rt[-1])
##    #plt.plot(eps_t[i],sig_f(eps_t[i],A,B,C,D))
##    
##    plt.xlabel("$\epsilon^{p}$")
##    plt.ylabel("$\sigma_{e}$")
##    plt.legend(["Virtual test","HAH fit",],loc="lower right")
##    #plt.legend(["$0^\circ$","$45^\circ$","$90^\circ$","$135^\circ$","$180^\circ$","$225^\circ$","$270^\circ$","$315^\circ$"],loc='lower right')
##    #plt.legend(["$0^\circ$","$90^\circ$","$180^\circ$","$270^\circ$"],loc='lower right')
##    plt.grid(True)
##    plt.show()
    
    i_p = i
    HAH_points = []
    for j,v in enumerate(eps_t[i]):
        if v >= 0.125:
            break
    x = HAH_state(eps_t[i][:j],S_t[i][:j],alpha,a,k1,k2,k3,k4,k5,L,kL,Sc,kS,k,R,kR,kRp,A,B,C,D,q,kp)
    g1,g2,g3,g4,gL,gS,gR,h1,h2,h3,h4,h5,h6 = x
    H = [h1,h2,h3,h4,h5,h6]
    sig = HAH_phi(Yld2000_phi,alpha,a,g1,g2,gL,gS,S_t[i][j],H,q,kp)
    #print(sig)

    for theta in np.linspace(0,2*np.pi,200):
        g = lambda Y: HAH_phi(Yld2000_phi,alpha,a,g1,g2,gL,gS,[Y*np.cos(theta)-1/3.0*(Y*np.cos(theta)+Y*np.sin(theta)),Y*np.sin(theta)-1/3.0*(Y*np.cos(theta)+Y*np.sin(theta)),-1/3.0*(Y*np.cos(theta)+Y*np.sin(theta)),0,0,0],H,q,kp) - sig
        Init = sig_rt[i][j]
        arc = opt.fsolve(g,Init)
        R = arc[0]
        HAH_points.append([R*np.cos(theta),R*np.sin(theta)])

        
    k1,k2,k3,k4,k5,k, L,kL,Sc,kS, = [  3.78165839e+02,   6.90650000e+00,   4.21876073e-01,
         1.00000000e+00,   0.00000000e+00,  -6.07925671e-01,
         3.09971940e+00,   7.88075782e+02,   1.00504838e+00,
         3.02999006e+02]
    HAH_compare = []
    for j,v in enumerate(eps_t[i]):
        if v >= 0.125:
            break
    x = HAH_state(eps_t[i][:j],S_t[i][:j],alpha,a,k1,k2,k3,k4,k5,L,kL,Sc,kS,k,R,kR,kRp,A,B,C,D,q,kp)
    g1,g2,g3,g4,gL,gS,gR,h1,h2,h3,h4,h5,h6 = x
    H = [h1,h2,h3,h4,h5,h6]
    sig = HAH_phi(Yld2000_phi,alpha,a,g1,g2,gL,gS,S_t[i][j],H,q,kp)
    #print(sig)

    for theta in np.linspace(0,2*np.pi,200):
        g = lambda Y: HAH_phi(Yld2000_phi,alpha,a,g1,g2,gL,gS,[Y*np.cos(theta)-1/3.0*(Y*np.cos(theta)+Y*np.sin(theta)),Y*np.sin(theta)-1/3.0*(Y*np.cos(theta)+Y*np.sin(theta)),-1/3.0*(Y*np.cos(theta)+Y*np.sin(theta)),0,0,0],H,q,kp) - sig
        Init = sig_rt[i][j]
        arc = opt.fsolve(g,Init)
        R = arc[0]
        HAH_compare.append([R*np.cos(theta),R*np.sin(theta)])
        
    points = []
    for i,eps in enumerate(eps_t[:16]):
        for j,v in enumerate(eps):
            if v >= 0.125:
                sigmas = Sig_t[i][j].reshape((3,3))
                eigens = np.linalg.eigvals(sigmas)
                #print(i,eigens)
                break
        #if  i==13 or i==18 or i==19:
        #    points.append([eigens[1],eigens[0]])
        #else:
        #    points.append([eigens[0],eigens[1]])
        points.append([eigens[0],eigens[1]])
    points0 = []
    for theta in np.linspace(0,2*np.pi,200):
        g = lambda Y: Yld2000_phi(alpha,a,[Y*np.cos(theta)-1/3.0*(Y*np.cos(theta)+Y*np.sin(theta)),Y*np.sin(theta)-1/3.0*(Y*np.cos(theta)+Y*np.sin(theta)),-1/3.0*(Y*np.cos(theta)+Y*np.sin(theta)),0,0,0]) - 117.3
        Init = 117.3
        arc = opt.fsolve(g,Init)
        R = arc[0]
        points0.append([R*np.cos(theta),R*np.sin(theta)])
        
    plt.plot([p[0] for p in points],[p[1] for p in points],'o')
    #plt.plot(points[i_p][0],points[i_p][1],'ro')
    #plt.plot(points[-1][0],points[-1][1],'x',color='r')
    plt.plot([p[0] for p in points0],[p[1] for p in points0])
    plt.plot([p[0] for p in HAH_points],[p[1] for p in HAH_points])
    #plt.plot([p[0] for p in HAH_compare],[p[1] for p in HAH_compare])
    plt.xlabel("$\sigma_{1}$")
    plt.ylabel("$\sigma_{2}$")
        
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    #plt.savefig("HAH_yield_180_compare.png")

if __name__ == "__main__":
    main()


        
        
            
            
    

                      
        
        
