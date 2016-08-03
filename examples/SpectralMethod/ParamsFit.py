#! /usr/bin/python
# -*- coding: UTF-8 -*-
# DAMASK plastic_phenopowerlaw 
# material parameters fitting

import os
import numpy as np
import damask
from damask.config.material import Material
from scipy.interpolate import interp1d
import scipy.optimize as opt
import subprocess
import shlex
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

class MaterialFile(Material):
    def __init__(self,filename="material.config"):
        Material.__init__(self)
        self.read(filename)
        self.filename = filename
    def change_params(self,params):
        for s in params.keys():
            if s not in self.data["phase"]:
                print("material file: section %s not exists"%(s))
                continue
            for k,v in params[s].items():
                self.change_value(part="phase",section=s,key=k,value=v)
        self.write(self.filename,overwrite=True)

class FlowCurve():
    def __init__(self,filename=None,data=None,Unit="Pa",E=None,nu=None,tol=1e-5):
        if filename is not None:
            data = np.loadtxt(fname=filename) # strain, effective stress
        print("#data: %d\n"%(len(data)))
        strain = np.array([d[0] for d in data])
        if Unit=="Pa":
            stress = np.array([d[1]/1e6 for d in data]) # MPa
        if Unit=="MPa":
            stress = np.array([d[1] for d in data])
        dstrain = np.diff(strain) # forward difference
        dstress = np.diff(stress) 
        tangent = dstress/dstrain
        curvature = np.gradient(tangent,dstrain) # central difference
        
        # E approx 200GPa -> tangent 200*e3 -> tol approx 0.5e-5
        # tol = 1e-5
        loading_points = []
        tmp_max_stress = 0
        tmp_max_index = 0
        for i,v in enumerate(tangent):
            if abs(v) > 1.0/tol:
                # linear elastic part
                if abs(strain[i]) <= 2e-3:
                    if np.isfinite(abs(v)):
                        E = abs(v)
                    loading_points.append([i,strain[i],stress[i],dstrain[i]]) # initial loading point
                else:
                    if abs(curvature[i]) > tol: # critical points
                        if stress[i] > tmp_max_stress: # unloading point
                            tmp_max_stress = stress[i]
                            tmp_max_index = i
        loading_points.append([tmp_max_index,strain[tmp_max_index],stress[tmp_max_index],dstrain[tmp_max_index]]) # unloading point
        
        print("#loadcases: %d\n"%(len(loading_points)))
        for v in loading_points:
            print(" (%d) strain: %f stress: %f MPa dstrain: %f \n"%(v[0],v[1],v[2],v[3]))
        
        # assign the velocity for the final point
        tangent = np.append(tangent,tangent[-1]+curvature[-1]*dstrain[-1])
        dstrain = np.append(dstrain,dstrain[-1])
        dstress = np.append(dstress,dstress[-1])
        for i,v in enumerate(stress):
            if dstrain[i] != 0:
                stress[i] = np.sign(tangent[i])*v # work hardening material
        
        if E is not None:
            strain = strain - stress/E # plastic strain
            print("Young's modulus: %f GPa\n"%(E/1e3))

        # plastic strain -> effective/accumulated plastic strain
        dstrain = np.diff(strain)
        acc_dstrain = np.insert(abs(dstrain),0,abs(strain[0]))
        acc_strain = np.add.accumulate(acc_dstrain)

        self.strain = strain
        self.stress = stress
        self.loadcases = loading_points
        self.data = data
        self.acc_strain = acc_strain
        self.von_stress = abs(stress)
        
    def sig_f(self,eps,f1,f2,f3,f4):
        # Hockett-Sherby
        # sig_f=f1+f2*(1-exp(-f3*eps**f4))
        eps = np.asarray(eps)
        sig = f1 +f2*(1-np.exp(-f3*eps**f4))

        return sig

    def powerlaw(self,dot_eps,sig_f,dot_eps0,m):
        # viscoplastic power law
        # dot_eps = dot_eps0*(sig_e/sig_f)**m
        dot_eps = np.asarray(dot_eps)
        sig_e = (dot_eps/dot_eps0)**(1./m)*sig_f

        return sig_e

    def sig_b(self,eps,b1,b2):
        # kinematic hardening
        # backstress: dot_sig_b = b1*dot_eps -b2*abs(dot_eps)*sig_b
        eps = np.asarray(eps)
        dot_eps = np.diff(eps)
        dot_eps = np.append(dot_eps,dot_eps[-1]) # final strain rate is continuous

        sig_b0 = 0 # initial bacstress = 0
        eps0 = eps[0] # initial strain

        sig = np.zeros(eps.shape)
        sig[0] = sig_b0
        for i in range(1,len(eps)):
            sig[i] = b1/b2*np.sign(dot_eps[i]) + (sig_b0-b1/b2*np.sign(dot_eps[i]))*np.exp(-b2*(eps[i]-eps0)*np.sign(dot_eps[i]))
            # critical loading points        
            if i in [v[0] for v in self.loadcases]:
               sig_b0 = sig[i]
               eps0 = eps[i]
        return sig

    def sigma(self,epsilon,params):
        # sig_e = abs(sig - sig_b)
        epsilon = np.asarray(epsilon) # plastic strain

        dot_epsilon = np.diff(epsilon)
        acc_dot_epsilon = np.insert(abs(dot_epsilon),0,abs(epsilon[0]))
        eps = np.add.accumulate(acc_dot_epsilon) # effective(accumulated) plastic strain positive

        dot_epsilon = np.append(dot_epsilon,dot_epsilon[-1]) # final strain rate is continuous
        
        dot_eps0,m,f1,f2,f3,f4,b1,b2 = params
        
        sig_f = self.sig_f(eps,f1,f2,f3,f4)
        sig_e = self.powerlaw(abs(dot_epsilon),sig_f,dot_eps0,m)
        sig_b = self.sig_b(epsilon,b1,b2)
        sig = sig_e*np.sign(dot_epsilon) + sig_b

        return sig
    
    def obj(self,params):
        dot_eps0,m,f1,f2,f3,f4,b1,b2 = params
        return np.sum(np.square(self.sigma(self.strain,params)-self.stress))

    def fit(self,params0,bounds):

        res = opt.differential_evolution(self.obj,bounds=bounds)

        return res
    
    def write_loadfile(self,loadfile,direction,fdot):
        line = {"fdot": ['*','0','0',
                         '0','*','0',
                         '0','0','*',],
                "p":   ['*','0','0',
                        '0','*','0',
                        '0','0','*',],
                "t":   '0',
                "N":   '1',}
        if direction.lower()=='x':
            f_index=0
            p_index=[4,8]
        elif direction.lower()=='y':
            f_index=4
            p_index=[0,8]
        elif direction.lower()=='z':
            f_index=8
            p_index=[0,4]
        else:
            print("uniaxial direction ('x','y','z') not specified")
        with open(loadfile,'w') as load:
            t0 = 0
            N0 = 0
            F0 = 1
            for case in self.loadcases:
                line['fdot'][f_index] = str(fdot*np.sign(self.stress[case[0]])) # fdot//stress
                for i in p_index:
                    line['p'][i] = '0'
                F1 = np.exp(case[1]) # true strain -> deformation gradient
                t = (F1-F0)/fdot
                N1 = case[0]
                t1 = t0+t
                N = N1-N0+1
                line['t'] = str(t)
                line['N'] = str(N)
                load.write(" fdot " + ' '.join( e for e in line['fdot'] ) +
                        " p "    + ' '.join( e for e in line['p'] ) +
                        " t "    + line['t'] +
                        " N "    + line['N'] +
                        "\n"
                        )                    
                t0 = t1
                N0 = N1
                F0 = F1
            # final point
            line['fdot'][f_index] = str(fdot*np.sign(self.stress[-1]))
            for i in p_index:
                line['p'][i] = '0'
            F1 = np.exp(self.data[-1][0])
            t = (F1-F0)/fdot
            N1 = len(self.data)
            t1 = t0+t
            N = N1-N0+1
            line['t'] = str(t)
            line['N'] = str(N)
            load.write(" fdot " + ' '.join( e for e in line['fdot'] ) +
                        " p "    + ' '.join( e for e in line['p'] ) +
                        " t "    + line['t'] +
                        " N "    + line['N'] +
                        "\n"
                        )                    
            t0 = t1
            N0 = N1
            F0 = F1
            
def damask_sim(params,geomfile,loadfile,materialfile,sections):
    material = MaterialFile(materialfile) # read from previous file
    # plastic_phenopowerlaw
    # slip
    params_write = {}
    j = 0 # params count
    #s0 = 0 # params section count start
    
    for section in sections:
        phenopowerlaw_slip = material.data["phase"][section.lower()]
        if "phenopowerlaw" not in phenopowerlaw_slip["plasticity"]:
            print("section %s requires plasticty phenopowerlaw")
            exit
        
        nslip_a = [ i for i,n in enumerate(phenopowerlaw_slip["nslip"]) if n!=0] # active slip systems
        params_index =[["gdot0_slip",[0]],
                       ["n_slip",[0]],
                       ["tau0_slip",nslip_a],
                       ["tausat_slip",nslip_a],
                       ["a_slip",[0]],
                       ["h0_slipslip",[0]],
                       ["taub_c",[0]],
                       ["taub_d",[0]],]
        
        #s1 = s0 + sum([len(v[1]) for v in params_index])
        params_check = {}
        params_write[section] = phenopowerlaw_slip
        for p in params_index:
            k = p[0]
            v = p[1]
            for i in v:
                params_check.setdefault(k,[]).append(params[j]) 
                params_write[section][k][i] = params[j]
                j += 1
        # check
        #try:
        #    assert(params_check["gdot0_slip"]>0)
        #    assert(params_check["n_slip"]>0)
        #    for i in range(len(nslip_a)):
        #        assert(params_check["tau0_slip"][i]>0)
        #        assert(params_check["tausat_slip"][i]>params_check["tau0_slip"][i])
        #    assert(params_check["a_slip"]>0)
        #    assert(params_check["taub_c"]>0)
        #    assert(params_check["taub_d"]>0)
        #except AssertionError, e:
        #    raise Exception(e.args)
        
    material.change_params(params_write)

    # simulation
    check = subprocess.call(shlex.split("DAMASK_spectral --geometry %s --load %s" %(geomfile,loadfile)))
    jobname = geomfile.split('.')[0] + '_' + loadfile.split('.')[0]
    if check == 0:
        subprocess.call(shlex.split("postResults --cr f,p %s.spectralOut" %(jobname)))
        postfn = jobname+".txt"
        subprocess.call(shlex.split("addCauchy ./postProc/%s" %(postfn)))
        subprocess.call(shlex.split("addStrainTensors -0 -v ./postProc/%s" %(postfn)))
        subprocess.call(shlex.split("addMises -s Cauchy -e ln'('V')' ./postProc/%s" %(postfn)))

        table = damask.ASCIItable("./postProc/%s" %(postfn),readonly=True)
        table.head_read()
        table.data_readArray(["Mises(ln(V))","Mises(Cauchy)"])

        sim_flow = FlowCurve(data=table.data,Unit='Pa')
        sim_data_n = len(sim_flow.data)
        sim_data_acc = [[sim_flow.acc_strain[i],sim_flow.von_stress[i]] for i in range(sim_data_n)]
        np.savetxt("./postProc/%s_acc.dat"%(jobname),sim_data_acc)
        sim_data_plastic =[[sim_flow.strain[i],sim_flow.stress[i]] for i in range(sim_data_n)]
        np.savetxt("./postProc/%s_plastic.dat"%(jobname),sim_data_plastic)
        return sim_data_plastic
    else:
        print("damask failure")
        return [[float("inf"),float("inf")]]

def damask_obj(params,geomfile,loadfile,materialfile,sections,macroparams,Unit="MPa"):
    sim_data = damask_sim(params,geomfile,loadfile,materialfile,sections)
    # Use exp_data instead of macroparams for interpolation
    #if Unit=="Pa":
    #    for point in exp_data:
    #        point[1] = point[1]/1e6
    #exp_strain = [point[0] for point in exp_data]
    #exp_stress = [point[1] for point in exp_data]
    #interp = interp1d(exp_strain,exp_stress)
    flowcurve = FlowCurve(data=sim_data,Unit=Unit)
    obj = np.zeros(len(sim_data))
    eps = [point[0] for point in sim_data]
    sig = [point[1] for point in sim_data]
    obj = sig-flowcurve.sigma(eps,macroparams)
    #for i,point in enumerate(sim_data):
    #    eps = point[0]
    #    sig = point[1]
    #    if eps>=min(exp_strain) and eps<=max(exp_strain):
    #        obj[i]=(point[1]-interp(eps))

    jobname = geomfile.split('.')[0]+'_'+loadfile.split('.')[0]
    if len(obj)!=1:
        with open("%s.log"%(jobname),'a') as errfile:
            errfile.write("params:")
            for p in params:
                errfile.write(" %e " %(p)) 
            errfile.write(" error: %e MPa\n" %(sum(np.square(obj))))
        return sum(np.square(obj*1e6)) # Pa
    else:
        return float("inf") # will cause ValueError!

def damask_fit(params0,args,bounds,constraints):
    res = opt.minimize(damask_obj,x0=params0,args=args,method='Nelder-Mead') #bounds=bounds) #,constraints=constraints)
    return res

def damask_run(geomfile,loadfile):
    # simulation
    check = subprocess.call(shlex.split("DAMASK_spectral --geometry %s --load %s" %(geomfile,loadfile)))
    jobname = geomfile.split('.')[0] + '_' + loadfile.split('.')[0]
    if check == 0:
        subprocess.call(shlex.split("postResults --cr f,p %s.spectralOut" %(jobname)))
        postfn = jobname+".txt"
        subprocess.call(shlex.split("addCauchy ./postProc/%s" %(postfn)))
        subprocess.call(shlex.split("addStrainTensors -0 -v ./postProc/%s" %(postfn)))
        subprocess.call(shlex.split("addMises -s Cauchy -e ln'('V')' ./postProc/%s" %(postfn)))

        table = damask.ASCIItable("./postProc/%s" %(postfn),readonly=True)
        table.head_read()
        table.data_readArray(["Mises(ln(V))","Mises(Cauchy)"])

        sim_flow = FlowCurve(data=table.data,Unit='Pa')
        sim_data_n = len(sim_flow.data)
        sim_data_acc = [[sim_flow.acc_strain[i],sim_flow.von_stress[i]] for i in range(sim_data_n)]
        np.savetxt("./postProc/%s_acc.dat"%(jobname),sim_data_acc)
        sim_data_plastic =[[sim_flow.strain[i],sim_flow.stress[i]] for i in range(sim_data_n)]
        np.savetxt("./postProc/%s_plastic.dat"%(jobname),sim_data_plastic) 
        #return sim_data_acc
    else:
        print("damask failure")
        #return [[float("inf"),float("inf")]]

            
def damask_plot(datafile,geomfile,loadfile): # datafile Unit="MPa"
    
    flowcurve = FlowCurve(filename=datafile,Unit="MPa")

    jobname = geomfile.split('.')[0] + '_' + loadfile.split('.')[0]
    sim_data = np.loadtxt(fname="./postProc/%s_acc.dat"%(jobname))
    sim_accstrain = [ d[0] for d in sim_data]
    sim_vonstress = [ d[1] for d in sim_data]
    plt.plot(flowcurve.acc_strain,flowcurve.von_stress,marker='x',color='b',ls='x-')
    plt.plot(sim_accstrain,sim_vonstress,marker='.',color='r',ls='-')
    plt.xlabel('$\epsilon^p$')
    plt.ylabel('$\sigma_{e}[MPa]$')
    plt.grid(True)
    plt.savefig("./postProc/%s.png"%(jobname))

def test():
    datafile = "eps_sig_kin_fit.dat"
    geomfile = "20grains8x8x8.geom"
    loadfile = "tcXfit.load"
    materialfile = "material.config"
    sections = ["bcc_ferrite"]
           
    damask_run(geomfile,loadfile)
    damask_plot(datafile,geomfile,loadfile)
    ##flowcurve.write_loadfile("tcXfit_test.load",direction='X',fdot=2e-3)
            
    ## sim = flowcurve.sigma(flowcurve.strain,res.x)
    ##plt.plot(flowcurve.acc_strain,abs(sim),color='r')    
    ##plt.show()
            
def main():
    datafile = "eps_sig_kin0.dat"
    geomfile = "20grains8x8x8.geom"
    loadfile = "tcXfit.load"
    materialfile = "material.config"
    sections = ["bcc_ferrite"]
    flowcurve = FlowCurve(filename=datafile,Unit="MPa")
    params0_de = [1e-3,20,497.12,336.52,22.84,1.0,3421.46,32.50]
    bounds_de = [(0,0.1),(0,1e4),(0,1e4),(0,1e4),(0,1e4),(0,1),(0,1e5),(0,1e2),]                                                                     
    res = flowcurve.fit(params0_de,bounds_de)
    exp_data_num = len(flowcurve.data)
    exp_data = [[flowcurve.strain[i],flowcurve.stress[i]] for i in range(exp_data_num)]
    print(datafile)
    print(res)
    ##"eps_sig_kin_fit3.dat"
    ##    nfev: 17229
    ## success: True
    ##     fun: 873069.18280964217
    ##       x: array([  2.73723410e+00,   6.48945959e+03,   5.01814287e+02,
    ##         2.28211115e+02,   1.74776548e+01,   9.84842425e-01,
    ##         6.98743308e+03,   3.28492291e+01])
    ## message: 'Optimization terminated successfully.'
    ##     nit: 141


    if res.success==True:
        jobname = geomfile.split('.')[0] + '_' + loadfile.split('.')[0]
        M = random.randint(20,30)/10.0 
        dot_eps0,m,f1,f2,f3,f4,b1,b2 = res.x
        if m > 20: 
            params0_lq = [1e-3,20,2*f1/(M**2)*1e6,2*f1/(M**2)*1e6,2*(f1+f2+b1/b2)/(M**2)*1e6,2*(f1+f2+b1/b2)/(M**2)*1e6,2.0,((f1+f2)*f3+b1)/(M**2)*1e6,b1/(M**2)*1e6,b2] # Unit="Pa"
        else:
            params0_lq = [dot_eps0*M,m,2*f1/(M**2)*1e6,2*f1/(M**2)*1e6,2*(f1+f2+b1/b2)/(M**2)*1e6,2*(f1+f2+b1/b2)/(M**2)*1e6,2.0,((f1+f2)*f3+b1)/(M**2)*1e6,b1/(M**2)*1e6,b2] # Unit="Pa"
        print("macroscopic parameters: ")
        print(res.x)
        print("M factor: ")
        print(M)
        print("initial params: ")
        print(params0_lq)
        with open("%s.log"%(jobname),'a') as errfile:
            errfile.write("initial:")
            for p in res.x:
                errfile.write(" %e " %(p)) 
            errfile.write(" error: %e MPa\n" %(res.fun))

        #bounds_lq = ([0, 1,   0,     0,     100e6,   100e6,   1,  0,    0,   1],
        #             [0.1, 1e4, 500e6, 500e6, 1e9,     1e9,     3,  20e9, 5e9, 1e2],)
        #print("bounds: ")
        #print(bounds_lq)
        bounds_lq = [(0,0.1),(1,1e4),(0,500e6),(0,500e6),(100e6,1e9),(100e6,1e9),(1,3),(0,20e9),(0,5e9),(1,1e2)]
        cons = ({'type':'ineq','fun': lambda x: x[4]-x[2],'jac': lambda x: np.array([0,0,-1,0,1,0,0,0,0,0])},
                {'type':'ineq','fun': lambda x: x[5]-x[3],'jac': lambda x: np.array([0,0,0,-1,0,1,0,0,0,0])},)
        args = (geomfile,loadfile,materialfile,sections,res.x,"MPa")
        damask_res = damask_fit(params0_lq,args,bounds_lq,cons)
        print(damask_res)
        sim_data = np.loadtxt(fname="./postProc/%s_plastic.dat"%(jobname))
        sim_strain = [ d[0] for d in sim_data]
        sim_stress = [ d[1] for d in sim_data]
        plt.plot(flowcurve.strain,flowcurve.stress,marker='x',color='b',ls='x-')
        plt.plot(sim_strain,sim_stress,marker='.',color='r',ls='-')
        plt.xlabel('$\epsilon^p$')
        plt.ylabel('$\sigma_{e}[MPa]$')
        plt.grid(True)
        plt.savefig("./postProc/%s.png"%(jobname))

    
if __name__ == "__main__":
    main()
    #test()

        
        

