from __future__ import division
import numpy as np 
import sympy as sy
from sympy import sqrt
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum import InnerProduct, OuterProduct
from scipy.integrate import odeint
from scipy.constants import physical_constants




def SMA_GPE_F12_simulation(t,wavefunction1,wavefunction2,Veff_,q_,g1_1_, g2_1_,g2_2_,g12_1_,g12_2_,hbar_=physical_constants["Planck constant over 2 pi"][0]):

	##############################################################################################
	# Order parameters, coeff, operators...
	##############################################################################################

	hbar=sy.Symbol("hbar")

	# 8 component order parameter
	w1_p1 = sy.Symbol('w1_p1')
	w1_0 = sy.Symbol('w1_0')
	w1_m1 = sy.Symbol('w1_m1')

	w2_p2 = sy.Symbol('w2_p2')
	w2_p1 = sy.Symbol('w2_p1')
	w2_0 = sy.Symbol('w2_0')
	w2_m1 = sy.Symbol('w2_m1')
	w2_m2 = sy.Symbol('w2_m2')

	# Complex conjugate of the order parameter
	w1_p1_cc = sy.Symbol('w1_p1_cc')
	w1_0_cc = sy.Symbol('w1_0_cc')
	w1_m1_cc = sy.Symbol('w1_m1_cc')

	w2_p2_cc = sy.Symbol('w2_p2_cc')
	w2_p1_cc = sy.Symbol('w2_p1_cc')
	w2_0_cc = sy.Symbol('w2_0_cc')
	w2_m1_cc = sy.Symbol('w2_m1_cc')
	w2_m2_cc = sy.Symbol('w2_m2_cc')

	# vectorial order parameters

	w1=sy.Matrix([w1_p1,w1_0,w1_m1])
	w1_cc=(sy.Matrix([w1_p1_cc,w1_0_cc,w1_m1_cc])).transpose()

	w2=sy.Matrix([w2_p2,w2_p1,w2_0,w2_m1,w2_m2])
	w2_cc=(sy.Matrix([w2_p2_cc,w2_p1_cc,w2_0_cc,w2_m1_cc,w2_m2_cc])).transpose()

	# Global param
	Veff=sy.Symbol("Veff")

	# Single particle contributions 
	q=sy.Symbol("q") # QZS
	
	
	# Interaction coefficients
	g1_1=sy.Symbol("g1_1")

	g2_1=sy.Symbol("g2_1")
	g2_2=sy.Symbol("g2_2")

	g12_1=sy.Symbol("g12_1")
	g12_2=sy.Symbol("g12_2")


	# Spin matrices
	F1_x=1/sqrt(2)*sy.Matrix([[0,1,0],[1,0,1],[0,1,0]])
	F1_y=1/sqrt(2)*sy.Matrix([[0,-1j,0],[1j,0,-1j],[0,1j,0]])
	F1_z=sy.Matrix([[1,0,0],[0,0,0],[0,0,-1]])

	F2_x=1/2*sy.Matrix([[0,2,0,0,0],[2,0,sqrt(6),0,0],[0,sqrt(6),0,sqrt(6),0],[0,0,sqrt(6),0,2],[0,0,0,2,0]])
	F2_y=1/2*sy.Matrix([[0,-1j*2,0,0,0],[1j*2,0,-1j*sqrt(6),0,0],[0,1j*sqrt(6),0,-1j*sqrt(6),0],[0,0,1j*sqrt(6),0,-1j*2],[0,0,0,1j*2,0]])
	F2_z=sy.Matrix([[2,0,0,0,0],[0,1,0,0,0],[0,0,0,0,0],[0,0,0,-1,0],[0,0,0,0,-2]])



	#Mean field operators

	F1_x_mf=(w1_cc*F1_x*w1)[0,0]
	F1_y_mf=(w1_cc*F1_y*w1)[0,0]
	F1_z_mf=(w1_cc*F1_z*w1)[0,0]
	Q1=q*(w1_cc*F1_z*F1_z*w1)[0,0]


	F2_x_mf=(w2_cc*F2_x*w2)[0,0]
	F2_y_mf=(w2_cc*F2_y*w2)[0,0]
	F2_z_mf=(w2_cc*F2_z*w2)[0,0]

	Q2=-q*(w2_cc*F2_z*F2_z*w2)[0,0]
	A2_0=1/sqrt(5)*(2*w2_p2*w2_m2-2*w2_p1*w2_m1+w2_0*w2_0)
	A2_0_cc=1/sqrt(5)*(2*w2_p2_cc*w2_m2_cc-2*w2_p1_cc*w2_m1_cc+w2_0_cc*w2_0_cc)

		
	P12_1=(1/sqrt(10)*w1_p1*w2_0)*(1/sqrt(10)*w1_p1_cc*w2_0_cc) + (-sqrt(3/10)*w1_0*w2_p1)*(-sqrt(3/10)*w1_0_cc*w2_p1_cc) + (sqrt(3/5)*w1_m1*w2_p2)*(sqrt(3/5)*w1_m1_cc*w2_p2_cc)
	P12_1+=(sqrt(3/10)*w1_p1*w2_m1)*(sqrt(3/10)*w1_p1_cc*w2_m1_cc) + (-sqrt(2/5)*w1_0*w2_0)*(-sqrt(2/5)*w1_0_cc*w2_0_cc) + (sqrt(3/10)*w1_m1*w2_p1)*(sqrt(3/10)*w1_m1_cc*w2_p1_cc)
	P12_1+=(sqrt(3/5)*w1_p1*w2_m2)*(sqrt(3/5)*w1_p1_cc*w2_m2_cc) + (-sqrt(3/10)*w1_0*w2_m1)*(-sqrt(3/10)*w1_0_cc*w2_m1_cc) + (1/sqrt(10)*w1_m1*w2_0)*(1/sqrt(10)*w1_m1_cc*w2_0_cc)



	##############################################################################################
	# Energy per particle (e)
	##############################################################################################

	#Single particle
	e=Q1+Q2 

	#F=1
	e+=1/(2*Veff)*g1_1*(F1_x_mf**2+F1_y_mf**2+F1_z_mf**2)

	#F=2
	e+=1/(2*Veff)*(g2_1*(F2_x_mf**2+F2_y_mf**2+F2_z_mf**2)+g2_2*A2_0*A2_0_cc)

	#F=1 & F=2
	e+=1/Veff*(g12_1*(F1_z_mf*F2_z_mf)+g12_2*P12_1)

	##############################################################################################
	# Time derivative functions
	##############################################################################################


	Dw1_p1__Dt_func=sy.lambdify((w1_p1,w1_0,w1_m1,w2_p2,w2_p1,w2_0,w2_m1,w2_m2,w1_p1_cc,w1_0_cc,w1_m1_cc,w2_p2_cc,w2_p1_cc,w2_0_cc,w2_m1_cc,w2_m2_cc,g1_1,g2_1,g2_2,g12_1,g12_2,hbar,q,Veff),1/(1j*hbar)*sy.diff(e,w1_p1_cc))
	Dw1_0__Dt_func= sy.lambdify((w1_p1,w1_0,w1_m1,w2_p2,w2_p1,w2_0,w2_m1,w2_m2,w1_p1_cc,w1_0_cc,w1_m1_cc,w2_p2_cc,w2_p1_cc,w2_0_cc,w2_m1_cc,w2_m2_cc,g1_1,g2_1,g2_2,g12_1,g12_2,hbar,q,Veff),1/(1j*hbar)*sy.diff(e,w1_0_cc))
	Dw1_m1__Dt_func=sy.lambdify((w1_p1,w1_0,w1_m1,w2_p2,w2_p1,w2_0,w2_m1,w2_m2,w1_p1_cc,w1_0_cc,w1_m1_cc,w2_p2_cc,w2_p1_cc,w2_0_cc,w2_m1_cc,w2_m2_cc,g1_1,g2_1,g2_2,g12_1,g12_2,hbar,q,Veff),1/(1j*hbar)*sy.diff(e,w1_m1_cc))

	Dw2_p2__Dt_func=sy.lambdify((w1_p1,w1_0,w1_m1,w2_p2,w2_p1,w2_0,w2_m1,w2_m2,w1_p1_cc,w1_0_cc,w1_m1_cc,w2_p2_cc,w2_p1_cc,w2_0_cc,w2_m1_cc,w2_m2_cc,g1_1,g2_1,g2_2,g12_1,g12_2,hbar,q,Veff),1/(1j*hbar)*sy.diff(e,w2_p2_cc))
	Dw2_p1__Dt_func=sy.lambdify((w1_p1,w1_0,w1_m1,w2_p2,w2_p1,w2_0,w2_m1,w2_m2,w1_p1_cc,w1_0_cc,w1_m1_cc,w2_p2_cc,w2_p1_cc,w2_0_cc,w2_m1_cc,w2_m2_cc,g1_1,g2_1,g2_2,g12_1,g12_2,hbar,q,Veff),1/(1j*hbar)*sy.diff(e,w2_p1_cc))
	Dw2_0__Dt_func= sy.lambdify((w1_p1,w1_0,w1_m1,w2_p2,w2_p1,w2_0,w2_m1,w2_m2,w1_p1_cc,w1_0_cc,w1_m1_cc,w2_p2_cc,w2_p1_cc,w2_0_cc,w2_m1_cc,w2_m2_cc,g1_1,g2_1,g2_2,g12_1,g12_2,hbar,q,Veff),1/(1j*hbar)*sy.diff(e,w2_0_cc))
	Dw2_m1__Dt_func=sy.lambdify((w1_p1,w1_0,w1_m1,w2_p2,w2_p1,w2_0,w2_m1,w2_m2,w1_p1_cc,w1_0_cc,w1_m1_cc,w2_p2_cc,w2_p1_cc,w2_0_cc,w2_m1_cc,w2_m2_cc,g1_1,g2_1,g2_2,g12_1,g12_2,hbar,q,Veff),1/(1j*hbar)*sy.diff(e,w2_m1_cc))
	Dw2_m2__Dt_func=sy.lambdify((w1_p1,w1_0,w1_m1,w2_p2,w2_p1,w2_0,w2_m1,w2_m2,w1_p1_cc,w1_0_cc,w1_m1_cc,w2_p2_cc,w2_p1_cc,w2_0_cc,w2_m1_cc,w2_m2_cc,g1_1,g2_1,g2_2,g12_1,g12_2,hbar,q,Veff),1/(1j*hbar)*sy.diff(e,w2_m2_cc))



	##############################################################################################
	# Time evolution
	##############################################################################################
		





	def timeEvolution (wavefunction,t):
			
		#Unpacking data

		w1_p1=wavefunction[0]+1j*wavefunction[1]
		w1_0=wavefunction[2]+1j*wavefunction[3]
		w1_m1=wavefunction[4]+1j*wavefunction[5]
		w2_p2=wavefunction[6]+1j*wavefunction[7]
		w2_p1=wavefunction[8]+1j*wavefunction[9]
		w2_0=wavefunction[10]+1j*wavefunction[11]
		w2_m1=wavefunction[12]+1j*wavefunction[13]
		w2_m2=wavefunction[14]+1j*wavefunction[15]
		
		w1_p1_cc=np.conjugate(w1_p1)
		w1_0_cc=np.conjugate(w1_0)
		w1_m1_cc=np.conjugate(w1_m1)
		w2_p2_cc=np.conjugate(w2_p2)
		w2_p1_cc=np.conjugate(w2_p1)
		w2_0_cc=np.conjugate(w2_0)
		w2_m1_cc=np.conjugate(w2_m1)
		w2_m2_cc=np.conjugate(w2_m2)
		
		
		Dw1_p1__Dt=complex(Dw1_p1__Dt_func(w1_p1,w1_0,w1_m1,w2_p2,w2_p1,w2_0,w2_m1,w2_m2,w1_p1_cc,w1_0_cc,w1_m1_cc,w2_p2_cc,w2_p1_cc,w2_0_cc,w2_m1_cc,w2_m2_cc,g1_1_,g2_1_,g2_2_,g12_1_,g12_2_,hbar_,q_,Veff_))
		Dw1_0__Dt=  complex(Dw1_0__Dt_func(w1_p1,w1_0,w1_m1,w2_p2,w2_p1,w2_0,w2_m1,w2_m2,w1_p1_cc,w1_0_cc,w1_m1_cc,w2_p2_cc,w2_p1_cc,w2_0_cc,w2_m1_cc,w2_m2_cc,g1_1_,g2_1_,g2_2_,g12_1_,g12_2_,hbar_,q_,Veff_))
		Dw1_m1__Dt=complex(Dw1_m1__Dt_func(w1_p1,w1_0,w1_m1,w2_p2,w2_p1,w2_0,w2_m1,w2_m2,w1_p1_cc,w1_0_cc,w1_m1_cc,w2_p2_cc,w2_p1_cc,w2_0_cc,w2_m1_cc,w2_m2_cc,g1_1_,g2_1_,g2_2_,g12_1_,g12_2_,hbar_,q_,Veff_))
		Dw2_p2__Dt=complex(Dw2_p2__Dt_func(w1_p1,w1_0,w1_m1,w2_p2,w2_p1,w2_0,w2_m1,w2_m2,w1_p1_cc,w1_0_cc,w1_m1_cc,w2_p2_cc,w2_p1_cc,w2_0_cc,w2_m1_cc,w2_m2_cc,g1_1_,g2_1_,g2_2_,g12_1_,g12_2_,hbar_,q_,Veff_))
		Dw2_p1__Dt=complex(Dw2_p1__Dt_func(w1_p1,w1_0,w1_m1,w2_p2,w2_p1,w2_0,w2_m1,w2_m2,w1_p1_cc,w1_0_cc,w1_m1_cc,w2_p2_cc,w2_p1_cc,w2_0_cc,w2_m1_cc,w2_m2_cc,g1_1_,g2_1_,g2_2_,g12_1_,g12_2_,hbar_,q_,Veff_))
		Dw2_0__Dt=  complex(Dw2_0__Dt_func(w1_p1,w1_0,w1_m1,w2_p2,w2_p1,w2_0,w2_m1,w2_m2,w1_p1_cc,w1_0_cc,w1_m1_cc,w2_p2_cc,w2_p1_cc,w2_0_cc,w2_m1_cc,w2_m2_cc,g1_1_,g2_1_,g2_2_,g12_1_,g12_2_,hbar_,q_,Veff_))
		Dw2_m1__Dt=complex(Dw2_m1__Dt_func(w1_p1,w1_0,w1_m1,w2_p2,w2_p1,w2_0,w2_m1,w2_m2,w1_p1_cc,w1_0_cc,w1_m1_cc,w2_p2_cc,w2_p1_cc,w2_0_cc,w2_m1_cc,w2_m2_cc,g1_1_,g2_1_,g2_2_,g12_1_,g12_2_,hbar_,q_,Veff_))
		Dw2_m2__Dt=complex(Dw2_m2__Dt_func(w1_p1,w1_0,w1_m1,w2_p2,w2_p1,w2_0,w2_m1,w2_m2,w1_p1_cc,w1_0_cc,w1_m1_cc,w2_p2_cc,w2_p1_cc,w2_0_cc,w2_m1_cc,w2_m2_cc,g1_1_,g2_1_,g2_2_,g12_1_,g12_2_,hbar_,q_,Veff_))
		
		Dw_Dt=[]
		Dw_Dt+=[np.real(Dw1_p1__Dt),np.imag(Dw1_p1__Dt)]
		Dw_Dt+=[np.real(Dw1_0__Dt),np.imag(Dw1_0__Dt)]
		Dw_Dt+=[np.real(Dw1_m1__Dt),np.imag(Dw1_m1__Dt)]
		Dw_Dt+=[np.real(Dw2_p2__Dt),np.imag(Dw2_p2__Dt)]
		Dw_Dt+=[np.real(Dw2_p1__Dt),np.imag(Dw2_p1__Dt)]
		Dw_Dt+=[np.real(Dw2_0__Dt),np.imag(Dw2_0__Dt)]
		Dw_Dt+=[np.real(Dw2_m1__Dt),np.imag(Dw2_m1__Dt)]
		Dw_Dt+=[np.real(Dw2_m2__Dt),np.imag(Dw2_m2__Dt)]
		return np.array(Dw_Dt,dtype=float)

			
	#Generating  single array containing real and complex values of the wavefunctions in F=1 and F=2
	wavefunctionAux=np.append(wavefunction1,wavefunction2)
	wavefunction=[]
	for i in range (0,8):
		wavefunction+=[np.real(wavefunctionAux[i]),np.imag(wavefunctionAux[i])]

		
	sol=odeint(timeEvolution,wavefunction,t)	

	w1Array=np.array(np.zeros((len(sol),3)),dtype=complex)
	w2Array=np.array(np.zeros((len(sol),5)),dtype=complex)

	w1Array[:,0]=sol[:,0]+1j*sol[:,1]
	w1Array[:,1]=sol[:,2]+1j*sol[:,3]
	w1Array[:,2]=sol[:,4]+1j*sol[:,5]

	w2Array[:,0]=sol[:,6]+1j*sol[:,7]
	w2Array[:,1]=sol[:,8]+1j*sol[:,9]
	w2Array[:,2]=sol[:,10]+1j*sol[:,11]
	w2Array[:,3]=sol[:,12]+1j*sol[:,13]
	w2Array[:,4]=sol[:,14]+1j*sol[:,15]

	F1_xArray=np.zeros(len(sol))
	F1_yArray=np.zeros(len(sol))
	F1_zArray=np.zeros(len(sol))

	F2_xArray=np.zeros(len(sol))
	F2_yArray=np.zeros(len(sol))
	F2_zArray=np.zeros(len(sol))	

	angle1Array=np.zeros(len(sol))	
	angle2Array=np.zeros(len(sol))


	F1_x=np.array(sy.N(F1_x))
	F1_y=np.array(sy.N(F1_y))
	F1_z=np.array(sy.N(F1_z))
	F1_x=np.array(F1_x,dtype=complex)
	F1_y=np.array(F1_y,dtype=complex)
	F1_z=np.array(F1_z,dtype=complex)


	F2_x=np.array(sy.N(F2_x))
	F2_y=np.array(sy.N(F2_y))
	F2_z=np.array(sy.N(F2_z))
	F2_x=np.array(F2_x,dtype=complex)
	F2_y=np.array(F2_y,dtype=complex)
	F2_z=np.array(F2_z,dtype=complex)



	for i in range(0,len(sol)):
		w1=w1Array[i]
		F1_xArray[i]=np.real(np.dot(np.conjugate(w1),np.dot(F1_x,w1)))
		F1_yArray[i]=np.real(np.dot(np.conjugate(w1),np.dot(F1_y,w1)))
		F1_zArray[i]=np.real(np.dot(np.conjugate(w1),np.dot(F1_z,w1)))
		
		if np.abs(F1_xArray[i])==0 and np.abs(F1_yArray[i])==0: #Avoids error in arctan2
			angle1Array[i]=0
		else:
			angle1Array[i]=np.arctan2(F1_yArray[i],F1_xArray[i])
	
		w2=w2Array[i]
		F2_xArray[i]=np.real(np.dot(np.conjugate(w2),np.dot(F2_x,w2)))
		F2_yArray[i]=np.real(np.dot(np.conjugate(w2),np.dot(F2_y,w2)))
		F2_zArray[i]=np.real(np.dot(np.conjugate(w2),np.dot(F2_z,w2)))
		
		if np.abs(F2_xArray[i])==0 and np.abs(F2_yArray[i])==0: #Avoids error in arctan2
			angle2Array[i]=0 
		else:
			angle2Array[i]=np.arctan2(F2_yArray[i],F2_xArray[i])
		
	return [w1Array,F1_xArray,F1_yArray,F1_zArray,angle1Array,w2Array,F2_xArray,F2_yArray,F2_zArray,angle2Array]



	

