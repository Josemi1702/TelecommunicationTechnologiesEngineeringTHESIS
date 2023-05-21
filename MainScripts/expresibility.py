"""
EXPRESSITIVITY CALCULATION MODULE: this module is used to calculate the expressivity of custom circuits.
"""

from qiskit import  Aer, execute
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.providers.aer import  UnitarySimulator

import numpy as np
import math
from math import pi
from scipy.special import rel_entr
from random import seed
from random import random

import matplotlib.pyplot as plt


global HISTOGRAM_COLUMNS, N_TETHAS

#Number of columns in histograms 
HISTOGRAM_COLUMNS = 74

#Number of angles within the tetha vector
N_TETHAS=80000




def calculate_expresability(num_qcir:int, num_qbits:int, L:int):
    """
    Calculates the expressivity of a circuit of given qubits and depth.

    Args:
        num_qcir (int): Id of the circuit selected from the created list.
        num_qbits (int): Number of qubits in the circuit.
        L (int): Circuit depth.

    Returns:
        float: Expresability
    """
    
    #Variables initialization
    simulator = Aer.get_backend('qasm_simulator')
    shots=1000
    str_ceros=str(0)*num_qbits
    P_Harr,fidelity, hist_cols, hist_center = [], [], [], []
    
    #Setting histograms
    hist_cols.append((0)/HISTOGRAM_COLUMNS)
    for i in range(HISTOGRAM_COLUMNS):
        hist_cols.append((i+1)/HISTOGRAM_COLUMNS)
        hist_center.append(hist_cols[1]+hist_cols[i])
        P_Harr.append((1-hist_cols[i])**((2**num_qbits)-1)-(1-hist_cols[i+1])**((2**num_qbits)-1)) #P_Harr formula

    #Circuit simulation and fidelity calculation
    for x in range(2000):
        theta = [2*pi*random() for i in range(N_TETHAS)]
        count=0
        qr,cr = QuantumRegister(num_qbits), ClassicalRegister(num_qbits)
        qc=circuit_generator(qr, cr, theta, num_qcir, num_qbits, L) #Custom circuits with differents gate topologies
        qc.measure(qr[:],cr[:])
        count =execute(qc, simulator, shots=shots).result().get_counts()
        if str_ceros in count and '1' in count:
            fidelity.append(count[str_ceros]/shots)
        elif str_ceros in count and '1' not in count:
            fidelity.append(count[str_ceros]/shots)
        else:
            fidelity.append(0)
    
    #Histogram
    weights = np.ones_like(fidelity)/float(len(fidelity))
    ranges=[0, 1]
    name_circuit="Circuit: "+str(num_qcir)
    
    P_1_hist=np.histogram(fidelity, bins=hist_cols, weights=weights, range=ranges)[0]
    kl_pq = rel_entr(P_1_hist, P_Harr) #Relative entropy between two probability distributions
    
    plt.figure()
    plt.plot(hist_center, P_Harr, label='P_Harr')
    plt.hist(fidelity, bins=hist_cols, weights=weights, range=ranges, label=name_circuit)
    
    #Save histogram
    save_string=f"./expresibilities/Expr_Circ_{num_qcir}_Qubits_{num_qbits}_Depth_{L}.pdf"
    plt.savefig(save_string, format='pdf')
    plt.close()
    
    return sum(kl_pq)




def circuit_generator(qr:QuantumRegister, cr:ClassicalRegister, theta, num_qcir:int, num_qbits:int, L:int):
    """
    Generates the selected circuit.

    Args:
        qr (QuantumRegister): Quantum registers.
        cr (ClassicalRegister): Classical Registers.
        theta (list): Tetha vector.
        num_qcir (int): Id of the circuit selected from the created list.
        num_qbits (int): Number of qubits in the circuit.
        L (int): Circuit depth.

    Returns:
        QuantumCircuit: Circuit
    """
    
    switch_dict = {
        0: ZZFeatureMap,
        1: circ1,
        2: circ2,
        3: circ3,
        4: circ4,
        5: circ5,
        6: circ6,
        7: circ7,
        8: circ8,
        9: circ9,
        10: circ10,
    }

    if num_qcir in switch_dict:
        return switch_dict[num_qcir](qr, cr, theta, num_qbits, L)
    else:
        print("Circuit not defined.")

def circuit_example_generator(qr:QuantumRegister, cr:ClassicalRegister, theta, num_qcir:int, num_qbits:int, L:int):
    """
    Generates the selected circuit of the examples set (to prove that the expressibility algorithm obtains results similar to those of the paper). 

    Args:
        qr (QuantumRegister): Quantum registers.
        cr (ClassicalRegister): Classical Registers.
        theta (list): Tetha vector.
        num_qcir (int): Id of the circuit selected from the created list.
        num_qbits (int): Number of qubits in the circuit.
        L (int): Circuit depth.

    Returns:
        QuantumCircuit: Circuit
    """
    
    switch_dict = {
        0: example2,
        1: example2,
        2: example3,
        3: example4,
        4: example5,
        5: example6,
        6: example7,
        7: example8,
        8: example9,
        9: example10,
        10: example,
    }
    

    if num_qcir in switch_dict:
        return switch_dict[num_qcir](qr, cr, theta, num_qbits, L)
    else:
        print("Circuit not defined.")

#--------------------------------------Custom Circuits--------------------------------------#
#                                                                                           #
#-------------------------------------------------------------------------------------------#
def ZZFeatureMap( qr, cr, theta, num_qubits:int , L:int):
    qc= QuantumCircuit(qr, cr)
    count=0

    for r in range(0,L):   
        for i in range(0, num_qubits):
            qc.h(qr[i])
            qc.p(2*pi*theta[count], qr[i])
            count=count+1

            
            
        for i in range(0, num_qubits):
            
            if(i>0):
                j=0
                while(j<i):
                    qc.cnot(qr[j],qr[i])
                    qc.p(2*(pi-theta[count])*(pi-theta[count+1]), qr[i])
                    count=count+3
                    qc.cnot(qr[j],qr[i])
                    j+=1                
            
        qc.barrier()
        
        
    #inverse repeat
    qc.barrier()
    for r in range(0,L):
        for i in range(num_qubits-1,-1, -1 ):
            
            if(i>0):
                j=i-1
                while(j>-1):
                    qc.cnot(qr[j],qr[i])
                    qc.p(2*(pi-theta[count])*(pi-theta[count+1]), qr[i])
                    count=count+3
                    qc.cnot(qr[j],qr[i])
                    j=j-1
                    
  
        
        for i in range(0, num_qubits):
            qc.p(2*pi*theta[count], qr[i])
            count=count+1
            qc.h(qr[i])
        qc.barrier()
    return qc
    
    
def circ1( qr, cr, theta, num_qubits:int , L:int): 
    qc= QuantumCircuit(qr, cr)
    count=0

    
    for l in range(L):    
        
        for i in range(0, num_qubits):
            qc.rz(theta[count],qr[i])
            count=count+1
        qc.barrier()


   #inverse repeat          
    qc.barrier(qr)
    for l in range(L):
        for i in range(0, num_qubits):
            qc.rz(theta[count],qr[i])
            count=count+1
        qc.barrier()

    return qc

def circ2( qr, cr, theta, num_qubits:int , L:int): 
    qc= QuantumCircuit(qr, cr)
    count=0

    
    for l in range(L):    
        
        for i in range(0, num_qubits):
            qc.rz(theta[count],qr[i])
            count=count+1
            qc.ry(theta[count],qr[i])
            count=count+1
        qc.barrier()


   #inverse repeat          
    qc.barrier(qr)
    for l in range(L):
        for i in range(0, num_qubits):
            qc.ry(theta[count],qr[i])
            count=count+1
            qc.rz(theta[count],qr[i])
            count=count+1
        qc.barrier()

    return qc

def circ3( qr, cr, theta, num_qubits:int , L:int):
    qc= QuantumCircuit(qr, cr)
    count=0

    for r in range(0,L):
        for i in range(0, num_qubits):
            qc.ry(theta[count],qr[i])
            count=count+1
            qc.rz(theta[count],qr[i])
            count=count+1
            if i<num_qubits-1:
                qc.cnot(qr[i],qr[i+1] )
        qc.barrier()
    #inverse repeat
    qc.barrier()
    for r in range(0,L):
        for i in range(num_qubits-1, -1, -1):
            if i<num_qubits-1:
                qc.cnot(qr[i], qr[i+1])
            qc.rz(theta[count],qr[i])
            count=count+1
            qc.ry(theta[count],qr[i])
            count=count+1
        qc.barrier()
    return qc

def circ4( qr, cr, theta, num_qubits:int , L:int):
    qc= QuantumCircuit(qr, cr)
    count=0

    for r in range(0,L):
        for i in range(0, num_qubits):
            qc.rx(theta[count],qr[i])
            count=count+1
            if i<num_qubits-1:
                qc.cnot(qr[i+1], qr[i])
            qc.rz(theta[count],qr[i])
            count=count+1
        qc.barrier()
    #inverse repeat
    qc.barrier()
    for r in range(0,L):
        for i in range(num_qubits-1, -1, -1):
            qc.rz(theta[count],qr[i])
            count=count+1
            if i<num_qubits-1:
                qc.cnot(qr[i+1], qr[i])
            qc.rx(theta[count],qr[i])
            count=count+1
        qc.barrier()
    return qc


def circ5( qr, cr, theta, num_qubits:int , L:int):
    qc= QuantumCircuit(qr, cr)
    count=0

    for r in range(0,L):
        for i in range(0, num_qubits):
            qc.ry(theta[count],qr[i])
            count=count+1
            qc.rz(theta[count],qr[i])
            count=count+1
            if i<num_qubits-1:
                qc.crx(theta[count],qr[i+1], qr[i])
                count=count+1
        qc.barrier()
    #inverse repeat
    qc.barrier()
    for r in range(0,L):
        for i in range(num_qubits-1, -1, -1):
            if i<num_qubits-1:
                qc.crx(theta[count], qr[i+1], qr[i])
                count=count+1
            qc.rz(theta[count],qr[i])
            count=count+1
            qc.ry(theta[count],qr[i])
            count=count+1
        qc.barrier()
    return qc

def circ6( qr, cr, theta, num_qubits:int , L:int):
    qc= QuantumCircuit(qr, cr)
    count=0

    for r in range(0,L):
        for i in range(0, num_qubits):
            qc.rx(theta[count],qr[i])
            count=count+1
            qc.ry(theta[count],qr[i])
            count=count+1
            if i<num_qubits-1:
                qc.cnot(qr[i], qr[i+1])
            qc.rz(theta[count],qr[i])
            count=count+1
            qc.rx(theta[count],qr[i])
            count=count+1
        qc.barrier()
    #inverse repeat
    qc.barrier()
    for r in range(0,L):
        for i in range(num_qubits-1, -1, -1):
            qc.rx(theta[count],qr[i])
            count=count+1
            qc.rz(theta[count],qr[i])
            count=count+1
            if i<num_qubits-1:
                qc.cnot(qr[i], qr[i+1])
            qc.ry(theta[count],qr[i])
            count=count+1
            qc.rx(theta[count],qr[i])
            count=count+1
        qc.barrier()
    return qc

def circ7( qr, cr, theta, num_qubits:int , L:int):
    qc= QuantumCircuit(qr, cr)
    count=0

    for r in range(0,L):
        for i in range(0, num_qubits):
            qc.h(qr[i])
        for i in range(0, num_qubits):
            qc.rx(theta[count],qr[i])
            count=count+1
            if i<num_qubits-1:
                qc.crz(theta[count], qr[i], qr[i+1])
                count=count+1
            qc.ry(theta[count],qr[i])
            count=count+1
        qc.barrier()
    #inverse repeat
    qc.barrier()
    for r in range(0,L):
        for i in range(num_qubits-1, -1, -1):
            qc.ry(theta[count],qr[i])
            count=count+1
            if i<num_qubits-1:
                qc.crz(theta[count], qr[i], qr[i+1])
                count=count+1
            qc.rx(theta[count],qr[i])
            count=count+1
            
        for i in range(0, num_qubits):
            qc.h(qr[i])
        qc.barrier()
    return qc

def circ8( qr, cr, theta, num_qubits:int , L:int):
    qc= QuantumCircuit(qr, cr)
    count=0

    for r in range(0,L):
        for i in range(0, num_qubits):
            qc.h(qr[i])
            qc.p(2*theta[count], qr[i])
            count=count+1
        for i in range(0, num_qubits): 
            if(i>0):
                j=i
                while(j>0):
                    qc.cnot(qr[j-1],qr[i]) 
                    j-=1
            qc.rx(theta[count],qr[i])
            count=count+1
            qc.rz(theta[count],qr[i])
            count=count+1
        qc.barrier()
        
        
    #inverse repeat
    qc.barrier()
    for r in range(0,L):
        for i in range(num_qubits-1, -1, -1):
            qc.rz(theta[count],qr[i])
            count=count+1
            qc.rx(theta[count],qr[i])
            count=count+1
            if(i>0):
                j=0
                while(j<i):
                    qc.cnot(qr[j],qr[i]) 
                    j+=1
            
        for i in range(0, num_qubits):
            qc.p(2*theta[count], qr[i])
            count=count+1
            qc.h(qr[i])
        qc.barrier()
    return qc

def circ9( qr, cr, theta, num_qubits:int , L:int):
    qc= QuantumCircuit(qr, cr)
    count=0

    for r in range(0,L):   
        for i in range(0, num_qubits):
            qc.h(qr[i])
            if i>0:
                qc.rx(theta[count]-theta[count+1], qr[i])
                count=count+2
            else:
                qc.rx(theta[count], qr[i])
                count=count+1 
            qc.h(qr[i])
            
            
        for i in range(0, num_qubits):
            if(i>0):
                j=0
                while(j<i):
                    qc.cz(qr[j],qr[i]) 
                    j+=1
                qc.ry((pi+theta[count])*(pi-theta[count+1]),qr[i])
                count=count+2
                qc.rz((pi-theta[count])*(pi+theta[count+1]),qr[i])
                count=count+2
                j=i
                while(j>0):
                    qc.cz(qr[j-1],qr[i])
                    j-=1 
            
            
        qc.barrier()
        
        
    #inverse repeat
    qc.barrier()
    for r in range(0,L):
        for i in range(num_qubits-1,-1, -1 ):
            if(i>0):
                j=0
                while(j<i):
                    qc.cz(qr[j],qr[i]) 
                    j+=1
                 
                qc.rz((pi-theta[count])*(pi+theta[count]),qr[i])
                count=count+2
                qc.ry((pi+theta[count])*(pi-theta[count+1]),qr[i])
                count=count+2
                j=i
                while(j>0):
                    qc.cz(qr[j-1],qr[i])
                    j-=1
                
        for i in range(0, num_qubits):
            qc.h(qr[i])
            if i>0:
                qc.rx(theta[count]-theta[count+1], qr[i])
                count=count+2
            else:
                qc.rx(theta[count], qr[i])
                count=count+1 
            qc.h(qr[i])
        qc.barrier()
    return qc


def circ10( qr, cr, theta, num_qubits:int , L:int):
    qc= QuantumCircuit(qr, cr)
    count=0

    for r in range(0,L):   
        for i in range(0, num_qubits):
            qc.h(qr[i])
            qc.ry(2*pi*theta[count], qr[i])
            count=count+1
            qc.h(qr[i])
            
            
        for i in range(0, num_qubits):
            
            if(i>0):
                j=0
                while(j<i):
                    qc.cz(qr[j],qr[i])
                    qc.cnot(qr[j],qr[i])
                    
                    if i>1:
                        qc.p(theta[count]*theta[count+1]*theta[count+2], qr[i])
                        count=count+3
                    else:
                        qc.p(theta[count]*theta[count+1], qr[i])
                        count=count+2 
                    qc.cnot(qr[i],qr[j] )
                    qc.cz(qr[i],qr[j])
                    j+=1
            
        for i in range(0, num_qubits):
            qc.h(qr[i])
            
            
            
        qc.barrier()
        
        
    #inverse repeat
    qc.barrier()
    for r in range(0,L):
        for i in range(0, num_qubits):
            qc.h(qr[i])
            
        for i in range(num_qubits-1,-1, -1 ):
            
            if(i>0):
                j=i-1
                while(j>-1):
                    qc.cz(qr[i],qr[j])
                    qc.cnot(qr[i],qr[j] )
                    if i>1:
                        qc.p(theta[count]*theta[count+1]*theta[count+2], qr[i])
                        count=count+3
                    else:
                        qc.p(theta[count]*theta[count+1], qr[i])
                        count=count+2
                    qc.cnot(qr[j],qr[i])
                    qc.cz(qr[j],qr[i])
                    j=j-1
                    
  
        
        for i in range(0, num_qubits):
            qc.h(qr[i])
            qc.ry(2*pi*theta[count], qr[i])
            count=count+1
            qc.h(qr[i])
        qc.barrier()
    return qc


#----------------------------------Examples from the paper----------------------------------#
#                                                                                           #
#-------------------------------------------------------------------------------------------#
def example( qr, cr, theta, num_qubits:int , L:int): #Circ 11 paper
    qc= QuantumCircuit(qr, cr)
    count=0
    repeat=1
    
    for l in range(L):    

        for i in range(4):
            qc.ry(theta[count],qr[i])
            count=count+1

        for i in range(4):
            qc.rz(theta[count],qr[i])
            count=count+1    


        qc.cx(qr[1],qr[0])
        qc.cx(qr[3],qr[2])



        qc.ry(theta[count],qr[1])
        count=count+1 
        qc.ry(theta[count],qr[2])
        count=count+1 
        qc.rz(theta[count],qr[1])
        count=count+1 
        qc.rz(theta[count],qr[2])
        count=count+1 
        qc.cx(qr[2],qr[1])


    if repeat!=0:             
        qc.barrier(qr)
        for l in range(L):
            qc.cx(qr[2],qr[1])

            qc.rz(theta[count],qr[2])
            count=count+1 
            qc.rz(theta[count],qr[1])
            count=count+1 
            qc.ry(theta[count],qr[2])
            count=count+1 
            qc.ry(theta[count],qr[1])
            count=count+1 

            qc.cx(qr[3],qr[2])
            qc.cx(qr[1],qr[0])


            for i in range(4):
                qc.rz(theta[count],qr[i])
                count=count+1            

            for i in range(4):
                qc.ry(theta[count],qr[i])
                count=count+1
    
    return qc

def example2( qr, cr, theta, num_qubits:int , L:int): #1 qubit, iddle
    qc= QuantumCircuit(qr, cr)
    qc.i(qr[0])
    qc.i(qr[0])
    return qc

def example3( qr, cr, theta, num_qubits:int , L:int): #1 qubit, h, rz
    qc= QuantumCircuit(qr, cr)
    qc.h(qr[0])
    qc.rz(theta[0], qr[0])
    
    qc.rz(-theta[1], qr[0])
    qc.h(qr[0])
    return qc
    
    
def example4( qr, cr, theta, num_qubits:int , L:int): # 1 qubit, h, rz y rx
    qc= QuantumCircuit(qr, cr)
    qc.h(qr[0])
    qc.rz(theta[0], qr[0])
    qc.rx(theta[1], qr[0])
    
    qc.rx(-theta[2], qr[0])
    qc.rz(-theta[3], qr[0])
    qc.h(qr[0])
    return qc
    
    
def example5( qr, cr, theta, num_qubits:int , L:int): #1 quibit, unitario
    from qiskit.quantum_info import random_unitary
    from qiskit.extensions import UnitaryGate
    
    qc= QuantumCircuit(qr, cr)
    u13=UnitaryGate(random_unitary(2))
    qc.append(u13, [qr[0]] )
    
    u13=UnitaryGate(random_unitary(2))
    qc.append(u13, [qr[0]] )
    return qc

#-----------------------------Volvemos a los de 4
def example6( qr, cr, theta, num_qubits:int , L:int): # Circ 2 paper
    qc= QuantumCircuit(qr, cr)
    count=0
    repeat=1
    for l in range(L):

        for i in range(4):
            qc.rx(theta[count],qr[i])
            count=count+1    
        for i in range(4):
            qc.rz(theta[count],qr[i])
            count=count+1            
        qc.cx(qr[3],qr[2])
        qc.cx(qr[2],qr[1])
        qc.cx(qr[1],qr[0])
    
    
    if repeat!=0:    
        qc.barrier(qr)
        for l in range(L):
        
            qc.cx(qr[1],qr[0])
            qc.cx(qr[2],qr[1])
            qc.cx(qr[3],qr[2])
    
            for i in range(4):
                qc.rz(theta[count],qr[i])
                count=count+1         
            for i in range(4):
                qc.rx(theta[count],qr[i])
                count=count+1     
    
    return qc

def example7( qr, cr, theta, num_qubits:int , L:int): #Circ 4 paper 
    qc= QuantumCircuit(qr, cr)
    repeat=1
    count=0
    for l in range(L):

        for i in range(4):
            qc.rx(theta[count],qr[i])
            count=count+1    
        for i in range(4):
            qc.rz(theta[count],qr[i])
            count=count+1            
        qc.crx(theta[count],qr[3],qr[2])
        count=count+1 
        qc.crx(theta[count],qr[2],qr[1])
        count=count+1 
        qc.crx(theta[count],qr[1],qr[0])
        count=count+1 

    if repeat!=0:               
        qc.barrier(qr)
    
        
        for l in range(L):
        
            qc.crx(theta[count],qr[1],qr[0])
            count=count+1 
            qc.crx(theta[count],qr[2],qr[1])
            count=count+1 
            qc.crx(theta[count],qr[3],qr[2])
            count=count+1 
        
            for i in range(4):
                qc.rz(theta[count],qr[i])
                count=count+1         
            for i in range(4):
                qc.rx(theta[count],qr[i])
                count=count+1                
    
    
    return qc


def example8( qr, cr, theta, num_qubits:int , L:int): #Circ 6 paper
    qc= QuantumCircuit(qr, cr)
    count=0
    repeat=1
    for l in range(L):

        for i in range(4):
            qc.rx(theta[count],qr[i])
            count=count+1    
        for i in range(4):
            qc.rz(theta[count],qr[i])
            count=count+1            

            
        for j in range(4):
            for i in range(4):
                if i!=j:
                    qc.crx(theta[count],qr[3-j],qr[3-i])
                    count=count+1

 

        for i in range(4):
            qc.rx(theta[count],qr[i])
            count=count+1
        
        
        for i in range(4):
            qc.rz(theta[count],qr[i])
            count=count+1            
            
    if repeat!=0:             
        qc.barrier(qr)
    
        
        for l in range(L):
        

            for i in range(4):
                qc.rz(theta[count],qr[i])
                count=count+1     
        
            for i in range(4):
                qc.rx(theta[count],qr[i])
                count=count+1
        

            for j in range(4):
                for i in range(4):
                    if i!=j:
                        qc.crx(theta[count],qr[j],qr[i])
                        count=count+1
    
            for i in range(4):
                qc.rz(theta[count],qr[i])
                count=count+1         
            for i in range(4):
                qc.rx(theta[count],qr[i])
                count=count+1                
    
    return qc

def example9( qr, cr, theta, num_qubits:int , L:int): #Circ 8 paper 
    qc= QuantumCircuit(qr, cr)
    count=0
    repeat=1
    for l in range(L):

        for i in range(4):
            qc.rx(theta[count],qr[i])
            count=count+1    
        for i in range(4):
            qc.rz(theta[count],qr[i])
            count=count+1            

            
        qc.crx(theta[count],qr[1],qr[0])
        count=count+1

        qc.crx(theta[count],qr[3],qr[2])
        count=count+1



        for i in range(4):
            qc.rx(theta[count],qr[i])
            count=count+1
        
        
        for i in range(4):
            qc.rz(theta[count],qr[i])
            count=count+1

        
        qc.crx(theta[count],qr[2],qr[1])    
        count=count+1
        
            
    if repeat!=0:             
        qc.barrier(qr)
    
        
        for l in range(L):
        
            qc.crx(theta[count],qr[2],qr[1])    
            count=count+1
        
            for i in range(4):
                qc.rz(theta[count],qr[i])
                count=count+1        
        
            for i in range(4):
                qc.rx(theta[count],qr[i])
                count=count+1
                
        
            qc.crx(theta[count],qr[3],qr[2])
            count=count+1
        
            qc.crx(theta[count],qr[1],qr[0])
            count=count+1
        
        
        
    
            for i in range(4):
                qc.rz(theta[count],qr[i])
                count=count+1         
            for i in range(4):
                qc.rx(theta[count],qr[i])
                count=count+1           
    
    
    
    return qc

def example10( qr, cr, theta, num_qubits:int , L:int): #Circ 10 paper
    qc= QuantumCircuit(qr, cr)
    count=0
    repeat=1
    for i in range(4):
        qc.ry(theta[count],qr[i])
        count=count+1
    
    for l in range(L):

        qc.cz(qr[3],qr[2])
        qc.cz(qr[2],qr[1])
        qc.cz(qr[1],qr[0])
        qc.cz(qr[3],qr[0])



        
        for i in range(4):
            qc.ry(theta[count],qr[i])
            count=count+1

        
            
    if repeat!=0:             
        qc.barrier(qr)
        for l in range(L):
            for i in range(4):
                qc.ry(theta[count],qr[i])
                count=count+1
            
            qc.cz(qr[3],qr[0])            
            qc.cz(qr[1],qr[0])
            qc.cz(qr[2],qr[1])
            qc.cz(qr[3],qr[2])
        
        for i in range(4):
            qc.ry(theta[count],qr[i])
            count=count+1
    
    return qc
