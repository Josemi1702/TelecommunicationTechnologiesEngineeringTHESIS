"""
Machine Learning MODULE: this module is used to use machine learning functions 
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import time
from math import pi
from pylab import cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

#Clasification and Clustering tools
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import cross_validate,cross_val_score, StratifiedKFold
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, homogeneity_score, completeness_score, f1_score #Some clustering metrics



#Quiskit basics
from qiskit.circuit import QuantumCircuit, Parameter, ParameterVector
from qiskit.circuit.library import ZZFeatureMap
from qiskit.visualization import circuit_drawer

#Quantum kernel tools
from qiskit_machine_learning.algorithms import PegasosQSVC
from qiskit.algorithms.optimizers import SPSA
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.datasets import ad_hoc_data
from qiskit.primitives import Sampler
from qiskit.utils import algorithm_globals
from qiskit_machine_learning.kernels import TrainableFidelityQuantumKernel
from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainer

#------------------------Pegasus---------------------------
#(I need to add a new function in quiskit class to execute cross validation

import logging
from datetime import datetime
from typing import Dict
import inspect
from sklearn.base import ClassifierMixin
from qiskit_machine_learning.algorithms.serializable_model import SerializableModelMixin
from qiskit_machine_learning.exceptions import QiskitMachineLearningError
from qiskit_machine_learning.kernels import BaseKernel
from qiskit.algorithms.state_fidelities import ComputeUncompute


global LINEAL_KERNEL, POLINOMIC_KERNEL, RBF_KERNEL
LINEAL_KERNEL = 11
POLINOMIC_KERNEL=LINEAL_KERNEL+1
RBF_KERNEL= LINEAL_KERNEL+2

n=1
optimicer_glob=False

def execute_SVM(x, y, num_qcir, L:int, adhoc, pegasus, optimicer, dataset:str):
    
    #Random Seed
    algorithm_globals.random_seed = 1234809
    
    
    #Data preprocesing
    if adhoc: #Dataset adhoc
        x, y, x_test, y_test, adhoc_total = ad_hoc_data(
            training_size=50,
            test_size=0,
            n=2,
            gap=0.3,
            plot_data=False,
            one_hot=False,
            include_sample_total=True,
        )
    else: #Other datasets
        x = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(x)
        x=np.array(x)
        y=np.array(y)
    
    #Creating a feature map and procesing kernel
    feature_dimension = x.shape[1]
    
    #skf for cross validation
    skf= StratifiedKFold(5, shuffle=True, random_state=1)
    
    if num_qcir<LINEAL_KERNEL:
        
        feature_map =featuremap_generator(num_qcir, feature_dimension, L)
            
        sampler = Sampler()
        fidelity = ComputeUncompute(sampler=sampler)       
        kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)
            
        if pegasus==False:
            kernel=kernel.evaluate
            
    elif num_qcir==LINEAL_KERNEL:
        kernel='linear'
    elif num_qcir==POLINOMIC_KERNEL:
        kernel='poly'
    elif num_qcir==RBF_KERNEL:
        kernel='rbf'
    else:
        print("Type of kernel incorrect, please intoduce a valid number.")
        return
      
    #Selection of SVM algorithm 
        
    if pegasus:
        tau = 100 # number of steps performed during the training procedure
        C = 1000  # regularization parameter
        
        if optimicer:
            global optimicer_glob
            optimicer_glob=optimicer
            save_string=f"{dataset}: Circuit {num_qcir} (depth={L})"
            
            SVM = PegasosQSVC(quantum_kernel=kernel, C=C, num_steps=tau, feature_map=feature_map, save_string=save_string)
            folder="OPTIMICER_results"
        else:
            SVM = PegasosQSVC(quantum_kernel=kernel, C=C, num_steps=tau)
            folder="PEGASOS_results"
        #Model cross validation
        cv_results = cross_validate(SVM, x, y, cv=skf, scoring="accuracy", return_estimator=False, return_train_score=True)
        
        
    else:
        if optimicer:
            save_string=f"{dataset}: Circuit {num_qcir} (depth={L})"
            SVM = OptimizedQuantumKernelSVM(feature_map, save_string)
            folder="OPTIMICER_results"
            
        else:
            SVM = SVC(kernel=kernel)
            folder="SVM_results"
        #Model cross validation
        cv_results = cross_validate(SVM, x, y, cv=skf, scoring="accuracy", return_estimator=False, return_train_score=True)
        #n_jobs=5,
    
        
    
    
    #Terminal log
    print(f"-------Circuit number: {num_qcir}----L: {L}----Dataset: {dataset}---------------------")
    print(f"Training time: {cv_results['fit_time']} seconds --> ({round(np.mean(cv_results['fit_time']), 4)} +- {round(np.std(cv_results['fit_time']),4)})")
    print(f"Test time: {cv_results['score_time']} seconds --> ({round(np.mean(cv_results['score_time']), 4)} +- {round(np.std(cv_results['score_time']),4)})")
    print(f"SVM classification train score: {cv_results['train_score']} --> ({round(np.mean(cv_results['train_score']), 4)} +- {round(np.std(cv_results['train_score']),4)}) ")
    print(f"SVM classification test score: {cv_results['test_score']} --> ({round(np.mean(cv_results['test_score']), 4)} +- {round(np.std(cv_results['test_score']),4)})")
    if num_qcir<LINEAL_KERNEL:
        print(feature_map.decompose().draw(output="text"))

    
    
    #File log
    
    with open(f"./{folder}/logs/Depth_{L}.log", "a") as f:
        f.seek(0, 2) 
        f.write(f"-------Circuit number: {num_qcir}----L: {L}----Dataset: {dataset}---------------------\n")
        f.write(f"Training time: {cv_results['fit_time']} seconds --> ({round(np.mean(cv_results['fit_time']), 4)} +- {round(np.std(cv_results['fit_time']),4)})\n")
        f.write(f"Test time: {cv_results['score_time']} seconds --> ({round(np.mean(cv_results['score_time']), 4)} +- {round(np.std(cv_results['score_time']),4)})\n")
        f.write(f"SVM classification train score: {cv_results['train_score']} --> ({round(np.mean(cv_results['train_score']), 4)} +- {round(np.std(cv_results['train_score']),4)})\n")
        f.write(f"SVM classification test score: {cv_results['test_score']} --> ({round(np.mean(cv_results['test_score']), 4)} +- {round(np.std(cv_results['test_score']),4)})\n\n\n\n")

    
    #Save data
    #1. Training time
    with open(f"./{folder}/fit_times/Fit_time_{dataset}_depth_{L}.csv", "a") as f:
        f.seek(0, 2)
        f.write(f"{round(np.mean(cv_results['fit_time']), 4)} +- {round(np.std(cv_results['fit_time']),4)},")
    #2. Test time
    with open(f"./{folder}/score_times/Score_time_{dataset}_depth_{L}.csv", "a") as f:
        f.seek(0, 2)
        f.write(f"{round(np.mean(cv_results['score_time']), 4)} +- {round(np.std(cv_results['score_time']),4)},")
    #3. Train score
    with open(f"./{folder}/train_scores/Train_score_{dataset}_depth_{L}.csv", "a") as f:
        f.seek(0, 2)
        f.write(f"{round(np.mean(cv_results['train_score']), 4)} +- {round(np.std(cv_results['train_score']),4)},")
        
    #1. Test score
    with open(f"./{folder}/test_scores/Test_score_{dataset}_depth_{L}.csv", "a") as f:
        f.seek(0, 2)
        f.write(f"{round(np.mean(cv_results['test_score']), 4)} +- {round(np.std(cv_results['test_score']),4)},")

    with open(f"./{folder}/test_scores_mean/Test_score_{dataset}_depth_{L}.csv", "a") as f:
        f.seek(0, 2)
        f.write(f"{round(np.mean(cv_results['test_score']), 4)},")
        
    #2D representation
    #save_string=f"./SVM_results/plots/Expr_Circ_{num_qcir}_Depth_{L}_{dataset}.pdf"
    #title_string= f"{dataset}: Circuit {num_qcir} (depth={L})"
    #plot_2D(x,y, estimators[scores.index(max(scores))], save_string, title_string)
    #print("Representation finished.")



class QKTCallback:
    """Callback wrapper class."""

    def __init__(self) -> None:
        self._data = [[] for i in range(5)]

    def callback(self, x0, x1=None, x2=None, x3=None, x4=None):
        """
        Args:
            x0: number of function evaluations
            x1: the parameters
            x2: the function value
            x3: the stepsize
            x4: whether the step was accepted
        """
        self._data[0].append(x0)
        self._data[1].append(x1)
        self._data[2].append(x2)
        self._data[3].append(x3)
        self._data[4].append(x4)

    def get_callback_data(self):
        return self._data

    def clear_callback_data(self):
        self._data = [[] for i in range(5)]
        
def optimice_kernel(feature_map:QuantumCircuit, x, y, save_string):
    num_qbits= feature_map.num_qubits
    
    X_train, X_test, y_train, y_test = train_test_split(x, y)
    # Concatenate optimicer circuit
    training_params = ParameterVector("α", 2)
    fm0 = QuantumCircuit(num_qbits)
    for i in range(num_qbits):
        fm0.ry(training_params[0], i)
        fm0.rz(training_params[1], i)
             
    fm = fm0.compose(feature_map) #fm0+feature_map
    print(f"Trainable parameters: {training_params}")
    circuit_drawer(fm)
    
    # Instantiate quantum kernel
    quant_kernel = TrainableFidelityQuantumKernel(feature_map=fm, training_parameters=training_params)

    # Set up the optimizer
    cb_qkt = QKTCallback()
    spsa_opt = SPSA(maxiter=10, callback=cb_qkt.callback, learning_rate=0.05, perturbation=0.05)

    # Instantiate a quantum kernel trainer.
    qkt = QuantumKernelTrainer(
        quantum_kernel=quant_kernel, loss="svc_loss", optimizer=spsa_opt, initial_point=[np.pi / 2, np.pi / 2])
    
    # Train the kernel using QKT directly
    y_train2 = y_train.ravel()
    qka_results = qkt.fit(X_train, y_train2)
    optimized_kernel = qka_results.quantum_kernel
    print(qka_results)
    
    #----------plot---------
    
    plot_data = cb_qkt.get_callback_data()  # callback data
    fig, ax = plt.subplots(1, 1)
    ax.set_title(save_string)
    ax.plot([i + 1 for i in range(len(plot_data[0]))], np.array(plot_data[2]), c="k", marker="o")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")
    fig.tight_layout()
    
    global n
    
    save_string="./OPTIMICER_results/plots/"+str(n)+save_string+".pdf"
    n=n+1
    plt.savefig(save_string, format='pdf')
    plt.show()
    plt.close()

    return optimized_kernel

#New SVM class to add quantum kernel optimizations
class OptimizedQuantumKernelSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, feature_map, save_string):
        self.feature_map = feature_map
        self.clf = SVC()
        self.save_string=save_string
        
    def fit(self, X, y):
        qkernel= optimice_kernel(self.feature_map, X, y, self.save_string)
        self.clf.kernel = qkernel.evaluate
        self.clf.fit(X, y)
        return self
    
    def predict(self, X):
        return self.clf.predict(X)


def execute_SpectralClustering(x, y, num_qcir, L:int, n_clusters:int,  dataset:str):
    #Random Seed
    seed = 1234809
    algorithm_globals.random_seed = seed
    
    x = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(x)
    x=np.array(x)
    y=np.array(y)
    # Aprendizaje no supervisado no necesita test
    
    #Creating a feature map
    feature_dimension = x.shape[1]
    feature_map =featuremap_generator(num_qcir, feature_dimension, L)
    #print(feature_map.decompose().draw(output="latex", fold=20))
    
    #Procesing kernel
    if num_qcir<LINEAL_KERNEL:
        feature_map =featuremap_generator(num_qcir, feature_dimension, L)
        #print(feature_map.decompose().draw(output="latex", fold=20))
        
        sampler = Sampler()
        fidelity = ComputeUncompute(sampler=sampler)
        kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)
        
        matrix = kernel.evaluate(x_vec=x)
        affinity="precomputed"
    elif num_qcir==LINEAL_KERNEL:
        affinity='linear'
    elif num_qcir==POLINOMIC_KERNEL:
        affinity='poly'
    elif num_qcir==RBF_KERNEL:
        affinity='rbf'
    else:
        print("Type of kernel incorrect, please intoduce a valid number.")
        return
    
    
    spectralClustering = SpectralClustering(n_clusters, affinity=affinity, n_jobs=8)
    
    
    if num_qcir<LINEAL_KERNEL:
        start = time.time()
        cluster_labels = spectralClustering.fit_predict(matrix)
        elapsed = time.time() - start
        
        
        silhouette_avg = silhouette_score(matrix, cluster_labels)
        calinski_harabasz_metric= calinski_harabasz_score(matrix, cluster_labels) #Distancia entre clusteres
        davies_bouldin_metric= davies_bouldin_score(matrix, cluster_labels)
    else:
        start = time.time()
        cluster_labels = spectralClustering.fit_predict(x)
        elapsed = time.time() - start
        
        silhouette_avg = silhouette_score(x, cluster_labels)
        calinski_harabasz_metric= calinski_harabasz_score(x, cluster_labels) #Distancia entre clusteres
        davies_bouldin_metric= davies_bouldin_score(x, cluster_labels)
        
    score = normalized_mutual_info_score(cluster_labels, y)
    adjusted_rand_metric= adjusted_rand_score(y, cluster_labels)
    p_i = [sum(cluster_labels == i) / len(cluster_labels) for i in set(cluster_labels)]
    p_i=np.array(p_i)
    entropy= np.sum(-p_i * np.log2(p_i))

    
    # Homogeneidad, completitud y puntuación F
    homogeneity = homogeneity_score(y, cluster_labels)
    completeness = completeness_score(y, cluster_labels)
    f1 = f1_score(y, cluster_labels, average='weighted')
 
    



    

    print(f"------------------------------Circuit number: {num_qcir}------------------------------")
    print(f"Fit and predict time: {round(elapsed)} seconds")
    print(f"Spectral Clustering  test score: {score}")
    print(f"Spectral Clustering  silhouette: {silhouette_avg}")
    print(f"Spectral Clustering  calinski harabasz metric: {calinski_harabasz_metric}")
    print(f"Spectral Clustering  davies bouldin metric: {davies_bouldin_metric}")
    print(f"Spectral Clustering  adjusted rand metric: {adjusted_rand_metric}")
    print(f"Spectral Clustering  entropy: {entropy}")
    print(f"Homogeneity: {homogeneity}")
    print(f"Completness: {completeness}")
    print(f"F score: {f1}")
    print("\n\n")
    
    #File log
    
    with open(f"./SpectralClustering2/logs/Depth_{L}.log", "a") as f:
        f.seek(0, 2) 
        f.write
        f.write(f"--------------------------Dataset {dataset}   Circuit number: {num_qcir}------------------------------\n")
        f.write(f"Fit and predict time: {round(elapsed)} seconds\n")
        f.write(f"Spectral Clustering  test score: {score}\n")
        f.write(f"Spectral Clustering  silhouette: {silhouette_avg}\n")
        f.write(f"Spectral Clustering  calinski harabasz metric: {calinski_harabasz_metric}\n")
        f.write(f"Spectral Clustering  davies bouldin metric: {davies_bouldin_metric}\n")
        f.write(f"Spectral Clustering  adjusted rand metric: {adjusted_rand_metric}\n")
        f.write(f"Spectral Clustering  entropy: {entropy}\n")
        f.write(f"Homogeneity: {homogeneity}\n")
        f.write(f"Completness: {completeness}\n")
        f.write(f"F score: {f1}\n")
        f.write("\n\n")
        
    #save results
    s = pd.Series(cluster_labels)
    s.to_csv(f"./SpectralClustering2/results/{dataset}_Circuit_{num_qcir}_(depth={L}).csv", index=False)


    
    #Plot
    clusters = np.unique(cluster_labels)
    data_by_cluster = [x[cluster_labels == c] for c in clusters]
    
    colors = ['#ff7f0e', '#1f77b4', '#2ca02c'] 
    for i, data in enumerate(data_by_cluster):
        plt.scatter(data[:, 0], data[:, 1], color=colors[i], label=f'Cluster {i}')
    plt.legend()
    plt.title(f"{dataset}: Circuit {num_qcir} (depth={L})")
    plt.xticks([])
    plt.yticks([])
    save_string=f"./SpectralClustering2/plots/{dataset}_Circuit_{num_qcir}_(depth={L}).pdf"
    plt.savefig(save_string, format='pdf')
    plt.show()
    plt.close()
    plt.show()
    return 


def plot_2D(x,y, SVM, save_string, title_string):
    feature_dimension = x.shape[1]
    if feature_dimension==2:
        #Color map that I like
        cmap_colors, cmap_name, colors = ['#1f77b4', '#ff7f0e'], 'my_cmap', []
        for i, color in enumerate(cmap_colors):
            colors.append((i / (len(cmap_colors)-1), color))
        mycmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors)
        
        x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
        y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
        
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))
        
        
        Z = SVM.predict(np.c_[xx.ravel(), yy.ravel()])
        #paralel computing to predict
        #pool = mp.Pool(processes=5)
        #results = [pool.apply_async(predict, args=(X_chunk, SVM)) for X_chunk in np.array_split(np.c_[xx.ravel(), yy.ravel()], 5)]
        #Z = np.concatenate([r.get() for r in results])
        
        
        Z = Z.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, cmap=mycmap, alpha=0.75)  
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolor='black')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        
        plt.title(title_string)
        colores = np.array(['#1f77b4',  '#ff7f0e'])
        plt.scatter(x[:, 0], x[:, 1], c=colores[y], edgecolor='black')
        
        #Labels
        class_labels = ['Class 1', 'Class 2']  # Cambie estas etiquetas según las clases de sus datos
        scatter_handles = []
        for i in range(len(class_labels)):
            handle = plt.scatter([], [], c=colores[i], label=class_labels[i], edgecolor='black')
            scatter_handles.append(handle)
        plt.legend(handles=scatter_handles)


        #plt.show()
         #Save pic    
        plt.savefig(save_string, format='pdf')
        plt.close()
        


def featuremap_generator(num_qcir, num_qbits, L:int):
    
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
        10: circ10
    

    }
    
    if num_qcir in switch_dict:
        return switch_dict[num_qcir](feature_dimension=num_qbits, reps=L)
    else:
        print("LETS PROBE NORMAL KERNELS")


def circ1(feature_dimension, reps:int):
    
    theta = Parameter('θ')
    theta_list = ParameterVector('θ', length=feature_dimension)


    qc = QuantumCircuit(feature_dimension)
    for r in range(0,reps):
        for i in range(0, feature_dimension):
            qc.rz(theta_list[i],i)
        qc.barrier()

    return qc

def circ2(feature_dimension, reps:int):
    
    theta = Parameter('θ')
    theta_list = ParameterVector('θ', length=feature_dimension)


    qc = QuantumCircuit(feature_dimension)
    for r in range(0,reps):
        for i in range(0, feature_dimension):
            qc.ry(theta_list[i],i)
            qc.rz(theta_list[i],i)
        qc.barrier()

    return qc

def circ3(feature_dimension, reps:int):
    
    theta = Parameter('θ')
    theta_list = ParameterVector('θ', length=feature_dimension)


    qc = QuantumCircuit(feature_dimension)
    for r in range(0,reps):
        for i in range(0, feature_dimension):
            qc.ry(theta_list[i],i)
            qc.rz(theta_list[i],i)
            if i<feature_dimension-1:
                qc.cnot(i, i+1)
        qc.barrier()

    return qc

def circ4(feature_dimension, reps:int):
    
    theta = Parameter('θ')
    theta_list = ParameterVector('θ', length=feature_dimension)


    qc = QuantumCircuit(feature_dimension)
    for r in range(0,reps):
        for i in range(0, feature_dimension):
            qc.rx(theta_list[i],i)
            if i<feature_dimension-1:
                qc.cnot(i+1, i)
            qc.rz(theta_list[i],i)
        qc.barrier()

    return qc

def circ5(feature_dimension, reps:int):
    
    theta = Parameter('θ')
    theta_list = ParameterVector('θ', length=feature_dimension)


    qc = QuantumCircuit(feature_dimension)
    for r in range(0,reps):
        for i in range(0, feature_dimension):
            qc.ry(theta_list[i],i)
            qc.rz(theta_list[i],i)
            if i<feature_dimension-1:
                qc.crx(theta_list[i+1],i+1, i)
        qc.barrier()

    return qc

def circ6(feature_dimension, reps:int):
    
    theta = Parameter('θ')
    theta_list = ParameterVector('θ', length=feature_dimension)


    qc = QuantumCircuit(feature_dimension)
    for r in range(0,reps):
        for i in range(0, feature_dimension):
            qc.rx(theta_list[i],i)
            qc.ry(theta_list[i],i)
            if i<feature_dimension-1:
                qc.cnot(i, i+1)
            qc.rz(theta_list[i],i)
            qc.rx(theta_list[i],i)
        qc.barrier()

    return qc

def circ7(feature_dimension, reps:int):
    
    theta = Parameter('θ')
    theta_list = ParameterVector('θ', length=feature_dimension)


    qc = QuantumCircuit(feature_dimension)
    for r in range(0,reps):
        for i in range(0, feature_dimension):
            qc.h(i)
        for i in range(0, feature_dimension):
            qc.rx(theta_list[i],i)
            if i<feature_dimension-1:
                qc.crz(theta_list[i], i, i+1)
            qc.ry(theta_list[i],i)
        qc.barrier()

    return qc

def circ8(feature_dimension, reps:int):
    
    theta = Parameter('θ')
    theta_list = ParameterVector('θ', length=feature_dimension)
    

    qc = QuantumCircuit(feature_dimension)
    for r in range(reps):
        for i in range(0, feature_dimension):
            qc.h(i)
            qc.p(2*theta_list[i], i)

        for i in range(0, feature_dimension): 
            if(i>0):
                j=i
                while(j>0):
                    qc.cnot(j-1,i)
                    j-=1
            qc.rx(theta_list[i],i)
            qc.rz(theta_list[i],i)
        qc.barrier()

    return qc

def circ9(feature_dimension, reps:int):
    
    theta = Parameter('θ')
    theta_list = ParameterVector('θ', length=feature_dimension)
    

    qc = QuantumCircuit(feature_dimension)
    for r in range(reps):
        for i in range(0, feature_dimension):
            qc.h(i)
            if i>0:
                qc.rx(theta_list[i]-theta_list[i-1], i)
            else:
                qc.rx(theta_list[i], i)
            qc.h(i)
                

        for i in range(0, feature_dimension):
            if(i>0):
                j=0
                
                while(j<i):
                    qc.cz(j,i) 
                    j+=1
                qc.ry((pi+theta_list[i-1])*(pi-theta_list[i]),i)
                qc.rz((pi-theta_list[i-1])*(pi+theta_list[i]),i)
                j=i
                while(j>0):
                    qc.cz(j-1,i)
                    j-=1
            
        qc.barrier()

    return qc

def circ10(feature_dimension, reps:int):
    
    theta = Parameter('θ')
    theta_list = ParameterVector('θ', length=feature_dimension)
    

    qc = QuantumCircuit(feature_dimension)
    for r in range(reps):
        for i in range(0, feature_dimension):
            qc.h(i)
            qc.ry(2*pi* theta_list[i], i)
            qc.h(i)
            

        for i in range(0, feature_dimension):
            if i>0:
                j=0
                while(j<i):
                    qc.cz(j,i)
                    qc.cnot(j,i )
                    if i>1:
                        qc.p(theta_list[i]*theta_list[i-1]*theta_list[i-2], i)
                    else:
                        qc.p(theta_list[i]*theta_list[i-1], i) 
                    qc.cnot(i,j )
                    qc.cz(i,j)
                    j+=1
            qc.barrier()
           
        for i in range(0, feature_dimension):
            qc.h(i)
            
        qc.barrier()

    return qc




# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Pegasos Quantum Support Vector Classifier."""



logger = logging.getLogger(__name__)


class PegasosQSVC(ClassifierMixin, SerializableModelMixin):
    r"""
    Implements Pegasos Quantum Support Vector Classifier algorithm. The algorithm has been
    developed in [1] and includes methods ``fit``, ``predict`` and ``decision_function`` following
    the signatures
    of `sklearn.svm.SVC <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_.
    This implementation is adapted to work with quantum kernels.

    **Example**

    .. code-block:: python

        quantum_kernel = FidelityQuantumKernel()

        pegasos_qsvc = PegasosQSVC(quantum_kernel=quantum_kernel)
        pegasos_qsvc.fit(sample_train, label_train)
        pegasos_qsvc.predict(sample_test)

    **References**
        [1]: Shalev-Shwartz et al., Pegasos: Primal Estimated sub-GrAdient SOlver for SVM.
            `Pegasos for SVM <https://home.ttic.edu/~nati/Publications/PegasosMPB.pdf>`_

    """

    FITTED = 0
    UNFITTED = 1

    # pylint: disable=invalid-name
    def __init__(
        self,
        quantum_kernel: BaseKernel | None = None,
        C: float = 1.0,
        num_steps: int = 1000,
        precomputed: bool = False,
        seed: int | None = None,
        feature_map: QuantumCircuit | None = None,
        save_string: str | None = None,
    ) -> None:
        """
        Args:
            quantum_kernel: a quantum kernel to be used for classification. Has to be ``None`` when
                a precomputed kernel is used.
            C: Positive regularization parameter. The strength of the regularization is inversely
                proportional to C. Smaller ``C`` induce smaller weights which generally helps
                preventing overfitting. However, due to the nature of this algorithm, some of the
                computation steps become trivial for larger ``C``. Thus, larger ``C`` improve
                the performance of the algorithm drastically. If the data is linearly separable
                in feature space, ``C`` should be chosen to be large. If the separation is not
                perfect, ``C`` should be chosen smaller to prevent overfitting.

            num_steps: number of steps in the Pegasos algorithm. There is no early stopping
                criterion. The algorithm iterates over all steps.
            precomputed: a boolean flag indicating whether a precomputed kernel is used. Set it to
                ``True`` in case of precomputed kernel.
            seed: a seed for the random number generator

        Raises:
            ValueError:
                - if ``quantum_kernel`` is passed and ``precomputed`` is set to ``True``. To use
                a precomputed kernel, ``quantum_kernel`` has to be of the ``None`` type.
            TypeError:
                - if ``quantum_kernel`` neither instance of
                  :class:`~qiskit_machine_learning.kernels.BaseKernel` nor ``None``.
        """
        
        if precomputed:
            if quantum_kernel is not None:
                raise ValueError("'quantum_kernel' has to be None to use a precomputed kernel")
        

        self._quantum_kernel = quantum_kernel
        self._precomputed = precomputed
        self._num_steps = num_steps
        if seed is not None:
            algorithm_globals.random_seed = seed

        if C > 0:
            self.C = C
        else:
            raise ValueError(f"C has to be a positive number, found {C}.")
        
         #New atributes for optimization in fit function
        if feature_map is None:
            self.feature_map=QuantumCircuit()
        else:
            self.feature_map= feature_map
        if save_string is None:
            self.save_string=""
        else:
            self.save_string=save_string

        # these are the parameters being fit and are needed for prediction
        self._alphas: Dict[int, int] | None = None
        self._x_train: np.ndarray | None = None
        self._n_samples: int | None = None
        self._y_train: np.ndarray | None = None
        self._label_map: Dict[int, int] | None = None
        self._label_pos: int | None = None
        self._label_neg: int | None = None

        # added to all kernel values to include an implicit bias to the hyperplane
        self._kernel_offset = 1

        # for compatibility with the base SVC class. Set as unfitted.
        self.fit_status_ = PegasosQSVC.UNFITTED
        
       

    # pylint: disable=invalid-name
    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> "PegasosQSVC":
        """Fit the model according to the given training data.

        Args:
            X: Train features. For a callable kernel (an instance of
               :class:`~qiskit_machine_learning.kernels.BaseKernel`) the shape
               should be ``(n_samples, n_features)``, for a precomputed kernel the shape should be
               ``(n_samples, n_samples)``.
            y: shape (n_samples), train labels . Must not contain more than two unique labels.
            sample_weight: this parameter is not supported, passing a value raises an error.

        Returns:
            ``self``, Fitted estimator.

        Raises:
            ValueError:
                - X and/or y have the wrong shape.
                - X and y have incompatible dimensions.
                - y includes more than two unique labels.
                - Pre-computed kernel matrix has the wrong shape and/or dimension.

            NotImplementedError:
                - when a sample_weight which is not None is passed.
        """
        #Optimicer
        global optimicer_glob
        if optimicer_glob:
            qkernel= optimice_kernel(self.feature_map, X, y, self.save_string)
            self.quantum_kernel = qkernel
        
        # check whether the data have the right format
        if np.ndim(X) != 2:
            raise ValueError("X has to be a 2D array")
        if np.ndim(y) != 1:
            raise ValueError("y has to be a 1D array")
        if len(np.unique(y)) != 2:
            raise ValueError("Only binary classification is supported")
        if X.shape[0] != y.shape[0]:
            raise ValueError("'X' and 'y' have to contain the same number of samples")
        if self._precomputed and X.shape[0] != X.shape[1]:
            raise ValueError(
                "For a precomputed kernel, X should be in shape (n_samples, n_samples)"
            )
        if sample_weight is not None:
            raise NotImplementedError(
                "Parameter 'sample_weight' is not supported. All samples have to be weighed equally"
            )
        # reset the fit state
        self.fit_status_ = PegasosQSVC.UNFITTED

        # the algorithm works with labels in {+1, -1}
        self._label_pos = np.unique(y)[0]
        self._label_neg = np.unique(y)[1]
        self._label_map = {self._label_pos: +1, self._label_neg: -1}

        # the training data are later needed for prediction
        self._x_train = X
        self._y_train = y
        self._n_samples = X.shape[0]

        # empty dictionary to represent sparse array
        self._alphas = {}

        t_0 = datetime.now()
        # training loop
        for step in range(1, self._num_steps + 1):
            # for every step, a random index (determining a random datum) is fixed
            i = algorithm_globals.random.integers(0, len(y))

            value = self._compute_weighted_kernel_sum(i, X, training=True)

            if (self._label_map[y[i]] * self.C / step) * value < 1:
                # only way for a component of alpha to become non zero
                self._alphas[i] = self._alphas.get(i, 0) + 1

        self.fit_status_ = PegasosQSVC.FITTED

        logger.debug("fit completed after %s", str(datetime.now() - t_0)[:-7])

        return self
   
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        out["quantum_kernel"] = self.quantum_kernel
        out["C"] = self.C
        out["num_steps"] = self.num_steps
        out["precomputed"] = self.precomputed
        out["save_string"] = self.save_string
        out["feature_map"] = self.feature_map
        #out["seed"] = self.seed
        return out
    
    # pylint: disable=invalid-name
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Perform classification on samples in X.

        Args:
            X: Features. For a callable kernel (an instance of
               :class:`~qiskit_machine_learning.kernels.BaseKernel`) the shape
               should be ``(m_samples, n_features)``, for a precomputed kernel the shape should be
               ``(m_samples, n_samples)``. Where ``m`` denotes the set to be predicted and ``n`` the
               size of the training set. In that case, the kernel values in X have to be calculated
               with respect to the elements of the set to be predicted and the training set.

        Returns:
            An array of the shape (n_samples), the predicted class labels for samples in X.

        Raises:
            QiskitMachineLearningError:
                - predict is called before the model has been fit.
            ValueError:
                - Pre-computed kernel matrix has the wrong shape and/or dimension.
        """

        t_0 = datetime.now()
        values = self.decision_function(X)
        y = np.array([self._label_pos if val > 0 else self._label_neg for val in values])
        logger.debug("prediction completed after %s", str(datetime.now() - t_0)[:-7])

        return y


    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the decision function for the samples in X.

        Args:
            X: Features. For a callable kernel (an instance of
               :class:`~qiskit_machine_learning.kernels.BaseKernel`) the shape
               should be ``(m_samples, n_features)``, for a precomputed kernel the shape should be
               ``(m_samples, n_samples)``. Where ``m`` denotes the set to be predicted and ``n`` the
               size of the training set. In that case, the kernel values in X have to be calculated
               with respect to the elements of the set to be predicted and the training set.

        Returns:
            An array of the shape (n_samples), the decision function of the sample.

        Raises:
            QiskitMachineLearningError:
                - the method is called before the model has been fit.
            ValueError:
                - Pre-computed kernel matrix has the wrong shape and/or dimension.
        """
        if self.fit_status_ == PegasosQSVC.UNFITTED:
            raise QiskitMachineLearningError("The PegasosQSVC has to be fit first")
        if np.ndim(X) != 2:
            raise ValueError("X has to be a 2D array")
        if self._precomputed and self._n_samples != X.shape[1]:
            raise ValueError(
                "For a precomputed kernel, X should be in shape (m_samples, n_samples)"
            )

        values = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            values[i] = self._compute_weighted_kernel_sum(i, X, training=False)

        return values


    def _compute_weighted_kernel_sum(self, index: int, X: np.ndarray, training: bool) -> float:
        """Helper function to compute the weighted sum over support vectors used for both training
        and prediction with the Pegasos algorithm.

        Args:
            index: fixed index distinguishing some datum
            X: Features
            training: flag indicating whether the loop is used within training or prediction

        Returns:
            Weighted sum of kernel evaluations employed in the Pegasos algorithm
        """
        # non-zero indices corresponding to the support vectors
        support_indices = list(self._alphas.keys())

        # for training
        if training:
            # support vectors
            x_supp = X[support_indices]
        # for prediction
        else:
            x_supp = self._x_train[support_indices]
        if not self._precomputed:
            # evaluate kernel function only for the fixed datum and the support vectors
            kernel = self._quantum_kernel.evaluate(X[index], x_supp) + self._kernel_offset
        else:
            kernel = X[index, support_indices]

        # map the training labels of the support vectors to {-1,1}
        y = np.array(list(map(self._label_map.get, self._y_train[support_indices])))
        # weights for the support vectors
        alphas = np.array(list(self._alphas.values()))
        # this value corresponds to a sum of kernel values weighted by their labels and alphas
        value = np.sum(alphas * y * kernel)

        return value

    @property
    def quantum_kernel(self) -> BaseKernel:
        """Returns quantum kernel"""
        return self._quantum_kernel

    @quantum_kernel.setter
    def quantum_kernel(self, quantum_kernel: BaseKernel):
        """
        Sets quantum kernel. If previously a precomputed kernel was set, it is reset to ``False``.
        """

        self._quantum_kernel = quantum_kernel
        # quantum kernel is set, so we assume the kernel is not precomputed
        self._precomputed = False

        # reset training status
        self._reset_state()

    @property
    def num_steps(self) -> int:
        """Returns number of steps in the Pegasos algorithm."""
        return self._num_steps

    @num_steps.setter
    def num_steps(self, num_steps: int):
        """Sets the number of steps to be used in the Pegasos algorithm."""
        self._num_steps = num_steps

        # reset training status
        self._reset_state()

    @property
    def precomputed(self) -> bool:
        """Returns a boolean flag indicating whether a precomputed kernel is used."""
        return self._precomputed

    @precomputed.setter
    def precomputed(self, precomputed: bool):
        """Sets the pre-computed kernel flag. If ``True`` is passed then the previous kernel is
        cleared. If ``False`` is passed then a new instance of
        :class:`~qiskit_machine_learning.kernels.FidelityQuantumKernel` is created."""
        self._precomputed = precomputed
        if precomputed:
            # remove the kernel, a precomputed will
            self._quantum_kernel = None
        else:
            # re-create a new default quantum kernel
            self._quantum_kernel = FidelityQuantumKernel()

        # reset training status
        self._reset_state()

    def _reset_state(self):
        """Resets internal data structures used in training."""
        self.fit_status_ = PegasosQSVC.UNFITTED
        self._alphas = None
        self._x_train = None
        self._n_samples = None
        self._y_train = None
        self._label_map = None
        self._label_pos = None
        self._label_neg = None