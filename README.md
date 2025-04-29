# Frequency-adaptive-Multi-scale-Deep-Neural-Networks

1D poisson example

Multi-scale deep neural networks (MscaleDNNs) with downing-scaling mapping have demonstrated superiority over traditional DNNs in approximating target functions characterized by high frequency features. However, the performance of MscaleDNNs heavily depends on the parameters in the downing-scaling mapping, which limits their broader application. In this work, we establish a fitting error bound to explain why MscaleDNNs are advantageous for approximating high frequency functions. Building on this insight, we construct a hybrid feature embedding to enhance the accuracy and robustness of the downing-scaling mapping. 
To reduce the dependency of MscaleDNNs on parameters in the downing-scaling mapping, we propose frequency-adaptive MscaleDNNs, which adaptively adjust these parameters based on a posterior error estimate that captures the frequency information of the fitted functions. Numerical examples, including wave propagation and  the propagation of a localized solution of the schr$\ddot{\text{o}}$dinger equation with a smooth potential near the semi-classical limit, are presented. 
These examples demonstrate that the frequency-adaptive MscaleDNNs improve accuracy by two to three orders of magnitude compared to standard MscaleDNNs.
