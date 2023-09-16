# dtPINN: Unified temporal propagation strategies for Physics Informed Neural Networks (PINNs) and their decompositions

## Abstract

Physics-informed neural networks (PINNs) as a means of solving partial differential equations (PDE) have garnered much attention in the Computational Science and Engineering (CS&E) world. However, a recent topic of interest is exploring various training (i.e., optimization) challenges – in particular, arriving at poor local minima in the optimization landscape results in a PINN approximation giving an inferior, and sometimes trivial, solution when solving forward time-dependent PDEs with no data. This problem is also found in, and in some sense more difficult, with domain decomposition strategies such as temporal decomposition using XPINNs. We furnish examples and explanations for different training challenges, their cause, and how they relate to information propagation and temporal decomposition. We then propose a new stacked-decomposition method that bridges the gap between time-marching PINNs and XPINNs. We also introduce significant computational speed-ups by using  transfer learning concepts to initialize subnetworks in the domain and loss tolerance-based propagation for the subdomains. Finally, we formulate a new time-sweeping collocation point algorithm inspired by the previous PINNs causality literature, which our framework can still describe, and provides a significant computational speed-up via reduced-cost collocation point segmentation. The proposed methods form our unified framework, which overcomes training challenges in PINNs and XPINNs for time-dependent PDEs by respecting the causality in multiple forms and improving scalability by limiting the computation required per optimization iteration. Finally, we provide numerical results for these methods on baseline PDE problems for which unmodified PINNs and XPINNs struggle to train.

## Citation

Penwarden, M., Jagtap A.D., Zhe, S., Karniadakis, G.E., Kirby, R.M., 2023. A unified scalable framework for causal sweeping strategies for Physics-Informed Neural Networks (PINNs) and their temporal decompositions. Journal of Computational Physics, 493, 2023, 112464.

https://doi.org/10.1016/j.jcp.2023.112464

## Description of codebase
This repository is split into four main parts: data, results, source, and example. If you would like to train new models on new problems, please refer to the format for the model trained in the Example folder. Over the course of the project, the source files were modified and functionality was added. Therefore, the format of earlier model runs may be deprecated due to changes in function inputs, file locations, etc. Older files in the Results folder will still run if these are updated. However, they are primarily left there for documentation purposes since the Jupyter Notebooks and resulting figures archive the manuscript results.

Data - The data folder contains the reference PDE solutions to the problems solved in the manuscript and the MATLAB code used to generate the solution data from the Chebfun toolbox (https://www.chebfun.org/).

Results - The results folder contains PINN model runs that directly contributed to the results reported in the manuscript and runs that did not make it into the manuscript. Models run in this folder (Convection & Allen-Cahn) may be deprecated and won't run without modification since the codebase (source files) was updated throughout the project to add more functionality.  

Source - The source folder contains the final version of the unified framework for PINNs and their temporal decompositions. These foundational classes and helper functions are called in the Jupyter Notebook files to run differently configured models on various problems under various conditions. 

Example - The example folder contains an up-to-date Jupyter Notebook that should run without modification and calls the source files to define and run models per the proposed manuscript framework. If you want to solve new problems with this code, modify the settings in this example. *Note: Animation setting not currently working with window-sweeping on. 

## Examples

### Example: Convection

https://github.com/mpenwarden/dtPINN/assets/74904442/16b3223a-1210-4733-8dda-45ab3e9fd788

https://github.com/mpenwarden/dtPINN/assets/74904442/6b5bf089-b9fa-4952-9997-3780d3e9734a

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
