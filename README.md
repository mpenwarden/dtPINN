# dtPINN: Unified temporal propagation strategies for Physics Informed Neural Networks (PINNs) and their decompositions

## Abstract

Physics-informed neural networks (PINNs) as a means of solving partial differential equations (PDE) have garnered much attention in the Computational Science and Engineering (CS&E) world. However, a recent topic of interest is exploring various training (i.e., optimization) challenges – in particular, arriving at poor local minima in the optimization landscape results in a PINN approximation giving an inferior, and sometimes trivial, solution when solving forward time-dependent PDEs with no data. This problem is also found in, and in some sense more difficult, with domain decomposition strategies such as temporal decomposition using XPINNs. We furnish examples and explanations for different training challenges, their cause, and how they relate to information propagation and temporal decomposition. We then propose a new stacked-decomposition method that bridges the gap between time-marching PINNs and XPINNs. We also introduce significant computational speed-ups by using  transfer learning concepts to initialize subnetworks in the domain and loss tolerance-based propagation for the subdomains. Finally, we formulate a new time-sweeping collocation point algorithm inspired by the previous PINNs causality literature, which our framework can still describe, and provides a significant computational speed-up via reduced-cost collocation point segmentation. The proposed methods form our unified framework, which overcomes training challenges in PINNs and XPINNs for time-dependent PDEs by respecting the causality in multiple forms and improving scalability by limiting the computation required per optimization iteration. Finally, we provide numerical results for these methods on baseline PDE problems for which unmodified PINNs and XPINNs struggle to train.

## Citation

Penwarden, M., Jagtap A.D., Zhe, S., Karniadakis, G.E., Kirby, R.M., 2023. A unified scalable framework for causal sweeping strategies for Physics-Informed Neural Networks (PINNs) and their temporal decompositions. Journal of Computational Physics, 2023, 112464.

https://doi.org/10.1016/j.jcp.2023.112464

## Examples

### Example: Convection

https://user-images.githubusercontent.com/74904442/202097110-31b74d6b-888f-4b32-ada7-9f13b145e53b.mp4

https://user-images.githubusercontent.com/74904442/202097173-3bb20175-1862-4e39-967e-4a9759720d06.mp4

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
