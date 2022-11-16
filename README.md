# dtPINN: Unified temporal propagation strategies for Physics Informed Neural Networks (PINNs) and their decompositions

## Abstract

Physics-informed neural networks (PINNs) as a means of solving partial differential equations (PDE) have garnered much attention in the Computational Science and Engineering (CS\&E) world. However, a recent topic of interest in this area has developed around PINNs optimization difficulty and its failure to solve problems once stuck in trivial solutions, which act as poor local minima. This issue also extends to and is, in some sense, more difficult in domain decomposition strategies such as temporal decomposition using XPINNs. We also introduce significant computational speed-ups by using transfer learning concepts to initialize sub-networks in the domain. We also formulate a new time-sweeping collocation point algorithm, inspired by previous PINNs causality literature, which provides significant computational speed up. We reframe these causality concepts into a generalized information propagation framework in which any method or combination of such can be described. Using the idea of information propagation, we propose a new stacked decomposition model which bridges the gap between time-marching PINNs and standard xPINNs. These proposed methods overcome failure modes in PINNs and xPINNs by respecting causality in multiple forms as well as improve scalably by limiting the amount of computation that has to be done per optimization iteration. Finally, we provide numerical results for these methods on novel PDE problems which PINNs struggle to optimize. 

## Citation

TBD

## Examples

### Example 1 (preliminary): Convection

https://github.com/mpenwarden/dtPINN/tree/main/figures/convection_test_1.mp4

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
