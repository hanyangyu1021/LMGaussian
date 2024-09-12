# LM-Gaussian: Boost Sparse-view 3D Gaussian Splatting with Large Model Priors
[**Paper**](https://arxiv.org/abs/2409.03456) | [**Project Page**](https://hanyangyu1021.github.io/lm-gaussian.github.io/) | [**Video**](https://www.youtube.com/watch?v=ic4luAY_Hvk)

Official implementation of LM-Gaussian: Boost Sparse-view 3D Gaussian Splatting with Large Model Priors

[Hanyang Yu](https://hanyangyu1021.github.io/), [Xiaoxiao Long](https://www.xxlong.site/) and [Ping Tan](https://ece.hkust.edu.hk/pingtan).


<p align="center"> All Code will be released soon...  🔨 🚧 </p>

Abstract: *We aim to address sparse-view reconstruction of a 3D scene by leveraging priors from large-scale vision models. While recent advancements such as 3D Gaussian Splatting (3DGS) have demonstrated remarkable success in 3D reconstruction, these methods typically necessitate hundreds of input images that densely capture the underlying scene, making them time-consuming and impractical for real-world applications. However, sparse-view reconstruction is inherently ill-posed and under-constrained, often resulting in inferior and incomplete outcomes. This is due to issues such as failed initialization, overfitting to input images, and a lack of detail. To mitigate these challenges, we introduce LM-Gaussian, a method capable of generating high-quality reconstructions from a limited number of images. Specifically, we propose a robust initialization module that leverages stereo priors to aid in the recovery of camera poses and the reliable initialization of point clouds. Additionally, a diffusion-based refinement is iteratively applied to incorporate image diffusion priors into the Gaussian optimization process to preserve intricate scene details. Finally, we utilize video diffusion priors to further enhance the rendered images for realistic visual effects. Overall, our approach significantly reduces the data acquisition requirements compared to previous 3DGS methods. We validate the effectiveness of our framework through experiments on various public datasets, demonstrating its potential for high- quality 360-degree scene reconstruction.*


## Rendering Results
<p align="center">
    <video controls>
        <source src="assets/barn3.mp4" type="video/mp4">
    </video>
</p>

<p align="center">
    <video controls>
        <source src="assets/horse3.mp4" type="video/mp4">
    </video>
</p>

<p align="center">
    <video controls>
        <source src="assets/family3.mp4" type="video/mp4">
    </video>
</p>

<p align="center">
    <video controls>
        <source src="assets/garden3.mp4" type="video/mp4">
    </video>
</p>

<p align="center">
    <video controls>
        <source src="assets/trunk3.mp4" type="video/mp4">
    </video>
</p>



## Method

Our method takes unposed sparse images as inputs. For example, we select 8 images from the Horse Scene to cover a 360-degree view. Initially, we utilize a Background-Aware Depth-guided Initialization Module to generate dense point clouds and camera poses (see Section IV-B). These variables act as the initialization for the Gaussian kernels. Subsequently, in the Multi-modal Regularized Gaussian Reconstruction Module (see Section IV-C), we collectively optimize the Gaussian network through depth, normal, and virtual-view regularizations. After this stage, we train a Gaussian Repair model capable of enhancing Gaussian-rendered new view images. These improved images serve as guides for the training network, iteratively restoring Gaussian details (see Section IV-D). Finally, we employ a scene enhancement module to further enhance the rendered images for realistic visual effects (see Section IV-E).
<p align="center">
    <img src="assets/overall.png">
</p>






## BibTeX

```bibtex
@misc{yu2024lmgaussianboostsparseview3d,
      title={LM-Gaussian: Boost Sparse-view 3D Gaussian Splatting with Large Model Priors}, 
      author={Hanyang Yu and Xiaoxiao Long and Ping Tan},
      year={2024},
      eprint={2409.03456},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.03456}, 
}
```
