Explain what task you struggled with the most, and how you overcame it.
The most challenging part was interpreting the one-line question with limited context. Later, while working on the solution, I initially struggled with noise in the output. I resolved this by switching from bilateral filtering to WLS filtering in the post-processing step.

Research NeRF analysis methods and write an overview of how it works in no longer than 2 paragraphs. 

Neural Radiance Fields (NeRF) are a deep learning–based method for synthesizing novel views of a 3D scene from a sparse set of 2D images. Instead of explicitly modeling geometry, NeRF represents a scene as a continuous volumetric function that maps 3D spatial coordinates and viewing directions to color and density. A multilayer perceptron (MLP) is trained using differentiable volume rendering: rays are cast from a virtual camera through the scene, and the MLP predicts radiance and opacity along each sampled point. The predicted colors are integrated to form the final pixel value, and the difference between rendered and real images provides the training loss.

Analysis methods for NeRF often focus on efficiency, quality, and generalization. Standard NeRF is computationally expensive, so variants like Instant-NGP use hash grids to accelerate training and inference. Others incorporate depth supervision, semantic priors, or hybrid representations (e.g., combining point clouds or voxels with NeRF) to improve accuracy and robustness. Researchers also analyze NeRF through evaluations such as rendering fidelity, reconstruction accuracy, generalization across viewpoints, and scalability to large or dynamic scenes. Together, these methods demonstrate how NeRF transforms sparse 2D data into high-fidelity 3D representations while balancing trade-offs between detail, speed, and generalization..

Write an overview of an aspect you found most interesting in no longer than 1 paragraph.

One of the most interesting aspects I found was how disparity maps can be converted into 3D point clouds using camera calibration parameters. It’s fascinating that just by knowing the focal length, baseline, and principal points from calibration, we can reconstruct the real-world geometry of a scene from two flat images.
