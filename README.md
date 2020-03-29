# DistanceDrivenFDK
A Fast Reconstruction Algorithm Research For CBCT

This repo is used as a record of a research on 3d reconstruction algorithm (Distance-Driven Back Projection of FDK Algorithm) in 2018

In the field of cone-beam CT reconstruction, FDK type reconstruction algorithm is widely used. 
For this kind of algorithm, the image back-projection step is the most time-consuming one. 
In this research, a variety of back-projection methods are compared, and a version that uses spatial geometry properties, engineering skills, and derived algorithms is introduced for the Distance Driven back-projection algorithm. 
Both the algorithm and the optimized version are implemented using the CPU and the CUDA parallel computing platform.

Techniques:
   - Algorithms/Models: Three-dimensional reconstruction, back-projection reconstruction, FDK, Distance-driven
   - Frameworks/Libraries: RTK, ITK
   - Languages: C++, CUDA C
   
Highlight:
   - The Distance driven method with no open source code is implemented using C++ based on which improvement is made in the selection of voxels' boundary points using CPU. 
   - An optimized version is implemented using texture memory, hardware interpolation and integral images on the CUDA parallel computing platform.
   
Important References:
   - [Distance-driven projection and backprojection in three dimensions](https://iopscience.iop.org/article/10.1088/0031-9155/49/11/024/pdf)
   - [GPU-based Branchless Distance-Driven Projection and Backprojection](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5761753/)

<br>Part of the codes for cpu and gpu versions of distance-driven fdk algorithm are shown in this repo.
<br>You are welcome to contact [me](https://www.linkedin.com/in/yuhua-angela-ma-676a73184/) and ask any questions about this project.

# Further explanation

## Abstract
<br>Cone beam computed tomography (or CBCT) projects cone-shape X rays onto the objects and reconstructs the object using projected images. This imaging technique plays an important role in position verification of tumor radiotherapy. Studying the rapid imaging method of cone beam CT is of great significance for promoting its applications. The common algorithms for 3D reconstruction of cone beam CT include iterative reconstruction algorithm and back projection reconstruction algorithm. Iterative reconstruction algorithm is high in accuracy, but the reconstruction is slow while the reconstruction accuracy of back projection reconstruction algorithm can meet the needs of use while having a moderate reconstruction speed. How to improve the back projection speed is the focus of fast algorithm research. 
<br><br>This research analyzes the mathematical principle of CT reconstruction based on the introduction of parallel beam, fan beam and cone beam reconstruction. The principle of cone beam CT reconstruction and the FDK algorithm are emphatically analyzed. Then three back projection methods of FDK algorithm are discussed which are Voxel driven method, Pixel driven method and Distance driven method. The Distance driven method with no open source code is implemented using C++ based on which improvement is made in the selection of voxels' boundary points using CPU. Then, an optimized version is implemented using texture memory, hardware interpolation and integral images on the CUDA parallel computing platform. The result shows that the reconstruction speed of optimized Distance driven algorithm is improved while the image quality remains unchanged.

## Key words
Three-dimensional reconstruction, CBCT, back-projection reconstruction, FDK, Distance-driven
