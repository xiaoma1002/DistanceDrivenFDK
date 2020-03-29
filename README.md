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

Important References are:(dd, cuda feature)
<br>[Distance-driven projection and backprojection in three dimensions](https://iopscience.iop.org/article/10.1088/0031-9155/49/11/024/pdf)
<br>[GPU-based Branchless Distance-Driven Projection and Backprojection](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5761753/)

<br>Part of the codes for cpu and gpu versions of distance-driven fdk algorithm are shown in this repo.

<br>You are welcome to contact [me](https://www.linkedin.com/in/yuhua-angela-ma-676a73184/) and ask any questions about this project.
