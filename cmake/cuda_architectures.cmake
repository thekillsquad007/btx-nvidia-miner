# Fat-binary SM targets for prebuilt releases (CUDA 12.8+ / nvcc max sm_120).
# Pascal (6.x) through Blackwell; matches README "Pascal or newer".
# Maxwell (sm_50/52) is omitted — deprecated in CUDA 12 and not listed in README.
#
#  60 P100          75 RTX 20xx / GTX 16xx    89 RTX 40xx
#  61 GTX 10xx      80 A100 / CMP 170HX       90 H100 / H200
#  70 V100          86 RTX 30xx / CMP 30-90   100 Blackwell datacenter
#  72 Jetson Xavier 87 Jetson Orin            120 RTX 50xx
set(BTX_RELEASE_CUDA_ARCHITECTURES
    "60;61;70;72;75;80;86;87;89;90;100;120"
    CACHE STRING "CUDA SM architectures for release fat binaries")