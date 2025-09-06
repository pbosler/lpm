#
#   set up virtual machine / base container
#
FROM ubuntu:22.04 AS base_build 
ENV DEBIAN_FRONTEND=noninteractive

# Set build arguments for proxy
ARG http_proxy

# Set proxy environment variables
ENV http_proxy=${http_proxy}
ENV https_proxy=${http_proxy}
ENV no_proxy=localhost,127.0.0.1

# Update the package list and install necessary tools
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt-get update && apt-get install -y --no-install-recommends --fix-missing \
  software-properties-common \
  openssl curl ca-certificates \
  autoconf \
  build-essential \
  clang-format \
  gcc-12 g++-12 gfortran-12 \
  libopenmpi-dev openmpi-bin \
  git cmake ninja-build \
  make pkg-config \
  netcdf-bin libnetcdf-mpi-dev \
  mesa-common-dev libglu1-mesa-dev freeglut3-dev \
  libblas-dev liblapack-dev libvtk9-dev && \
  apt-get clean && rm -rf /var/lib/apt/lists/* 

# Set the default GCC, G++, and GFortran to version 12
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 60 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-12 \
    --slave /usr/bin/gfortran gfortran /usr/bin/gfortran-12

#   copy all TPL tarballs to container
WORKDIR /tpl-src
COPY spdlog-1.15.3.tar.gz \
     Catch2-3.10.0.tar.gz \ #     VTK-9.5.1.tar.gz \
     kokkos-4.7.00.tar.gz \
     kokkos-kernels-4.7.00.tar.gz \
     compadre-1.6.2.tar.gz \
     COMPOSE-pb-lpm-kokkos-4.7.tar.gz \
     fftw-3.3.10.tar.gz \
     finufft-2.4.1.tar.gz \
     /tpl-src/
## VTK
#RUN cd /tpl-src && \
#    tar -xzf VTK-9.5.1.tar.gz && \
#    cd VTK-9.5.1 && \
#    cmake -B ./build \
#    -DCMAKE_C_COMPILER=gcc \
#    -DCMAKE_CXX_COMPILER=mpicxx \
#    -DMPI_C_COMPILER=mpicc \
#    -DMPI_CXX_COMPILER=mpicxx \
#    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
#    -DBUILD_TESTING=OFF \
#    -DVTK_BUILD_EXAMPLES=OFF \
#    -DVTK_BUILD_ALL_MODULES=OFF \
#    -DVTK_USE_CUDA=OFF \
#    -DVTK_USE_MPI=ON \
#    -DVTK_GROUP_ENABLE_MPI=YES \
#    -DVTK_USE_LARGE_DATA=OFF \
#    -DVTK_WRAP_PYTHON=OFF \
#    -DVTK_USE_EXTERNAL=OFF \
#    -DOpenGL_GL_PREFERENCE=GLVND \ 
#    -GNinja . && \
#    cmake --build ./build -j 8 && \
#    cmake --install ./build --prefix /tpl && rm -rf ./build
# spdlog, Catch2, Kokkos, KokkosKernels, Compadre
RUN cd /tpl-src && \
    tar -xzf spdlog-1.15.3.tar.gz && \
    cd spdlog-1.15.3 && \
    cmake -B ./build \
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUILD_SHARED_LIBS=OFF \
    -DSPDLOG_BUILD_EXAMPLE=OFF \
    -DSPDLOG_BUILD_TESTS=OFF \
    -DSPDLOG_ENABLE_PCH=ON \
    -DSPDLOG_INSTALL=ON . && \
    cmake --build ./build -j 8 && \
    cmake --install ./build --prefix /tpl && rm -rf ./build && \
    cd /tpl-src && \
    tar -xzf Catch2-3.10.0.tar.gz && \
    cd Catch2-3.10.0 && \
    cmake -B ./build \
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUILD_SHARED_LIBS=OFF \
    -DCATCH_INSTALL_DOCS=OFF \
    -DCATCH_INSTALL_EXTRAS=OFF \
    -DCATCH_BUILD_TESTING=OFF . && \
    cmake --build ./build -j 8 && \
    cmake --install ./build --prefix /tpl && rm -rf ./build && \
    cd /tpl-src && \
    tar -xzf kokkos-4.7.00.tar.gz && \
    cd kokkos-4.7.00 && \
    cmake -B ./build \
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CXX_FLAGS=-fPIC \
    -DKokkos_ENABLE_SERIAL=ON \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ENABLE_CUDA=OFF \
    -DKokkos_ARCH_NATIVE=ON \
    -DKokkos_ENABLE_DEPRECATED_CODE_4=ON \
    -DKokkos_ENABLE_DEPRECATION_WARNINGS=ON \
    -DKokkos_ENABLE_TESTS=OFF . && \
    cmake --build ./build -j 8 && \
    cmake --install ./build --prefix /tpl && rm -rf ./build &&  \
    cd /tpl-src && \
    tar -xzf kokkos-kernels-4.7.00.tar.gz && \
    cd kokkos-kernels-4.7.00 && \
    cmake -B ./build \ 
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DKokkos_ROOT=/tpl . && \
    cmake --build ./build -j 8 && \
    cmake --install ./build --prefix /tpl && rm -rf ./build && \
    cd /tpl-src && \
    tar -xzf compadre-1.6.2.tar.gz && \
    cd compadre-1.6.2 && \
    cmake -B ./build \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DKokkosCore_PREFIX=/tpl \
    -DKokkosKernels_PREFIX=/tpl \
    -DCompadre_USE_CUDA=OFF \
    -DCompadre_USE_PYTHON=OFF . && \
    cmake --build ./build -j 8 && \
    cmake --install ./build --prefix /tpl && rm -rf ./build
# COMPOSE    
#RUN cd /tpl-src && \
#    tar -xzf COMPOSE-pb-lpm-kokkos-4.7.tar.gz && \
#    cd COMPOSE-pb-lpm-kokkos-4.7 && \
#    cmake -B ./build \
#    -DCMAKE_CXX_COMPILER=mpicxx \
#    -DCMAKE_CXX_STANDARD=17 \
#    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
#    -DCMAKE_CXX_FLAGS="-std=c++17 -fPIC" \
#    -DKokkos_DIR=/tpl . && \
#    cmake --build ./build -j 8 && \
#    cmake --install ./build --prefix /tpl && rm -rf ./build
## FFTW3 and finufft    
#RUN cd /tpl-src && \
#    tar -xzf fftw-3.3.10.tar.gz && \
#    cd fftw-3.3.10 && \
#    ./configure --prefix=/tpl --enable-openmp --enable-threads --enable-mpi \
#    --enable-float --enable-type-prefix --disable-dependency-tracking && \
#    make -j 8 && make install && make clean && \
#    ./configure --prefix=/tpl --enable-openmp --enable-threads --enable-mpi \
#    --enable-type-prefix --disable-dependency-tracking && \
#    make -j 8 && make install && make clean && \
#    cd /tpl-src && \
#    tar -xzf finufft-2.4.1.tar.gz && \
#    cd finufft-2.4.1 && \
#    cmake -B ./build \
#    -DCMAKE_CXX_STANDARD=17 \
#    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
#    -DCMAKE_CXX_COMPILER=mpicxx \
#    -DCMAKE_INSTALL_PREFIX=/tpl \
#    -DCMAKE_C_FLAGS="-I/tpl/include" \
#    -DCMAKE_CXX_FLAGS="-I/tpl/include" \
#    -DFINUFFT_USE_OPENMP=ON \
#    -DFINUFFT_ENABLE_SANITIZERS=OFF \
#    -DFINUFFT_BUILD_TESTS=OFF \
#    -DFINUFFT_USE_CPU=ON \
#    -DFINUFFT_USE_CUDA=OFF \
#    -DFINUFFT_FFTW_LIBRARIES=/tpl/lib64/libfftw3.a . && \
#    cmake --build ./build -j 8 && \
#    cmake --install ./build --prefix /tpl && rm -rf ./build && \
#    rm -rf /tpl-src
VOLUME /tpl  
  