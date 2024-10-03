FROM ubuntu:22.04
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt-get update && apt-get install -y --no-install-recommends --fix-missing \
  ssh \
  openssl curl ca-certificates \
  autoconf \
  clang-format \
  gcc g++ gfortran \
  libopenmpi-dev \
  git cmake \
  libnetcdf-mpi-dev \
  make pkg-config \
  mesa-common-dev libglu1-mesa-dev freeglut3-dev \
  libblas-dev liblapack-dev  && mkdir -p /lpm-tpl && mkdir /tpl-src \
  && rm -rf /var/lib/apt/lists/*
ENV VTK_ROOT=/lpm-tpl
ENV KOKKOS_ROOT=/lpm-tpl
ENV FFTW_ROOT=/lpm-tpl
ENV GIT_SSL_NO_VERIFY=1

WORKDIR /tpl-src
ADD https://github.com/Kitware/VTK.git#v9.3.1 /tpl-src/vtk-9.3.1
ADD https://github.com/kokkos/kokkos.git#4.4.00 /tpl-src/kokkos
ADD https://github.com/kokkos/kokkos-kernels.git#4.4.00 /tpl-src/kokkos-kernels
ADD https://github.com/sandialabs/compadre.git /tpl-src/compadre
ADD https://github.com/pbosler/COMPOSE.git#lpm-tpl /tpl-src/compose

WORKDIR /tpl-src/vtk-9.3.1
RUN mkdir build && cd build && \
cmake \
-DCMAKE_C_COMPILER=gcc \
-DCMAKE_CXX_COMPILER=g++ \
-DMPI_C_COMPILER=mpicc \
-DMPI_CXX_COMPILER=mpicxx \
-DCMAKE_INSTALL_PREFIX=${VTK_ROOT} \
-DBUILD_SHARED_LIBS=OFF \
-DCMAKE_BUILD_TYPE=RelWithDebInfo \
-DBUILD_TESTING=OFF \
-DVTK_BUILD_EXAMPLES=OFF \
-DVTK_BUILD_ALL_MODULES=OFF \
-DVTK_USE_MPI=ON \
-DVTK_GROUP_ENABLE_MPI=YES \
-DVTK_USE_LARGE_DATA=OFF \
-DVTK_WRAP_JAVA=OFF \
-DVTK_WRAP_PYTHON=OFF \
-DOpenGL_GL_PREFERENCE=GLVND \
/tpl-src/vtk-9.3.1 && \
make -j 8 && make install && cd .. && rm -rf build && ls /lpm-tpl

WORKDIR /tpl-src/kokkos
RUN mkdir build && cd build && \
cmake \
-DCMAKE_BUILD_TYPE=RelWithDebInfo \
-DCMAKE_CXX_STANDARD=17 \
-DCMAKE_C_COMPILER=mpicc \
-DCMAKE_CXX_COMPILER=mpicxx \
-DCMAKE_INSTALL_PREFIX=${KOKKOS_ROOT} \
-DKokkos_ENABLE_SERIAL=ON \
-DKokkos_ENABLE_OPENMP=ON \
-DKokkos_ENABLE_LIBDL=ON \
-DKokkos_ARCH_NATIVE=ON \
-DBUILD_SHARED_LIBS=OFF \
/tpl-src/kokkos && \
make -j 8 && make install && cd .. && rm -rf build


WORKDIR /tpl-src/kokkos-kernels
RUN mkdir build && cd build && \
cmake \
-DCMAKE_INSTALL_PREFIX=${KOKKOS_ROOT} \
-DCMAKE_CXX_STANDARD=17 \
-DCMAKE_BUILD_TYPE=RelWithDebInfo \
-DKokkos_ROOT=${KOKKOS_ROOT} \
-DKokkos_DIR=${KOKKOS_ROOT} \
/tpl-src/kokkos-kernels && \
make -j 8 && make install && rm -rf build


WORKDIR /tpl-src/compadre
RUN mkdir build && cd build && \
cmake \
-DCMAKE_BUILD_TYPE=RelWithDebInfo \
-DBUILD_SHARED_LIBS=OFF \
-DCMAKE_INSTALL_PREFIX=${KOKKOS_ROOT} \
-DCMAKE_C_COMPILER=mpicc \
-DCMAKE_CXX_COMPILER=mpicxx \
-DKokkosCore_PREFIX=${KOKKOS_ROOT} \
-DCompadre_EXAMPLES=OFF \
-DCompadre_TESTS=OFF \
-DCompadre_USE_PYTHON=OFF \
-DKokkosKernels_PREFIX=${KOKKOS_ROOT} \
/tpl-src/compadre && \
make -j 8 && make install && cd .. && rm -rf build


WORKDIR /tpl-src/compose
RUN mkdir build && cd build && \
cmake \
-DCMAKE_C_COMPILER=mpicc \
-DCMAKE_CXX_COMPILER=mpicxx \
-DCMAKE_CXX_STANDARD=17 \
-DBUILD_SHARED_LIBS=OFF \
-DCMAKE_BUILD_TYPE=RelWithDebInfo \
-DCMAKE_INSTALL_PREFIX=${KOKKOS_ROOT} \
-DKokkos_DIR=${KOKKOS_ROOT} \
/tpl-src/compose && \
make -j 8 && make install && cd .. && rm -rf build && ls /lpm-tpl/lib

WORKDIR /tpl-src
COPY fftw-3.3.10.tar.gz /tpl-src/fftw-3.3.10.tar.gz
RUN tar -xzf fftw-3.3.10.tar.gz && \
cd fftw-3.3.10 && \
./configure --prefix=${FFTW_ROOT} --enable-openmp --enable-threads --enable-mpi \
--enable-float --enable-type-prefix --disable-dependency-tracking && \
make -j 8 && make install && make clean && \
./configure --prefix=${FFTW_ROOT} --enable-openmp --enable-threads --enable-mpi \
--enable-type-prefix --disable-dependency-tracking && \
make -j 8 && make install && cd .. && rm -rf build && \
cd /tpl-src && rm -rf fftw-3.3.10/ && ls /lpm-tpl/lib

