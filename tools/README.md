# LPM Tools

This folder contains tools used by LPM for constructing metadata about the status of the repository and building third-party libraries (TPL).

## Docker images

To facilitate auto-testing via github actions, we use a docker image to contain pre-built TPLs.
To use the docker file, you must first download the tarballs for the TPLs you want to build.
Currently, all of them.

Current (04SEP2025) TPLs in the container:
- spdlog 1.15.3
- Catch2 3.10.0
- Kokkos 4.7.00
- KokkosKernels 4.7.00
- Compadre 1.6.2
- COMPOSE branch pb-lpm-kokkos-4.7
- fftw 3.3.10
- finufft 2.4.1

To build the docker image, use a command similar to the following, run inside the tools folder:
`docker build --platform=linux/amd64 --network=host -t lpm-tpl:0 .`
In the above, `lpm-tpl` is the name of the image, and `0` is its tag or version number. The platform type must match one of the available github action runners.

To troubleshoot a docker build, intermediate images can be listed with `docker images`.  
Then, one of those images can be run interactively to debug problems with `docker run --rm -it <image-name> bash`.  

Once the whole build completes successfully, tag the image to upload it to DockerHub:
- `docker login`
- `docker tag lpm-tpl:0 pbosler/lpm-tpl:0`
- `docker tag lpm-tpl:0 pbosler/lpm-tpl:latest`
- `docker push pbosler/lpm-tpl:0`
- `docker push pbosler/lpm-tpl:latest`

