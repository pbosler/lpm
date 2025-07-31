#ifndef RBF_NEIGHBOR_SEARCH_HPP
#define RBF_NEIGHBOR_SEARCH_HPP

#include "Kokkos_Core.hpp"
#include <utility>
#include <fstream>
#include <typeinfo>
#include "Compadre_NeighborLists.hpp"
#include "Compadre_PointCloudSearch.hpp"


//Rbf InterpOps and DiffOps
template<class Input>
struct rbf_neighbor_search{

  //member variables and data structures
  Int N,N_eval,l,stencil_size,dimensions;
  Real tau;
  ViewVectorType source_data_sites,target_data_sites;
  ko::View<Int**> nbr_lists_N; // neighbor list
  
  
  //Differential Operators constructor 
  rbf_neighbor_search(Input input) : nbr_lists_N("nbr_lists_N",input.N,input.stencil_size), source_data_sites("Xs",input.N,input.dimensions), target_data_sites("Xt",input.N_eval,input.dimensions)
  {
    N = input.N;
    N_eval = input.N_eval;
    l = input.l;
    tau = input.tau;
    dimensions = input.dimensions;
  }

  void knn_search(){
    double epsilon_multiplier = tau;
    int min_neighbors = stencil_size/tau;
    // Point cloud construction for neighbor search
    // CreatePointCloudSearch constructs an object of type PointCloudSearch, but deduces the templates for you
//    Compadre::PointCloudSearch<ViewVectorType>(source_data_sites, dimensions,-1);
    auto point_cloud_search(    Compadre::PointCloudSearch<ViewVectorType>(source_data_sites, dimensions,-1));


    Kokkos::View<int*> neighbor_lists_device("neighbor lists", 
            0); // first column is # of neighbors
    Kokkos::View<int*>::HostMirror neighbor_lists = Kokkos::create_mirror_view(neighbor_lists_device);
    // number_of_neighbors_list must be the same size as the number of target sites so that it can be populated
    // with the number of neighbors for each target site.
    Kokkos::View<int*> number_of_neighbors_list_device("number of neighbor lists", 
            stencil_size); // first column is # of neighbors
    Kokkos::View<int*>::HostMirror number_of_neighbors_list = Kokkos::create_mirror_view(number_of_neighbors_list_device);

    // each target site has a window size
    Kokkos::View<double*, Kokkos::DefaultExecutionSpace> epsilon_device("h supports", stencil_size);
    Kokkos::View<double*>::HostMirror epsilon = Kokkos::create_mirror_view(epsilon_device);

    size_t storage_size = point_cloud_search.generateCRNeighborListsFromKNNSearch(true /*dry run*/, target_data_sites, neighbor_lists, 
            number_of_neighbors_list, epsilon, min_neighbors, epsilon_multiplier);
    
    // resize neighbor_lists_device so as to be large enough to contain all neighborhoods
    Kokkos::resize(neighbor_lists_device, storage_size);
    neighbor_lists = Kokkos::create_mirror_view(neighbor_lists_device);

    // query the point cloud a second time, but this time storing results into neighbor_lists
    point_cloud_search.generateCRNeighborListsFromKNNSearch(false /*not dry run*/, target_data_sites, neighbor_lists, 
            number_of_neighbors_list, epsilon, min_neighbors, epsilon_multiplier);
    nbr_lists_N = Kokkos::create_mirror_view(neighbor_lists);
  }
};

#endif RBF_NEIGHBOR_SEARCH_HPP

