#ifndef RBF_TEAM_WGTS_HPP
#define RBF_TEAM_WGTS_HPP

#include "Rbf_BaseDefs.hpp"
#include "Rbf_PointSet.hpp"
#include "Rbf_PointSet_Impl.hpp"
#include "Kokkos_Core.hpp"
#include <utility>
#include <fstream>
#include <typeinfo>
#include "KokkosBlas2_gemv.hpp"
#include "KokkosBlas3_gemm.hpp"
#include "KokkosBlas1_scal.hpp"
#include "KokkosBlas1_abs.hpp"
#include "KokkosBlas1_dot.hpp"
#include "KokkosBlas1_nrm2.hpp"
#include "KokkosBlas1_update.hpp"
#include "Compadre_NeighborLists.hpp"
#include "Compadre_PointCloudSearch.hpp"
#include "lpm_rbf_nl.hpp"

using namespace Rbf;
using AlgoTagType = KokkosBatched::Algo::QR::Unblocked;

//Rbf InterpOps and DiffOps
template<class Input>
struct rbf_team_matrices{

  //member variables and data structures
  Int N,N_eval,l,maxiter;
  Real tol;

  ViewVectorType dpts,nrls,Lk_wgts,Ik_wgts; 
  ko::View<Real*> u,v,Lu;
  ko::View<Int*> number_neighbors_list;
  ko::View<Int*> cr_neighbor_lists;
  
  
  //Differential Operators constructor 
  rbf_team_matrices(Input input) : dpts("dpts",input.N,3),nrls("nrls",input.N,3),\
                                   u("u",input.N),v("v",input.N),Lu("Lu",input.N)
  {
    //set input variables for rbf class
    N = input.N;
    l = input.l;
    N_eval = input.N_eval;

  }

  //member functions
  Kokkos::View<Real**> project_stencils(Real nrl_1, Real nrl_2, Real nrl_3, Int stencil_size, ko::View<Int*> neighbors);
  void tpm_lap_wgts();
  Real max_element();

};

/**
Tangent Plane Projection
**/
template <class Input>
Kokkos::View<Real**> rbf_team_matrices<Input>::project_stencils(Real nrl_1, Real nrl_2, Real nrl_3, Int stencil_size, ko::View<Int*> neighbors)
{
  ko::View<Real**> proj_xnbrs("proj_xnbrs",2,stencil_size);
  //project stencils in tangent space
  ViewScalarType nrl("nrl",3);
  ViewScalarType nrla("nrla",3);
  nrl(0) = nrl_1;
  nrl(1) = nrl_2;
  nrl(2) = nrl_3;
  Int imax = 0;
  for(Int i = 0; i < 3; ++i){
    nrla(i) = std::abs(nrl(i));
  }
  for(Int i = 0; i < 3; ++i){
    if (nrla(0) < nrla(i)){
      imax = i;
    }
  }
  ViewScalarType e1("e1",3);
  ViewScalarType e2("e2",3);
  ViewScalarType t1("t1",3);
  ViewScalarType t2("t2",3);
  ViewVectorType R("R",2,3);
  ViewVectorType xnbrs("xnbrs",stencil_size,3);

  if (imax == 0) {
    e1(1) = 1;
    e2(2) = 1;
  }

  else if (imax == 1){
    e1(0) = 1;
    e2(2) = 1;
  }

  else{
    e1(0) = 1;
    e2(1) = 1;
  }

  //Compute two orthogonal vectors to the normal using Gram Schmidt
  Real a = nrl_1*e1(0) + nrl_2*e1(1) + nrl_3*e1(2);
  Real b = nrl_1*e2(0) + nrl_2*e2(1) + nrl_3*e2(2);

  t1(0) = e1(0) - a * nrl_1;
  t1(1) = e1(1) - a * nrl_2;
  t1(2) = e1(2) - a * nrl_3;

  Real nrmt1 = normh(t1);
  t1(0) = t1(0)/nrmt1;
  t1(1) = t1(1)/nrmt1;
  t1(2) = t1(2)/nrmt1;

  Real c = t1(0)*e2(0) + t1(1)*e2(1) + t1(2)*e2(2);
  t2(0) = e2(0) - b * nrl_1 - c * t1(0);
  t2(1) = e2(1) - b * nrl_2 - c * t1(1);
  t2(2) = e2(2) - b * nrl_3 - c * t1(2);

  Real nrmt2 = normh(t2);
  t2(0) = t2(0)/nrmt2;
  t2(1) = t2(1)/nrmt2;
  t2(2) = t2(2)/nrmt2;

  // Form rotation matrix
  R(0,0) = t1(0);
  R(0,1) = t1(1);
  R(0,2) = t1(2);

  R(1,0) = t2(0);
  R(1,1) = t2(1);
  R(1,2) = t2(2);
  ko::parallel_for( stencil_size, [=] (const Int i) {
    xnbrs( i , 0 ) = ( dpts( neighbors(i), 0 ) - nrl_1 );
    xnbrs( i , 1 ) = ( dpts( neighbors(i), 1 ) - nrl_2 );
    xnbrs( i , 2 ) = ( dpts( neighbors(i), 2 ) - nrl_3 );
  });

  //projection xp -> R*x
  Real alpha = 1 , beta = 0;
  KokkosBlas::gemm("N","T",alpha,R,xnbrs,beta,proj_xnbrs);

  return proj_xnbrs;
}




/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Find max
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
/*template<class Input>
Real rbf_team_matrices<Input>::max_element(ViewScalarType& v)
{
  Real v_max = 0;
  ko::parallel_for( v.extent(0), [=] (Int i) {
    if (v(i) > v_max){
      v_max = v(i);
    }
  });
  return v_max;
}
*/


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
RBF FINITE DIFFERENCE (FD) WEIGHTS USING TPM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
template<class Input>
void rbf_team_matrices<Input>::tpm_lap_wgts()
{


  /**
  data structures for storing stencil neighbors and aux. matrices and vectors
  */
  Int L = (l+1)*(l+2)/2;
  auto nla(Compadre::CreateNeighborLists(cr_neighbor_lists,number_neighbors_list));

  // Compute RBF-FD weights by assembly of \[ A & P \\ P^T & O\]  and  \[ D\phi \\ Dp \]
  std::cout << "Begin Weight generation" << std::endl;
  for(Int k = 0; k < N; ++k){

    //batch system matrices and vectors
    Int stencil_size = nla.getNumberOfNeighborsHost(k);
    Int n = stencil_size;
    Int nl = n+L;
    ViewMatrixType a("a", 1,nl,nl);
    ViewVectorType x("x", 1,nl);
    ViewVectorType b("b", 1,nl);
    ViewVectorType t("t", 1, nl);
    PivotViewType  p("p", 1, nl);
    WorkViewType   w("w", 1, 2*nl);
    
  
    // STENCILS and PROJECTION
    ko::View<Int*> neighbors("neighbors",stencil_size);
    ko::View<Real*> u_nbrs("u_nbrs",stencil_size);

    for(Int j=0; j < stencil_size; ++j){
      neighbors(j) =  nla.getNeighborHost(k,j); // nearest neighbors in kth stencil 
      u_nbrs(j) = u(neighbors(j)); //target values
    }
    ko::View<Real**> proj_xnbrs = project_stencils(dpts(k,0),dpts(k,1),dpts(k,2),stencil_size,neighbors);
  
    // RBF DM MATRIX
    ViewVectorType A("A",stencil_size,stencil_size);
    ViewVectorType P("P",stencil_size,L);
    ViewScalarType r2("r2", stencil_size*stencil_size);
    ko::parallel_for( stencil_size, [=] (const Int i) {
      for (Int j = 0; j < stencil_size; ++j){
        Real rij = std::sqrt(std::pow(proj_xnbrs(0,i) - proj_xnbrs(0,j),2) + \
                    std::pow(proj_xnbrs(1,i) - proj_xnbrs(1,j),2));
        r2(i*stencil_size+j) = rij;
        A(i,j) = std::pow(rij,2.0*l+1);
      }
    });

    //    Real stencil_support = max_element(r2);
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    //BIVARIATE POLYNOMIALS
    Int idx = 0;
    for(Int ii = 0; ii < l+1; ii++)
    {
      for(Int jj = 0; jj < ii+1; jj++)
      {
        Int power1 = (jj);
        Int power2 = (ii - jj);
        for(Int kk = 0; kk < stencil_size; kk++)
        {
          Real dx = proj_xnbrs(0,kk) - proj_xnbrs(0,0);
          Real dy = proj_xnbrs(1,kk) - proj_xnbrs(1,0);
          P(kk,idx) = std::pow(dx,power1)*std::pow(dy,power2);
        }
        idx++;
      }
    }

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    //SADDLE POINT MATRIX FOR QUADRATIC MINIMIZATION
    ko::parallel_for( n , [=] (const Int i) {
      for(Int j = 0; j < n; ++j)
      {
        a(0, i , j ) = A( i , j );
      }
      for(Int j = 0; j < L; ++j)
      {
        a(0, i , n + j ) = P( i , j );
      }
    });
    ko::parallel_for( L , [=] (const Int i) {
      for(Int j = 0; j < L; ++j)
      {
        a(0, n + i , n + j ) = 0;
      }
      for(Int j = 0; j < n; ++j)
      {
        a(0, n + i , j ) = P( j , i );
      }
    });

    for(Int i = 0; i < stencil_size; ++i)
    {
      Real rc = std::sqrt(std::pow(proj_xnbrs(0,i) - proj_xnbrs(0,0),2) + std::pow(proj_xnbrs(1,i) - proj_xnbrs(0,1),2));
      Real lap_phi = std::pow(2.0*l+1.0,2)*std::pow(rc,(2.0*l-1.0));
      b(0,i) = lap_phi;
    }
    b(0,stencil_size + 3) = 2;
    b(0,stencil_size + 5) = 2;

    // Solve C U = B for the weights  
    Functor_SolverBatchedTeamVectorQR_WithColumnPivoting
    <DeviceType,ViewMatrixType,ViewVectorType,
    PivotViewType,WorkViewType,AlgoTagType>(a,x,b,t,p,w).run();

    //compute lap_u_k \approx weights_k*u_k
    auto wgts_k = ko::subview(x,0,ko::ALL);
    ko::parallel_for( stencil_size , [=] (const Int i) {
        Lu(k) += wgts_k(i)*u_nbrs(i);
    });
    // lap_u_k computed

  }


}

#endif
