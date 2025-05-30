#ifndef RBF_TEAM_WGTS_HPP
#define RBF_TEAM_WGTS_HPP

//#include "Rbf_BaseDefs.hpp"
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
#include "lpm_rbf_nl.hpp"
#include "lpm_compadre.hpp"

using namespace Rbf;
using AlgoTagType = KokkosBatched::Algo::QR::Unblocked;
//typedef ParamTag<Trans::NoTranspose,Trans::NoTranspose> param_tag_type;

//Rbf InterpOps and DiffOps
template<class Input>
struct rbf_team_matrices{
  //member variables and data structures
	Int N,N_eval,l,stencil_size,maxiter;
	Real tol;

	ViewMatrixType rot_nbrs_N,x_nbrs_N;
	ViewVectorType rot_nbrs_k,rot_xc_N,x_nbrs_k,xc_N,dpts,epts,dpts_reordered,nrls,Lk_wgts,Ik_wgts; 
	ViewScalarType rot_xc_k,xc_k,u,v,Lu;  
  PivotViewType nbr_list_N; // neighbor list
  
  
  //Operators constructor 
  rbf_team_matrices(Input input) : rot_nbrs_N("rot_nbrs_N",input.N,input.stencil_size,3),x_nbrs_N("x_nbrs_N",input.N,input.stencil_size,3),\
                                   rot_nbrs_k("rot_nbrs_k",2,input.stencil_size),x_nbrs_k("x_nbrs_k",3,input.stencil_size),\
                                   rot_xc_N("rot_xc_N",input.N,3),rot_xc_k("rot_xc_k",3),xc_N("xc_N",input.N,3),xc_k("xc_k",3),\
                                    nbr_list_N("nbr_list_N",input.N,input.stencil_size),\
                                    dpts("dpts",input.N,3),dpts_reordered("Xbi",input.N,3),nrls("nrls",input.N,3),\
                                   Lk_wgts("Lk_wgts",input.N,input.stencil_size),u("u",input.N),v("v",input.N),Lu("Lu",input.N)
  {
    //set input variables for rbf class
    N = input.N;
    stencil_size = input.stencil_size; 
    l = input.l;
    N_eval = 2*N;


		/**		Build the tree		*/
		std::vector<MyPoint> ptlist(N);
		std::vector<Int> nbr_list;
		fill_ptlist(ptlist,dpts,N);
		kdt::KDTree<MyPoint> kdtree(ptlist);

		/**
	 	Find stencil neighbors
	  */
    for(Int k = 0; k < N; ++k){   
      /**
      Find k-nearest neighbors of xc
      */
      knn_search_ballpoint(kdtree,ptlist,nbr_list,k,stencil_size);

      /**
      Construct neighbor pointlist X_n of xc
      */
      auto nbr_list_k = ko::subview(nbr_list_N,k,ko::ALL);

      //for (Int i = 0; i < stencil_size; ++i){
      ko::parallel_for(stencil_size, [=] (const Int i) 
      {
        nbr_list_k(i) = nbr_list[i];
      });
      if (input.pde_type == "surface")
      {
        if (input.domain_name == "sphere"){
          project_stencils(dpts(k,0),dpts(k,1),dpts(k,2),k);
        }
        else{
  //        std::cout << "Computing Normals" << std::endl;
          compute_normals(input.domain_name);
    //      std::cout << "Completed Normals Computation" << std::endl;

      //    std::cout << "Projecting stencils" << std::endl;
          project_stencils(nrls(k,0),nrls(k,1),nrls(k,2),k);
        //  std::cout << "Stencil Projection complete" << std::endl;

        }
      }
      else{
        //setup node reordering for boundary nodes
        //extract boundary indices
      }
		}
    rbf_team_matrices(Input input) : rot_nbrs_N("rot_nbrs_N",input.N,input.stencil_size,2),x_nbrs_N("x_nbrs_N",input.N,input.stencil_size,3),\
                                     rot_nbrs_k("rot_nbrs_k",2,input.stencil_size),x_nbrs_k("x_nbrs_k",3,input.stencil_size),\
                                     rot_xc_N("rot_xc_N",input.N,2),rot_xc_k("rot_xc_k",2),xc_N("xc_N",input.N,3),xc_k("xc_k",3),\
                                     nbr_list_N("nbr_list_N",input.N,input.stencil_size),\
                                     dpts("dpts",input.N,3),epts("epts",input.N_eval,3),nrls("nrls",input.N,3),\
                                     Lk_wgts("Lk_wgts",input.N,input.stencil_size),
                                     Ik_wgts("Lk_wgts",input.N,input.stencil_size),
                                     ,u("u",input.N),v("v",input.N),Lu("Lu",input.N){}



  {}

  };

  //member functions
	void fill_N_nbrs(Int k);
	void nbr_search_sph();
	void nbr_search_eval();
//  template<typename VVT>
  //void tri_poly(VVT P);
	void tpm_fd_wgts();
	void tpm_interp_wgts();
	void compute_spmv();
	void compute_normals(std::string domain_name);
	void project_stencils(Real nrl_1, Real nrl_2, Real nrl_3, Int k);
	Real max_element();
  template<typename VST>
	void bicgstab(VST rhs);
	template<typename crsMat_t>
	void makeSparseMatrix (
      typename crsMat_t::StaticCrsGraphType::row_map_type::non_const_type & ptr,
      typename crsMat_t::StaticCrsGraphType::entries_type::non_const_type & ind,
      typename crsMat_t::values_type::non_const_type & val,
      typename crsMat_t::ordinal_type &numRows,
      typename crsMat_t::ordinal_type &numCols,
      typename crsMat_t::size_type &nnz,
      const int whichMatrix);
	template<typename crsMat_t>
	crsMat_t  makeCrsMatrix (int numRows);

  
};

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
	Projection of stencils to tangent plane
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
template<class Input>
void rbf_team_matrices<Input>::project_stencils( Real nrl_1, Real nrl_2, Real nrl_3, Int k )
{

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
	ViewVectorType xn1("xn1",stencil_size,3);

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
		xn1( i , 0 ) = ( dpts( nbr_list_N( k , i ) , 0 ) - nrl_1 );
		xn1( i , 1 ) = ( dpts( nbr_list_N( k , i ) , 1 ) - nrl_2 );
		xn1( i , 2 ) = ( dpts( nbr_list_N( k , i ) , 2 ) - nrl_3 );
	});

	Real alpha = 1 , beta = 0;
	KokkosBlas::gemm("N","T",alpha,R,xn1,beta,rot_nbrs_k);

//	for(int i = 0; i < 3; ++i){	
//		std::cout << R( 0 , i ) << " " << R( 1 , i ) << std::endl; 
//	}	

	ko::parallel_for( stencil_size, [=] (const Int i) {
		rot_nbrs_N(k,i,0) = rot_nbrs_k(0,i);
		rot_nbrs_N(k,i,1) = rot_nbrs_k(1,i);
	});
//	std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" <<std::endl;
}


/**
Fill rotated neighbors
*/


template<class Input>
void rbf_team_matrices<Input>::fill_N_nbrs(Int k)
{
	for( Int i = 0; i < 3; i++){
		xc_N(k,i) = xc_k(i);
	}
	for( Int i = 0; i < 3; i++){
		for( Int j = 0; j < stencil_size; j++){
			x_nbrs_N(k,j,i) = x_nbrs_k(i,j);
		}
	}
}

//template<class Input>
//void rbf_team_matrices<Input>::fill_N_nbrs(Int k)
//{
//	for( Int i = 0; i < 3; i++){
//		rot_xc_N(k,i) = rot_xc_k(i);
//	}
//	for( Int i = 0; i < 3; i++){
//		for( Int j = 0; j < stencil_size; j++){
//			rot_nbrs_N(k,j,i) = rot_nbrs_k(i,j);
//		}
//	}
//}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Compute sparse matvec 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
template<class Input>
void rbf_team_matrices<Input>::compute_spmv()
{
	ko::parallel_for( N, [=] (const Int i) {
		Real sum_i = 0;
		for (Int j = 0; j < stencil_size; ++j){
			Int nbr_j = nbr_list_N(i,j);
			sum_i += Lk_wgts(i,j)*v(nbr_j);
		}
		Lu(i) = sum_i;
	});
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
void rbf_team_matrices<Input>::tpm_fd_wgts()
{


	/**
	data structures for storing stencil neighbors and aux. matrices and vectors
	*/
	Int L = (l+1)*(l+2)/2;
	Int nl = stencil_size + L;
	ViewVectorType A("A",stencil_size,stencil_size);
	ViewVectorType P("P",stencil_size,L);

  ViewMatrixType a("a", N, nl, nl);
  ViewVectorType x("x", N, nl);
  ViewVectorType b("b", N, nl);
  ViewScalarType r2("r2", stencil_size*stencil_size);

  Int n = stencil_size;
  // Compute RBF-FD weights by assembly of \[ A & P \\ P^T & O\]  and  \[ D\phi \\ Dp \]
  for(Int k = 0; k < N; ++k){
    // STENCILS
    auto rot_nbrs_j = ko::subview(rot_nbrs_N,k,ko::ALL,ko::ALL);
    auto rot_xc = ko::subview(rot_xc_N,k,ko::ALL);

    // RBF DM MATRIX

    ko::parallel_for( stencil_size, [=] (const Int i) {
      for (Int j = 0; j < stencil_size; ++j){
        Real rij = std::sqrt(std::pow(rot_nbrs_j(i,0) - rot_nbrs_j(j,0),2) + \
    			std::pow(rot_nbrs_j(i,1) - rot_nbrs_j(j,1),2));
        r2(i*stencil_size+j) = rij;
        A(i,j) = std::pow(rij,2.0*l+1);
      }
    });
    //    Real stencil_support = max_element(r2);
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    //Polynomials of degree l
    /*    Real dx_max = 0.00000000000001;
    Real dy_max = dx_max;
    for(Int kk = 0; kk < stencil_size; kk++)
    {
    Real dx = rot_nbrs_j(kk,0) - rot_nbrs_j(0,0);
    Real dy = rot_nbrs_j(kk,1) - rot_nbrs_j(0,1);
    if (dx_max < dx){
    dx_max = std::fabs(dx);
    }
    if (dy_max < dy){
    dy_max = std::fabs(dy);
    }
    }*/
    Int idx = 0;
    for(Int ii = 0; ii < l+1; ii++)
    {
      for(Int jj = 0; jj < ii+1; jj++)
      {
        Int power1 = (jj);
        Int power2 = (ii - jj);
        for(Int kk = 0; kk < stencil_size; kk++)
        {
          Real dx = rot_nbrs_j(kk,0) - rot_nbrs_j(0,0);
          Real dy = rot_nbrs_j(kk,1) - rot_nbrs_j(0,1);
          P(kk,idx) = std::pow(dx,power1)*std::pow(dy,power2);
        }
        idx++;
      }
    }

		//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		auto aa = ko::subview(a, k, ko::ALL(), ko::ALL());
    auto bb = ko::subview(b, k, ko::ALL());

  	auto aablk0 = ko::subview(aa,std::make_pair(0,n),\
  				std::make_pair(0,n));
  	auto aablk1 = ko::subview(aa,std::make_pair(0,n),\
  				std::make_pair( n , n + L));
  	auto aablk2 = ko::subview(aa,std::make_pair(n,n + L),\
  				std::make_pair(0,n));

  	ko::parallel_for( n , [=] (const Int i) {
  		for(Int j = 0; j < n; ++j)
      {
  			aablk0( i , j ) = A( i , j );
  		}
  	});
  	ko::parallel_for( n , [=] (const Int i) {
  		for(Int j = 0; j < L; ++j)
  		{
  			aablk1( i , j ) = P( i , j );
  		}
  	});
  	ko::parallel_for( L , [=] (const Int i) {
  		for(Int j = 0; j < n; ++j)
  		{
  			aablk2( i , j ) = P( j , i );
  		}
  	});

    
//		gen_blkmat(a,A,P,stencil_size,L); //Additional rewrite needed

		for(Int i = 0; i < stencil_size; ++i)
    {
			Real rc = std::sqrt(std::pow(rot_nbrs_j(i,0) - rot_nbrs_j(0,0),2) + \
				std::pow(rot_nbrs_j(i,1) - rot_nbrs_j(0,1),2));
			bb(i) = std::pow(2.0*l+1.0,2)*std::pow(rc,(2.0*l-1.0));
		}
		bb(stencil_size + 3) = 2;
		bb(stencil_size + 5) = 2;

  }
  std::cout << "Accumulating entries for batched solve" << std::endl; 
	//END LOOP FOR LHS AND RHS
	/* Solve Aw = L\phi for the weights  */
  ViewVectorType t("t", N, nl);
  PivotViewType  p("p", N, nl);
  WorkViewType   w("w", N, 2*nl);

  Functor_SolverBatchedTeamVectorQR_WithColumnPivoting
  <DeviceType,ViewMatrixType,ViewVectorType,
  PivotViewType,WorkViewType,AlgoTagType>(a,x,b,t,p,w).run();
  std::cout << "Batched solve complete. Weights generated" << std::endl; 

  ko::parallel_for( N, [=] (const Int i) {
    for (Int j = 0; j < stencil_size; ++j)
    {
      Lk_wgts(i , j) = x(i , j);
    }
  });

}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
RBF LOCAL INTERPOLATION WEIGHTS USING TPM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
template<class Input>
void rbf_team_matrices<Input>::tpm_interp_wgts()
{

	
	//data structures for storing stencil neighbors and aux. matrices and vectors
	
	Int L = (l+1)*(l+2)/2;
	Int nl = stencil_size + L;
	ViewVectorType A("A",stencil_size,stencil_size);
	ViewVectorType P("P",stencil_size,L);

  ViewMatrixType a("a", N_eval, nl, nl);
  ViewVectorType x("x", N_eval, nl);
  ViewVectorType b("b", N_eval, nl);
  Int n = stencil_size;

  // Compute RBF-FD weights
  for(Int k = 0; k < N_eval; ++k){

    // STENCILS
		auto rot_nbrs_j = ko::subview(rot_nbrs_N,k,ko::ALL,ko::ALL);
		auto rot_xc = ko::subview(rot_xc_N,k,ko::ALL);

    // RBF DM MATRIX
		ko::parallel_for( stencil_size, [=] (const Int i) {
			for (Int j = 0; j < stencil_size; ++j){
				Real rij = std::sqrt(std::pow(rot_nbrs_j(i,0) - rot_nbrs_j(j,0),2) + \
					std::pow(rot_nbrs_j(i,1) - rot_nbrs_j(j,1),2));
				A(i,j) = std::pow(rij,2*l+1);
			}
		});

		//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		Int idx = 0;
		for(Int ii = 0; ii < l+1; ii++)
		{
			for(Int jj = 0; jj < ii+1; jj++)
			{
				Int power1 = (jj);
				Int power2 = (ii - jj);
				for(Int kk = 0; kk < stencil_size; kk++)
				{
					Real dx = rot_nbrs_j(kk,0) - rot_xc(0);
					Real dy = rot_nbrs_j(kk,1) - rot_xc(1);
					P(kk,idx) = std::pow(dx,power1)*std::pow(dy,power2);
				}
				idx++;
			}
		}

		//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		auto aa = ko::subview(a, k, ko::ALL(), ko::ALL());
    auto bb = ko::subview(b, k, ko::ALL());

  	auto aablk0 = ko::subview(aa,std::make_pair(0,n),\
  				std::make_pair(0,n));
  	auto aablk1 = ko::subview(aa,std::make_pair(0,n),\
  				std::make_pair( n , n + L));
  	auto aablk2 = ko::subview(aa,std::make_pair(n,n + L),\
  				std::make_pair(0,n));

  	ko::parallel_for( n , [=] (const Int i) {
  		for(Int j = 0; j < n; ++j)
      {
  			aablk0( i , j ) = A( i , j );
  		}
  	});
  	ko::parallel_for( n , [=] (const Int i) {
  		for(Int j = 0; j < L; ++j)
  		{
  			aablk1( i , j ) = P( i , j );
  		}
  	});
  	ko::parallel_for( L , [=] (const Int i) {
  		for(Int j = 0; j < n; ++j)
  		{
  			aablk2( i , j ) = P( j , i );
  		}
  	});

		//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		for(Int i = 0; i < stencil_size; ++i)
    {
			Real rc = std::sqrt(std::pow(rot_nbrs_j(i,0) - rot_nbrs_j(0,0),2) + \
				std::pow(rot_nbrs_j(i,1) - rot_nbrs_j(0,1),2));
			bb(i) = std::pow(rc,(2.0*l+1.0));
		}
		bb(stencil_size) = 1.0;
  }

	// Solve Aw = I\phi for the weights  
  ViewVectorType t("t", N_eval, nl);
  PivotViewType  p("p", N_eval, nl);
  WorkViewType   w("w", N_eval, 2*nl);
  Functor_SolverBatchedTeamVectorQR_WithColumnPivoting
  <DeviceType,ViewMatrixType,ViewVectorType,
  PivotViewType,WorkViewType,AlgoTagType>(a,x,b,t,p,w).run();

	ko::parallel_for( N_eval, [=] (const Int i) {
		ko::parallel_for( stencil_size, [=] (const Int j) {
        Ik_wgts(i , j) = x(i , j);
      });
  });

}
/* Compute Ip*u via dot product 
//	ko::parallel_for( N_eval, [=] (const Int k) {
//		ko::parallel_for( stencil_size, [=] (const Int i) {
//			Lu(k) += wgts(k,i)*u(nbr_list_N(k,i));
//		});
//  });
*/

template<class Input>
void rbf_team_matrices<Input>::nbr_search_sph()
{

	/**
	Build the tree
	*/
	std::vector<MyPoint> ptlist(N);
	std::vector<Int> nbr_list;
	fill_ptlist(ptlist,dpts,N);
	kdt::KDTree<MyPoint> kdtree(ptlist);
	Int dim = 3;

	/*
	Build input data structures
	**/
	ViewScalarType xc("xc",dim);
	ViewScalarType rot_xc("rot_xc",dim);
	ViewVectorType xc_nbrs("xc_nbrs",dim,stencil_size);
	ViewVectorType rot_xc_nbrs("rot_xc_nbrs",dim,stencil_size);
	ViewVectorType qw("qw",dim,dim);
	ViewVectorType rw("rw",dim,dim);
	ViewVectorType W("W",dim,dim);

  Real alpha = 1;
  Real beta = 0;
	/**
  Begin weight computation and collection
  */
	for(Int k = 0; k < N; ++k){

		/**
    Select stencil center xc
    */
		auto tmp = ko::subview(dpts,k,ko::ALL);
		xc(0) = tmp(0);
		xc(1) = tmp(1);
		xc(2) = tmp(2);

		/**
    Find k-nearest neighbors of xc
    */
		knn_search_ballpoint(kdtree,ptlist,nbr_list,k,stencil_size);

		/**
     Construct neighbor pointlist X_n of xc
    */
		auto nbr_list_k = ko::subview(nbr_list_N,k,ko::ALL);
		for (Int i = 0; i < stencil_size; ++i){
			xc_nbrs(0,i) = dpts(nbr_list[i],0);
			xc_nbrs(1,i) = dpts(nbr_list[i],1);
			xc_nbrs(2,i) = dpts(nbr_list[i],2);
			nbr_list_k(i) = nbr_list[i];
		}
		/**
    Projection Matrix
    */
		W(0,0) = 1.0 - xc(0)*xc(0);
		W(1,1) = 1.0 - xc(1)*xc(1);
		W(2,2) = 1.0 - xc(2)*xc(2);

		W(0,1) = - xc(0)*xc(1);
		W(0,2) = - xc(0)*xc(2);
		W(1,2) = - xc(1)*xc(2);

		W(1,0) = - xc(1)*xc(0);
		W(2,0) = - xc(2)*xc(0);
		W(2,1) = - xc(2)*xc(1);

    /**
    Projection of stencils into TP 
    */
		householder_qr(W,qw);
		KokkosBlas::gemv("N",alpha,W,xc,beta,rot_xc);
		rot_xc_k = rot_xc;
		KokkosBlas::gemm("N","N",alpha,W,xc_nbrs,beta,rot_xc_nbrs);
		rot_nbrs_k = rot_xc_nbrs;
		fill_N_nbrs(k);

	}
}

template<class Input>
void rbf_team_matrices::nbr_search_eval()
{

	
	//Build the tree
	
	std::vector<MyPoint> ptlist(N);
	std::vector<Int> nbr_list;
	fill_ptlist(ptlist,dpts,N);
	kdt::KDTree<MyPoint> kdtree(ptlist);
	Int dim = 3;


	//Build input data structures

	ViewScalarType xc("xc",dim);
	ViewScalarType rot_xc("rot_xc",dim);
	ViewVectorType xc_nbrs("xc_nbrs",dim,stencil_size);
	ViewVectorType rot_xc_nbrs("rot_xc_nbrs",dim,stencil_size);
	ViewVectorType qw("qw",dim,dim);
	ViewVectorType rw("rw",dim,dim);
	ViewVectorType W("W",dim,dim);

	//BLAS inputs
  
	Real alpha = 1,beta = 0;

	
  //Begin weight computation and collection
  
	for(Int k = 0; k < N_eval; ++k){

		
    //Select stencil center xc
    
		MyPoint tmp = MyPoint(epts(k,0),epts(k,1),epts(k,2));
		xc(0) = tmp[0];
		xc(1) = tmp[1];
		xc(2) = tmp[2];

		
    //Find k-nearest neighbors of xc
    
		knn_search_ballpoint_interp(kdtree,tmp,nbr_list,stencil_size);

		
    // Construct neighbor pointlist X_n of xc
    
		auto nbr_list_k = ko::subview(nbr_list_N,k,ko::ALL);
		for (Int i = 0; i < stencil_size; ++i){
			xc_nbrs(0,i) = dpts(nbr_list[i],0);
			xc_nbrs(1,i) = dpts(nbr_list[i],1);
			xc_nbrs(2,i) = dpts(nbr_list[i],2);
			nbr_list_k(i) = nbr_list[i];
		}

		//Projection Matrix
		W(0,0) = 1.0 - xc(0)*xc(0);
		W(1,1) = 1.0 - xc(1)*xc(1);
		W(2,2) = 1.0 - xc(2)*xc(2);

		W(0,1) = - xc(0)*xc(1);
		W(0,2) = - xc(0)*xc(2);
		W(1,2) = - xc(1)*xc(2);

		W(1,0) = - xc(1)*xc(0);
		W(2,0) = - xc(2)*xc(0);
		W(2,1) = - xc(2)*xc(1);

    
    //Projection of stencils into TP 
    
		householder_qr(W,qw);
		KokkosBlas::gemv("N",alpha,W,xc,beta,rot_xc);
		rot_xc_k = rot_xc;
		KokkosBlas::gemm("N","N",alpha,W,xc_nbrs,beta,rot_xc_nbrs);
		rot_nbrs_k = rot_xc_nbrs;
		fill_N_nbrs(k);

	}
}
*/
#endif
