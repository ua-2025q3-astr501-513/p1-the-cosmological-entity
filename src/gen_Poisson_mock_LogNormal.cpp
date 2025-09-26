/* 
	This program calculate the poisson sample from given power spectrum.
	[1]
	A Gaussian random field is generated in Fourier space for given
	LogNormal power spectrum, then real space density for each grid is 
   calculated by the Fourier + Log transform.
	[2]
	Algorithm for poisson sampling for total number of galaxies of Ngals
   is explained in HETDEX_mock.pdf

	19 May 2012
	Donghui Jeong
	[v2] Generate velocities from the matter density, rather than the galaxy density divided by the linear bias, by Aniket Agrawal, March 2016

    - Fix the bug in the interpolator.
    Using one "accelerator" for four spline function screwed up the interpolation when Pmax is large. Define one "acc" for each spline.

    - In MacOS10.15, FFTW_ESTIMATE gave Segmentation fault, and change to FFTW_MEASURE works fine. Perhaps a bug in the default version?

    26 Nov. 2019
    Donghui Jeong

    - Change if(!(n2%2)) to if(n2%2==0)

    - The segmentation faulat problem with FFTW_ESTIMATE is resolved
    by recompiling fftw with the correct variants.
    (e.g.) MacPort example: port install fftw-3 +gcc9 +openmp

    - For high-resolution simulation, changed the index from int to long long.

    2 April 2020
    Donghui Jeong

*/

#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <fftw3.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_linalg.h>
#include <omp.h>
#include "check_var_type.h"
#include <gsl/gsl_spline.h>

#include <iomanip>      // std::setprecision


using namespace std;

int count_num_line(const string &fname);

void ReadParams(int argc, char* argv[]);
string pkGfname;
string mpkGfname;
string cpkGfname;
double lengthx,lengthy,lengthz;
int nmax;
int Ngalaxies;
double aH;
string ffname;
double bias;
int seed;
int Pseed;
int useed;
string Poissonfname;
string Densityfname;
int use_cpkG;
int output_gal;
int output_matter;

void calc_pkvalues(double *pkvalue, double *mpkvalue, double *cpkvalue,
                   gsl_spline *pks, gsl_interp_accel *acc_pk,
                   gsl_spline *mpks, gsl_interp_accel *acc_mpk,
                   gsl_spline *cpks, gsl_interp_accel *acc_cpk,
                   double kvalue);
void calc_deltak(double pkG, double mpkG, double cpkG, double factor, 
				 double *deltak, double *deltamk, const gsl_rng *r);
void interpolation(string ifname, int ndata, gsl_spline *pk);

int main(int argc, char* argv[])
{

	ReadParams(argc, argv);

	// count the line number of input files
	int ndatapk = count_num_line(pkGfname);
	int ndatampk = count_num_line(mpkGfname);
	int ndatacpk = count_num_line(cpkGfname);
	int ndataf = count_num_line(ffname);

// -------------------------------------------------------------------------
// gsl variables
// -------------------------------------------------------------------------
	// initialize the gsl random generator
	const gsl_rng_type *rt;
	// random number in k-space sampling
	gsl_rng *r;			// with 'seed'
	// random number for position
	gsl_rng *rpos;		// with 'Pseed'
	// random number for selecting galaxies
	gsl_rng *rselect;	// with 'useed'
	// random number for flux

	gsl_rng_env_setup();
	rt = gsl_rng_mt19937; // use mt19937 generator

	cout << "Setting up the arrays......."<<endl;
	double max_xy = lengthx > lengthy? lengthx: lengthy; 
	double maxlength = lengthz > max_xy? lengthz: max_xy;
// -------------------------------------------------------------------------
// Allocating the array
//
// n0,n1,n2 is determined so that 
// 1. max(n0,n1,n2) = nmax
// 2. Dbin is constant for all three directions so that the Nyquist frequency
//		is the same for all three directions.
// -------------------------------------------------------------------------
	int n0 = int(ceil(nmax * lengthx / maxlength));
	int n1 = int(ceil(nmax * lengthy / maxlength));
	int n2 = int(ceil(nmax * lengthz / maxlength));

	cout << "n0,n1,n2=" << n0 << "\t" << n1 << "\t" << n2 << endl;

	double Dbin = maxlength / double(nmax); 

	double Lx = Dbin * double(n0);
	double Ly = Dbin * double(n1);
	double Lz = Dbin * double(n2);

	cout << "size of Fourier grid is (n0,n1,n2)"<< endl;
	cout << "("<<n0<<","<<n1<<","<<n2<<")"<<endl;
	cout << "Fourier resolution is " << Dbin << "[Mpc/h]" << endl;

	int cn2 = n2/2+1;				// Size of last column in complex array
	long long Ntotal = n0 * n1;
	Ntotal = Ntotal * n2;			// overall number of real cells
	long long Nc= n0 * n1;
	Nc = Nc * cn2;					// overall number of complex cells (padding)
	double pi = 3.14159265358979;
	double kF0 = 2. * pi / (Lx); // size of x cell in k-space
	double kF1 = 2. * pi / (Ly); // size of y cell in k-space
	double kF2 = 2. * pi / (Lz); // size of z cell in k-space
	double volume = Lx*Ly*Lz;

	double d3r = Dbin*Dbin*Dbin;
	double d3k = 1./volume;

/*-------------------------------------------------------------------------*/
/* allocate arrays and plans for Fourier transformation                    */
/*-------------------------------------------------------------------------*/

	int nproc = 1;
	#pragma omp parallel
	{
		nproc = omp_get_max_threads();
	}
	fftw_init_threads();
	fftw_plan_with_nthreads(nproc);

	fftw_complex *deltak, *deltamk;// complex array in k-space	//modified by Aniket
	double *deltar, *deltamr; 	// storage arrays for positions and velocities	//modified by Aniket

	deltak = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nc);
	deltamk = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nc);	//modified by Aniket

	// aliasing to the double pointer deltar and deltamr for in-place tranform
	deltar = (double *)deltak;
	deltamr = (double *)deltamk;	//modified by Aniket

	fftw_plan Pfftw_c2r, mPfftw_c2r;	//modified by Aniket

	Pfftw_c2r = fftw_plan_dft_c2r_3d(n0, n1, n2, deltak, deltar, FFTW_ESTIMATE);
    mPfftw_c2r = fftw_plan_dft_c2r_3d(n0, n1, n2, deltamk, deltamr,FFTW_ESTIMATE);	//modified by Aniket
//	Pfftw_c2r = fftw_plan_dft_c2r_3d(n0, n1, n2, deltak, deltar, FFTW_MEASURE); 
//   mPfftw_c2r = fftw_plan_dft_c2r_3d(n0, n1, n2, deltamk, deltamr, FFTW_MEASURE);	//modified by Aniket

	//	declare velocity variables
	fftw_complex *vxk;
	fftw_complex *vyk;
	fftw_complex *vzk;
	vxk = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nc);
	vyk = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nc);
	vzk = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nc);

	double *vxr;
	double *vyr;
	double *vzr;
	vxr = (double *) vxk;
	vyr = (double *) vyk;
	vzr = (double *) vzk;

	fftw_plan vx_r2c;
	vx_r2c = fftw_plan_dft_r2c_3d(n0, n1, n2, vxr, vxk, FFTW_ESTIMATE);
	fftw_plan vx_c2r;
	vx_c2r = fftw_plan_dft_c2r_3d(n0, n1, n2, vxk, vxr, FFTW_ESTIMATE);
	fftw_plan vy_c2r;
	vy_c2r = fftw_plan_dft_c2r_3d(n0, n1, n2, vyk, vyr, FFTW_ESTIMATE);
	fftw_plan vz_c2r;
	vz_c2r = fftw_plan_dft_c2r_3d(n0, n1, n2, vzk, vzr, FFTW_ESTIMATE);

/*-----------------------------------------------------------------------------------------------------------------------------*/
/* opening and reading Gussian power spectra and logarithimic growth rate                              */
/*-----------------------------------------------------------------------------------------------------------------------------*/
    gsl_interp_accel *acc_pk  = gsl_interp_accel_alloc ();
    gsl_interp_accel *acc_mpk = gsl_interp_accel_alloc ();
    gsl_interp_accel *acc_cpk = gsl_interp_accel_alloc ();
    gsl_interp_accel *acc_f   = gsl_interp_accel_alloc ();

    gsl_spline *pks = gsl_spline_alloc (gsl_interp_cspline, ndatapk);
    gsl_spline *mpks = gsl_spline_alloc (gsl_interp_cspline, ndatampk);
    gsl_spline *cpks = gsl_spline_alloc (gsl_interp_cspline, ndatacpk);
    gsl_spline *fs = gsl_spline_alloc (gsl_interp_cspline, ndataf);

	interpolation(pkGfname, ndatapk, pks);
	interpolation(mpkGfname, ndatampk, mpks);
	interpolation(cpkGfname, ndatacpk, cpks);
	interpolation(ffname, ndataf, fs);

/*-------------------------------------------------------------------------*/
/* Generating the mock density field in k-space                            */
/*-------------------------------------------------------------------------*/
	cout<<"Generating mock density field in k-space"<<endl;

	r  = gsl_rng_alloc(rt);
	gsl_rng_set(r,seed);

    // Initialize the complex array
	for(int i0 = 0; i0 < n0; i0++){
		for(int i1 = 0; i1 < n1; i1++){
			for(int i2 = 0; i2 < cn2; i2++){
				long long indx = i1 + n1*(i0);
                indx = i2 + cn2*indx;
                deltak[indx][0]=0.0;
                deltak[indx][1]=0.0;
                deltamk[indx][0]=0.0;
                deltamk[indx][1]=0.0;
			}
		}
	}

    // First, generate the mock density field for 
	// deltak(0:n0-1,0:n1-1,1:cn2-2) when n2 is even
	// deltak(0:n0-1,0:n1-1,1:cn2-1) when n2 is odd
	// CAUTION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	// 	Do NOT parallelize this loop. 
	// 	Order of random numbers will not be reproducable.
	// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	// Note that we use the periodicity of the density field 
	// in Fourier space :: deltak[i0,i1,i2] = deltak[i0-n0,i1-n1,i2]
	int cn2max = (n2%2==1? cn2 : cn2-1);
	double k0 = 0.0, k1 = 0.0, k2 = 0.0;
    double kvalue = 0.0, pkvalue = 0.0, mpkvalue = 0.0, cpkvalue = 0.0;
	for(int i0 = 0; i0 < n0; i0++){
		k0 = kF0 * double((i0 < n0/2) ? i0 : i0-n0);
		for(int i1 = 0; i1 < n1; i1++){
			k1 = kF1 * double((i1 < n1/2) ? i1 : i1-n1);
			for(int i2 = 1; i2 < cn2max; i2++){
				k2 = kF2 * double(i2);
				kvalue = sqrt(k0*k0 + k1*k1 + k2*k2);
                calc_pkvalues(&pkvalue, &mpkvalue, &cpkvalue,
                              pks, acc_pk, mpks, acc_mpk, cpks, acc_cpk,
                              kvalue);
				long long indx = i1 + n1*(i0);
                indx = i2 + cn2*indx;
				calc_deltak(pkvalue, mpkvalue, cpkvalue, 0.5*volume,
							&deltak[indx][0], &deltamk[indx][0], r);
				calc_deltak(pkvalue, mpkvalue, cpkvalue, 0.5*volume,
							&deltak[indx][1], &deltamk[indx][1],r );
			}
		}
	}

	// Generating deltak for boundary plane.
	// i2 = 0 && i2 = cn2-1 for even n2
	// i2 = 0               for odd  n2

	/* For those boundary plane, we have to impose the Hermitian condition
		explicitly. That is, we have to impose
		for n2=0 plane,

			deltak[i0,i1,0] = deltak*[-i0,-i1,0] = deltak*[n0-i0,n1-i1,0]

		and, in addition, for even n2,

			deltak[i0,i1,n2/2] = deltak*[-i0,-i1,-n2/2]
                               = deltak*[n0-i0,n1-i1,n2/2]

		<<Pseudo code>>
		do i0=0,n0-1
			x0 = mod(n0-i0,n0)
			do i1=0,n1-1
				x1 = mod(n1-i1,n0)
				if(x0==i0 && x1==i1)
					if(i0==0 && i1==0 && i2==0) then delta[0]=0
					else
					assign real number (Nyquest)
				if(i1+n1*i0 < x1+n1*x0) 
					then assign C to deltak[i0,i1] and C* to deltak[x0,x1]
				else
					Do not assign any value since delta[i0,i1] is already assigned.
			enddo
		enddo
	*/
	// For odd n2, fill only i2=0 plane
	// For even n2, fill i2=0, i2=n2/2 plane

	for(int i0 = 0; i0 < n0; i0++){
		int x0=(n0-i0)%n0;
		k0 = kF0 * double((i0 < n0/2) ? i0 : i0-n0);
		for(int i1 = 0; i1 < n1; i1++){
			int x1=(n1-i1)%n1;
			k1 = kF1 * double((i1 < n1/2) ? i1 : i1-n1);
			double kvalue0 = sqrt(k0*k0 + k1*k1);
			double kvalue1 = sqrt(k0*k0 + k1*k1 + pow(kF2*n2/2.,2.));
			if(i0==x0 && i1==x1){
				if(i0==0 && i1==0){
					deltak[0][0] = 0.;	// DC mode = 0 'cause <delta>=0
					deltak[0][1] = 0.;
					deltamk[0][0] = 0.;	//modified by Aniket
					deltamk[0][1] = 0.;	//modified by Aniket
					// For even n2, generate the real number thing deltak(0,0,n2/2)
					if(n2%2==0){
				        long long indx = i1 + n1*(i0);
                        indx = n2/2 + cn2*indx;
                        calc_pkvalues(&pkvalue, &mpkvalue, &cpkvalue,
                                      pks, acc_pk, mpks, acc_mpk, cpks, acc_cpk,
                                      kvalue1);
						calc_deltak(pkvalue, mpkvalue, cpkvalue, volume,
									&deltak[indx][0], &deltamk[indx][0], r);
						deltak[indx][1] = 0.0; deltamk[indx][1] = 0.0;
					}
				}
				else{ // Nyquist frequency modes are real.
					long long indx = cn2*(i1+n1*(i0));
                    calc_pkvalues(&pkvalue, &mpkvalue, &cpkvalue,
                                pks, acc_pk, mpks, acc_mpk, cpks, acc_cpk,
                                kvalue0);
					calc_deltak(pkvalue, mpkvalue, cpkvalue, volume,
								&deltak[indx][0], &deltamk[indx][0], r);
					deltak[indx][1] = 0.0; deltamk[indx][1] = 0.0;
					// For even n2, repeat the same procedure for i2=n2/2 
					if(n2%2==0){
						long long indx = n2/2+cn2*(i1+n1*(i0));
                        calc_pkvalues(&pkvalue, &mpkvalue, &cpkvalue,
                                    pks, acc_pk, mpks, acc_mpk, cpks, acc_cpk,
                                    kvalue1);	
						calc_deltak(pkvalue, mpkvalue, cpkvalue, volume,
									&deltak[indx][0], &deltamk[indx][0], r);
						deltak[indx][1] = 0.0; deltamk[indx][1] = 0.0;
					}
				}
			}
			else{
				if(i1+n1*i0 < x1+n1*x0){
					// generate the gaussian random field for
					// deltak(i0,i1,0)
				    long long indx = i1 + n1*(i0);
                    indx = cn2*indx;
				    long long indx2 = x1 + n1*(x0);
                    indx2 = cn2*indx2;
                    calc_pkvalues(&pkvalue, &mpkvalue, &cpkvalue,
                                pks, acc_pk, mpks, acc_mpk, cpks, acc_cpk,
                                kvalue0);
					calc_deltak(pkvalue, mpkvalue, cpkvalue, 0.5*volume,
						    	&deltak[indx][0], &deltamk[indx][0], r);
					calc_deltak(pkvalue, mpkvalue, cpkvalue, 0.5*volume,
						    	&deltak[indx][1], &deltamk[indx][1], r);

					
                    // assign the complex conjugate to
					// deltak(x0,x1,0)
					deltak[indx2][0] = deltak[indx][0];
					deltamk[indx2][0] = deltamk[indx][0];	//modified by Aniket
					deltak[indx2][1] = -deltak[indx][1];
					deltamk[indx2][1] = -deltamk[indx][1];	//modified by Aniket

					// For even n2, repeat the same procedure for i2=n2/2 
					if(n2%2==0){
				        long long indx = i1 + n1*(i0);
                        indx = n2/2 + cn2*indx;
				        long long indx2 = x1 + n1*(x0);
                        indx2 = n2/2 + cn2*indx2;
                        calc_pkvalues(&pkvalue, &mpkvalue, &cpkvalue,
                                    pks, acc_pk, mpks, acc_mpk, cpks, acc_cpk,
                                    kvalue1);
						calc_deltak(pkvalue, mpkvalue, cpkvalue, 0.5*volume,
							    	&deltak[indx][0], &deltamk[indx][0], r);
						calc_deltak(pkvalue, mpkvalue, cpkvalue, 0.5*volume,
						        	&deltak[indx][1], &deltamk[indx][1], r);
                
						deltak[indx2][0]=deltak[indx][0];
						deltamk[indx2][0]=deltamk[indx][0];
						deltak[indx2][1]=-deltak[indx][1];
						deltamk[indx2][1]=-deltamk[indx][1];
					}
				}
			}
		}
	}

	// free random generator
	gsl_rng_free(r);
	cout << "Finished generating mock density field." << endl;

/*-------------------------------------------------------------------------*/
/* calculate density map and store it to the file                          */
/*-------------------------------------------------------------------------*/
	// c2r Fourier transform to generate the real space density map
	cout << "Doing FFT for density field." << endl;
	fftw_execute(Pfftw_c2r);
	fftw_execute(mPfftw_c2r);	//modified by Aniket
	cout << "Done FFT for density field." << endl;

	// Physical size of the last column of deltar is cn2*2.
	// However, the last 2 (even n2) or 1 (odd n2) elements are junks,
	// which is the result of the in-place fftw padding.
	int cn22=cn2*2;

	// Normalize the Fourier output to get the density field
	// also calculate the density variance
	long double averageG = 0., averagemG = 0.;	//modified by Aniket
	long double sigmasqG = 0., sigmasqmG = 0.;	//modified by Aniket
	for(int i0 = 0; i0 < n0; i0++){
		for(int i1 = 0; i1 < n1; i1++){
			for(int i2 = 0; i2 < n2; i2++){
				long long ijk = i1 + n1*(i0);
                ijk = i2 + cn22*ijk;
				// normalizing deltar
				deltar[ijk] = deltar[ijk]*d3k;
				deltamr[ijk] = deltamr[ijk]*d3k;			//modified by Aniket
				averageG += (long double)deltar[ijk];
				sigmasqG += (long double)deltar[ijk]*deltar[ijk];
				averagemG += (long double)deltamr[ijk];			//modified by Aniket
				sigmasqmG += (long double)deltamr[ijk]*deltamr[ijk];	//modified by Aniket
			}
		}
	}
	cout<<"Average of Log-normal density field  :"<<averageG<<endl;
	cout<<"Variance of Log-normal density field :"<<sqrt(sigmasqG)<<endl;
	averageG = averageG/(long double)(Ntotal);
	sigmasqG = sigmasqG/(long double)(Ntotal) - averageG*averageG;
	cout<<"Average of Log-normal density field  :"<<averageG<<endl;
	cout<<"Variance of Log-normal density field :"<<sqrt(sigmasqG)<<endl;
	
	cout<<"Average of matter Log-normal density field :"<<averagemG<<endl;		//modified by Aniket
	cout<<"Variance of matter Log-normal density field :"<<sqrt(sigmasqmG)<<endl;	//modified by Aniket
	averagemG = averagemG/(long double)(Ntotal);					//modified by Aniket
	sigmasqmG = sigmasqmG/(long double)(Ntotal) - averagemG*averagemG;		//modified by Aniket
	cout<<"Average of matter Log-normal density field :"<<averagemG<<endl;		//modified by Aniket
	cout<<"Variance of matter Log-normal density field :"<<sqrt(sigmasqmG)<<endl;	//modified by Aniket
	
	long double GtoLN = exp(-0.5*sigmasqG), mGtoLN = exp(-0.5*sigmasqmG);		//modified by Aniket
	double deltar_max=0.;
	double deltar_min=100.;
	long double avg_dr = 0;
	long double var_dr = 0;

	for(int i0 = 0; i0 < n0; i0++){
		for(int i1 = 0; i1 < n1; i1++){
			for(int i2 = 0; i2 < n2; i2++){
				// int ijk = i2 + cn22*(i1 + n1*(i0));
				long long ijk = i1 + n1*(i0);
                ijk = i2 + cn22*ijk;
				// Transform to the density field
				deltar[ijk] = GtoLN*exp(deltar[ijk]) - 1.;
				deltamr[ijk] = mGtoLN*exp(deltamr[ijk]) - 1.;		//modified by Aniket
				long double deltar_l = (long double)(deltar[ijk]);
				avg_dr += deltar_l;
				var_dr += deltar_l*deltar_l;
				// calculate the maximum deltar
				deltar_max = ((deltar_max > deltar[ijk])? deltar_max:deltar[ijk]);
				deltar_min = ((deltar_min < deltar[ijk])? deltar_min:deltar[ijk]);
				vxr[ijk] = deltamr[ijk];				//modified by Aniket
			}
		}
	}

	cout << "density maximum = " << deltar_max << endl;
	cout << "density minimum = " << deltar_min << endl;
	avg_dr = avg_dr/(long double)(Ntotal);
	var_dr = var_dr/(long double)(Ntotal)-avg_dr*avg_dr;
	cout << "Average of density field: " << avg_dr << endl;
	cout << "Variance of density field: " << var_dr << endl;

/*-------------------------------------------------------------------------*/
// Output density grid to get the density power spectrum
/*-------------------------------------------------------------------------*/	

	// if output_matter == 0, skip to output matter density file
	if(output_matter == 1){
    	ofstream densityout(Densityfname.c_str(),ios::binary);			//modified by Aniket
    	if(!densityout.is_open()){
    		cerr<<"Output file cannot be opened!"<<endl;
    	}
    
    	densityout.seekp(0);							//modified by Aniket
    	densityout.write((char*)&lengthx, sizeof(double));
    	densityout.write((char*)&lengthy, sizeof(double));
    	densityout.write((char*)&lengthz, sizeof(double));
    	densityout.write((char*)&n0, sizeof(int));
    	densityout.write((char*)&n1, sizeof(int));
    	densityout.write((char*)&n2, sizeof(int));
    
    	for(int i0 = 0; i0 < n0; i0++){
    	  for(int i1 = 0; i1 < n1; i1++){
    	    for(int i2 = 0; i2 < n2; i2++){
    	      // int ijk = i2 + cn22*(i1 + n1*(i0));
			  long long ijk = i1 + n1*(i0);
              ijk = i2 + cn22*ijk;
    	      densityout.write((char*) &deltamr[ijk], sizeof(double));
    	    }
    	  }
    	}
    	densityout.close();
	}

	// If output_gal == 0, skip the following procedure..
	if(output_gal == 0){
	  return 0;
	}

/*-------------------------------------------------------------------------*/
/* calculate velocity field by FFT-ing the density field                   */
/*-------------------------------------------------------------------------*/
	cout << "Doing FFT for the density field." << endl;
	fftw_execute(vx_r2c);

	cout << "Calculating velocity field in Fourier space..." << endl;
	#pragma omp parallel for
	for(int i0 = 0; i0 < n0; i0++){
		double k0 = kF0 * double((i0 < n0/2) ? i0 : i0-n0);
		for(int i1 = 0; i1 < n1; i1++){
			double k1 = kF1 * double((i1 < n1/2) ? i1 : i1-n1);
			for(int i2 = 0; i2 < cn2; i2++){
				double k2 = kF2 * double(i2);
				double k = sqrt(k0*k0 + k1*k1 + k2*k2);
                double fvalue = (k>0)? gsl_spline_eval (fs, k, acc_f) : 1.0;
				double const_fac = (k>0)? aH/k/k*d3r*fvalue : 0;	//modified by Aniket - smoothing doesn't seem to work
				long long indx = i1+n1*i0;
                indx = i2 + cn2*indx;
				double deltakr = vxk[indx][0]*const_fac;
				double deltaki = vxk[indx][1]*const_fac;
				vxk[indx][0] = -k0*deltaki;
				vyk[indx][0] = -k1*deltaki;
				vzk[indx][0] = -k2*deltaki;
				vxk[indx][1] =  k0*deltakr;
				vyk[indx][1] =  k1*deltakr;
				vzk[indx][1] =  k2*deltakr;
			}
		}
	}

	cout << "Doing FFT for the vx field." << endl;
	fftw_execute(vx_c2r);
	cout << "Doing FFT for the vy field." << endl;
	fftw_execute(vy_c2r);
	cout << "Doing FFT for the vz field." << endl;
	fftw_execute(vz_c2r);

// -------------------------------------------------------------------------
// Initialize the gsl random generator
// -------------------------------------------------------------------------
	cout << "Initializing random generater.." << endl;

	rpos  = gsl_rng_alloc(rt);
	gsl_rng_set(rpos,Pseed);
	rselect = gsl_rng_alloc(rt);
	gsl_rng_set(rselect,useed);

// -------------------------------------------------------------------------
// generating random samples for given total number of galaxies
// -------------------------------------------------------------------------	
	float xx,yy,zz;
	long double vx,vy,vz;
	float* galdata = new float[6*Ngalaxies*2];
	double urand;

	cout << "Generating Poisson particles..........."<<endl;

	long double ngalbar = (long double)(Ngalaxies)/(long double)(Ntotal);

	long double avg_vx = 0;
	long double var_vx = 0;
	double vx_min = 1e5;
	double vx_max = -1e5;
	long double avg_vy = 0;
	long double var_vy = 0;
	double vy_min = 1e5;
	double vy_max = -1e5;
	long double avg_vz = 0;
	long double var_vz = 0;
	double vz_min = 1e5;
	double vz_max = -1e5;

	int nPoisson = 0;
	for(int i0 = 0; i0 < n0; i0++){
		for(int i1 = 0; i1 < n1; i1++){
			for(int i2 = 0; i2 < n2; i2++){
				// int ijk = i2 + cn22*(i1 + n1*(i0));
				long long ijk = i1 + n1*(i0);
                ijk = i2 + cn22*ijk;

				double xmin_thiscell = i0 * Dbin;
				double ymin_thiscell = i1 * Dbin;
				double zmin_thiscell = i2 * Dbin;

				double Nthiscell = ngalbar*(1.+deltar[ijk]);
				int Nthiscell_int = gsl_ran_poisson(rselect,Nthiscell);

				vx = vxr[ijk]*d3k;	//modified by Aniket
				vy = vyr[ijk]*d3k;	//modified by Aniket
				vz = vzr[ijk]*d3k;	//modified by Aniket

				for(int ithiscell=0;ithiscell<Nthiscell_int;ithiscell++){
					urand = gsl_rng_uniform(rpos);
					xx = xmin_thiscell + urand * Dbin;
					urand = gsl_rng_uniform(rpos);
					yy = ymin_thiscell + urand * Dbin;
					urand = gsl_rng_uniform(rpos);
					zz = zmin_thiscell + urand * Dbin;
					// store position of this galaxy into array
					galdata[6*nPoisson  ] = xx;
					galdata[6*nPoisson+1] = yy;
					galdata[6*nPoisson+2] = zz;
					galdata[6*nPoisson+3] = vx;
					galdata[6*nPoisson+4] = vy;
					galdata[6*nPoisson+5] = vz;
					nPoisson++;

					avg_vx += vx;
					avg_vy += vy;
					avg_vz += vz;
					var_vx += vx*vx;
					var_vy += vy*vy;
					var_vz += vz*vz;
					if (vx<=vx_min) vx_min = vx;
					if (vy<=vy_min) vy_min = vy;
					if (vz<=vz_min) vz_min = vz;
					if (vx>=vx_max) vx_max = vx;
					if (vy>=vy_max) vy_max = vy;
					if (vz>=vz_max) vz_max = vz;
				}
			}
		}
	}

	cout << "Total number of " << nPoisson << " galaxies are generated!" << endl;

	fftw_free(deltak);
	fftw_free(deltamk);	//modified by Aniket
	fftw_free(vxk);
	fftw_free(vyk);
	fftw_free(vzk);

	avg_vx = avg_vx/(long double)(nPoisson);
	var_vx = var_vx/(long double)(nPoisson)-avg_vx*avg_vx;
	avg_vy = avg_vy/(long double)(nPoisson);
	var_vy = var_vy/(long double)(nPoisson)-avg_vy*avg_vy;
	avg_vz = avg_vz/(long double)(nPoisson);
	var_vz = var_vz/(long double)(nPoisson)-avg_vz*avg_vz;

	cout << "min[vx] = " << vx_min << "\tmax[vx] = " << vx_max << endl;
	cout << "avg[vx] = " << avg_vx << "\tvar[vx] = " << var_vx << endl;
	cout << "min[vy] = " << vy_min << "\tmax[vy] = " << vy_max << endl;
	cout << "avg[vy] = " << avg_vy << "\tvar[vy] = " << var_vy << endl;
	cout << "min[vz] = " << vz_min << "\tmax[vz] = " << vz_max << endl;
	cout << "avg[vz] = " << avg_vz << "\tvar[vz] = " << var_vz << endl;

	ofstream fout(Poissonfname.c_str(),ios::binary);
	if(!fout.is_open()){
		cerr << "Output file cannot be opened!" << endl;
	}
	// output to file
	fout.seekp(0);
	fout.write((char*) &Lx, sizeof(double));
	fout.write((char*) &Ly, sizeof(double));
	fout.write((char*) &Lz, sizeof(double));
	fout.write((char*) &nPoisson, sizeof(int));
	fout.write((char*) galdata, sizeof(float)*6*nPoisson);
	fout.close();

	delete [] galdata;

	fftw_destroy_plan(Pfftw_c2r);
	fftw_destroy_plan(mPfftw_c2r);	

	fftw_destroy_plan(vx_r2c);
	fftw_destroy_plan(vx_c2r);
	fftw_destroy_plan(vy_c2r);
	fftw_destroy_plan(vz_c2r);

	return 0;
}

int count_num_line(const string &fname){
	int i = 0;
	ifstream ifs(fname.c_str());
	if(ifs){
		string line;
		while(true){
			getline(ifs,line);
	        if(ifs.eof()) break;
	        i++;
		}
	}
	return i;
}

void ReadParams(int argc, char* argv[]){
	if (argc == 1){
		cout << "Please enter the name of the Log-normal Pk file "
                "[1st: k in units of h/Mpc; 2nd: P(k) in units of Mpc^3/h^3]" << endl;
		cin >> pkGfname;
		cout << "Please enter the name of the Log-normal matter Pk file "
                "[1st: k in units of h/Mpc; 2nd: P(k) in units of Mpc^3/h^3]" << endl;
		cin >> mpkGfname;
		cout << "use the galaxy-matter cross pkG as an input? (0:no, 1:yes)" << endl;
		cin >> use_cpkG;
		if (use_cpkG == 1){
			cout << "Please enter the name of the Log-normal galaxy-matter cross Pk file "
                    "[1st: k in units of h/Mpc; 2nd: P(k) in units of Mpc^3/h^3]" << endl;
			cin >> cpkGfname;
		}else{
			cpkGfname = mpkGfname; // dummy
		}
		cout << "Please enter the length of the Fourier box in x, y, and z:" << endl;
		cin >> lengthx >> lengthy >> lengthz;
		cout << "Please enter the maximum size of Fourier grid (1D) !" << endl;
		cin >> nmax;
		cout << "Please enter the total number of galaxies to sample!" << endl;
		cin >> Ngalaxies;
		cout << "Please enter the velocity to distance factor (aH) [km/s/(Mpc/h)]:" << endl;
		cin >> aH;
		cout << "Please enter the logarithmic growth rate file [1st: k in units of h/Mpc; 2nd: f(k,z)]:" << endl;
		cin >> ffname;
		cout << "Please enter the linear bias:" << endl;
		cin >> bias;
		cout << "Please enter the random seed for Fourier space sampling!" << endl;
		cin >> seed;
		cout << "Please enter the random seed for position of galaxy!" << endl;
		cin >> Pseed;
		cout << "Please enter the random seed for selecting galaxies!" << endl;
		cin >> useed;
		cout << "Output the galaxy catalog? (0:no, 1:yes)" << endl;
		cin >> output_gal;
		if (output_gal == 1){
     		cout << "Please enter the output file name to store the galaxy catalog!" <<endl;
	    	cin >> Poissonfname;
		}else{
			Poissonfname = "dummy.dat"; // dummy
		}
		cout << "Output the matter density file? (0:no, 1:yes)" << endl;
		cin >> output_matter;
		if (output_matter == 1){
     		cout << "Please enter the output file name to store the matter density!" <<endl;
	    	cin >> Densityfname;
		}else{
			Densityfname = "dummy.dat"; // dummy
		}
	}else if (argc == 20){
		pkGfname = argv[1];
		mpkGfname = argv[2];
		if(check_int(argv[3],"use_cpkG")) use_cpkG = atoi(argv[3]);
		if(use_cpkG != 0 and use_cpkG != 1){
			cout << "use_cpkG should be 0 or 1!!!" << endl;
			exit(1);
		}
		cpkGfname = argv[4];
		if(check_float(argv[5],"lengthx")) lengthx = atof(argv[5]);
		if(check_float(argv[6],"lengthy")) lengthy = atof(argv[6]);
		if(check_float(argv[7],"lengthz")) lengthz = atof(argv[7]);
		if(check_int(argv[8],"nmax")) nmax = atoi(argv[8]);
		if(check_int(argv[9],"Ngalaxies")) Ngalaxies = atoi(argv[9]);
		if(check_float(argv[10],"aH")) aH = atof(argv[10]);
		ffname = argv[11];
		if(check_float(argv[12],"bias")) bias = atof(argv[12]);
		if(check_int(argv[13],"seed")) seed = atoi(argv[13]);
		if(check_int(argv[14],"Pseed")) Pseed = atoi(argv[14]);
		if(check_int(argv[15],"useed")) useed = atoi(argv[15]);
		Poissonfname = argv[16];
		Densityfname = argv[17];
		if(check_int(argv[18],"output_matter")) output_matter = atoi(argv[18]);
		if(output_matter != 0 and output_matter != 1){
			cout << "output_matter should be 0 or 1!!!" << endl;
			exit(1);
		}
		if(check_int(argv[19],"output_gal")) output_gal = atoi(argv[19]);
		if(output_gal != 0 and output_gal != 1){
			cout << "output_gal should be 0 or 1!!!" << endl;
			exit(1);
		}
	}else{
		cout << "number of arguments should be 0 or 19!!!" << endl;
		exit(1);
	}
}

void calc_deltak(double pkG, double mpkG, double cpkG, double factor, 
                 double *deltak, double *deltamk, const gsl_rng *r){
	if (use_cpkG == 1){
		double cov_data[] = {pkG*factor, cpkG*factor, cpkG*factor, mpkG*factor}; 
		gsl_matrix_view cov_mat = gsl_matrix_view_array(cov_data,2,2);
		gsl_linalg_cholesky_decomp(&cov_mat.matrix);
		double c00 = gsl_matrix_get(&cov_mat.matrix,0,0);
        double c10 = gsl_matrix_get(&cov_mat.matrix,1,0);
		double c11 = gsl_matrix_get(&cov_mat.matrix,1,1);

		double gauss = gsl_ran_gaussian(r,1.0), gauss2 = gsl_ran_gaussian(r,1.0);
		*deltak = c00*gauss;
		*deltamk = c10*gauss+c11*gauss2;
	}else{
		double gauss = gsl_ran_gaussian(r,1.0);
		*deltak = gauss*sqrt(pkG*factor);
		*deltamk = gauss*sqrt(mpkG*factor);
	}
}

// compute interpolating function gsl_spline *pk
void interpolation(string ifname, int ndata, gsl_spline *pk){
    int i = 0;
	string str;
    ifstream ifs(ifname.c_str());
	double xin[ndata]; double yin[ndata];
	while(getline(ifs, str)) {
		sscanf(str.data(), "%lf %lf", &xin[i], &yin[i]);
		i++;
	}
    gsl_spline_init (pk, xin, yin, ndata);
}

// calc pk, mpk and cpk at kvalue using gsl interpolation
void calc_pkvalues(double *pkvalue, double *mpkvalue, double *cpkvalue,
                   gsl_spline *pks, gsl_interp_accel *acc_pk,
                   gsl_spline *mpks, gsl_interp_accel *acc_mpk,
                   gsl_spline *cpks, gsl_interp_accel *acc_cpk,
                   double kvalue){
    *mpkvalue = gsl_spline_eval (mpks, kvalue, acc_mpk);
	if (output_gal == 1){
		*pkvalue = gsl_spline_eval (pks, kvalue, acc_pk);
	}else{
		// in this case don't need to calculate pkvalue,
		// so assign dammy value
		*pkvalue = *mpkvalue;
	}
	if (use_cpkG == 0){
		// in this case don't need to calculate cpkvalue,
		// so assign dammy value
		*cpkvalue = *mpkvalue;
	}else{
    	*cpkvalue = gsl_spline_eval (cpks, kvalue, acc_cpk);
	}
}
