/*****************************************************************
* Copyright (C) by Regents of the University of Minnesota.       *
*                                                                *
* This Software is released under GNU General Public License 2.0 *
* http://www.gnu.org/licenses/gpl-2.0.html                       *
*                                                                *
******************************************************************/

static char help[] = "Testing programming!";

#include <vector>
#include "petscda.h"
#include "petscts.h"
#include "petscpc.h"
#include "petscsnes.h"
#include <stdio.h>
#include <stdlib.h>

#define NEWMETRIC

#ifdef TECIO
	#include "TECIO.h"
#endif

#define max(x, y) (((x) > (y)) ? (x) : (y))
#define min(x, y) (((x) < (y)) ? (x) : (y))

PetscInt ti, block_number, Flux_in;
int binary_input=0;
int xyz_input=0;
PetscInt tis, tie, tsteps=5;
PetscReal angle;
PetscReal delta=0;
int nv_once=0;
int onlyV=0;
int k_average=0;
int j_average=0;
int i_average=0;
int ik_average=0;
int ikc_average=0;	// conditional spatial averaging in ik directions (channel flow)
int reynolds=0;	// 1: contravariant reynolds stress

int i_begin, i_end;
int j_begin, j_end;
int k_begin, k_end;

int pcr=0;
int avg=0, rans=0, rans_output=0, levelset=0;
int vc = 1;
int vtkOutput = 0;
int ASCII = 0;
int surf = 0;

int cs=0;
int i_periodic=0;
int j_periodic=0;
int k_periodic=0;
int kk_periodic=0;
int averaging_option=0;
int pi=-1, pk=-1;
int shear=0;

PetscInt elmtCheck1 = 4317, elmtCheck2 = 3359;

char prefix[256];

//int l, m, n;
/* Symmetric matrix A -> eigenvectors in columns of V, corresponding eigenvalues in d. */
void eigen_decomposition(double A[3][3], double V[3][3], double d[3]);

typedef struct {
	PetscReal t, f;
} FlowWave;

typedef struct {
	PassiveScalar u, v, w, p;
} PassiveField;

typedef struct {
	PetscScalar u, v, w;
} Field;

typedef struct {
	PetscScalar x, y, z;
} Cmpnts;

typedef struct {
	PetscScalar x, y;
} Cmpnts2;

typedef struct {
	PassiveScalar csi[3], eta[3], zet[3], aj;
} Metrics;

typedef struct {
	Vec	Ubcs; // An ugly hack, waste of memory
} BCS;

typedef struct {
	PetscReal	x, y;
} Cpt2D;

typedef struct {
	PetscReal	P;    //Press on the surface elmt
	PetscInt		n_P; //number of Press Pts on the elmt
	PetscReal	Tow_ws, Tow_wt; //wall shear stress of the elmt
	PetscReal	Tow_wn; // normal stress 

	PetscInt		Clsnbpt_i,Clsnbpt_j,Clsnbpt_k; //Closest Near Bndry Pt to each surf elmt 
	PetscInt		icell,jcell,kcell;
	PetscInt		FoundAroundcell, FoundAround2ndCell;
	PetscInt		Need3rdPoint,Need3rdPoint_2,Need3rdPoint_3;
	//PetscInt      Aroundcellnum;
	PetscInt		rank;
} SurfElmtInfo;


typedef struct {
  PetscInt	i1, j1, k1;
  PetscInt	i2, j2, k2;
  PetscInt	i3, j3, k3;
  PetscReal	cr1, cr2, cr3; // coefficients
  PetscReal	d_i; // distance to interception point on grid cells
  PetscInt	imode; // interception mode

  PetscInt	ni, nj, nk;	// fluid cell
  PetscReal	d_s; // shortest distance to solid surfaces
  Cmpnts	pmin;
  PetscInt	cell; // shortest distance surface cell
  PetscReal	cs1, cs2, cs3;

  PetscInt	i11, j11, k11;
  PetscInt	i22, j22, k22;
  PetscInt	i33, j33, k33;
  PetscReal	cr11, cr22, cr33; // coefficients
  PetscReal	d_ii; // distance to interception point on grid cells
  PetscInt	iimode; // interception mode
  PetscReal	cs11, cs22, cs33;

  PetscInt	ii1, jj1, kk1;
  PetscInt	ii2, jj2, kk2;
  PetscInt	ii3, jj3, kk3;
  PetscReal	ct1, ct2, ct3; // coefficients
  //PetscReal	d_s; // distance to interception point on grid cells
  PetscInt	smode; // interception mode

  PetscInt	ii11, jj11, kk11;
  PetscInt	ii22, jj22, kk22;
  PetscInt	ii33, jj33, kk33;
  PetscReal	ct11, ct22, ct33; // coefficients
  PetscReal	d_ss; // distance to interception point on grid cells
  PetscInt	ssmode; // interception mode

  
/*   PetscInt      bi1, bj1, bk1; */
/*   PetscInt      bi2, bj2, bk2; */
/*   PetscInt      bi3, bj3, bk3; */
/*   PetscInt      bi4, bj4, bk4; */

/*   PetscReal     bcr1,bcr2,bcr3,bcr4; */
} IBMInfo;

typedef struct IBMListNode {
	IBMInfo ibm_intp;
	struct IBMListNode* next;
} IBMListNode;

typedef struct IBMList {
	IBMListNode *head;
} IBMList;

typedef struct {
	PetscReal    S_new[6],S_old[6],S_real[6],S_realm1[6];
	PetscReal    S_ang_n[6],S_ang_o[6],S_ang_r[6],S_ang_rm1[6];
	PetscReal    red_vel, damp, mu_s; // reduced vel, damping coeff, mass coeff
	PetscReal    F_x,F_y,F_z, A_tot; //Forces & Area
	PetscReal    F_x_old,F_y_old,F_z_old; //Forces & Area
	PetscReal    F_x_real,F_y_real,F_z_real; //Forces & Area
	PetscReal    M_x,M_y,M_z; // Moments
	PetscReal    M_x_old,M_y_old,M_z_old; //Forces & Area
	PetscReal    M_x_real,M_y_real,M_z_real; //Forces & Area
	PetscReal    M_x_rm2,M_y_rm2,M_z_rm2; //Forces & Area
	PetscReal    M_x_rm3,M_y_rm3,M_z_rm3; //Forces & Area
	PetscReal    x_c,y_c,z_c; // center of rotation(mass)
	PetscReal    Mdpdn_x, Mdpdn_y,Mdpdn_z;
	PetscReal    Mdpdn_x_old, Mdpdn_y_old,Mdpdn_z_old;

	// Aitkin's iteration
	PetscReal    dS[6],dS_o[6],atk,atk_o;

	// for force calculation
	SurfElmtInfo  *elmtinfo;
	IBMInfo       *ibminfo;

	//Max & Min of ibm domain where forces are calculated
	PetscReal    Max_xbc,Min_xbc;
	PetscReal    Max_ybc,Min_ybc;
	PetscReal    Max_zbc,Min_zbc;

	// CV bndry
	PetscInt     CV_ys,CV_ye,CV_zs,CV_ze;

	// add begin (xiaolei)
	PetscReal    omega_x, omega_y, omega_z; 
	PetscReal    nx_tb, ny_tb, nz_tb; // direction vector of rotor axis rotor_model
	PetscReal    angvel_z, angvel_x, angvel_y, angvel_axis;
	PetscReal    x_c0,y_c0,z_c0;
	PetscReal    Torque_generator, J_rotation, CP_max, TSR_max, r_rotor, Torque_aero, ang_axis, angvel_fixed, Force_axis;  
	int rotate_alongaxis;
	// add end (xiaolei)
	PetscReal    Force_rotor_z, Mom_rotor_x;
	
}	FSInfo;

typedef struct {
	PetscInt	nbnumber;
	PetscInt	n_v, n_elmt;	// number of vertices and number of elements
	PetscInt	my_n_v, my_n_elmt;	// seokkoo, my proc
	PetscInt	*nv1, *nv2, *nv3;	// Node index of each triangle
	PetscReal	*nf_x, *nf_y, *nf_z;	// Normal direction
	//PetscReal	*nf_x0, *nf_y0, *nf_z0;	// Normal direction
	PetscReal	*x_bp, *y_bp, *z_bp;	// Coordinates of IBM surface nodes
	PetscReal	*x_bp0, *y_bp0, *z_bp0;
	PetscReal *x_bp_i, *y_bp_i, *z_bp_i;
	
	PetscReal	*x_bp_o, *y_bp_o, *z_bp_o;
	//PetscReal	x_bp_in[101][3270], y_bp_in[101][3270], z_bp_in[101][3270];
	Cmpnts	*u, *uold, *urm1;
  
	// Added 4/1/06 iman
	PetscReal     *dA ;         // area of an element
	// Added 4/2/06 iman
	PetscReal     *nt_x, *nt_y, *nt_z; //tangent dir
	PetscReal     *ns_x, *ns_y, *ns_z; //azimuthal dir
	// Added 6/4/06 iman
	//Cmpnts        *cent; // center of the element 
	PetscReal     *cent_x,*cent_y,*cent_z;

	// for radius check
	Cmpnts *qvec;
	PetscReal *radvec; 
	
	//seokkoo
	
	PetscInt *_nv1, *_nv2, *_nv3;	// for rank 0
	PetscInt *count;
	PetscInt *local2global_elmt;
	int total_n_elmt, total_n_v;
	
	PetscReal *_x_bp, *_y_bp, *_z_bp;	// for rank 0
	PetscReal *shear, *mean_shear;
	PetscReal *reynolds_stress1; //at each element
	PetscReal *reynolds_stress2; //at each element
	PetscReal *reynolds_stress3; //at each element
	PetscReal *pressure; // pressure at each element
		// add begin (xiaolei)
	/* for calculating surface force */
	int	*ib_elmt, *jb_elmt, *kb_elmt; // the lowest interpolation point for element
	PetscReal	*xb_elmt, *yb_elmt, *zb_elmt; // normal extension from the surface element center
	PetscReal	*p_elmt; // only for rotor model
	Cmpnts		*tau_elmt; // only for rotor model

	// add being (xiaolei)
	//
	PetscReal 	*Tmprt_lagr, *Ftmprt_lagr, *tmprt;
	PetscReal 	*F_lagr_x, *F_lagr_y, *F_lagr_z; // force at the IB surface points (lagrange points)
	PetscReal 	*Ft_lagr_avg, *Fa_lagr_avg, *Fr_lagr_avg; // force at the IB surface points (lagrange points)
	PetscReal 	*U_lagr_x, *U_lagr_y, *U_lagr_z;
	PetscReal 	*U_rel; // relative incoming velocity for actuator model 
	int        *i_min, *i_max, *j_min, *j_max, *k_min, *k_max;
	/* ACL */
	PetscReal       *angle_attack, *angle_twist, *chord_blade; // twist angle and chord length at each grid point
	PetscReal       *CD, *CL;
	PetscReal	pitch[3];  // Maximum number of blades: 3
	PetscReal 	U_ref;
	PetscReal	*dhx, *dhy, *dhz;
	PetscReal 	CD_bluff;
	PetscReal 	friction_factor, pressure_factor;
	PetscReal 	axialforcecoefficient, tangentialforcecoefficient;
	PetscReal 	axialprojectedarea, tangentialprojectedarea;
	PetscReal	dh;
	PetscReal	indf_axis, Tipspeedratio, CT, indf_tangent;

	PetscReal 	*Fr_mean, *Fa_mean, *Ft_mean; // force at the IB surface points (lagrange points)
	PetscReal 	*Ur_mean, *Ua_mean, *Ut_mean;
	PetscReal 	*AOA_mean, *Urel_mean;
        // add end (xiaolei)
    int  *color;
    int *s2l; // actuator line element index for each actuator surface element 
} IBMNodes;

typedef struct {
	PetscInt	IM, JM, KM; // dimensions of grid
	DA da;	/* Data structure for scalars (include the grid geometry
						informaion, to obtain the grid information,
						use DAGetCoordinates) */
	DA fda, fda2;	// Data Structure for vectors
	DALocalInfo info;

	Vec	Cent;	// Coordinates of cell centers
	Vec 	Csi, Eta, Zet, Aj;
	Vec 	ICsi, IEta, IZet, IAj;
	Vec 	JCsi, JEta, JZet, JAj;
	Vec 	KCsi, KEta, KZet, KAj;
	Vec 	Ucont;	// Contravariant velocity components
	Vec 	Ucat;	// Cartesian velocity components
	Vec 	WCV;	// vorticity cross velocity	
	Vec	Ucat_o;
	Vec 	P;
	Vec	Phi;
	Vec	GridSpace;
	Vec	Nvert;
	Vec	Nvert_o;
	BCS	Bcs;

	Vec 	WX; //ASR - vorticity
	Vec 	WY;
	Vec 	WZ;
	Vec 	WM;	
	Vec 	HELICITY;
	Vec 	tempValue;
	
	PetscInt	*nvert;//ody property

	PetscReal	ren;	// Reynolds number
	PetscReal	dt; 	// time step
	PetscReal	st;	// Strouhal number

	PetscReal	r[101], tin[101], uinr[101][1001];

	Vec	lUcont, lUcat, lP, lPhi;
	Vec	lCsi, lEta, lZet, lAj;
	Vec	lICsi, lIEta, lIZet, lIAj;
	Vec	lJCsi, lJEta, lJZet, lJAj;
	Vec	lKCsi, lKEta, lKZet, lKAj;
	Vec	lGridSpace;
	Vec	lNvert, lNvert_o;
	Vec	lCent;

	Vec Ucat_sum;		// u, v, w
	Vec Ucat_cross_sum;		// uv, vw, wu
	Vec Ucat_square_sum;	// u^2, v^2, w^2

	PetscInt _this;

	FlowWave *inflow, *kinematics;
	PetscInt number_flowwave, number_kinematics;
	
	IBMList *ibmlist;
} UserCtx;

PetscErrorCode ReadCoordinates(UserCtx *user);
PetscErrorCode QCriteria(UserCtx *user);
PetscErrorCode writeJacobian(UserCtx *user);
PetscErrorCode ASCIIOutPut(UserCtx *user, PetscInt ti, PetscInt component);
PetscErrorCode Vort(UserCtx *user, int dir);
PetscErrorCode Velocity_Magnitude(UserCtx *user, PetscInt component);
PetscErrorCode Lambda2(UserCtx *user);
PetscErrorCode FormMetrics(UserCtx *user);
void Calc_avg_shear_stress(UserCtx *user);
PetscErrorCode ibm_read_ucd_asr(IBMNodes *ibm, PetscInt ibi);
PetscErrorCode fsi_interpolation_coeff(UserCtx *user, IBMNodes *ibm, IBMInfo *ibminfo, SurfElmtInfo *elmtinfo, FSInfo *fsi);
PetscErrorCode Closest_NearBndryPt_ToSurfElmt(UserCtx *user, IBMNodes *ibm, SurfElmtInfo *elmtinfo, FSInfo *fsi, PetscInt ibi);
PetscErrorCode Closest_NearBndryPt_ToSurfElmt_delta(UserCtx *user, IBMNodes *ibm, SurfElmtInfo *elmtinfo, FSInfo *fsi, PetscInt ibi);
PetscErrorCode Find_fsi_interp_Coeff(IBMInfo *ibminfo, UserCtx *user, IBMNodes *ibm, SurfElmtInfo *elmtinfo);
PetscErrorCode Find_fsi_interp_Coeff_delta(IBMInfo *ibminfo, UserCtx *user, IBMNodes *ibm, SurfElmtInfo *elmtinfo);
PetscTruth ISInsideCell(Cmpnts p, Cmpnts cell[8], PetscReal d[6]);
PetscInt ISPointInTriangle(Cmpnts p, Cmpnts p1, Cmpnts p2, Cmpnts p3, PetscReal nfx, PetscReal nfy, PetscReal nfz);
PetscErrorCode triangle_intp_fsi(Cpt2D p, Cpt2D p1, Cpt2D p2, Cpt2D p3, IBMInfo *ibminfo, PetscInt number);
PetscErrorCode triangle_intp2_fsi(Cpt2D p, Cpt2D p1, Cpt2D p2, Cpt2D p3, IBMInfo *ibminfo, PetscInt number);
PetscErrorCode triangle_intp_fsi_2(Cpt2D p, Cpt2D p1, Cpt2D p2, Cpt2D p3, IBMInfo *ibminfo, PetscInt number);
PetscErrorCode linear_intp(Cpt2D p, Cpt2D p1, Cpt2D p2, IBMInfo *ibminfo, PetscInt number, PetscInt nvert);
PetscErrorCode linear_intp_2(Cpt2D p, Cpt2D p1, Cpt2D p2, IBMInfo *ibminfo, PetscInt number, PetscInt nvert);
PetscErrorCode fsi_InterceptionPoint(Cmpnts p, Cmpnts pc[8], PetscReal nvertpc[8], PetscReal nfx, PetscReal nfy, PetscReal nfz, IBMInfo *ibminfo, PetscInt number, Cmpnts *intp, PetscInt *Need3rdPoint);
PetscErrorCode Calc_fsi_surf_stress2(IBMInfo *ibminfo, UserCtx *user, IBMNodes *ibm, SurfElmtInfo *elmtinfo);
PetscErrorCode GridCellaround2ndElmt(UserCtx *user, IBMNodes *ibm, SurfElmtInfo *elmtinfo, Cmpnts pc,PetscInt elmt, PetscInt knbn,PetscInt jnbn, PetscInt inbn, PetscInt *kin, PetscInt *jin, PetscInt *iin, PetscInt *foundFlag);
PetscErrorCode Find_fsi_2nd_interp_Coeff(PetscInt foundFlag, PetscInt i, PetscInt j, PetscInt k, PetscInt elmt, Cmpnts p, IBMInfo *ibminfo, UserCtx *user, IBMNodes *ibm, SurfElmtInfo *elmtinfo);
PetscErrorCode Find_fsi_3rd_interp_Coeff(PetscInt i, PetscInt j, PetscInt k, PetscInt elmt, Cmpnts p, IBMInfo *ibminfo, UserCtx *user, IBMNodes *ibm, SurfElmtInfo *elmtinfo);
PetscErrorCode distance(Cmpnts p1, Cmpnts p2, Cmpnts p3, Cmpnts p4, Cmpnts p, PetscReal *d);
PetscInt ISInsideTriangle2D(Cpt2D p, Cpt2D pa, Cpt2D pb, Cpt2D pc);
PetscInt ISSameSide2D(Cpt2D p, Cpt2D p1, Cpt2D p2, Cpt2D p3);
PetscErrorCode fsi_2nd_InterceptionPoint(Cmpnts p, Cmpnts pc[8], PetscReal nvertpc[8], PetscReal nfx, PetscReal nfy, PetscReal nfz, IBMInfo *ibminfo, PetscInt number, Cmpnts *intp, Cmpnts pOriginal, PetscInt *Need3rdPoint_2);
PetscErrorCode fsi_3rd_InterceptionPoint(Cmpnts p, Cmpnts pc[8], PetscReal nvertpc[8], PetscReal nfx, PetscReal nfy, PetscReal nfz, IBMInfo *ibminfo, PetscInt number, Cmpnts *intp, Cmpnts pOriginal, PetscInt *Need3rdPoint_3);
PetscErrorCode triangle_2nd_intp_fsi(Cpt2D p, Cpt2D p1, Cpt2D p2, Cpt2D p3, IBMInfo *ibminfo, PetscInt number);
void InitIBMList(IBMList *ilist);
void AddIBMNode(IBMList *ilist, IBMInfo ibm_intp);
void DestroyIBMList(IBMList *ilist);
PetscErrorCode ibm_search_advanced(UserCtx *user, IBMNodes *ibm, PetscInt ibi);

//==================================================================================================================
void InitIBMList(IBMList *ilist) {
  ilist->head = PETSC_NULL;
}

void AddIBMNode(IBMList *ilist, IBMInfo ibm_intp)
{
/*	IBMListNode *new;
	PetscMalloc( sizeof(IBMListNode), &new);
	new->next = ilist->head;
	new->ibm_intp = ibm_intp;
	ilist->head = new;
	*/
}

void DestroyIBMList(IBMList *ilist)
{
  /*IBMListNode *current;
  while (ilist->head) {
    current = ilist->head->next;
    PetscFree(ilist->head);
    ilist->head = current;
  }*/
}

PetscErrorCode ibm_search_advanced(UserCtx *user, IBMNodes *ibm, PetscInt ibi)
{

  DA	da = user->da, fda = user->fda;

  DALocalInfo	info = user->info;
  PetscInt	xs = info.xs, xe = info.xs + info.xm;
  PetscInt  	ys = info.ys, ye = info.ys + info.ym;
  PetscInt	zs = info.zs, ze = info.zs + info.zm;
  PetscInt	mx = info.mx, my = info.my, mz = info.mz;
  PetscInt	lxs, lxe, lys, lye, lzs, lze;
  
  PetscInt	i, j, k;

  PetscReal	***nvert;

  lxs = xs; lxe = xe;
  lys = ys; lye = ye;
  lzs = zs; lze = ze;

  if (xs==0) lxs = xs+1;
  if (ys==0) lys = ys+1;
  if (zs==0) lzs = zs+1;

  if (xe==mx) lxe = xe-1;
  if (ye==my) lye = ye-1;
  if (ze==mz) lze = ze-1;

  Cmpnts ***coor;
  Vec Coor;
  DAVecGetArray(fda, Coor, &coor);

  if (user->ibmlist[ibi].head) DestroyIBMList(&(user->ibmlist[ibi]));
  InitIBMList(&user->ibmlist[ibi]); 
  IBMInfo ibm_intp;

  DAVecGetArray(da, user->Nvert, &nvert);

  for (k=lzs; k<lze; k++) {
    for (j=lys; j<lye; j++) {
      for (i=lxs; i<lxe; i++) {
		if ((int)(nvert[k][j][i]+0.5) == 1) {
		  ibm_intp.ni = i;
		  ibm_intp.nj = j;
		  ibm_intp.nk = k;
		  AddIBMNode(&user->ibmlist[ibi], ibm_intp);
		}
      }
    }
  }

  DAVecRestoreArray(fda, Coor,&coor);
  DAVecRestoreArray(da, user->Nvert, &nvert);
  
  return 0;
}

//==================================================================================================================

PetscErrorCode VtkOutput(UserCtx *user, int only_V)
{
	PetscInt	i, j, k, bi, numbytes;
	Cmpnts		***ucat, ***coor, ***ucat_o;
	PetscReal	***p, ***nvert, ***level;
	Vec			Coor, Levelset, K_Omega;
	FILE		*f, *fblock;
	char		filen[80], filen2[80];
	PetscInt	rank;
	size_t		offset = 0;

	printf("Writing VTK Output\n");

	if (cs) {
		printf("Option \"cs\" not implemented in VTK output!\n");
		return -1;
	}

	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	if (!rank) {

		if (block_number > 0) {
			sprintf(filen2, "Result%6.6d.vtm", ti);
			fblock = fopen(filen2, "w");

			// VTK header
			PetscFPrintf(PETSC_COMM_WORLD, fblock, "<VTKFile type=\"vtkMultiBlockDataSet\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt32\">\n");
			PetscFPrintf(PETSC_COMM_WORLD, fblock, "  <vtkMultiBlockDataSet>\n");
		}

		for (bi=0; bi<block_number; bi++) {
			DA			da   = user[bi].da;
			DA			fda  = user[bi].fda;
			DALocalInfo	info = user[bi].info;

			// grid count
			PetscInt	xs = info.xs, xe = info.xs + info.xm;
			PetscInt	ys = info.ys, ye = info.ys + info.ym;
			PetscInt	zs = info.zs, ze = info.zs + info.zm;
			PetscInt	mx = info.mx, my = info.my, mz = info.mz;

			i_begin = 1, i_end = mx-1;	// cross section in tecplot
			j_begin = 1, j_end = my-1;
			k_begin = 1, k_end = mz-1;

			// use options if specified
			PetscOptionsGetInt(PETSC_NULL, "-i_begin", &i_begin, PETSC_NULL);
			PetscOptionsGetInt(PETSC_NULL, "-i_end", &i_end, PETSC_NULL);
			PetscOptionsGetInt(PETSC_NULL, "-j_begin", &j_begin, PETSC_NULL);
			PetscOptionsGetInt(PETSC_NULL, "-j_end", &j_end, PETSC_NULL);
			PetscOptionsGetInt(PETSC_NULL, "-k_begin", &k_begin, PETSC_NULL);
			PetscOptionsGetInt(PETSC_NULL, "-k_end", &k_end, PETSC_NULL);

			xs = i_begin - 1, xe = i_end+1;
			ys = j_begin - 1, ye = j_end+1;
			zs = k_begin - 1, ze = k_end+1;

			DAGetCoordinates(da, &Coor);
			DAVecGetArray(fda, Coor, &coor);
			if (only_V != 2) DAVecGetArray(da,  user[bi].Nvert, &nvert);
			if (!only_V) DAVecGetArray(da,  user[bi].P, &p);
			if (!vc) DAVecGetArray(fda, user[bi].Ucat_o, &ucat_o);
			else DAVecGetArray(fda, user[bi].Ucat, &ucat);

			// Check if petsc_real is a double variable. If this is not the case and you want to make the code work
			// then you need to change the type="Float64" output below to the correct size
			if (PETSC_REAL != PETSC_DOUBLE) {
				printf("PETSC_REAL is not equal to PETSC_DOUBLE which conflicts with the vtk Output\n");
				break;
			}

			sprintf(filen, "Result%6.6d_%2.2d.vts", ti, bi);
			f = fopen(filen, "w");

			// write entry to multiblock file
			if (block_number > 0) {
				PetscFPrintf(PETSC_COMM_WORLD, fblock, "    <DataSet index=\"%d\" file=\"%s\">\n", bi, filen);
				PetscFPrintf(PETSC_COMM_WORLD, fblock, "    </DataSet>\n");
			}

			// VTK header
			PetscFPrintf(PETSC_COMM_WORLD, f, "<VTKFile type=\"StructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt32\">\n");
			PetscFPrintf(PETSC_COMM_WORLD, f, "  <StructuredGrid WholeExtent=\"%d %d %d %d %d %d\">\n", xs, xe-2, ys, ye-2, zs, ze-2);
			PetscFPrintf(PETSC_COMM_WORLD, f, "    <Piece Extent=\"%d %d %d %d %d %d\">\n", xs, xe-2, ys, ye-2, zs, ze-2);

			if (!vc) {
				// point data header
				PetscFPrintf(PETSC_COMM_WORLD, f, "      <PointData>\n");
				// u ca
				PetscFPrintf(PETSC_COMM_WORLD, f, "        <DataArray type=\"Float64\" Name=\"Ucat_o\" NumberOfComponents=\"3\" format=\"appended\" offset=\"%d\"/>\n", offset);
				offset += (ze-1-zs)*(ye-1-ys)*(xe-1-xs)*sizeof(double)*3+sizeof(int);
				// point data footer
				PetscFPrintf(PETSC_COMM_WORLD, f, "      </PointData>\n");
			}

			// cell data header
			PetscFPrintf(PETSC_COMM_WORLD, f, "      <CellData>\n");
			// pressure
			if (!only_V) {
				PetscFPrintf(PETSC_COMM_WORLD, f, "        <DataArray type=\"Float64\" Name=\"P\" format=\"appended\" offset=\"%d\"/>\n", offset);
				offset += (ze-2-zs)*(ye-2-ys)*(xe-2-xs)*sizeof(double)+sizeof(int);
			}
			// ucat
			if (vc) {
				PetscFPrintf(PETSC_COMM_WORLD, f, "        <DataArray type=\"Float64\" Name=\"Ucat\" NumberOfComponents=\"3\" format=\"appended\" offset=\"%d\"/>\n", offset);
				offset += (ze-2-zs)*(ye-2-ys)*(xe-2-xs)*sizeof(double)*3+sizeof(int);
			}
			// nvert
			if (only_V != 2) {
				PetscFPrintf(PETSC_COMM_WORLD, f, "        <DataArray type=\"Float64\" Name=\"Nvert\" format=\"appended\" offset=\"%d\"/>\n", offset);
				offset += (ze-2-zs)*(ye-2-ys)*(xe-2-xs)*sizeof(double)+sizeof(int);
			}
			// rans (k, omega, nut)
			if (!onlyV && rans) {
				PetscFPrintf(PETSC_COMM_WORLD, f, "        <DataArray type=\"Float64\" Name=\"k\" format=\"appended\" offset=\"%d\"/>\n", offset);
				offset += (ze-2-zs)*(ye-2-ys)*(xe-2-xs)*sizeof(double)+sizeof(int);
				PetscFPrintf(PETSC_COMM_WORLD, f, "        <DataArray type=\"Float64\" Name=\"omega\" format=\"appended\" offset=\"%d\"/>\n", offset);
				offset += (ze-2-zs)*(ye-2-ys)*(xe-2-xs)*sizeof(double)+sizeof(int);
				PetscFPrintf(PETSC_COMM_WORLD, f, "        <DataArray type=\"Float64\" Name=\"nut\" format=\"appended\" offset=\"%d\"/>\n", offset);
				offset += (ze-2-zs)*(ye-2-ys)*(xe-2-xs)*sizeof(double)+sizeof(int);
			}
			// levelset
			if (levelset) {
				PetscFPrintf(PETSC_COMM_WORLD, f, "        <DataArray type=\"Float64\" Name=\"Levelset\" format=\"appended\" offset=\"%d\"/>\n", offset);
				offset += (ze-2-zs)*(ye-2-ys)*(xe-2-xs)*sizeof(double)+sizeof(int);
			}
			// point data footer
			PetscFPrintf(PETSC_COMM_WORLD, f, "      </CellData>\n");

			// coordinate saving
			// header
			PetscFPrintf(PETSC_COMM_WORLD, f, "      <Points>\n");
			PetscFPrintf(PETSC_COMM_WORLD, f, "        <DataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"3\" format=\"appended\" offset=\"%d\"/>\n", offset);
			offset += (ze-1-zs)*(ye-1-ys)*(xe-1-xs)*sizeof(double)*3+sizeof(int);
			// footer
			PetscFPrintf(PETSC_COMM_WORLD, f, "      </Points>\n");

			// piece footer
			PetscFPrintf(PETSC_COMM_WORLD, f, "    </Piece>\n");
			PetscFPrintf(PETSC_COMM_WORLD, f, "  </StructuredGrid>\n");

			// data is following next
			PetscFPrintf(PETSC_COMM_WORLD, f, "  <AppendedData encoding=\"raw\">\n_");

			// ucat_o
			if (!vc) {
				numbytes = sizeof(double)*(xe-1-xs)*(ye-1-ys)*(ze-1-zs)*3;
				fwrite(&numbytes, sizeof(int), 1, f);
				for (k=zs; k<ze-1; k++) {
					for (j=ys; j<ye-1; j++) {
						for (i=xs; i<xe-1; i++) {
							double value[3];
							value[0] = ucat_o[k][j][i].x;
							value[1] = ucat_o[k][j][i].y;
							value[2] = ucat_o[k][j][i].z;
							fwrite(value, sizeof(double), 3, f);
						}
					}
				}
			}

			// pressure
			if (!only_V) {
				numbytes = sizeof(double)*(xe-2-xs)*(ye-2-ys)*(ze-2-zs);
				fwrite(&numbytes, sizeof(int), 1, f);
				for (k=zs; k<ze-2; k++) {
					for (j=ys; j<ye-2; j++) {
						for (i=xs; i<xe-2; i++) {
							double value = p[k+1][j+1][i+1];
							fwrite(&value, sizeof(value), 1, f);
						}
					}
				}
			}

			// ucat
			if (vc) {
				numbytes = sizeof(double)*(xe-2-xs)*(ye-2-ys)*(ze-2-zs)*3;
				fwrite(&numbytes, sizeof(int), 1, f);
				for (k=zs; k<ze-2; k++) {
					for (j=ys; j<ye-2; j++) {
						for (i=xs; i<xe-2; i++) {
							double value[3];
							value[0] = ucat[k+1][j+1][i+1].x;
							value[1] = ucat[k+1][j+1][i+1].y;
							value[2] = ucat[k+1][j+1][i+1].z;
							fwrite(value, sizeof(double), 3, f);
						}
					}
				}
			}

			// nvert
			if (only_V != 2) {
				numbytes = sizeof(double)*(xe-2-xs)*(ye-2-ys)*(ze-2-zs);
				fwrite(&numbytes, sizeof(int), 1, f);
				for (k=zs; k<ze-2; k++) {
					for (j=ys; j<ye-2; j++) {
						for (i=xs; i<xe-2; i++) {
							double value = nvert[k+1][j+1][i+1];
							fwrite(&value, sizeof(value), 1, f);
						}
					}
				}
			}

			// rans (k, omega, nut)
			if (!onlyV && rans) {
				Cmpnts2 ***komega;
				DACreateGlobalVector(user[bi].fda2, &K_Omega);
				PetscViewer	viewer;
				sprintf(filen, "kfield%06d_%1d.dat", ti, user->_this);
				PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
				VecLoadIntoVector(viewer, K_Omega);
				PetscViewerDestroy(viewer);
				DAVecGetArray(user[bi].fda2, K_Omega, &komega);

				numbytes = sizeof(double)*(xe-2-xs)*(ye-2-ys)*(ze-2-zs);
				fwrite(&numbytes, sizeof(int), 1, f);
				for (k=zs; k<ze-2; k++) {
					for (j=ys; j<ye-2; j++) {
						for (i=xs; i<xe-2; i++) {
							double value = komega[k+1][j+1][i+1].x;
							fwrite(&value, sizeof(value), 1, f);
						}
					}
				}

				numbytes = sizeof(double)*(xe-2-xs)*(ye-2-ys)*(ze-2-zs);
				fwrite(&numbytes, sizeof(int), 1, f);
				for (k=zs; k<ze-2; k++) {
					for (j=ys; j<ye-2; j++) {
						for (i=xs; i<xe-2; i++) {
							double value = komega[k+1][j+1][i+1].y;
							fwrite(&value, sizeof(value), 1, f);
						}
					}
				}

				numbytes = sizeof(double)*(xe-2-xs)*(ye-2-ys)*(ze-2-zs);
				fwrite(&numbytes, sizeof(int), 1, f);
				for (k=zs; k<ze-2; k++) {
					for (j=ys; j<ye-2; j++) {
						for (i=xs; i<xe-2; i++) {
							double value = komega[k+1][j+1][i+1].x/(komega[k+1][j+1][i+1].y+1e-20);;
							fwrite(&value, sizeof(value), 1, f);
						}
					}
				}

				DAVecRestoreArray(user[bi].fda2, K_Omega, &komega);
				VecDestroy(K_Omega);
			}

			// levelset
			if (levelset) {
				// get levelset data
				DACreateGlobalVector(da, &Levelset);
				PetscViewer viewer;
				sprintf(filen, "lfield%06d_%1d.dat", ti, user->_this);
				PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
				VecLoadIntoVector(viewer, Levelset);
				PetscViewerDestroy(viewer);
				DAVecGetArray(da, Levelset, &level);

				// write data
				numbytes = sizeof(double)*(xe-2-xs)*(ye-2-ys)*(ze-2-zs);
				fwrite(&numbytes, sizeof(int), 1, f);
				for (k=zs; k<ze-2; k++) {
					for (j=ys; j<ye-2; j++) {
						for (i=xs; i<xe-2; i++) {
							double value = level[k+1][j+1][i+1];
							fwrite(&value, sizeof(value), 1, f);
						}
					}
				}

				// cleanup
				DAVecRestoreArray(da, Levelset, &level);
				VecDestroy(Levelset);
			}

			// coordinates
			numbytes = sizeof(double)*(xe-1-xs)*(ye-1-ys)*(ze-1-zs)*3;
			fwrite(&numbytes, sizeof(int), 1, f);
			for (k=zs; k<ze-1; k++) {
				for (j=ys; j<ye-1; j++) {
					for (i=xs; i<xe-1; i++) {
						double value[3];
						value[0] = coor[k][j][i].x;
						value[1] = coor[k][j][i].y;
						value[2] = coor[k][j][i].z;
						fwrite(value, sizeof(double), 3, f);
					}
				}
			}

			// end of data
			PetscFPrintf(PETSC_COMM_WORLD, f, "  </AppendedData>\n");

			DAVecRestoreArray(fda, Coor, &coor);
			if (only_V != 2) DAVecRestoreArray(da , user[bi].Nvert, &nvert);
			if (!only_V) DAVecRestoreArray(da , user[bi].P, &p);
			if (vc) DAVecRestoreArray(fda, user[bi].Ucat, &ucat);
			else DAVecRestoreArray(fda, user[bi].Ucat_o, &ucat_o);

			// VTK footer
			PetscFPrintf(PETSC_COMM_WORLD, f, "</VTKFile>\n");

			// close structured grid file
			fclose(f);

		} // end of block loop

		if (block_number > 0) {
			// VTK header
			PetscFPrintf(PETSC_COMM_WORLD, fblock, "  </vtkMultiBlockDataSet>\n");
			PetscFPrintf(PETSC_COMM_WORLD, fblock, "</VTKFile>\n");

			// close multiblock file
			fclose(fblock);
		}

	} // end of if (!rank)

	return(0);

}

int file_exist(char *str)
{
	int r=0;

	/*if (!my_rank)*/ {
		FILE *fp=fopen(str, "r");
		if (!fp) {
			r=0;
			printf("\nFILE !!! %s does not exist !!!\n", str);
		}
		else {
			fclose(fp);
			r=1;
		}
	}
	MPI_Bcast(&r, 1, MPI_INT, 0, PETSC_COMM_WORLD);
	return r;
};

void Calculate_Covariant_metrics(double g[3][3], double G[3][3])
{
	/*
		| csi.x  csi.y csi.z |-1		| x.csi  x.eta x.zet |
		| eta.x eta.y eta.z |	 =	| y.csi   y.eta  y.zet |
		| zet.x zet.y zet.z |		| z.csi  z.eta z.zet |
	*/

	const double a11=g[0][0], a12=g[0][1], a13=g[0][2];
	const double a21=g[1][0], a22=g[1][1], a23=g[1][2];
	const double a31=g[2][0], a32=g[2][1], a33=g[2][2];

	double det= a11*(a33*a22-a32*a23)-a21*(a33*a12-a32*a13)+a31*(a23*a12-a22*a13);

	G[0][0] = (a33*a22-a32*a23)/det,	G[0][1] = - (a33*a12-a32*a13)/det, 	G[0][2] = (a23*a12-a22*a13)/det;
	G[1][0] = -(a33*a21-a31*a23)/det, G[1][1] = (a33*a11-a31*a13)/det,	G[1][2] = - (a23*a11-a21*a13)/det;
	G[2][0] = (a32*a21-a31*a22)/det,	G[2][1] = - (a32*a11-a31*a12)/det,	G[2][2] = (a22*a11-a21*a12)/det;
};

void Calculate_normal(Cmpnts csi, Cmpnts eta, Cmpnts zet, double ni[3], double nj[3], double nk[3])
{
	double g[3][3];
	double G[3][3];

	g[0][0]=csi.x, g[0][1]=csi.y, g[0][2]=csi.z;
	g[1][0]=eta.x, g[1][1]=eta.y, g[1][2]=eta.z;
	g[2][0]=zet.x, g[2][1]=zet.y, g[2][2]=zet.z;

	Calculate_Covariant_metrics(g, G);
	double xcsi=G[0][0], ycsi=G[1][0], zcsi=G[2][0];
	double xeta=G[0][1], yeta=G[1][1], zeta=G[2][1];
	double xzet=G[0][2], yzet=G[1][2], zzet=G[2][2];

	double nx_i = xcsi, ny_i = ycsi, nz_i = zcsi;
	double nx_j = xeta, ny_j = yeta, nz_j = zeta;
	double nx_k = xzet, ny_k = yzet, nz_k = zzet;

	double sum_i=sqrt(nx_i*nx_i+ny_i*ny_i+nz_i*nz_i);
	double sum_j=sqrt(nx_j*nx_j+ny_j*ny_j+nz_j*nz_j);
	double sum_k=sqrt(nx_k*nx_k+ny_k*ny_k+nz_k*nz_k);

	nx_i /= sum_i, ny_i /= sum_i, nz_i /= sum_i;
	nx_j /= sum_j, ny_j /= sum_j, nz_j /= sum_j;
	nx_k /= sum_k, ny_k /= sum_k, nz_k /= sum_k;

	ni[0] = nx_i, ni[1] = ny_i, ni[2] = nz_i;
	nj[0] = nx_j, nj[1] = ny_j, nj[2] = nz_j;
	nk[0] = nx_k, nk[1] = ny_k, nk[2] = nz_k;
}

double Contravariant_Reynolds_stress(double uu, double uv, double uw, double vv, double vw, double ww,
	double csi0, double csi1, double csi2, double eta0, double eta1, double eta2)
{
	double A = uu*csi0*eta0 + vv*csi1*eta1 + ww*csi2*eta2 + uv * (csi0*eta1+csi1*eta0)	+ uw * (csi0*eta2+csi2*eta0) + vw * (csi1*eta2+csi2*eta1);
	double B = sqrt(csi0*csi0+csi1*csi1+csi2*csi2)*sqrt(eta0*eta0+eta1*eta1+eta2*eta2);

	return A/B;
}

void Compute_du_center (int i, int j, int k,  int mx, int my, int mz, Cmpnts ***ucat, PetscReal ***nvert,
				double *dudc, double *dvdc, double *dwdc,
				double *dude, double *dvde, double *dwde,
				double *dudz, double *dvdz, double *dwdz)
{
	if ((nvert[k][j][i+1])> 0.1) {
		*dudc = ( ucat[k][j][i].x - ucat[k][j][i-1].x );
		*dvdc = ( ucat[k][j][i].y - ucat[k][j][i-1].y );
		*dwdc = ( ucat[k][j][i].z - ucat[k][j][i-1].z );
	}
	else if ((nvert[k][j][i-1])> 0.1) {
		*dudc = ( ucat[k][j][i+1].x - ucat[k][j][i].x );
		*dvdc = ( ucat[k][j][i+1].y - ucat[k][j][i].y );
		*dwdc = ( ucat[k][j][i+1].z - ucat[k][j][i].z );
	}
	else {
		if (i_periodic && i==1) {
			*dudc = ( ucat[k][j][i+1].x - ucat[k][j][mx-2].x ) * 0.5;
			*dvdc = ( ucat[k][j][i+1].y - ucat[k][j][mx-2].y ) * 0.5;
			*dwdc = ( ucat[k][j][i+1].z - ucat[k][j][mx-2].z ) * 0.5;
		}
		else if (i_periodic && i==mx-2) {
			*dudc = ( ucat[k][j][1].x - ucat[k][j][i-1].x ) * 0.5;
			*dvdc = ( ucat[k][j][1].y - ucat[k][j][i-1].y ) * 0.5;
			*dwdc = ( ucat[k][j][1].z - ucat[k][j][i-1].z ) * 0.5;
		}
		else {
			*dudc = ( ucat[k][j][i+1].x - ucat[k][j][i-1].x ) * 0.5;
			*dvdc = ( ucat[k][j][i+1].y - ucat[k][j][i-1].y ) * 0.5;
			*dwdc = ( ucat[k][j][i+1].z - ucat[k][j][i-1].z ) * 0.5;
		}
	}

	if ((nvert[k][j+1][i])> 0.1) {
		*dude = ( ucat[k][j][i].x - ucat[k][j-1][i].x );
		*dvde = ( ucat[k][j][i].y - ucat[k][j-1][i].y );
		*dwde = ( ucat[k][j][i].z - ucat[k][j-1][i].z );
	}
	else if ((nvert[k][j-1][i])> 0.1) {
		*dude = ( ucat[k][j+1][i].x - ucat[k][j][i].x );
		*dvde = ( ucat[k][j+1][i].y - ucat[k][j][i].y );
		*dwde = ( ucat[k][j+1][i].z - ucat[k][j][i].z );
	}
	else {
		if (j_periodic && j==1) {
			*dude = ( ucat[k][j+1][i].x - ucat[k][my-2][i].x ) * 0.5;
			*dvde = ( ucat[k][j+1][i].y - ucat[k][my-2][i].y ) * 0.5;
			*dwde = ( ucat[k][j+1][i].z - ucat[k][my-2][i].z ) * 0.5;
		}
		else if (j_periodic && j==my-2) {
			*dude = ( ucat[k][1][i].x - ucat[k][j-1][i].x ) * 0.5;
			*dvde = ( ucat[k][1][i].y - ucat[k][j-1][i].y ) * 0.5;
			*dwde = ( ucat[k][1][i].z - ucat[k][j-1][i].z ) * 0.5;
		}
		else {
			*dude = ( ucat[k][j+1][i].x - ucat[k][j-1][i].x ) * 0.5;
			*dvde = ( ucat[k][j+1][i].y - ucat[k][j-1][i].y ) * 0.5;
			*dwde = ( ucat[k][j+1][i].z - ucat[k][j-1][i].z ) * 0.5;
		}
	}

	if ((nvert[k+1][j][i])> 0.1) {
		*dudz = ( ucat[k][j][i].x - ucat[k-1][j][i].x );
		*dvdz = ( ucat[k][j][i].y - ucat[k-1][j][i].y );
		*dwdz = ( ucat[k][j][i].z - ucat[k-1][j][i].z );
	}
	else if ((nvert[k-1][j][i])> 0.1) {
		*dudz = ( ucat[k+1][j][i].x - ucat[k][j][i].x );
		*dvdz = ( ucat[k+1][j][i].y - ucat[k][j][i].y );
		*dwdz = ( ucat[k+1][j][i].z - ucat[k][j][i].z );
	}
	else {
		if (k_periodic && k==1) {
			*dudz = ( ucat[k+1][j][i].x - ucat[mz-2][j][i].x ) * 0.5;
			*dvdz = ( ucat[k+1][j][i].y - ucat[mz-2][j][i].y ) * 0.5;
			*dwdz = ( ucat[k+1][j][i].z - ucat[mz-2][j][i].z ) * 0.5;
		}
		else if (k_periodic && k==mz-2) {
			*dudz = ( ucat[1][j][i].x - ucat[k-1][j][i].x ) * 0.5;
			*dvdz = ( ucat[1][j][i].y - ucat[k-1][j][i].y ) * 0.5;
			*dwdz = ( ucat[1][j][i].z - ucat[k-1][j][i].z ) * 0.5;
		}
		else {
			*dudz = ( ucat[k+1][j][i].x - ucat[k-1][j][i].x ) * 0.5;
			*dvdz = ( ucat[k+1][j][i].y - ucat[k-1][j][i].y ) * 0.5;
			*dwdz = ( ucat[k+1][j][i].z - ucat[k-1][j][i].z ) * 0.5;
		}
	}
}


void Compute_du_dxyz (	double csi0, double csi1, double csi2, double eta0, double eta1, double eta2, double zet0, double zet1, double zet2, double ajc,
					double dudc, double dvdc, double dwdc, double dude, double dvde, double dwde, double dudz, double dvdz, double dwdz,
					double *du_dx, double *dv_dx, double *dw_dx, double *du_dy, double *dv_dy, double *dw_dy, double *du_dz, double *dv_dz, double *dw_dz )
{
	*du_dx = (dudc * csi0 + dude * eta0 + dudz * zet0) * ajc;
	*du_dy = (dudc * csi1 + dude * eta1 + dudz * zet1) * ajc;
	*du_dz = (dudc * csi2 + dude * eta2 + dudz * zet2) * ajc;
	*dv_dx = (dvdc * csi0 + dvde * eta0 + dvdz * zet0) * ajc;
	*dv_dy = (dvdc * csi1 + dvde * eta1 + dvdz * zet1) * ajc;
	*dv_dz = (dvdc * csi2 + dvde * eta2 + dvdz * zet2) * ajc;
	*dw_dx = (dwdc * csi0 + dwde * eta0 + dwdz * zet0) * ajc;
	*dw_dy = (dwdc * csi1 + dwde * eta1 + dwdz * zet1) * ajc;
	*dw_dz = (dwdc * csi2 + dwde * eta2 + dwdz * zet2) * ajc;
};


void IKavg(float *x, int xs, int xe, int ys, int ye, int zs, int ze, int mx, int my, int mz)
{
	int i, j, k;

	for (j=ys; j<ye-2; j++) {
		double iksum=0;
		int count=0;
		for (i=xs; i<xe-2; i++)
		for (k=zs; k<ze-2; k++) {
			iksum += x[k * (mx-2)*(my-2) + j*(mx-2) + i];
			count++;
		}
		double ikavg = iksum/(double)count;
		for (i=xs; i<xe-2; i++)
		for (k=zs; k<ze-2; k++) x[k * (mx-2)*(my-2) + j*(mx-2) + i] = ikavg;
	}
}

/*
	pi, pk : # of grid points correcsponding to the period
	conditional averaging
*/
void IKavg_c (float *x, int xs, int xe, int ys, int ye, int zs, int ze, int mx, int my, int mz)
{
	//int i, j, k;

	if (pi<=0) pi = (xe-xs-2); // no averaging
	if (pk<=0) pk = (ze-zs-2); // no averaging

	int ni = (xe-xs-2) / pi;
	int nk = (ze-zs-2) / pk;

	std::vector< std::vector<float> > iksum (pk);

	for (int k=0; k<pk; k++) iksum[k].resize(pi);

	for (int j=ys; j<ye-2; j++) {

		for (int k=0; k<pk; k++) std::fill( iksum[k].begin(), iksum[k].end(), 0.0 );

		for (int i=xs; i<xe-2; i++)
		for (int k=zs; k<ze-2; k++) {
			iksum [ ( k - zs ) % pk ] [ ( i - xs ) % pi] += x [k * (mx-2)*(my-2) + j*(mx-2) + i];
		}

		for (int i=xs; i<xe-2; i++)
		for (int k=zs; k<ze-2; k++) x [k * (mx-2)*(my-2) + j*(mx-2) + i] = iksum [ ( k - zs ) % pk ] [ ( i - xs ) % pi ] / (ni*nk);
	}
}


void Kavg(float *x, int xs, int xe, int ys, int ye, int zs, int ze, int mx, int my, int mz)
{
	int i, j, k;

	for (j=ys; j<ye-2; j++)
	for (i=xs; i<xe-2; i++) {
		double ksum=0;
		int count=0;
		for (k=zs; k<ze-2; k++) {
			ksum += x[k * (mx-2)*(my-2) + j*(mx-2) + i];
			count++;
		}
		double kavg = ksum/(double)count;
		for (k=zs; k<ze-2; k++) x[k * (mx-2)*(my-2) + j*(mx-2) + i] = kavg;
	}

}

void Javg(float *x, int xs, int xe, int ys, int ye, int zs, int ze, int mx, int my, int mz)
{
	int i, j, k;

	for (k=zs; k<ze-2; k++)
	for (i=xs; i<xe-2; i++) {
		double jsum=0;
		int count=0;
		for (j=ys; j<ye-2; j++) {
			jsum += x[k * (mx-2)*(my-2) + j*(mx-2) + i];
			count++;
		}
		double javg = jsum/(double)count;
		for (j=ys; j<ye-2; j++) x[k * (mx-2)*(my-2) + j*(mx-2) + i] = javg;
	}

}

void Iavg(float *x, int xs, int xe, int ys, int ye, int zs, int ze, int mx, int my, int mz)
{
	int i, j, k;

	for (k=zs; k<ze-2; k++) {
		for (j=ys; j<ye-2; j++) {
			double isum=0;
			int count=0;
			for (i=xs; i<xe-2; i++) {
				isum += x[k * (mx-2)*(my-2) + j*(mx-2) + i];
				count++;
			}
			double iavg = isum/(double)count;
			for (i=xs; i<xe-2; i++) x[k * (mx-2)*(my-2) + j*(mx-2) + i] = iavg;
		}
	}
}

void Iavg(Cmpnts ***x, int xs, int xe, int ys, int ye, int zs, int ze, int mx, int my, int mz)
{
	int i, j, k;

	for (k=zs; k<ze-2; k++)
	for (j=ys; j<ye-2; j++) {
		Cmpnts isum, iavg;
		isum.x = isum.y = isum.z = 0;

		int count=0;
		for (i=xs; i<xe-2; i++) {
			 isum.x += x[k+1][j+1][i+1].x;
			 isum.y += x[k+1][j+1][i+1].y;
			 isum.z += x[k+1][j+1][i+1].z;
			 count++;
		}

		iavg.x = isum.x /(double)count;
		iavg.y = isum.y /(double)count;
		iavg.z = isum.z /(double)count;

		for (i=xs; i<xe-2; i++) x[k+1][j+1][i+1] = iavg;
	}
}

void Iavg(PetscReal ***x, int xs, int xe, int ys, int ye, int zs, int ze, int mx, int my, int mz)
{
	int i, j, k;

	for (k=zs; k<ze-2; k++)
	for (j=ys; j<ye-2; j++) {
		double isum, iavg;
		isum = 0;

		int count=0;
		for (i=xs; i<xe-2; i++) {
			 isum += x[k+1][j+1][i+1];
			 count++;
		}
		iavg = isum /(double)count;

		for (i=xs; i<xe-2; i++) x[k+1][j+1][i+1] = iavg;
	}
}

#ifdef TECIO
PetscErrorCode TECIOOut_V(UserCtx *user, int only_V)	// seokkoo
{
	PetscInt bi;

	char filen[80];
	sprintf(filen, "%sResult%06d.plt", prefix, ti);

	INTEGER4 I, Debug, VIsDouble, DIsDouble, III, IMax, JMax, KMax;
	VIsDouble = 0;
	DIsDouble = 0;
	Debug = 0;

	if (only_V)   {
		if (cs) I = TECINI100((char*)"Flow", (char*)"X Y Z UU Cs", filen, (char*)".", &Debug, &VIsDouble);
		else if (only_V==2) I = TECINI100((char*)"Flow", (char*)"X Y Z UU", filen, (char*)".", &Debug, &VIsDouble);
		else I = TECINI100((char*)"Flow", (char*)"X Y Z UU Nv", filen, (char*)".", &Debug, &VIsDouble);
	}
	else {
		if (cs) I = TECINI100((char*)"Flow", (char*)"X Y Z U V W UU P Cs", filen, (char*)".", &Debug, &VIsDouble);
		else if (levelset) I = TECINI100((char*)"Flow", (char*)"X Y Z U V W UU P Nv Level", filen, (char*)".", &Debug, &VIsDouble);
		else I = TECINI100((char*)"Flow", (char*)"X Y Z U V W UU P Nv", filen, (char*)".", &Debug, &VIsDouble);
	}

	for (bi=0; bi<block_number; bi++) {
		DA da = user[bi].da, fda = user[bi].fda;
		DALocalInfo info = user[bi].info;

		PetscInt	xs = info.xs, xe = info.xs + info.xm;
		PetscInt 	ys = info.ys, ye = info.ys + info.ym;
		PetscInt	zs = info.zs, ze = info.zs + info.zm;
		PetscInt	mx = info.mx, my = info.my, mz = info.mz;

		PetscInt	lxs, lys, lzs, lxe, lye, lze;
		PetscInt	i, j, k;
		PetscReal	***aj;
		Cmpnts	***ucat, ***coor, ***ucat_o, ***csi, ***eta, ***zet;
		PetscReal	***p, ***nvert, ***level;
		Vec		Coor, zCoor, nCoor;
		VecScatter	ctx;
		Vec K_Omega;

		DAVecGetArray(user[bi].da, user[bi].Nvert, &nvert);
		/*DAVecGetArray(user[bi].da, user[bi].Aj, &aj);
		DAVecGetArray(user[bi].fda, user[bi].Csi, &csi);
		DAVecGetArray(user[bi].fda, user[bi].Eta, &eta);
		DAVecGetArray(user[bi].fda, user[bi].Zet, &zet);*/
		DAVecGetArray(user[bi].fda, user[bi].Ucat, &ucat);

		INTEGER4	ZoneType=0, ICellMax=0, JCellMax=0, KCellMax=0;
		INTEGER4	IsBlock=1, NumFaceConnections=0, FaceNeighborMode=0;
		INTEGER4    ShareConnectivityFromZone=0;
		INTEGER4	LOC[40] = {1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0}; /* 1 is cell-centered 0 is node centered */

		/*************************/
		printf("mi=%d, mj=%d, mk=%d\n", mx, my, mz);
		printf("xs=%d, xe=%d\n", xs, xe);
		printf("ys=%d, ye=%d\n", ys, ye);
		printf("zs=%d, ze=%d\n", zs, ze);
		//exit(0);

		i_begin = 1, i_end = mx-1;	// cross section in tecplot
		j_begin = 1, j_end = my-1;
		k_begin = 1, k_end = mz-1;

		PetscOptionsGetInt(PETSC_NULL, "-i_begin", &i_begin, PETSC_NULL);
		PetscOptionsGetInt(PETSC_NULL, "-i_end", &i_end, PETSC_NULL);
		PetscOptionsGetInt(PETSC_NULL, "-j_begin", &j_begin, PETSC_NULL);
		PetscOptionsGetInt(PETSC_NULL, "-j_end", &j_end, PETSC_NULL);
		PetscOptionsGetInt(PETSC_NULL, "-k_begin", &k_begin, PETSC_NULL);
		PetscOptionsGetInt(PETSC_NULL, "-k_end", &k_end, PETSC_NULL);

		xs = i_begin - 1, xe = i_end+1;
		ys = j_begin - 1, ye = j_end+1;
		zs = k_begin - 1, ze = k_end+1;

		printf("xs=%d, xe=%d\n", xs, xe);
		printf("ys=%d, ye=%d\n", ys, ye);
		printf("zs=%d, ze=%d\n", zs, ze);
		//exit(0);
		//xs=0, xe=nsection+1;
		/*************************/

		if (vc) {
			LOC[3]=0; LOC[4]=0; LOC[5]=0; LOC[6]=0;
		}
		else if (only_V) {
			LOC[4]=0; LOC[5]=0; LOC[6]=0;
		}
		/*
		IMax = mx-1;
		JMax = my-1;
		KMax = mz-1;
		*/
		IMax = i_end - i_begin + 1;
		JMax = j_end - j_begin + 1;
		KMax = k_end - k_begin + 1;

		I = TECZNE100((char*)"Block 1",
			&ZoneType, 	/* Ordered zone */
			&IMax,
			&JMax,
			&KMax,
			&ICellMax,
			&JCellMax,
			&KCellMax,
			&IsBlock,	/* ISBLOCK  BLOCK format */
			&NumFaceConnections,
			&FaceNeighborMode,
			LOC,
			NULL,
			&ShareConnectivityFromZone); /* No connectivity sharing */

		//III = (mx-1) * (my-1) * (mz-1);
		III = IMax*JMax*KMax;

		DAGetCoordinates(da, &Coor);
		DAVecGetArray(fda, Coor, &coor);

		float *x;
		x = new float [III];

		for (k=zs; k<ze-1; k++)
		for (j=ys; j<ye-1; j++)
		for (i=xs; i<xe-1; i++) {
			x[(k-zs) * IMax*JMax + (j-ys)*IMax + (i-xs)] = coor[k][j][i].x;
		}
		I = TECDAT100(&III, &x[0], &DIsDouble);

		for (k=zs; k<ze-1; k++)
		for (j=ys; j<ye-1; j++)
		for (i=xs; i<xe-1; i++) {
			x[(k-zs) * IMax*JMax + (j-ys)*IMax + (i-xs)] = coor[k][j][i].y;
		}
		I = TECDAT100(&III, &x[0], &DIsDouble);

		for (k=zs; k<ze-1; k++)
		for (j=ys; j<ye-1; j++)
		for (i=xs; i<xe-1; i++) {
			x[(k-zs) * IMax*JMax + (j-ys)*IMax + (i-xs)] = coor[k][j][i].z;
		}
		I = TECDAT100(&III, &x[0], &DIsDouble);

		DAVecRestoreArray(fda, Coor, &coor);
		delete []x;

		if (!vc) {
			x = new float [(mx-1)*(my-1)*(mz-1)];
			DAVecGetArray(user[bi].fda, user[bi].Ucat_o, &ucat_o);
			for (k=0; k<mz-1; k++)
			for (j=0; j<my-1; j++)
			for (i=0; i<mx-1; i++) {
				ucat_o[k][j][i].x = 0.125 *
					(ucat[k][j][i].x + ucat[k][j][i+1].x +
					ucat[k][j+1][i].x + ucat[k][j+1][i+1].x +
					ucat[k+1][j][i].x + ucat[k+1][j][i+1].x +
					ucat[k+1][j+1][i].x + ucat[k+1][j+1][i+1].x);
				ucat_o[k][j][i].y = 0.125 *
					(ucat[k][j][i].y + ucat[k][j][i+1].y +
					ucat[k][j+1][i].y + ucat[k][j+1][i+1].y +
					ucat[k+1][j][i].y + ucat[k+1][j][i+1].y +
					ucat[k+1][j+1][i].y + ucat[k+1][j+1][i+1].y);
				ucat_o[k][j][i].z = 0.125 *
					(ucat[k][j][i].z + ucat[k][j][i+1].z +
					ucat[k][j+1][i].z + ucat[k][j+1][i+1].z +
					ucat[k+1][j][i].z + ucat[k+1][j][i+1].z +
					ucat[k+1][j+1][i].z + ucat[k+1][j+1][i+1].z);
			}

			for (k=zs; k<ze-1; k++)
			for (j=ys; j<ye-1; j++)
			for (i=xs; i<xe-1; i++) {
				x[k * (mx-1)*(my-1) + j*(mx-1) + i] = ucat_o[k][j][i].x;
			}
			if (!only_V) I = TECDAT100(&III, &x[0], &DIsDouble);

			for (k=zs; k<ze-1; k++)
			for (j=ys; j<ye-1; j++)
			for (i=xs; i<xe-1; i++) {
				x[k * (mx-1)*(my-1) + j*(mx-1) + i] = ucat_o[k][j][i].y;
			}
			if (!only_V) I = TECDAT100(&III, &x[0], &DIsDouble);

			for (k=zs; k<ze-1; k++)
			for (j=ys; j<ye-1; j++)
			for (i=xs; i<xe-1; i++) {
				x[k * (mx-1)*(my-1) + j*(mx-1) + i] = ucat_o[k][j][i].z;
			}

			if (!only_V) I = TECDAT100(&III, &x[0], &DIsDouble);

			for (k=zs; k<ze-1; k++)
			for (j=ys; j<ye-1; j++)
			for (i=xs; i<xe-1; i++) {
				x[k * (mx-1)*(my-1) + j*(mx-1) + i] = sqrt( ucat_o[k][j][i].x*ucat_o[k][j][i].x + ucat_o[k][j][i].y*ucat_o[k][j][i].y + ucat_o[k][j][i].z*ucat_o[k][j][i].z );
			}
			I = TECDAT100(&III, &x[0], &DIsDouble);

			DAVecRestoreArray(user[bi].fda, user[bi].Ucat_o, &ucat_o);
			delete []x;
		}
		else {
			//x = new float [(mx-2)*(my-2)*(mz-2)];
			//III = (mx-2) * (my-2) * (mz-2);
			III = (IMax-1)*(JMax-1)*(KMax-1);
			x = new float [III];

			if (!only_V)  {
				for (k=zs; k<ze-2; k++)
				for (j=ys; j<ye-2; j++)
				for (i=xs; i<xe-2; i++) {
					x[ (k-zs) * (IMax-1)*(JMax-1) + (j-ys) * (IMax-1) + (i-xs)] = ucat[k+1][j+1][i+1].x;
				}
				I = TECDAT100(&III, &x[0], &DIsDouble);

				for (k=zs; k<ze-2; k++)
				for (j=ys; j<ye-2; j++)
				for (i=xs; i<xe-2; i++) {
					x[ (k-zs) * (IMax-1)*(JMax-1) + (j-ys) * (IMax-1) + (i-xs)] = ucat[k+1][j+1][i+1].y;
				}
				I = TECDAT100(&III, &x[0], &DIsDouble);

				for (k=zs; k<ze-2; k++)
				for (j=ys; j<ye-2; j++)
				for (i=xs; i<xe-2; i++) {
					x[ (k-zs) * (IMax-1)*(JMax-1) + (j-ys) * (IMax-1) + (i-xs)] = ucat[k+1][j+1][i+1].z;
				}
				I = TECDAT100(&III, &x[0], &DIsDouble);
			}

			for (k=zs; k<ze-2; k++)
			for (j=ys; j<ye-2; j++)
			for (i=xs; i<xe-2; i++) {
				x[ (k-zs) * (IMax-1)*(JMax-1) + (j-ys) * (IMax-1) + (i-xs)] = sqrt(ucat[k+1][j+1][i+1].x*ucat[k+1][j+1][i+1].x+ucat[k+1][j+1][i+1].y*ucat[k+1][j+1][i+1].y+ucat[k+1][j+1][i+1].z*ucat[k+1][j+1][i+1].z);
			}
			I = TECDAT100(&III, &x[0], &DIsDouble);
			delete []x;
		}


		III = (IMax-1)*(JMax-1)*(KMax-1);
		//III = (mx-2) * (my-2) * (mz-2);
	 		x = new float [III];
		//x.resize (III);

		if (!only_V) {
			DAVecGetArray(user[bi].da, user[bi].P, &p);
			for (k=zs; k<ze-2; k++)
			for (j=ys; j<ye-2; j++)
			for (i=xs; i<xe-2; i++) {
				x[ (k-zs) * (IMax-1)*(JMax-1) + (j-ys) * (IMax-1) + (i-xs)] = p[k+1][j+1][i+1];
			}
			I = TECDAT100(&III, &x[0], &DIsDouble);
			DAVecRestoreArray(user[bi].da, user[bi].P, &p);
		}


		if (only_V!=2) {
			for (k=zs; k<ze-2; k++)
			for (j=ys; j<ye-2; j++)
			for (i=xs; i<xe-2; i++) {
				x[ (k-zs) * (IMax-1)*(JMax-1) + (j-ys) * (IMax-1) + (i-xs)] = nvert[k+1][j+1][i+1];
			}
			I = TECDAT100(&III, &x[0], &DIsDouble);
		}

		if (only_V==2) {	// Z Vorticity
			/*
			for (k=zs; k<ze-2; k++)
			for (j=ys; j<ye-2; j++)
			for (i=xs; i<xe-2; i++) {
				double dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz;
				double du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz;
				double csi0 = csi[k][j][i].x, csi1 = csi[k][j][i].y, csi2 = csi[k][j][i].z;
				double eta0= eta[k][j][i].x, eta1 = eta[k][j][i].y, eta2 = eta[k][j][i].z;
				double zet0 = zet[k][j][i].x, zet1 = zet[k][j][i].y, zet2 = zet[k][j][i].z;
				double ajc = aj[k][j][i];

				if (i==0 || j==0 || k==0) x[k * (mx-2)*(my-2) + j*(mx-2) + i] = 0;
				else {
					Compute_du_center (i, j, k, mx, my, mz, ucat, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde, &dudz, &dvdz, &dwdz);
					Compute_du_dxyz (	csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz,
										&du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz );
					x[k * (mx-2)*(my-2) + j*(mx-2) + i] = dv_dx - du_dy;
				}
			}
			I = TECDAT100(&III, &x[0], &DIsDouble);
			*/
		}

		if (!onlyV && rans /*&& rans_output*/) {
			Cmpnts2 ***komega;
			DACreateGlobalVector(user->fda2, &K_Omega);
			PetscViewer	viewer;
			sprintf(filen, "kfield%06d_%1d.dat", ti, user->_this);
			PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
			VecLoadIntoVector(viewer, K_Omega);
			PetscViewerDestroy(viewer);
			DAVecGetArray(user[bi].fda2, K_Omega, &komega);

			for (k=zs; k<ze-2; k++)
			for (j=ys; j<ye-2; j++)
			for (i=xs; i<xe-2; i++) x[ (k-zs) * (IMax-1)*(JMax-1) + (j-ys) * (IMax-1) + (i-xs)] = komega[k+1][j+1][i+1].x;
			I = TECDAT100(&III, &x[0], &DIsDouble);

			for (k=zs; k<ze-2; k++)
			for (j=ys; j<ye-2; j++)
			for (i=xs; i<xe-2; i++) x[ (k-zs) * (IMax-1)*(JMax-1) + (j-ys) * (IMax-1) + (i-xs)] = komega[k+1][j+1][i+1].y;
			I = TECDAT100(&III, &x[0], &DIsDouble);

			for (k=zs; k<ze-2; k++)
			for (j=ys; j<ye-2; j++)
			for (i=xs; i<xe-2; i++) x[ (k-zs) * (IMax-1)*(JMax-1) + (j-ys) * (IMax-1) + (i-xs)] = komega[k+1][j+1][i+1].x/(komega[k+1][j+1][i+1].y+1.e-20);
			I = TECDAT100(&III, &x[0], &DIsDouble);

			DAVecRestoreArray(user[bi].fda2, K_Omega, &komega);
			VecDestroy(K_Omega);
		}
		if (levelset) {
			PetscReal ***level;
			Vec Levelset;
			DACreateGlobalVector(user->da, &Levelset);
			PetscViewer	viewer;
			sprintf(filen, "lfield%06d_%1d.dat", ti, user->_this);
			PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
			VecLoadIntoVector(viewer, Levelset);
			PetscViewerDestroy(viewer);
			DAVecGetArray(user[bi].da, Levelset, &level);

			for (k=zs; k<ze-2; k++)
			for (j=ys; j<ye-2; j++)
			for (i=xs; i<xe-2; i++) x[ (k-zs) * (IMax-1)*(JMax-1) + (j-ys) * (IMax-1) + (i-xs)] = level[k+1][j+1][i+1];
			I = TECDAT100(&III, &x[0], &DIsDouble);

			DAVecRestoreArray(user[bi].da, Levelset, &level);
			VecDestroy(Levelset);
		}

		delete []x;
		/*
		DAVecRestoreArray(user[bi].da, user[bi].Aj, &aj);
		DAVecRestoreArray(user[bi].fda, user[bi].Csi, &csi);
		DAVecRestoreArray(user[bi].fda, user[bi].Eta, &eta);
		DAVecRestoreArray(user[bi].fda, user[bi].Zet, &zet);*/
		DAVecRestoreArray(user[bi].fda, user[bi].Ucat, &ucat);
		DAVecRestoreArray(user[bi].da, user[bi].Nvert, &nvert);
	}
	I = TECEND100();
	return 0;
}
#else
PetscErrorCode TECIOOut_V(UserCtx *user, int only_V)
{
	PetscPrintf(PETSC_COMM_WORLD, "Compiled without Tecplot. Function TECIOOut_V not available!\n");
	return -1;
}
#endif


#ifdef TECIO
PetscErrorCode TECIOOut_Averaging(UserCtx *user)	// seokkoo
{
	PetscInt bi;

	char filen[80];
	sprintf(filen, "%sResult%06d-avg.plt", prefix, ti);

	INTEGER4 I, Debug, VIsDouble, DIsDouble, III, IMax, JMax, KMax;
	VIsDouble = 0;
	DIsDouble = 0;
	Debug = 0;

	if (pcr) I = TECINI100((char*)"Averaging", "X Y Z P Velocity_Magnitude Nv",  filen, (char*)".",  &Debug,  &VIsDouble);
	else if (avg==1) {
		if (averaging_option==1) {
			//I = TECINI100((char*)"Averaging", "X Y Z U V W uu vv ww uv vw uw UV_ VW_ UW_ Nv",  filen, (char*)".",  &Debug,  &VIsDouble); //OSL
			//I = TECINI100((char*)"Averaging", "X Y Z U V W uu vv ww uv vw uw UU_ VV_ WW_ Nv",  filen, (char*)".",  &Debug,  &VIsDouble); //OSL
			I = TECINI100((char*)"Averaging", "X Y Z U V W uu vv ww uv vw uw Nv",  filen, (char*)".",  &Debug,  &VIsDouble);
		}
		else if (averaging_option==2) {
			//I = TECINI100((char*)"Averaging", "X Y Z U V W uu vv ww uv vw uw  UV_ VW_ UW_ P pp Nv",  filen, (char*)".",  &Debug,  &VIsDouble);
			I = TECINI100((char*)"Averaging", "X Y Z U V W uu vv ww uv vw uw  P pp Nv",  filen, (char*)".",  &Debug,  &VIsDouble);
		}
		else if (averaging_option==3) {
			//I = TECINI100((char*)"Averaging", "X Y Z U V W uu vv ww uv vw uw UV_ VW_ UW_ P pp Vortx Vorty Vortz vortx2 vorty2 vortz2 Nv",  filen, (char*)".",  &Debug,  &VIsDouble);
			I = TECINI100((char*)"Averaging", "X Y Z U V W uu vv ww uv vw uw P pp Vortx Vorty Vortz vortx2 vorty2 vortz2 Q_avg Nv",  filen, (char*)".",  &Debug,  &VIsDouble);
		}
	}
	else if (avg==2) I = TECINI100((char*)"Averaging", "X Y Z U V W K Nv",  filen, (char*)".",  &Debug,  &VIsDouble);

	for (bi=0; bi<block_number; bi++) {
		DA da = user[bi].da, fda = user[bi].fda;
		DALocalInfo info = user[bi].info;

		PetscInt	xs = info.xs, xe = info.xs + info.xm;
		PetscInt  	ys = info.ys, ye = info.ys + info.ym;
		PetscInt	zs = info.zs, ze = info.zs + info.zm;
		PetscInt	mx = info.mx, my = info.my, mz = info.mz;

		PetscInt	lxs, lys, lzs, lxe, lye, lze;
		PetscInt	i, j, k;
		PetscReal	***aj;
		Cmpnts	***ucat, ***coor, ***ucat_o;
		Cmpnts	***u2sum, ***u1sum,  ***usum;
		PetscReal	***p, ***nvert;
		Vec		Coor, zCoor, nCoor;
		//VecScatter ctx;

		Vec X, Y, Z, U, V, W;

		INTEGER4	ZoneType=0, ICellMax=0, JCellMax=0, KCellMax=0;
		INTEGER4	IsBlock=1, NumFaceConnections=0, FaceNeighborMode=0;
		INTEGER4    ShareConnectivityFromZone=0;
		INTEGER4	LOC[100] = {1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
		/* 1 is cell-centered   0 is node centered */

		IMax = mx-1; JMax = my-1; KMax = mz-1;

		I = TECZNE100((char*)"Block 1",
			&ZoneType, 	/* Ordered zone */
			&IMax,
			&JMax,
			&KMax,
			&ICellMax,
			&JCellMax,
			&KCellMax,
			&IsBlock,	/* ISBLOCK  BLOCK format */
			&NumFaceConnections,
			&FaceNeighborMode,
			LOC,
			NULL,
			&ShareConnectivityFromZone); /* No connectivity sharing */

		float *x;

		x = new float [(mx-1)*(my-1)*(mz-1)];
		III = (mx-1) * (my-1) * (mz-1);

		DAGetCoordinates(da, &Coor);
		DAVecGetArray(fda, Coor, &coor);

		// X
		for (k=zs; k<ze-1; k++)
		for (j=ys; j<ye-1; j++)
		for (i=xs; i<xe-1; i++) x[k * (mx-1)*(my-1) + j*(mx-1) + i] = coor[k][j][i].x;
		I = TECDAT100(&III, x, &DIsDouble);

		// Y
		for (k=zs; k<ze-1; k++)
		for (j=ys; j<ye-1; j++)
		for (i=xs; i<xe-1; i++) x[k * (mx-1)*(my-1) + j*(mx-1) + i] = coor[k][j][i].y;
		I = TECDAT100(&III, x, &DIsDouble);

		// Z
		for (k=zs; k<ze-1; k++)
		for (j=ys; j<ye-1; j++)
		for (i=xs; i<xe-1; i++) x[k * (mx-1)*(my-1) + j*(mx-1) + i] = coor[k][j][i].z;
		I = TECDAT100(&III, x, &DIsDouble);
		
		DAVecRestoreArray(fda, Coor, &coor);

		//delete []x;
		double N=(double)tis+1.0;
		//x = new float [(mx-2)*(my-2)*(mz-2)];

		III = (mx-2) * (my-2) * (mz-2);

		if (pcr)  {
			DAVecGetArray(user[bi].da, user[bi].P, &p);
			for (k=zs; k<ze-2; k++)
			for (j=ys; j<ye-2; j++)
			for (i=xs; i<xe-2; i++) x[k * (mx-2)*(my-2) + j*(mx-2) + i] = p[k+1][j+1][i+1];
			I = TECDAT100(&III, x, &DIsDouble);
			DAVecRestoreArray(user[bi].da, user[bi].P, &p);

			// Load ucat
			PetscViewer	viewer;
			sprintf(filen, "ufield%06d_%1d.dat", ti, user->_this);
			PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
			VecLoadIntoVector(viewer, (user[bi].Ucat));
			PetscViewerDestroy(viewer);

			DAVecGetArray(user[bi].fda, user[bi].Ucat, &ucat);
			for (k=zs; k<ze-2; k++)
			for (j=ys; j<ye-2; j++)
			for (i=xs; i<xe-2; i++) x[k * (mx-2)*(my-2) + j*(mx-2) + i]
								= sqrt ( ucat[k+1][j+1][i+1].x*ucat[k+1][j+1][i+1].x + ucat[k+1][j+1][i+1].y*ucat[k+1][j+1][i+1].y + ucat[k+1][j+1][i+1].z*ucat[k+1][j+1][i+1].z );
			I = TECDAT100(&III, x, &DIsDouble);
			DAVecRestoreArray(user[bi].fda, user[bi].Ucat, &ucat);
		}
		else if (avg==1) {
			PetscViewer viewer;
			char filen[128];

			DACreateGlobalVector(user[bi].fda, &user[bi].Ucat_sum);
			DACreateGlobalVector(user[bi].fda, &user[bi].Ucat_cross_sum);
			DACreateGlobalVector(user[bi].fda, &user[bi].Ucat_square_sum);

			sprintf(filen, "su0_%06d_%1d.dat", ti, user[bi]._this);
			PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
			VecLoadIntoVector(viewer, (user[bi].Ucat_sum));
			PetscViewerDestroy(viewer);

			sprintf(filen, "su1_%06d_%1d.dat", ti, user[bi]._this);
			PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
			VecLoadIntoVector(viewer, (user[bi].Ucat_cross_sum));
			PetscViewerDestroy(viewer);

			sprintf(filen, "su2_%06d_%1d.dat", ti, user[bi]._this);
			PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
			VecLoadIntoVector(viewer, (user[bi].Ucat_square_sum));
			PetscViewerDestroy(viewer);

			DAVecGetArray(user[bi].fda, user[bi].Ucat_sum, &usum);
			DAVecGetArray(user[bi].fda, user[bi].Ucat_cross_sum, &u1sum);
			DAVecGetArray(user[bi].fda, user[bi].Ucat_square_sum, &u2sum);

			// U
			for (k=zs; k<ze-2; k++)
			for (j=ys; j<ye-2; j++)
			for (i=xs; i<xe-2; i++) x[k * (mx-2)*(my-2) + j*(mx-2) + i] = usum[k+1][j+1][i+1].x/N;
			if (i_average) Iavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (j_average) Javg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (k_average) Kavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (ik_average) IKavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (ikc_average) IKavg_c(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			
			if(ASCII>=1){
				FILE *f2;
				char filen2[80];
				sprintf(filen2, "UcatAvg.dat");
				if ((f2 = fopen(filen2, "r"))){
					fclose(f2);
				}
				else{
					f2 = fopen(filen2, "w");
					PetscFPrintf(PETSC_COMM_WORLD, f2, "%d %d %d\n",xe-2-xs,ye-2-ys,ze-2-zs);
					for (k=zs; k<ze-2; k++){
					for (j=ys; j<ye-2; j++){
					for (i=xs; i<xe-2; i++){
							PetscFPrintf(PETSC_COMM_WORLD, f2, "%e %e %e\n",usum[k+1][j+1][i+1].x/N, usum[k+1][j+1][i+1].y/N, usum[k+1][j+1][i+1].z/N );
					}}}
					fclose(f2);
				}
			}
			I = TECDAT100(&III, x, &DIsDouble);

			// V
			for (k=zs; k<ze-2; k++)
			for (j=ys; j<ye-2; j++)
			for (i=xs; i<xe-2; i++) x[k * (mx-2)*(my-2) + j*(mx-2) + i] = usum[k+1][j+1][i+1].y/N;
			if (i_average) Iavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (j_average) Javg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (k_average) Kavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (ik_average) IKavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (ikc_average) IKavg_c(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			I = TECDAT100(&III, x, &DIsDouble);

			// W
			for (k=zs; k<ze-2; k++)
			for (j=ys; j<ye-2; j++)
			for (i=xs; i<xe-2; i++) x[k * (mx-2)*(my-2) + j*(mx-2) + i] = usum[k+1][j+1][i+1].z/N;
			if (i_average) Iavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (j_average) Javg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (k_average) Kavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (ik_average) IKavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (ikc_average) IKavg_c(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			I = TECDAT100(&III, x, &DIsDouble);

			// uu, u rms
			for (k=zs; k<ze-2; k++)
			for (j=ys; j<ye-2; j++)
			for (i=xs; i<xe-2; i++) {
				double U = usum[k+1][j+1][i+1].x/N;
				x[k * (mx-2)*(my-2) + j*(mx-2) + i] = ( u2sum[k+1][j+1][i+1].x/N - U*U );
			}
			if (i_average) Iavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (j_average) Javg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (k_average) Kavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (ik_average) IKavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (ikc_average) IKavg_c(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			I = TECDAT100(&III, x, &DIsDouble);

			// vv, v rms
			for (k=zs; k<ze-2; k++)
			for (j=ys; j<ye-2; j++)
			for (i=xs; i<xe-2; i++) {
				double V = usum[k+1][j+1][i+1].y/N;
				x[k * (mx-2)*(my-2) + j*(mx-2) + i] = ( u2sum[k+1][j+1][i+1].y/N - V*V );
			}
			if (i_average) Iavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (j_average) Javg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (k_average) Kavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (ik_average) IKavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (ikc_average) IKavg_c(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			I = TECDAT100(&III, x, &DIsDouble);

			// ww, w rms
			for (k=zs; k<ze-2; k++)
			for (j=ys; j<ye-2; j++)
			for (i=xs; i<xe-2; i++) {
				double W = usum[k+1][j+1][i+1].z/N;
				x[k * (mx-2)*(my-2) + j*(mx-2) + i] = ( u2sum[k+1][j+1][i+1].z/N - W*W );
			}
			if (i_average) Iavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (j_average) Javg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (k_average) Kavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (ik_average) IKavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (ikc_average) IKavg_c(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			I = TECDAT100(&III, x, &DIsDouble);

			// uv
			for (k=zs; k<ze-2; k++)
			for (j=ys; j<ye-2; j++)
			for (i=xs; i<xe-2; i++) {
				double UV = usum[k+1][j+1][i+1].x*usum[k+1][j+1][i+1].y / (N*N);
				x[k * (mx-2)*(my-2) + j*(mx-2) + i] = u1sum[k+1][j+1][i+1].x/N - UV;
			}
			if (i_average) Iavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (j_average) Javg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (k_average) Kavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (ik_average) IKavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (ikc_average) IKavg_c(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			I = TECDAT100(&III, x, &DIsDouble);

			// vw
			for (k=zs; k<ze-2; k++)
			for (j=ys; j<ye-2; j++)
			for (i=xs; i<xe-2; i++) {
				double VW = usum[k+1][j+1][i+1].y*usum[k+1][j+1][i+1].z / (N*N);
				x[k * (mx-2)*(my-2) + j*(mx-2) + i] = u1sum[k+1][j+1][i+1].y/N - VW;
			}
			if (i_average) Iavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (j_average) Javg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (k_average) Kavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (ik_average) IKavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (ikc_average) IKavg_c(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			I = TECDAT100(&III, x, &DIsDouble);

			// wu
			for (k=zs; k<ze-2; k++)
			for (j=ys; j<ye-2; j++)
			for (i=xs; i<xe-2; i++) {
				double WU = usum[k+1][j+1][i+1].z*usum[k+1][j+1][i+1].x / (N*N);
				x[k * (mx-2)*(my-2) + j*(mx-2) + i] = u1sum[k+1][j+1][i+1].z/N - WU;
			}
			if (i_average) Iavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (j_average) Javg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (k_average) Kavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (ik_average) IKavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (ikc_average) IKavg_c(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			I = TECDAT100(&III, x, &DIsDouble);

			/*******************************
			//
			DACreateGlobalVector(user[bi].fda, &user[bi].Csi);
			DACreateGlobalVector(user[bi].fda, &user[bi].Eta);
			DACreateGlobalVector(user[bi].fda, &user[bi].Zet);
			DACreateGlobalVector(user[bi].da, &user[bi].Aj);
			FormMetrics(&(user[bi]));

			Cmpnts ***csi, ***eta, ***zet;
			DAVecGetArray(user[bi].fda, user[bi].Csi, &csi);
			DAVecGetArray(user[bi].fda, user[bi].Eta, &eta);
			DAVecGetArray(user[bi].fda, user[bi].Zet, &zet);

			//UV_ or UU_
			for (k=zs; k<ze-2; k++)
			for (j=ys; j<ye-2; j++)
			for (i=xs; i<xe-2; i++) {
				double U = usum[k+1][j+1][i+1].x/N;
				double V = usum[k+1][j+1][i+1].y/N;
				double W = usum[k+1][j+1][i+1].z/N;

				double uu = ( u2sum[k+1][j+1][i+1].x/N - U*U );
				double vv = ( u2sum[k+1][j+1][i+1].y/N - V*V );
				double ww = ( u2sum[k+1][j+1][i+1].z/N - W*W );
				double uv = u1sum[k+1][j+1][i+1].x/N - U*V;
				double vw = u1sum[k+1][j+1][i+1].y/N - V*W;
				double uw = u1sum[k+1][j+1][i+1].z/N - W*U;

				double csi0 = csi[k+1][j+1][i+1].x, csi1 = csi[k+1][j+1][i+1].y, csi2 = csi[k+1][j+1][i+1].z;
				double eta0 = eta[k+1][j+1][i+1].x, eta1 = eta[k+1][j+1][i+1].y, eta2 = eta[k+1][j+1][i+1].z;
				double zet0 = zet[k+1][j+1][i+1].x, zet1 = zet[k+1][j+1][i+1].y, zet2 = zet[k+1][j+1][i+1].z;

				// UV_
				//x[k * (mx-2)*(my-2) + j*(mx-2) + i] = Contravariant_Reynolds_stress(uu, uv, uw, vv, vw, ww,	csi0, csi1, csi2, eta0, eta1, eta2);
				// UU_
				x[k * (mx-2)*(my-2) + j*(mx-2) + i] = Contravariant_Reynolds_stress(uu, uv, uw, vv, vw, ww,	csi0, csi1, csi2, csi0, csi1, csi2);
			}
			if (i_average) Iavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (j_average) Javg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (k_average) Kavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (ik_average) IKavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (ikc_average) IKavg_c(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			I = TECDAT100(&III, x, &DIsDouble);

			// VW_ or VV_
			for (k=zs; k<ze-2; k++)
			for (j=ys; j<ye-2; j++)
			for (i=xs; i<xe-2; i++) {
				double U = usum[k+1][j+1][i+1].x/N;
				double V = usum[k+1][j+1][i+1].y/N;
				double W = usum[k+1][j+1][i+1].z/N;

				double uu = ( u2sum[k+1][j+1][i+1].x/N - U*U );
				double vv = ( u2sum[k+1][j+1][i+1].y/N - V*V );
				double ww = ( u2sum[k+1][j+1][i+1].z/N - W*W );
				double uv = u1sum[k+1][j+1][i+1].x/N - U*V;
				double vw = u1sum[k+1][j+1][i+1].y/N - V*W;
				double uw = u1sum[k+1][j+1][i+1].z/N - W*U;

				double csi0 = csi[k+1][j+1][i+1].x, csi1 = csi[k+1][j+1][i+1].y, csi2 = csi[k+1][j+1][i+1].z;
				double eta0 = eta[k+1][j+1][i+1].x, eta1 = eta[k+1][j+1][i+1].y, eta2 = eta[k+1][j+1][i+1].z;
				double zet0 = zet[k+1][j+1][i+1].x, zet1 = zet[k+1][j+1][i+1].y, zet2 = zet[k+1][j+1][i+1].z;

				// VW_
				//x[k * (mx-2)*(my-2) + j*(mx-2) + i] = Contravariant_Reynolds_stress(uu, uv, uw, vv, vw, ww,	zet0, zet1, zet2, eta0, eta1, eta2);
				// VV_
				x[k * (mx-2)*(my-2) + j*(mx-2) + i] = Contravariant_Reynolds_stress(uu, uv, uw, vv, vw, ww,	eta0, eta1, eta2, eta0, eta1, eta2);
			}
			if (i_average) Iavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (j_average) Javg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (k_average) Kavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (ik_average) IKavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (ikc_average) IKavg_c(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			I = TECDAT100(&III, x, &DIsDouble);

			//UW_ or WW_
			for (k=zs; k<ze-2; k++)
			for (j=ys; j<ye-2; j++)
			for (i=xs; i<xe-2; i++) {
				double U = usum[k+1][j+1][i+1].x/N;
				double V = usum[k+1][j+1][i+1].y/N;
				double W = usum[k+1][j+1][i+1].z/N;

				double uu = ( u2sum[k+1][j+1][i+1].x/N - U*U );
				double vv = ( u2sum[k+1][j+1][i+1].y/N - V*V );
				double ww = ( u2sum[k+1][j+1][i+1].z/N - W*W );
				double uv = u1sum[k+1][j+1][i+1].x/N - U*V;
				double vw = u1sum[k+1][j+1][i+1].y/N - V*W;
				double uw = u1sum[k+1][j+1][i+1].z/N - W*U;

				double csi0 = csi[k+1][j+1][i+1].x, csi1 = csi[k+1][j+1][i+1].y, csi2 = csi[k+1][j+1][i+1].z;
				double eta0 = eta[k+1][j+1][i+1].x, eta1 = eta[k+1][j+1][i+1].y, eta2 = eta[k+1][j+1][i+1].z;
				double zet0 = zet[k+1][j+1][i+1].x, zet1 = zet[k+1][j+1][i+1].y, zet2 = zet[k+1][j+1][i+1].z;

				// UW_
				//x[k * (mx-2)*(my-2) + j*(mx-2) + i] = Contravariant_Reynolds_stress(uu, uv, uw, vv, vw, ww,	csi0, csi1, csi2, zet0, zet1, zet2);
				// WW_
				x[k * (mx-2)*(my-2) + j*(mx-2) + i] = Contravariant_Reynolds_stress(uu, uv, uw, vv, vw, ww,	zet0, zet1, zet2, zet0, zet1, zet2);
			}
			if (i_average) Iavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (j_average) Javg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (k_average) Kavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (ik_average) IKavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (ikc_average) IKavg_c(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			I = TECDAT100(&III, x, &DIsDouble);

			DAVecRestoreArray(user[bi].fda, user[bi].Csi, &csi);
			DAVecRestoreArray(user[bi].fda, user[bi].Eta, &eta);
			DAVecRestoreArray(user[bi].fda, user[bi].Zet, &zet);

			VecDestroy(user[bi].Csi);
			VecDestroy(user[bi].Eta);
			VecDestroy(user[bi].Zet);
			VecDestroy(user[bi].Aj);

			DAVecRestoreArray(user[bi].fda, user[bi].Ucat_sum, &usum);
			DAVecRestoreArray(user[bi].fda, user[bi].Ucat_cross_sum, &u1sum);
			DAVecRestoreArray(user[bi].fda, user[bi].Ucat_square_sum, &u2sum);

			VecDestroy(user[bi].Ucat_sum);
			VecDestroy(user[bi].Ucat_cross_sum);
			VecDestroy(user[bi].Ucat_square_sum);
			//
			********************************/

			if (averaging_option>=2) {
				Vec P_sum, P_square_sum;
				PetscReal ***psum, ***p2sum;

				DACreateGlobalVector(user[bi].da, &P_sum);
				DACreateGlobalVector(user[bi].da, &P_square_sum);

				sprintf(filen, "sp_%06d_%1d.dat", ti, user[bi]._this);
				PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
				VecLoadIntoVector(viewer, P_sum);
				PetscViewerDestroy(viewer);

				sprintf(filen, "sp2_%06d_%1d.dat", ti, user[bi]._this);
				PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
				VecLoadIntoVector(viewer, P_square_sum);
				PetscViewerDestroy(viewer);

				DAVecGetArray(user[bi].da, P_sum, &psum);
				DAVecGetArray(user[bi].da, P_square_sum, &p2sum);

				for (k=zs; k<ze-2; k++)
				for (j=ys; j<ye-2; j++)
				for (i=xs; i<xe-2; i++) {
					double P = psum[k+1][j+1][i+1]/N;
					x[k * (mx-2)*(my-2) + j*(mx-2) + i] = P;
				}
				if (i_average) Iavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
				if (j_average) Javg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
				if (k_average) Kavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
				if (ik_average) IKavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
				if (ikc_average) IKavg_c(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
				I = TECDAT100(&III, x, &DIsDouble);


				for (k=zs; k<ze-2; k++)
				for (j=ys; j<ye-2; j++)
				for (i=xs; i<xe-2; i++) {
					double P = psum[k+1][j+1][i+1]/N;
					x[k * (mx-2)*(my-2) + j*(mx-2) + i] = ( p2sum[k+1][j+1][i+1]/N - P*P );
				}
				if (i_average) Iavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
				if (j_average) Javg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
				if (k_average) Kavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
				if (ik_average) IKavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
				if (ikc_average) IKavg_c(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
				I = TECDAT100(&III, x, &DIsDouble);

				DAVecRestoreArray(user[bi].da, P_sum, &psum);
				DAVecRestoreArray(user[bi].da, P_square_sum, &p2sum);

				VecDestroy(P_sum);
				VecDestroy(P_square_sum);
			}

			if (averaging_option>=3) {
				Vec Vort_sum, Vort_square_sum;
				Cmpnts ***vortsum, ***vort2sum;

				DACreateGlobalVector(user[bi].fda, &Vort_sum);
				DACreateGlobalVector(user[bi].fda, &Vort_square_sum);

				sprintf(filen, "svo_%06d_%1d.dat", ti, user[bi]._this);
				PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
				VecLoadIntoVector(viewer, Vort_sum);
				PetscViewerDestroy(viewer);

				sprintf(filen, "svo2_%06d_%1d.dat", ti, user[bi]._this);
				PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
				VecLoadIntoVector(viewer, Vort_square_sum);
				PetscViewerDestroy(viewer);

				DAVecGetArray(user[bi].fda, Vort_sum, &vortsum);
				DAVecGetArray(user[bi].fda, Vort_square_sum, &vort2sum);
				
				PetscPrintf(PETSC_COMM_WORLD, "Achtung!%f\n",vortsum[10][10][10].y/N);
				
				PetscReal ***nvert;
				DAVecGetArray(user[bi].da, user[bi].Nvert, &nvert);
				
                if(ASCII>=2){
                    FILE *f2;
                    char filen2[80];
                    sprintf(filen2, "VortAvg.dat");
                    if ((f2 = fopen(filen2, "r"))){
                        fclose(f2);
                    }
                    else{
						f2 = fopen(filen2, "w");
						PetscFPrintf(PETSC_COMM_WORLD, f2, "%d %d %d\n",xe-2-xs,ye-2-ys,ze-2-zs);
						for (k=zs; k<ze-2; k++){
						for (j=ys; j<ye-2; j++){
						for (i=xs; i<xe-2; i++){
							if ( nvert[k][j][i]>0.1 ){
								PetscFPrintf(PETSC_COMM_WORLD, f2, "0.0 0.0 0.0\n");
							}
							else {
								PetscFPrintf(PETSC_COMM_WORLD, f2, "%e %e %e\n",vortsum[k+1][j+1][i+1].x/N, vortsum[k+1][j+1][i+1].y/N, vortsum[k+1][j+1][i+1].z/N );
							}
						}}}
						fclose(f2);
                    }
                }

               if(ASCII>=3){
                    FILE *f2;
                    char filen2[80];
                    sprintf(filen2, "HelicityAvg.dat");
                    if ((f2 = fopen(filen2, "r"))){
                        fclose(f2);
                    }
                    else{
						f2 = fopen(filen2, "w");
						PetscFPrintf(PETSC_COMM_WORLD, f2, "%d %d %d\n",xe-2-xs,ye-2-ys,ze-2-zs);
						for (k=zs; k<ze-2; k++){
						for (j=ys; j<ye-2; j++){
						for (i=xs; i<xe-2; i++){
							if ( nvert[k][j][i]>0.1 ){
								PetscFPrintf(PETSC_COMM_WORLD, f2, "0.0 0.0 0.0\n");
							}
							else {							
								PetscFPrintf(PETSC_COMM_WORLD, f2, "%e %e %e\n",vortsum[k+1][j+1][i+1].x/N*usum[k+1][j+1][i+1].x/N, vortsum[k+1][j+1][i+1].y/N*usum[k+1][j+1][i+1].y/N, vortsum[k+1][j+1][i+1].z/N*usum[k+1][j+1][i+1].z/N );
							}
						}}}
						fclose(f2);
                    }
                }

				for (k=zs; k<ze-2; k++)
				for (j=ys; j<ye-2; j++)
				for (i=xs; i<xe-2; i++) {
					double vortx = vortsum[k+1][j+1][i+1].x/N;
						x[k * (mx-2)*(my-2) + j*(mx-2) + i] = vortx;
				}
				if (i_average) Iavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
				if (j_average) Javg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
				if (k_average) Kavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
				if (ik_average) IKavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
				if (ikc_average) IKavg_c(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
				I = TECDAT100(&III, x, &DIsDouble);

				for (k=zs; k<ze-2; k++)
				for (j=ys; j<ye-2; j++)
				for (i=xs; i<xe-2; i++) {
					double vorty = vortsum[k+1][j+1][i+1].y/N;
					x[k * (mx-2)*(my-2) + j*(mx-2) + i] = vorty;
				}
				if (i_average) Iavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
				if (j_average) Javg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
				if (k_average) Kavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
				if (ik_average) IKavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
				if (ikc_average) IKavg_c(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
				I = TECDAT100(&III, x, &DIsDouble);

				for (k=zs; k<ze-2; k++)
				for (j=ys; j<ye-2; j++)
				for (i=xs; i<xe-2; i++) {
					double vortz = vortsum[k+1][j+1][i+1].z/N;
					x[k * (mx-2)*(my-2) + j*(mx-2) + i] = vortz;
				}
				if (i_average) Iavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
				if (j_average) Javg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
				if (k_average) Kavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
				if (ik_average) IKavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
				if (ikc_average) IKavg_c(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
				I = TECDAT100(&III, x, &DIsDouble);

				for (k=zs; k<ze-2; k++)
				for (j=ys; j<ye-2; j++)
				for (i=xs; i<xe-2; i++) {
					double vortx = vortsum[k+1][j+1][i+1].x/N;
					x[k * (mx-2)*(my-2) + j*(mx-2) + i] = ( vort2sum[k+1][j+1][i+1].x/N - vortx*vortx );
				}
				if (i_average) Iavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
				if (j_average) Javg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
				if (k_average) Kavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
				if (ik_average) IKavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
				if (ikc_average) IKavg_c(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
				I = TECDAT100(&III, x, &DIsDouble);

				for (k=zs; k<ze-2; k++)
				for (j=ys; j<ye-2; j++)
				for (i=xs; i<xe-2; i++) {
					double vorty = vortsum[k+1][j+1][i+1].y/N;
					x[k * (mx-2)*(my-2) + j*(mx-2) + i] = ( vort2sum[k+1][j+1][i+1].y/N - vorty*vorty );
				}
				if (i_average) Iavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
				if (j_average) Javg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
				if (k_average) Kavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
				if (ik_average) IKavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
				if (ikc_average) IKavg_c(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
				I = TECDAT100(&III, x, &DIsDouble);

				for (k=zs; k<ze-2; k++)
				for (j=ys; j<ye-2; j++)
				for (i=xs; i<xe-2; i++) {
					double vortz = vortsum[k+1][j+1][i+1].z/N;
					x[k * (mx-2)*(my-2) + j*(mx-2) + i] = ( vort2sum[k+1][j+1][i+1].z/N - vortz*vortz );
				}
				if (i_average) Iavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
				if (j_average) Javg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
				if (k_average) Kavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
				if (ik_average) IKavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
				if (ikc_average) IKavg_c(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
				I = TECDAT100(&III, x, &DIsDouble);
								
				// Calculating q-criterion
				Cmpnts ***csi, ***eta, ***zet, ***uavg;
				PetscReal ***q; 
				DAVecGetArray(user[bi].fda, user[bi].Csi, &csi);
				DAVecGetArray(user[bi].fda, user[bi].Eta, &eta);
				DAVecGetArray(user[bi].fda, user[bi].Zet, &zet);
				DAVecGetArray(user[bi].da, user[bi].Aj, &aj);
				DAVecGetArray(user[bi].fda, user[bi].Ucat, &uavg);		
				DAVecGetArray(user[bi].da, user[bi].tempValue, &q);
			
				PetscReal w11, w12, w13, w21, w22, w23, w31, w32, w33;
				PetscReal so, wo;
				PetscReal csi1, csi2, csi3, eta1, eta2, eta3, zet1, zet2, zet3;	
				
				for (k=zs; k<ze-2; k++)
				for (j=ys; j<ye-2; j++)
				for (i=xs; i<xe-2; i++) {
					uavg[k+1][j+1][i+1].x = usum[k+1][j+1][i+1].x/N;
					uavg[k+1][j+1][i+1].y = usum[k+1][j+1][i+1].y/N;
					uavg[k+1][j+1][i+1].z = usum[k+1][j+1][i+1].z/N;
				}
				
				for (k=zs; k<ze-2; k++) {
					for (j=ys; j<ye-2; j++) {						
						for (i=xs; i<xe-2; i++) {
							
							double dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz;
							double du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz;
							double csi0 = csi[k+1][j+1][i+1].x, csi1 = csi[k+1][j+1][i+1].y, csi2 = csi[k+1][j+1][i+1].z;
							double eta0= eta[k+1][j+1][i+1].x, eta1 = eta[k+1][j+1][i+1].y, eta2 = eta[k+1][j+1][i+1].z;
							double zet0 = zet[k+1][j+1][i+1].x, zet1 = zet[k+1][j+1][i+1].y, zet2 = zet[k+1][j+1][i+1].z;
							double ajc = aj[k+1][j+1][i+1];
							Compute_du_center (i+1, j+1, k+1, mx, my, mz, uavg, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde, &dudz, &dvdz, &dwdz);
							Compute_du_dxyz (	csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz,
													&du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz );
							double Sxx = 0.5*( du_dx + du_dx ), Sxy = 0.5*(du_dy + dv_dx), Sxz = 0.5*(du_dz + dw_dx);
							double Syx = Sxy, Syy = 0.5*(dv_dy + dv_dy),	Syz = 0.5*(dv_dz + dw_dy);
							double Szx = Sxz, Szy=Syz, Szz = 0.5*(dw_dz + dw_dz);
							so = Sxx*Sxx + Sxy*Sxy + Sxz*Sxz + Syx*Syx + Syy*Syy + Syz*Syz + Szx*Szx + Szy*Szy + Szz*Szz;
							w11 = 0;
							w12 = 0.5*(du_dy - dv_dx);
							w13 = 0.5*(du_dz - dw_dx);
							w21 = -w12;
							w22 = 0.;
							w23 = 0.5*(dv_dz - dw_dy);
							w31 = -w13;
							w32 = -w23;
							w33 = 0.;
							wo = w11*w11 + w12*w12 + w13*w13 + w21*w21 + w22*w22 + w23*w23 + w31*w31 + w32*w32 + w33*w33;
							if ( nvert[k+1][j+1][i+1]>0.1 ) q[k+1][j+1][i+1] = 0;
							else q[k+1][j+1][i+1] = (wo - so) / 2.;
						}
					}
				}
				
				
				if(ASCII>=4){
                    FILE *f2;
                    char filen2[80];
                    sprintf(filen2, "QcriAvg.dat");
                    if ((f2 = fopen(filen2, "r"))){
                        fclose(f2);
                    }
                    else{
						f2 = fopen(filen2, "w");
						PetscFPrintf(PETSC_COMM_WORLD, f2, "%d\n%d\n%d\n",xe-2-xs,ye-2-ys,ze-2-zs);
						for (k=zs; k<ze-2; k++){
						for (j=ys; j<ye-2; j++){
						for (i=xs; i<xe-2; i++){
							if ( nvert[k][j][i]>0.1 ){
								PetscFPrintf(PETSC_COMM_WORLD, f2, "0.0\n");
							}
							else {							
								PetscFPrintf(PETSC_COMM_WORLD, f2, "%e\n", q[k+1][j+1][i+1]);
							}
						}}}
						fclose(f2);
                    }
                }
				
				
				
				for (k=zs; k<ze-2; k++)
				for (j=ys; j<ye-2; j++)
				for (i=xs; i<xe-2; i++) x[k * (mx-2)*(my-2) + j*(mx-2) + i] = q[k+1][j+1][i+1];
				I = TECDAT100(&III, x, &DIsDouble);
								
				DAVecRestoreArray(user[bi].da, user[bi].tempValue, &q);
				DAVecRestoreArray(user[bi].fda, user[bi].Csi, &csi);
				DAVecRestoreArray(user[bi].fda, user[bi].Eta, &eta);
				DAVecRestoreArray(user[bi].fda, user[bi].Zet, &zet);
				DAVecRestoreArray(user[bi].da, user[bi].Aj, &aj);
				DAVecRestoreArray(user[bi].da, user[bi].Nvert, &nvert);		
				DAVecRestoreArray(user[bi].fda, user[bi].Ucat, &uavg);
				
				DAVecRestoreArray(user[bi].fda, Vort_sum, &vortsum);
				DAVecRestoreArray(user[bi].fda, Vort_square_sum, &vort2sum);

				VecDestroy(Vort_sum);
				VecDestroy(Vort_square_sum);

				//haha
				/*
				//TKE budget
			 	Vec Udp_sum, dU2_sum, UUU_sum;
				PetscReal ***udpsum, ***aj;
				Cmpnts ***du2sum, ***uuusum;
				Cmpnts ***csi, ***eta, ***zet;

				DACreateGlobalVector(user[bi].da, &Udp_sum);
				DACreateGlobalVector(user[bi].fda, &dU2_sum);
				DACreateGlobalVector(user[bi].fda, &UUU_sum);
				DACreateGlobalVector(user[bi].fda, &user[bi].Ucat_sum);
				DACreateGlobalVector(user[bi].fda, &user[bi].Ucat_cross_sum);
				DACreateGlobalVector(user[bi].fda, &user[bi].Ucat_square_sum);
				DACreateGlobalVector(user[bi].fda, &user[bi].Csi);
				DACreateGlobalVector(user[bi].fda, &user[bi].Eta);
				DACreateGlobalVector(user[bi].fda, &user[bi].Zet);
				DACreateGlobalVector(user[bi].da, &user[bi].Aj);

				sprintf(filen, "su0_%06d_%1d.dat", ti, user[bi]._this);
				PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
				VecLoadIntoVector(viewer, (user[bi].Ucat_sum));
				PetscViewerDestroy(viewer);

				sprintf(filen, "su1_%06d_%1d.dat", ti, user[bi]._this);
				PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
				VecLoadIntoVector(viewer, (user[bi].Ucat_cross_sum));
				PetscViewerDestroy(viewer);

				sprintf(filen, "su2_%06d_%1d.dat", ti, user[bi]._this);
				PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
				VecLoadIntoVector(viewer, (user[bi].Ucat_square_sum));
				PetscViewerDestroy(viewer);

				sprintf(filen, "su3_%06d_%1d.dat", ti, user[bi]._this);
				PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
				VecLoadIntoVector(viewer, Udp_sum);
				PetscViewerDestroy(viewer);

				sprintf(filen, "su4_%06d_%1d.dat", ti, user[bi]._this);
				PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
				VecLoadIntoVector(viewer, dU2_sum);
				PetscViewerDestroy(viewer);

				sprintf(filen, "su5_%06d_%1d.dat", ti, user[bi]._this);
				PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
				VecLoadIntoVector(viewer, UUU_sum);
				PetscViewerDestroy(viewer);

				FormMetrics(&(user[bi]));

				DAVecGetArray(user[bi].da, user[bi].Aj, &aj);
				DAVecGetArray(user[bi].fda, user[bi].Csi, &csi);
				DAVecGetArray(user[bi].fda, user[bi].Eta, &eta);
				DAVecGetArray(user[bi].fda, user[bi].Zet, &zet);
				DAVecGetArray(user[bi].fda, user[bi].Ucat_sum, &usum);
				DAVecGetArray(user[bi].fda, user[bi].Ucat_cross_sum, &u1sum);
				DAVecGetArray(user[bi].fda, user[bi].Ucat_square_sum, &u2sum);
				DAVecGetArray(user[bi].da, Udp_sum, &udpsum);
				DAVecGetArray(user[bi].fda, dU2_sum, &du2sum);
				DAVecGetArray(user[bi].fda, UUU_sum, &uuusum);

				DAVecRestoreArray(user[bi].da, user[bi].Aj, &aj);
				DAVecRestoreArray(user[bi].fda, user[bi].Csi, &csi);
				DAVecRestoreArray(user[bi].fda, user[bi].Eta, &eta);
				DAVecRestoreArray(user[bi].fda, user[bi].Zet, &zet);
				DAVecRestoreArray(user[bi].fda, user[bi].Ucat_sum, &usum);
				DAVecRestoreArray(user[bi].fda, user[bi].Ucat_cross_sum, &u1sum);
				DAVecRestoreArray(user[bi].fda, user[bi].Ucat_square_sum, &u2sum);
				DAVecRestoreArray(user[bi].da, Udp_sum, &udpsum);
				DAVecRestoreArray(user[bi].fda, dU2_sum, &du2sum);
				DAVecRestoreArray(user[bi].fda, UUU_sum, &uuusum);

				VecDestroy(user[bi].Csi);
				VecDestroy(user[bi].Eta);
				VecDestroy(user[bi].Zet);
				VecDestroy(user[bi].Aj);
				VecDestroy(user[bi].Ucat_sum);
				VecDestroy(user[bi].Ucat_cross_sum);
				VecDestroy(user[bi].Ucat_square_sum);
				VecDestroy(Udp_sum);
				VecDestroy(dU2_sum);
				VecDestroy(UUU_sum);
				*/
			}
		}
		else if (avg==2) {
			PetscViewer viewer;
			Vec K_sum;
			PetscReal ***ksum;
			char filen[128];

			DACreateGlobalVector(user->fda, &user->Ucat_sum);
						            DACreateGlobalVector(user->fda, &user->Ucat_square_sum);
			if (rans) {
				DACreateGlobalVector(user->da, &K_sum);
			}

			sprintf(filen, "su0_%06d_%1d.dat", ti, user->_this);
			PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
			VecLoadIntoVector(viewer, (user->Ucat_sum));
			PetscViewerDestroy(viewer);

			if (rans) {
				sprintf(filen, "sk_%06d_%1d.dat", ti, user->_this);
				PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
				VecLoadIntoVector(viewer, K_sum);
				PetscViewerDestroy(viewer);
			}
			else {
				sprintf(filen, "su2_%06d_%1d.dat", ti, user->_this);
				PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
				VecLoadIntoVector(viewer, (user->Ucat_square_sum));
				PetscViewerDestroy(viewer);
			}

			DAVecGetArray(user[bi].fda, user[bi].Ucat_sum, &usum);

			if (rans) DAVecGetArray(user[bi].da, K_sum, &ksum);
			else DAVecGetArray(user[bi].fda, user[bi].Ucat_square_sum, &u2sum);

			// U
			for (k=zs; k<ze-2; k++)
			for (j=ys; j<ye-2; j++)
			for (i=xs; i<xe-2; i++) x[k * (mx-2)*(my-2) + j*(mx-2) + i] = usum[k+1][j+1][i+1].x/N;
			if (i_average) Iavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (j_average) Javg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (k_average) Kavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (ik_average) IKavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (ikc_average) IKavg_c(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			I = TECDAT100(&III, x, &DIsDouble);

			// V
			for (k=zs; k<ze-2; k++)
			for (j=ys; j<ye-2; j++)
			for (i=xs; i<xe-2; i++) x[k * (mx-2)*(my-2) + j*(mx-2) + i] = usum[k+1][j+1][i+1].y/N;
			if (i_average) Iavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (j_average) Javg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (k_average) Kavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (ik_average) IKavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (ikc_average) IKavg_c(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			I = TECDAT100(&III, x, &DIsDouble);

			// W
			for (k=zs; k<ze-2; k++)
			for (j=ys; j<ye-2; j++)
			for (i=xs; i<xe-2; i++) x[k * (mx-2)*(my-2) + j*(mx-2) + i] = usum[k+1][j+1][i+1].z/N;
			if (i_average) Iavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (j_average) Javg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (k_average) Kavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (ik_average) IKavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (ikc_average) IKavg_c(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			I = TECDAT100(&III, x, &DIsDouble);

			// k
			for (k=zs; k<ze-2; k++)
			for (j=ys; j<ye-2; j++)
			for (i=xs; i<xe-2; i++) {
				if (rans)  {
					x[k * (mx-2)*(my-2) + j*(mx-2) + i] = ksum[k+1][j+1][i+1]/N;
				}
				else {
					double U = usum[k+1][j+1][i+1].x/N;
					double V = usum[k+1][j+1][i+1].y/N;
					double W = usum[k+1][j+1][i+1].z/N;
					x[k * (mx-2)*(my-2) + j*(mx-2) + i] = ( u2sum[k+1][j+1][i+1].x/N - U*U ) + ( u2sum[k+1][j+1][i+1].y/N - V*V ) + ( u2sum[k+1][j+1][i+1].z/N - W*W );
					x[k * (mx-2)*(my-2) + j*(mx-2) + i] *= 0.5;
				}
			}
			if (i_average) Iavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (j_average) Javg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (k_average) Kavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (ik_average) IKavg(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			if (ikc_average) IKavg_c(x, xs, xe, ys, ye, zs, ze, mx, my, mz);
			I = TECDAT100(&III, x, &DIsDouble);

			DAVecRestoreArray(user[bi].fda, user[bi].Ucat_sum, &usum);

			if (rans) DAVecRestoreArray(user[bi].da, K_sum, &ksum);
			else DAVecRestoreArray(user[bi].fda, user[bi].Ucat_square_sum, &u2sum);

			VecDestroy(user->Ucat_sum);
			if (rans) VecDestroy(K_sum);
			else VecDestroy(user->Ucat_square_sum);
		}

		DAVecGetArray(user[bi].da, user[bi].Nvert, &nvert);
		for (k=zs; k<ze-2; k++)
		for (j=ys; j<ye-2; j++)
		for (i=xs; i<xe-2; i++) x[k * (mx-2)*(my-2) + j*(mx-2) + i] = nvert[k+1][j+1][i+1];
		I = TECDAT100(&III, x, &DIsDouble);
		DAVecRestoreArray(user[bi].da, user[bi].Nvert, &nvert);

		delete []x;
	}
	I = TECEND100();
	return 0;
} // end of TECIOOut_Averaging
#else
PetscErrorCode TECIOOut_Averaging(UserCtx *user)
{
	PetscPrintf(PETSC_COMM_WORLD, "Compiled without Tecplot. Function TECIOOut_Averaging not available!\n");
	return -1;
}
#endif

#ifdef TECIO
PetscErrorCode TECIOOutQ(UserCtx *user, int Q)
{
/*
To add a extra variable
1. 		Change this line: I = TECINI100((char*)"Result", (char*)"X Y Z Q WX WY WZ WM Helicity VM",   filen,  (char*)".",   &Debug,  &VIsDouble);
2.		INTEGER4	LOC[10] = {1, 1, 1, 0, 0, 0, 0, 0, 0}; /* 1 is cell-centered 0 is node centered 
*/		
	PetscInt bi;

	char filen[80];
	sprintf(filen, "QCriteria%06d.plt", ti);

	INTEGER4 I, Debug, VIsDouble, DIsDouble, III, IMax, JMax, KMax;
	VIsDouble = 0;
	DIsDouble = 0;
	Debug = 0;

	if (Q==1) {
		printf("qcr=%d, Q-Criterion\n", Q);
		//I = TECINI100((char*)"Result", (char*)"X Y Z Q Velocity_Magnitude Nv",   filen,  (char*)".",   &Debug,  &VIsDouble);
		if (ASCII==10) {
			I = TECINI100((char*)"Result", (char*)"X Y Z Q WX WY WZ WM Helicity u v w VM jacobian div_wCv",   filen,  (char*)".",   &Debug,  &VIsDouble);
		}
		else if(ASCII==11){
			I = TECINI100((char*)"Result", (char*)"X Y Z Q WX WY WZ WM Helicity u v w VM jacobian div_grad_uu_phi",   filen,  (char*)".",   &Debug,  &VIsDouble);
		}
		else if(ASCII>=20){
			I = TECINI100((char*)"Result", (char*)"X Y Z Q WX WY WZ WM Helicity u v w VM jacobian div_wCv div_grad_uu_phi",   filen,  (char*)".",   &Debug,  &VIsDouble);			
		}
		else{
			I = TECINI100((char*)"Result", (char*)"X Y Z Q WX WY WZ WM Helicity u v w VM",   filen,  (char*)".",   &Debug,  &VIsDouble);
		}
		//I = TECINI100((char*)"Result", (char*)"X Y Z Q wCvx wCvy wCvz aj WX WY WZ WM Helicity ",   filen,  (char*)".",   &Debug,  &VIsDouble);		
	}
	else if (Q==2) {
		printf("Lambda2-Criterion\n");
		//I = TECINI100((char*)"Result", (char*)"X Y Z Lambda2 Velocity_Magnitude Nv",   filen,  (char*)".",   &Debug,  &VIsDouble);
		I = TECINI100((char*)"Result", (char*)"X Y Z Lambda2 Velocity_Magnitude",   filen,  (char*)".",   &Debug,  &VIsDouble);
	}
	else if (Q==3) {
		printf("Q-Criterion from saved file\n");
		I = TECINI100((char*)"Result", (char*)"X Y Z Q Velocity_Magnitude",   filen,  (char*)".",   &Debug,  &VIsDouble);
	}

	for (bi=0; bi<block_number; bi++) {
		DA da = user[bi].da, fda = user[bi].fda;
		DALocalInfo info = user[bi].info;

		PetscInt	xs = info.xs, xe = info.xs + info.xm;
		PetscInt  	ys = info.ys, ye = info.ys + info.ym;
		PetscInt	zs = info.zs, ze = info.zs + info.zm;
		PetscInt	mx = info.mx, my = info.my, mz = info.mz;

		PetscInt	lxs, lys, lzs, lxe, lye, lze;
		PetscInt	i, j, k;
		PetscReal	***aj;
		Cmpnts	***ucat, ***coor, ***ucat_o, ***wCv;
		PetscReal	***p, ***nvert;
		Vec		Coor, zCoor, nCoor;
		Vec WCV;
		VecScatter	ctx;

		Vec X, Y, Z, U, V, W;

		INTEGER4	ZoneType=0, ICellMax=0, JCellMax=0, KCellMax=0;
		INTEGER4	IsBlock=1, NumFaceConnections=0, FaceNeighborMode=0;
		INTEGER4    ShareConnectivityFromZone=0;
		INTEGER4	LOC[15] = {1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; /* 1 is cell-centered 0 is node centered */

		IMax = mx-1; JMax = my-1; KMax = mz-1;

		I = TECZNE100((char*)"Block 1",
			&ZoneType, 	/* Ordered zone */
			&IMax,
			&JMax,
			&KMax,
			&ICellMax,
			&JCellMax,
			&KCellMax,
			&IsBlock,	/* ISBLOCK  BLOCK format */
			&NumFaceConnections,
			&FaceNeighborMode,
			LOC,
			NULL,
			&ShareConnectivityFromZone); /* No connectivity sharing */

		III = (mx-1) * (my-1) * (mz-1);
		float	*x;
		PetscMalloc(mx*my*mz*sizeof(float), &x);	// seokkoo

		DAGetCoordinates(da, &Coor);
		DAVecGetArray(fda, Coor, &coor);

		for (k=zs; k<ze-1; k++)
		for (j=ys; j<ye-1; j++)
		for (i=xs; i<xe-1; i++) {
			x[k * (mx-1)*(my-1) + j*(mx-1) + i] = coor[k][j][i].x;
		}
		I = TECDAT100(&III, x, &DIsDouble);

		for (k=zs; k<ze-1; k++)
		for (j=ys; j<ye-1; j++)
		for (i=xs; i<xe-1; i++) {
			x[k * (mx-1)*(my-1) + j*(mx-1) + i] = coor[k][j][i].y;
		}
		I = TECDAT100(&III, x, &DIsDouble);

		for (k=zs; k<ze-1; k++)
		for (j=ys; j<ye-1; j++)
		for (i=xs; i<xe-1; i++) {
			x[k * (mx-1)*(my-1) + j*(mx-1) + i] = coor[k][j][i].z;
		}

		I = TECDAT100(&III, x, &DIsDouble);
		
		
		if(ASCII){	
	 	    FILE *f2;
		    char filen2[80];
		    sprintf(filen2, "Coor.dat");
		    if ((f2 = fopen(filen2, "r"))){
		     	fclose(f2);
		    }
		    else{
				f2 = fopen(filen2, "w");
				PetscFPrintf(PETSC_COMM_WORLD, f2, "%d %d %d\n",xe-1-xs,ye-1-ys,ze-1-zs);
				for (k=zs; k<ze-1; k++){
				for (j=ys; j<ye-1; j++){
				for (i=xs; i<xe-1; i++){
					PetscFPrintf(PETSC_COMM_WORLD, f2, "%e %e %e\n",coor[k][j][i].x, coor[k][j][i].y, coor[k][j][i].z );
				}}}
				fclose(f2);
		    }
		}
		
		DAVecRestoreArray(fda, Coor, &coor);

		III = (mx-2) * (my-2) * (mz-2);

		if (Q==1) {
			QCriteria(user);
			DAVecGetArray(user[bi].da, user[bi].tempValue, &p);
			for (k=zs; k<ze-2; k++)
			for (j=ys; j<ye-2; j++)
			for (i=xs; i<xe-2; i++) {
				x[k * (mx-2)*(my-2) + j*(mx-2) + i] =p[k+1][j+1][i+1];
			}
			I = TECDAT100(&III, x, &DIsDouble);
			DAVecRestoreArray(user[bi].da, user[bi].tempValue, &p);
			
			
			//Vort(user,1); // writing vorticity in x-dir
			DAVecGetArray(user[bi].da, user[bi].WX, &p);
			for (k=zs; k<ze-2; k++)
			for (j=ys; j<ye-2; j++)
			for (i=xs; i<xe-2; i++) {
				x[k * (mx-2)*(my-2) + j*(mx-2) + i] = p[k+1][j+1][i+1];
			}
			I = TECDAT100(&III, x, &DIsDouble);
			DAVecRestoreArray(user[bi].da, user[bi].WX, &p);
			
			
			//Vort(user,2);
			DAVecGetArray(user[bi].da, user[bi].WY, &p);
			for (k=zs; k<ze-2; k++)
			for (j=ys; j<ye-2; j++)
			for (i=xs; i<xe-2; i++) {
				x[k * (mx-2)*(my-2) + j*(mx-2) + i] = p[k+1][j+1][i+1];
			}
			I = TECDAT100(&III, x, &DIsDouble);
			DAVecRestoreArray(user[bi].da, user[bi].WY, &p);
			
		
			//Vort(user,3);
			DAVecGetArray(user[bi].da, user[bi].WZ, &p);
			for (k=zs; k<ze-2; k++)
			for (j=ys; j<ye-2; j++)
			for (i=xs; i<xe-2; i++) {
				x[k * (mx-2)*(my-2) + j*(mx-2) + i] = p[k+1][j+1][i+1];
			}
			I = TECDAT100(&III, x, &DIsDouble);
			DAVecRestoreArray(user[bi].da, user[bi].WZ, &p);


			DAVecGetArray(user[bi].da, user[bi].WM, &p);
			for (k=zs; k<ze-2; k++)
			for (j=ys; j<ye-2; j++)
			for (i=xs; i<xe-2; i++) {
				x[k * (mx-2)*(my-2) + j*(mx-2) + i] = p[k+1][j+1][i+1];
			}
			I = TECDAT100(&III, x, &DIsDouble);
			DAVecRestoreArray(user[bi].da, user[bi].WM, &p);
			
			
			// writing helicity
			DAVecGetArray(user[bi].da, user[bi].HELICITY, &p);
			for (k=zs; k<ze-2; k++)
			for (j=ys; j<ye-2; j++)
			for (i=xs; i<xe-2; i++) {
				x[k * (mx-2)*(my-2) + j*(mx-2) + i] = p[k+1][j+1][i+1];
			}
			I = TECDAT100(&III, x, &DIsDouble);
			DAVecRestoreArray(user[bi].da, user[bi].HELICITY, &p);
		}
		else if (Q==2) {
			Lambda2(user);
			DAVecGetArray(user[bi].da, user[bi].tempValue, &p);
			for (k=zs; k<ze-2; k++)
			for (j=ys; j<ye-2; j++)
			for (i=xs; i<xe-2; i++) {
				x[k * (mx-2)*(my-2) + j*(mx-2) + i] = p[k+1][j+1][i+1];
			}
			I = TECDAT100(&III, x, &DIsDouble);
			DAVecRestoreArray(user[bi].da, user[bi].tempValue, &p);
		}
		else if (Q==3) {
			char filen2[128];
			PetscViewer	viewer;

			sprintf(filen2, "qfield%06d_%1d.dat", ti, user->_this);
			PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen2, FILE_MODE_READ, &viewer);
			VecLoadIntoVector(viewer, (user[bi].tempValue));
			PetscViewerDestroy(viewer);

			DAVecGetArray(user[bi].da, user[bi].tempValue, &p);
			for (k=zs; k<ze-2; k++)
			for (j=ys; j<ye-2; j++)
			for (i=xs; i<xe-2; i++) {
				x[k * (mx-2)*(my-2) + j*(mx-2) + i] = p[k+1][j+1][i+1];
			}
			I = TECDAT100(&III, x, &DIsDouble);
			DAVecRestoreArray(user[bi].da, user[bi].tempValue, &p);
		}

		Velocity_Magnitude(user,1);
		DAVecGetArray(user[bi].da, user[bi].tempValue, &p);
		for (k=zs; k<ze-2; k++)
		for (j=ys; j<ye-2; j++)
		for (i=xs; i<xe-2; i++) {
			x[k * (mx-2)*(my-2) + j*(mx-2) + i] = p[k+1][j+1][i+1];
		}
		I = TECDAT100(&III, x, &DIsDouble);
		DAVecRestoreArray(user[bi].da, user[bi].tempValue, &p);

		Velocity_Magnitude(user,2);
		DAVecGetArray(user[bi].da, user[bi].tempValue, &p);
		for (k=zs; k<ze-2; k++)
		for (j=ys; j<ye-2; j++)
		for (i=xs; i<xe-2; i++) {
			x[k * (mx-2)*(my-2) + j*(mx-2) + i] = p[k+1][j+1][i+1];
		}
		I = TECDAT100(&III, x, &DIsDouble);
		DAVecRestoreArray(user[bi].da, user[bi].tempValue, &p);
		
		Velocity_Magnitude(user,3);
		DAVecGetArray(user[bi].da, user[bi].tempValue, &p);
		for (k=zs; k<ze-2; k++)
		for (j=ys; j<ye-2; j++)
		for (i=xs; i<xe-2; i++) {
			x[k * (mx-2)*(my-2) + j*(mx-2) + i] = p[k+1][j+1][i+1];
		}
		I = TECDAT100(&III, x, &DIsDouble);
		DAVecRestoreArray(user[bi].da, user[bi].tempValue, &p);		
		
		Velocity_Magnitude(user,4);
		DAVecGetArray(user[bi].da, user[bi].tempValue, &p);
		for (k=zs; k<ze-2; k++)
		for (j=ys; j<ye-2; j++)
		for (i=xs; i<xe-2; i++) {
			x[k * (mx-2)*(my-2) + j*(mx-2) + i] = p[k+1][j+1][i+1];
		}
		I = TECDAT100(&III, x, &DIsDouble);
		DAVecRestoreArray(user[bi].da, user[bi].tempValue, &p);		
		
		if (ASCII>=10) {
			writeJacobian(user);
			DAVecGetArray(user[bi].da, user[bi].tempValue, &p);
			for (k=zs; k<ze-2; k++)
			for (j=ys; j<ye-2; j++)
			for (i=xs; i<xe-2; i++) {
					x[k * (mx-2)*(my-2) + j*(mx-2) + i] = p[k+1][j+1][i+1];
			}
			I = TECDAT100(&III, x, &DIsDouble);
			DAVecRestoreArray(user[bi].da, user[bi].tempValue, &p);		
			
			if(ASCII==10 || ASCII>=20){
				ASCIIOutPut(user,ti,4);
				DAVecGetArray(user[bi].da, user[bi].tempValue, &p);
				for (k=zs; k<ze-2; k++)
				for (j=ys; j<ye-2; j++)
				for (i=xs; i<xe-2; i++) {
						x[k * (mx-2)*(my-2) + j*(mx-2) + i] = p[k+1][j+1][i+1];
				}
				I = TECDAT100(&III, x, &DIsDouble);
				DAVecRestoreArray(user[bi].da, user[bi].tempValue, &p);
			}
			if(ASCII==11 || ASCII>=20){
				ASCIIOutPut(user,ti,5);
				DAVecGetArray(user[bi].da, user[bi].tempValue, &p);
				for (k=zs; k<ze-2; k++)
				for (j=ys; j<ye-2; j++)
				for (i=xs; i<xe-2; i++) {
						x[k * (mx-2)*(my-2) + j*(mx-2) + i] = p[k+1][j+1][i+1];
				}
				I = TECDAT100(&III, x, &DIsDouble);
				DAVecRestoreArray(user[bi].da, user[bi].tempValue, &p);
			}
		}

		/*
		DAVecGetArray(user[bi].da, user[bi].Nvert, &nvert);
		for (k=zs; k<ze-2; k++)
		for (j=ys; j<ye-2; j++)
		for (i=xs; i<xe-2; i++) {
			x[k * (mx-2)*(my-2) + j*(mx-2) + i] = nvert[k+1][j+1][i+1];
		}
		I = TECDAT100(&III, x, &DIsDouble);
		DAVecRestoreArray(user[bi].da, user[bi].Nvert, &nvert);
		*/

		PetscFree(x);
	}
	I = TECEND100();

	return 0;
}
#else
PetscErrorCode TECIOOutQ(UserCtx *user, int Q)
{
	PetscPrintf(PETSC_COMM_WORLD, "Compiled without Tecplot. Function TECIOOutQ not available!\n");
	return -1;
}
#endif

PetscErrorCode FormMetrics(UserCtx *user)
{
	DA		cda;
	Cmpnts	***csi, ***eta, ***zet;
	PetscScalar	***aj;
	Vec		coords;
	Cmpnts	***coor;

	DA		da = user->da, fda = user->fda;
	Vec		Csi = user->Csi, Eta = user->Eta, Zet = user->Zet;
	Vec		Aj = user->Aj;
	Vec		ICsi = user->ICsi, IEta = user->IEta, IZet = user->IZet;
	Vec		JCsi = user->JCsi, JEta = user->JEta, JZet = user->JZet;
	Vec		KCsi = user->KCsi, KEta = user->KEta, KZet = user->KZet;
	Vec		IAj = user->IAj, JAj = user->JAj, KAj = user->KAj;


	Cmpnts	***icsi, ***ieta, ***izet;
	Cmpnts	***jcsi, ***jeta, ***jzet;
	Cmpnts	***kcsi, ***keta, ***kzet;
	Cmpnts	***gs;
	PetscReal	***iaj, ***jaj, ***kaj;

	Vec		Cent = user->Cent; //local working array for storing cell center geometry

	Vec		Centx, Centy, Centz, lCoor;
	Cmpnts	***cent, ***centx, ***centy, ***centz;

	PetscInt	xs, ys, zs, xe, ye, ze;
	DALocalInfo	info;

	PetscInt	mx, my, mz;
	PetscInt	lxs, lxe, lys, lye, lzs, lze;

	PetscScalar	dxdc, dydc, dzdc, dxde, dyde, dzde, dxdz, dydz, dzdz;
	PetscInt	i, j, k, ia, ja, ka, ib, jb, kb;
	PetscInt	gxs, gxe, gys, gye, gzs, gze;
	PetscErrorCode	ierr;

	PetscReal	xcp, ycp, zcp, xcm, ycm, zcm;
	DAGetLocalInfo(da, &info);
	mx = info.mx; my = info.my; mz = info.mz;
	xs = info.xs; xe = xs + info.xm;
	ys = info.ys; ye = ys + info.ym;
	zs = info.zs; ze = zs + info.zm;

	gxs = info.gxs; gxe = gxs + info.gxm;
	gys = info.gys; gye = gys + info.gym;
	gzs = info.gzs; gze = gzs + info.gzm;

	DAGetCoordinateDA(da, &cda);
	DAVecGetArray(cda, Csi, &csi);
	DAVecGetArray(cda, Eta, &eta);
	DAVecGetArray(cda, Zet, &zet);
	DAVecGetArray(cda, Cent, &cent);	
	ierr = DAVecGetArray(da, Aj,  &aj); CHKERRQ(ierr);

	DAGetGhostedCoordinates(da, &coords);
	DAVecGetArray(fda, coords, &coor);

	//  VecDuplicate(coords, &Cent);
	lxs = xs; lxe = xe;
	lys = ys; lye = ye;
	lzs = zs; lze = ze;

	if (xs==0) lxs = xs+1;
	if (ys==0) lys = ys+1;
	if (zs==0) lzs = zs+1;

	if (xe==mx) lxe = xe-1;
	if (ye==my) lye = ye-1;
	if (ze==mz) lze = ze-1;

	/* Calculating transformation metrics in i direction */
	for (k=lzs; k<lze; k++) {
		for (j=lys; j<lye; j++) {
			for (i=xs; i<lxe; i++) {
				/* csi = de X dz */
				dxde = 0.5 * (coor[k  ][j  ][i  ].x + coor[k-1][j  ][i  ].x -
											coor[k  ][j-1][i  ].x - coor[k-1][j-1][i  ].x);
				dyde = 0.5 * (coor[k  ][j  ][i  ].y + coor[k-1][j  ][i  ].y -
											coor[k  ][j-1][i  ].y - coor[k-1][j-1][i  ].y);
				dzde = 0.5 * (coor[k  ][j  ][i  ].z + coor[k-1][j  ][i  ].z -
											coor[k  ][j-1][i  ].z - coor[k-1][j-1][i  ].z);

				dxdz = 0.5 * (coor[k  ][j-1][i  ].x + coor[k  ][j  ][i  ].x -
											coor[k-1][j-1][i  ].x - coor[k-1][j  ][i  ].x);
				dydz = 0.5 * (coor[k  ][j-1][i  ].y + coor[k  ][j  ][i  ].y -
											coor[k-1][j-1][i  ].y - coor[k-1][j  ][i  ].y);
				dzdz = 0.5 * (coor[k  ][j-1][i  ].z + coor[k  ][j  ][i  ].z -
											coor[k-1][j-1][i  ].z - coor[k-1][j  ][i  ].z);

				csi[k][j][i].x = dyde * dzdz - dzde * dydz;
				csi[k][j][i].y =-dxde * dzdz + dzde * dxdz;
				csi[k][j][i].z = dxde * dydz - dyde * dxdz;

			}
		}
	}

	// Need more work -- lg65
	/* calculating j direction metrics */
	for (k=lzs; k<lze; k++){
		for (j=ys; j<lye; j++){
			for (i=lxs; i<lxe; i++){

				/* eta = dz X de */
				dxdc = 0.5 * (coor[k  ][j  ][i  ].x + coor[k-1][j  ][i  ].x -
											coor[k  ][j  ][i-1].x - coor[k-1][j  ][i-1].x);
				dydc = 0.5 * (coor[k  ][j  ][i  ].y + coor[k-1][j  ][i  ].y -
											coor[k  ][j  ][i-1].y - coor[k-1][j  ][i-1].y);
				dzdc = 0.5 * (coor[k  ][j  ][i  ].z + coor[k-1][j  ][i  ].z -
											coor[k  ][j  ][i-1].z - coor[k-1][j  ][i-1].z);

				dxdz = 0.5 * (coor[k  ][j  ][i  ].x + coor[k  ][j  ][i-1].x -
											coor[k-1][j  ][i  ].x - coor[k-1][j  ][i-1].x);
				dydz = 0.5 * (coor[k  ][j  ][i  ].y + coor[k  ][j  ][i-1].y -
											coor[k-1][j  ][i  ].y - coor[k-1][j  ][i-1].y);
				dzdz = 0.5 * (coor[k  ][j  ][i  ].z + coor[k  ][j  ][i-1].z -
											coor[k-1][j  ][i  ].z - coor[k-1][j  ][i-1].z);

				eta[k][j][i].x = dydz * dzdc - dzdz * dydc;
				eta[k][j][i].y =-dxdz * dzdc + dzdz * dxdc;
				eta[k][j][i].z = dxdz * dydc - dydz * dxdc;

			}
		}
	}

	/* calculating k direction metrics */
	for (k=zs; k<lze; k++){
		for (j=lys; j<lye; j++){
			for (i=lxs; i<lxe; i++){
				dxdc = 0.5 * (coor[k  ][j  ][i  ].x + coor[k  ][j-1][i  ].x -
											coor[k  ][j  ][i-1].x - coor[k  ][j-1][i-1].x);
				dydc = 0.5 * (coor[k  ][j  ][i  ].y + coor[k  ][j-1][i  ].y -
											coor[k  ][j  ][i-1].y - coor[k  ][j-1][i-1].y);
				dzdc = 0.5 * (coor[k  ][j  ][i  ].z + coor[k  ][j-1][i  ].z -
											coor[k  ][j  ][i-1].z - coor[k  ][j-1][i-1].z);

				dxde = 0.5 * (coor[k  ][j  ][i  ].x + coor[k  ][j  ][i-1].x -
											coor[k  ][j-1][i  ].x - coor[k  ][j-1][i-1].x);
				dyde = 0.5 * (coor[k  ][j  ][i  ].y + coor[k  ][j  ][i-1].y -
											coor[k  ][j-1][i  ].y - coor[k  ][j-1][i-1].y);
				dzde = 0.5 * (coor[k  ][j  ][i  ].z + coor[k  ][j  ][i-1].z -
											coor[k  ][j-1][i  ].z - coor[k  ][j-1][i-1].z);

				zet[k][j][i].x = dydc * dzde - dzdc * dyde;
				zet[k][j][i].y =-dxdc * dzde + dzdc * dxde;
				zet[k][j][i].z = dxdc * dyde - dydc * dxde;

			}
		}
	}

		for (k=lzs; k<lze; k++)
        for (j=lys; j<lye; j++)
        for (i=lxs; i<lxe; i++) {
                cent[k][j][i].x = 0.125 *
                        (coor[k  ][j  ][i  ].x + coor[k  ][j-1][i  ].x +
                        coor[k-1][j  ][i  ].x + coor[k-1][j-1][i  ].x +
                        coor[k  ][j  ][i-1].x + coor[k  ][j-1][i-1].x +
                        coor[k-1][j  ][i-1].x + coor[k-1][j-1][i-1].x);
                cent[k][j][i].y = 0.125 *
                        (coor[k  ][j  ][i  ].y + coor[k  ][j-1][i  ].y +
                        coor[k-1][j  ][i  ].y + coor[k-1][j-1][i  ].y +
                        coor[k  ][j  ][i-1].y + coor[k  ][j-1][i-1].y +
                        coor[k-1][j  ][i-1].y + coor[k-1][j-1][i-1].y);
                cent[k][j][i].z = 0.125 *
                        (coor[k  ][j  ][i  ].z + coor[k  ][j-1][i  ].z +
                        coor[k-1][j  ][i  ].z + coor[k-1][j-1][i  ].z +
                        coor[k  ][j  ][i-1].z + coor[k  ][j-1][i-1].z +
                        coor[k-1][j  ][i-1].z + coor[k-1][j-1][i-1].z);
        }


	/* calculating Jacobian of the transformation */
	for (k=lzs; k<lze; k++){
		for (j=lys; j<lye; j++){
			for (i=lxs; i<lxe; i++){

				if (i>0 && j>0 && k>0) {
					dxdc = 0.25 * (coor[k  ][j  ][i  ].x + coor[k  ][j-1][i  ].x +
												 coor[k-1][j  ][i  ].x + coor[k-1][j-1][i  ].x -
												 coor[k  ][j  ][i-1].x - coor[k  ][j-1][i-1].x -
												 coor[k-1][j  ][i-1].x - coor[k-1][j-1][i-1].x);
					dydc = 0.25 * (coor[k  ][j  ][i  ].y + coor[k  ][j-1][i  ].y +
												 coor[k-1][j  ][i  ].y + coor[k-1][j-1][i  ].y -
												 coor[k  ][j  ][i-1].y - coor[k  ][j-1][i-1].y -
												 coor[k-1][j  ][i-1].y - coor[k-1][j-1][i-1].y);
					dzdc = 0.25 * (coor[k  ][j  ][i  ].z + coor[k  ][j-1][i  ].z +
												 coor[k-1][j  ][i  ].z + coor[k-1][j-1][i  ].z -
												 coor[k  ][j  ][i-1].z - coor[k  ][j-1][i-1].z -
												 coor[k-1][j  ][i-1].z - coor[k-1][j-1][i-1].z);

					dxde = 0.25 * (coor[k  ][j  ][i  ].x + coor[k  ][j  ][i-1].x +
												 coor[k-1][j  ][i  ].x + coor[k-1][j  ][i-1].x -
												 coor[k  ][j-1][i  ].x - coor[k  ][j-1][i-1].x -
												 coor[k-1][j-1][i  ].x - coor[k-1][j-1][i-1].x);
					dyde = 0.25 * (coor[k  ][j  ][i  ].y + coor[k  ][j  ][i-1].y +
												 coor[k-1][j  ][i  ].y + coor[k-1][j  ][i-1].y -
												 coor[k  ][j-1][i  ].y - coor[k  ][j-1][i-1].y -
												 coor[k-1][j-1][i  ].y - coor[k-1][j-1][i-1].y);
					dzde = 0.25 * (coor[k  ][j  ][i  ].z + coor[k  ][j  ][i-1].z +
												 coor[k-1][j  ][i  ].z + coor[k-1][j  ][i-1].z -
												 coor[k  ][j-1][i  ].z - coor[k  ][j-1][i-1].z -
												 coor[k-1][j-1][i  ].z - coor[k-1][j-1][i-1].z);

					dxdz = 0.25 * (coor[k  ][j  ][i  ].x + coor[k  ][j-1][i  ].x +
												 coor[k  ][j  ][i-1].x + coor[k  ][j-1][i-1].x -
												 coor[k-1][j  ][i  ].x - coor[k-1][j-1][i  ].x -
												 coor[k-1][j  ][i-1].x - coor[k-1][j-1][i-1].x);
					dydz = 0.25 * (coor[k  ][j  ][i  ].y + coor[k  ][j-1][i  ].y +
												 coor[k  ][j  ][i-1].y + coor[k  ][j-1][i-1].y -
												 coor[k-1][j  ][i  ].y - coor[k-1][j-1][i  ].y -
												 coor[k-1][j  ][i-1].y - coor[k-1][j-1][i-1].y);
					dzdz = 0.25 * (coor[k  ][j  ][i  ].z + coor[k  ][j-1][i  ].z +
												 coor[k  ][j  ][i-1].z + coor[k  ][j-1][i-1].z -
												 coor[k-1][j  ][i  ].z - coor[k-1][j-1][i  ].z -
												 coor[k-1][j  ][i-1].z - coor[k-1][j-1][i-1].z);

					aj[k][j][i] = dxdc * (dyde * dzdz - dzde * dydz) -
												dydc * (dxde * dzdz - dzde * dxdz) +
												dzdc * (dxde * dydz - dyde * dxdz);
					aj[k][j][i] = 1./aj[k][j][i];

					#ifdef NEWMETRIC
					csi[k][j][i].x = dyde * dzdz - dzde * dydz;
					csi[k][j][i].y =-dxde * dzdz + dzde * dxdz;
					csi[k][j][i].z = dxde * dydz - dyde * dxdz;

					eta[k][j][i].x = dydz * dzdc - dzdz * dydc;
					eta[k][j][i].y =-dxdz * dzdc + dzdz * dxdc;
					eta[k][j][i].z = dxdz * dydc - dydz * dxdc;

					zet[k][j][i].x = dydc * dzde - dzdc * dyde;
					zet[k][j][i].y =-dxdc * dzde + dzdc * dxde;
					zet[k][j][i].z = dxdc * dyde - dydc * dxde;
					#endif
				}
			}
		}
	}

	// mirror grid outside the boundary
	if (xs==0) {
		i = xs;
		for (k=zs; k<ze; k++)
		for (j=ys; j<ye; j++) {
			#ifdef NEWMETRIC
			csi[k][j][i] = csi[k][j][i+1];
			#endif
			eta[k][j][i] = eta[k][j][i+1];
			zet[k][j][i] = zet[k][j][i+1];
			aj[k][j][i] = aj[k][j][i+1];
			cent[k][j][i] = cent[k][j][i+1];
		}
	}

	if (xe==mx) {
		i = xe-1;
		for (k=zs; k<ze; k++)
		for (j=ys; j<ye; j++) {
			#ifdef NEWMETRIC
			csi[k][j][i] = csi[k][j][i-1];
			#endif
			eta[k][j][i] = eta[k][j][i-1];
			zet[k][j][i] = zet[k][j][i-1];
			aj[k][j][i] = aj[k][j][i-1];
			cent[k][j][i] = cent[k][j][i-1];			
		}
	}


	if (ys==0) {
		j = ys;
		for (k=zs; k<ze; k++)
		for (i=xs; i<xe; i++) {
			#ifdef NEWMETRIC
			eta[k][j][i] = eta[k][j+1][i];
			#endif
			csi[k][j][i] = csi[k][j+1][i];
			zet[k][j][i] = zet[k][j+1][i];
			aj[k][j][i] = aj[k][j+1][i];
			cent[k][j][i] = cent[k][j+1][i];
		}
	}


	if (ye==my) {
		j = ye-1;
		for (k=zs; k<ze; k++)
		for (i=xs; i<xe; i++) {
			#ifdef NEWMETRIC
			eta[k][j][i] = eta[k][j-1][i];
			#endif
			csi[k][j][i] = csi[k][j-1][i];
			zet[k][j][i] = zet[k][j-1][i];
			aj[k][j][i] = aj[k][j-1][i];
			cent[k][j][i] = cent[k][j-1][i];
		}
	}


	if (zs==0) {
		k = zs;
		for (j=ys; j<ye; j++)
		for (i=xs; i<xe; i++) {
			#ifdef NEWMETRIC
			zet[k][j][i] = zet[k+1][j][i];
			#endif
			eta[k][j][i] = eta[k+1][j][i];
			csi[k][j][i] = csi[k+1][j][i];
			aj[k][j][i] = aj[k+1][j][i];
			cent[k][j][i] = cent[k+1][j][i];
		}
	}


	if (ze==mz) {
		k = ze-1;
		for (j=ys; j<ye; j++)
		for (i=xs; i<xe; i++) {
			#ifdef NEWMETRIC
			zet[k][j][i] = zet[k-1][j][i];
			#endif
			eta[k][j][i] = eta[k-1][j][i];
			csi[k][j][i] = csi[k-1][j][i];
			aj[k][j][i] = aj[k-1][j][i];
			cent[k][j][i] = cent[k-1][j][i];
		}
	}

	DAVecRestoreArray(cda, Csi, &csi);
	DAVecRestoreArray(cda, Eta, &eta);
	DAVecRestoreArray(cda, Zet, &zet);
	DAVecRestoreArray(cda, Cent, &cent);	
	DAVecRestoreArray(da, Aj,  &aj);

	DAVecRestoreArray(cda, coords, &coor);

	VecAssemblyBegin(Csi);
	VecAssemblyEnd(Csi);
	VecAssemblyBegin(Eta);
	VecAssemblyEnd(Eta);
	VecAssemblyBegin(Zet);
	VecAssemblyEnd(Zet);
	VecAssemblyBegin(Cent);
	VecAssemblyEnd(Cent);	
	VecAssemblyBegin(Aj);
	VecAssemblyEnd(Aj);

	PetscBarrier(PETSC_NULL);
	return 0;
}

PetscErrorCode Ucont_P_Binary_Input(UserCtx *user)
{
	PetscViewer	viewer;

	char filen2[128];

	PetscOptionsClearValue("-vecload_block_size");
	sprintf(filen2, "pfield%06d_%1d.dat", ti, user->_this);

	PetscViewer	pviewer;
	//Vec temp;
	PetscInt rank;
	PetscReal norm;

	if (file_exist(filen2))
	if (!onlyV) {
		//DACreateNaturalVector(user->da, &temp);
	PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen2, FILE_MODE_READ, &pviewer);
	VecLoadIntoVector(pviewer, (user->P));
	VecNorm(user->P, NORM_INFINITY, &norm);
	PetscPrintf(PETSC_COMM_WORLD, "PIn %le\n", norm);
	PetscViewerDestroy(pviewer);
	//VecDestroy(temp);
	}

	if (nv_once) sprintf(filen2, "nvfield%06d_%1d.dat", 0, user->_this);
	else sprintf(filen2, "nvfield%06d_%1d.dat", ti, user->_this);

	if (cs) sprintf(filen2, "cs_%06d_%1d.dat", ti, user->_this);

	if ( !nv_once || (nv_once && ti==tis) ) {
		PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen2, FILE_MODE_READ, &pviewer);
		VecLoadIntoVector(pviewer, (user->Nvert));
		PetscViewerDestroy(pviewer);
	}

}

PetscErrorCode Ucont_P_Binary_Input1(UserCtx *user)
{
	PetscViewer viewer;
	char filen[128];

	sprintf(filen, "ufield%06d_%1d.dat", ti, user->_this);
	PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);

	PetscInt N;

	VecGetSize(user->Ucat, &N);
	PetscPrintf(PETSC_COMM_WORLD, "PPP %d\n", N);
	VecLoadIntoVector(viewer, (user->Ucat));
	PetscViewerDestroy(viewer);

	PetscBarrier(PETSC_NULL);
}

PetscErrorCode Ucont_P_Binary_Input_Averaging(UserCtx *user)
{
	PetscViewer viewer;
	char filen[128];
	/*
	sprintf(filen, "su0_%06d_%1d.dat", ti, user->_this);
	PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
	VecLoadIntoVector(viewer, (user->Ucat_sum));
	PetscViewerDestroy(viewer);

	sprintf(filen, "su1_%06d_%1d.dat", ti, user->_this);
	PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
	VecLoadIntoVector(viewer, (user->Ucat_cross_sum));
	PetscViewerDestroy(viewer);

	sprintf(filen, "su2_%06d_%1d.dat", ti, user->_this);
	PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
	VecLoadIntoVector(viewer, (user->Ucat_square_sum));
	PetscViewerDestroy(viewer);
	*/
	/*
	sprintf(filen, "sp_%06d_%1d.dat", ti, user->_this);
	PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
	VecLoadIntoVector(viewer, (user->P));
	PetscViewerDestroy(viewer);
	*/
	if (pcr) {
		Vec Ptmp;
		VecDuplicate(user->P, &Ptmp);

		sprintf(filen, "pfield%06d_%1d.dat", ti, user->_this);
		PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
		VecLoadIntoVector(viewer, user->P);
		PetscViewerDestroy(viewer);

		sprintf(filen, "sp_%06d_%1d.dat", ti, user->_this);
		PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
		VecLoadIntoVector(viewer, Ptmp);
		PetscViewerDestroy(viewer);


		VecScale(Ptmp, -1./((double)tis+1.0));
		VecAXPY(user->P, 1., Ptmp);

		VecDestroy(Ptmp);
	}

	if (nv_once) sprintf(filen, "nvfield%06d_%1d.dat", 0, user->_this);
	else sprintf(filen, "nvfield%06d_%1d.dat", ti, user->_this);

	//if (cs) sprintf(filen2, "cs_%06d_%1d.dat", ti, user->_this);

	if ( !nv_once || (nv_once && ti==tis) ) {
		PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
		VecLoadIntoVector(viewer, (user->Nvert));
		PetscViewerDestroy(viewer);
	}
	/*
	if ( !nv_once || (nv_once && ti==tis) ) {
		sprintf(filen, "nvfield%06d_%1d.dat", ti, user->_this);
		PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
		VecLoadIntoVector(viewer, (user->Nvert));
		PetscViewerDestroy(viewer);
	}
	*/
	PetscBarrier(PETSC_NULL);
}

#undef __FUNCT__
#define __FUNCT__ "main"
//=========================================================================================================
int main(int argc, char **argv)
{
	PetscTruth flag;

	DA	da, fda;
	Vec	qn, qnm;
	Vec	c;
	UserCtx	*user;

	PetscErrorCode ierr;

	IBMNodes	*ibm;
	IBMInfo *ibminfo;
	SurfElmtInfo *elmtinfo;
	FSInfo        *fsi;

	PetscInitialize(&argc, &argv, (char *)0, help);
	PetscOptionsInsertFile(PETSC_COMM_WORLD, "control.dat", PETSC_TRUE);

	char tmp_str[256];
	PetscOptionsGetString(PETSC_NULL, "-prefix", tmp_str, 256, &flag);
	if (flag)sprintf(prefix, "%s_", tmp_str);
	else sprintf(prefix, "");

	PetscOptionsGetInt(PETSC_NULL, "-vc", &vc, PETSC_NULL);
	PetscOptionsGetInt(PETSC_NULL, "-binary", &binary_input, &flag);
	PetscOptionsGetInt(PETSC_NULL, "-xyz", &xyz_input, &flag);
	PetscOptionsGetInt(PETSC_NULL, "-rans", &rans, PETSC_NULL);
	PetscOptionsGetInt(PETSC_NULL, "-ransout", &rans_output, PETSC_NULL);
	PetscOptionsGetInt(PETSC_NULL, "-levelset", &levelset, PETSC_NULL);
	PetscOptionsGetInt(PETSC_NULL, "-avg", &avg, &flag);
	PetscOptionsGetInt(PETSC_NULL, "-shear", &shear, &flag);
	PetscOptionsGetInt(PETSC_NULL, "-averaging", &averaging_option, &flag);	// from control.dat

	PetscOptionsGetInt(PETSC_NULL, "-cs", &cs, &flag);
	PetscOptionsGetInt(PETSC_NULL, "-i_periodic", &i_periodic, &flag);
	PetscOptionsGetInt(PETSC_NULL, "-j_periodic", &j_periodic, &flag);
	PetscOptionsGetInt(PETSC_NULL, "-k_periodic", &k_periodic, &flag);

	PetscOptionsGetInt(PETSC_NULL, "-ii_periodic", &i_periodic, &flag);
	PetscOptionsGetInt(PETSC_NULL, "-jj_periodic", &j_periodic, &flag);
	PetscOptionsGetInt(PETSC_NULL, "-kk_periodic", &k_periodic, &flag);

	PetscOptionsGetInt(PETSC_NULL, "-nv", &nv_once, &flag);
	PetscOptionsGetInt(PETSC_NULL, "-vtk", &vtkOutput, &flag);
	PetscOptionsGetInt(PETSC_NULL, "-ASCII", &ASCII, &flag);
	PetscOptionsGetInt(PETSC_NULL, "-surf", &surf, &flag);
	PetscOptionsGetReal(PETSC_NULL, "-delta", &delta, &flag);	
	printf("nv_once=%d\n", nv_once);

	int qcr = 0;
	PetscOptionsGetInt(PETSC_NULL, "-qcr", &qcr, PETSC_NULL);

	PetscOptionsGetInt(PETSC_NULL, "-tis", &tis, &flag);
	if (!flag) PetscPrintf(PETSC_COMM_WORLD, "Need the starting number!\n");

	PetscOptionsGetInt(PETSC_NULL, "-tie", &tie, &flag);
	if (!flag) tie = tis;

	PetscOptionsGetInt(PETSC_NULL, "-ts", &tsteps, &flag);
	if (!flag) tsteps = 5; /* Default increasement is 5 */

	PetscOptionsGetInt(PETSC_NULL, "-onlyV", &onlyV, &flag);
	PetscOptionsGetInt(PETSC_NULL, "-iavg", &i_average, &flag);
	PetscOptionsGetInt(PETSC_NULL, "-javg", &j_average, &flag);
	PetscOptionsGetInt(PETSC_NULL, "-kavg", &k_average, &flag);
	PetscOptionsGetInt(PETSC_NULL, "-ikavg", &ik_average, &flag);
	PetscOptionsGetInt(PETSC_NULL, "-pcr", &pcr, &flag);
	PetscOptionsGetInt(PETSC_NULL, "-reynolds", &reynolds, &flag);

	PetscOptionsGetInt(PETSC_NULL, "-ikcavg", &ikc_average, &flag);
	if (flag) {
		PetscTruth flag1, flag2;
		PetscOptionsGetInt(PETSC_NULL, "-pi", &pi, &flag1);
		PetscOptionsGetInt(PETSC_NULL, "-pk", &pk, &flag2);

		if (!flag1 || !flag2) {
			printf("To use -ikcavg you must set -pi and -pk, which are number of points in i- and k- directions.\n");
			exit(0);
		}
	}

	if (pcr) avg=1;
	if (i_average) avg=1;
	if (j_average) avg=1;
	if (k_average) avg=1;
	if (ik_average) avg=1;
	if (ikc_average) avg=1;


	if (i_average + j_average + k_average >1) PetscPrintf(PETSC_COMM_WORLD, "Iavg and Javg cannot be set together !! !\n"), exit(0);

	PetscInt rank, bi;
	int ibi = 0;		
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	if (xyz_input) block_number=1;
	else {
		FILE *fd;
		fd = fopen("grid.dat", "r");
		if (binary_input) fread(&block_number, sizeof(int), 1, fd);
		else fscanf(fd, "%i\n", &block_number);
		MPI_Bcast(&block_number, 1, MPI_INT, 0, PETSC_COMM_WORLD);
		fclose(fd);
	}

	PetscMalloc(block_number*sizeof(UserCtx), &user);
	PetscOptionsGetReal(PETSC_NULL, "-ren", &user->ren, PETSC_NULL);
	
	PetscMalloc(sizeof(IBMNodes), &ibm);
	PetscMalloc(sizeof(FSInfo), &fsi);

	ReadCoordinates(user);
	PetscPrintf(PETSC_COMM_WORLD, "read coord!\n");

	ibm_read_ucd_asr(&ibm[ibi], ibi);
	PetscMalloc(ibm[ibi].n_elmt*sizeof(SurfElmtInfo), &elmtinfo); //ASR
	PetscMalloc(ibm[ibi].n_elmt*sizeof(IBMInfo), &ibminfo); // ASR	

	for (bi=0; bi<block_number; bi++) {
		
		DACreateGlobalVector(user[bi].da, &user[bi].P);		
		DACreateGlobalVector(user[bi].da, &user[bi].Nvert);
		DACreateGlobalVector(user[bi].fda, &user[bi].Csi);
		DACreateGlobalVector(user[bi].fda, &user[bi].Eta);
		DACreateGlobalVector(user[bi].fda, &user[bi].Zet);
		DACreateGlobalVector(user[bi].fda, &user[bi].Cent);					
		DACreateGlobalVector(user[bi].da, &user[bi].Aj);
		DACreateGlobalVector(user[bi].fda, &user[bi].Ucat);		
		DACreateGlobalVector(user[bi].da, &user[bi].tempValue);		
		FormMetrics(&(user[bi]));		
		
		if (shear) {
			Calc_avg_shear_stress(&(user[bi]));
		}
		if (qcr) {
			DACreateGlobalVector(user[bi].da, &user[bi].WX);
			DACreateGlobalVector(user[bi].da, &user[bi].WY);
			DACreateGlobalVector(user[bi].da, &user[bi].WZ);
			DACreateGlobalVector(user[bi].da, &user[bi].WM);
			DACreateGlobalVector(user[bi].da, &user[bi].HELICITY);
			DACreateGlobalVector(user[bi].fda, &user[bi].WCV);			
			if (!vc) DACreateGlobalVector(user[bi].fda, &user[bi].Ucat_o);
		}
		if (avg) {
			if (i_average) PetscPrintf(PETSC_COMM_WORLD, "Averaging in I direction!\n");
			else if (j_average) PetscPrintf(PETSC_COMM_WORLD, "Averaging in J direction!\n");
			else if (k_average) PetscPrintf(PETSC_COMM_WORLD, "Averaging in K direction!\n");
			else if (ik_average) PetscPrintf(PETSC_COMM_WORLD, "Averaging in IK direction!\n");
			else PetscPrintf(PETSC_COMM_WORLD, "Averaging !\n");
			if (avg==1) {
				DACreateGlobalVector(user[bi].fda, &user[bi].Ucat_sum);
				DACreateGlobalVector(user[bi].fda, &user[bi].Ucat_cross_sum);
				DACreateGlobalVector(user[bi].fda, &user[bi].Ucat_square_sum);
			}
			else if (avg==2) {	// just compute k
				DACreateGlobalVector(user[bi].fda, &user[bi].Ucat_sum);
				DACreateGlobalVector(user[bi].fda, &user[bi].Ucat_square_sum);
			}
		}
	}

	for (ti=tis; ti<=tie; ti+=tsteps) {
		for (bi=0; bi<block_number; bi++) {
			if (avg) Ucont_P_Binary_Input_Averaging(&user[bi]);
			else {
				Ucont_P_Binary_Input(&user[bi]);
				Ucont_P_Binary_Input1(&user[bi]);
			}
		}
		if(surf){
			if(ti==tis){
				Closest_NearBndryPt_ToSurfElmt(user, ibm, elmtinfo, fsi, 0);
				Find_fsi_interp_Coeff(ibminfo, user, ibm, elmtinfo);			
				if(delta!=0){
					Closest_NearBndryPt_ToSurfElmt_delta(user, ibm, elmtinfo, fsi, 0);
					Find_fsi_interp_Coeff_delta(ibminfo, user, ibm, elmtinfo);							
				}
			}
			Calc_fsi_surf_stress2(ibminfo, user, ibm, elmtinfo);
		}
		if (avg) TECIOOut_Averaging(user);
		if (vtkOutput) VtkOutput(user, onlyV);
		if(onlyV) TECIOOut_V(user, onlyV);
		if(qcr) TECIOOutQ(user, qcr);
	}
	
	PetscFinalize();
}

PetscErrorCode ReadCoordinates(UserCtx *user)
{
	Cmpnts ***coor;

	Vec Coor;
	PetscInt bi, i, j, k, rank, IM, JM, KM;
	PetscReal *gc;
	FILE *fd;
	PetscReal	d0 = 1.;

	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	PetscReal	cl = 1.;
	PetscOptionsGetReal(PETSC_NULL, "-cl", &cl, PETSC_NULL);

	char str[256];

	if (xyz_input) sprintf(str, "xyz.dat");
	else sprintf(str, "grid.dat");

	fd = fopen(str, "r");

	if (fd==NULL) printf("Cannot open %s !\n", str),exit(0);

	printf("Begin Reading %s !\n", str);

	if (xyz_input) i=1;
	else if (binary_input) {
		fread(&i, sizeof(int), 1, fd);
		if (i!=1) PetscPrintf(PETSC_COMM_WORLD, "This seems to be a text file !\n"),exit(0);
	}
	else {
		fscanf(fd, "%i\n", &i);
		if (i!=1) PetscPrintf(PETSC_COMM_WORLD, "This seems to be a binary file !\n"),exit(0);
	}


	for (bi=block_number-1; bi>=0; bi--) {

		std::vector<double> X, Y,Z;
		double tmp;

		if (xyz_input) {
			fscanf(fd, "%i %i %i\n", &(user[bi].IM), &(user[bi].JM), &(user[bi].KM));
			X.resize(user[bi].IM);
			Y.resize(user[bi].JM);
			Z.resize(user[bi].KM);

			for (i=0; i<user[bi].IM; i++) fscanf(fd, "%le %le %le\n", &X[i], &tmp, &tmp);
			for (j=0; j<user[bi].JM; j++) fscanf(fd, "%le %le %le\n", &tmp, &Y[j], &tmp);
			for (k=0; k<user[bi].KM; k++) fscanf(fd, "%le %le %le\n", &tmp, &tmp, &Z[k]);
		}
		else if (binary_input) {
			fread(&(user[bi].IM), sizeof(int), 1, fd);
			fread(&(user[bi].JM), sizeof(int), 1, fd);
			fread(&(user[bi].KM), sizeof(int), 1, fd);
		}
		else fscanf(fd, "%i %i %i\n", &(user[bi].IM), &(user[bi].JM), &(user[bi].KM));

		IM = user[bi].IM; JM = user[bi].JM; KM = user[bi].KM;


		DACreate3d(PETSC_COMM_WORLD, DA_NONPERIODIC, DA_STENCIL_BOX,
			user[bi].IM+1, user[bi].JM+1, user[bi].KM+1, 1,1,
			PETSC_DECIDE, 1, 2, PETSC_NULL, PETSC_NULL, PETSC_NULL,
			&(user[bi].da));
		if (rans) {
			DACreate3d(PETSC_COMM_WORLD, DA_NONPERIODIC, DA_STENCIL_BOX,
				user[bi].IM+1, user[bi].JM+1, user[bi].KM+1, 1,1,
				PETSC_DECIDE, 2, 2, PETSC_NULL, PETSC_NULL, PETSC_NULL,
				&(user[bi].fda2));
		}
		DASetUniformCoordinates(user[bi].da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
		DAGetCoordinateDA(user[bi].da, &(user[bi].fda));

		DAGetLocalInfo(user[bi].da, &(user[bi].info));

		DALocalInfo	info = user[bi].info;
		PetscInt	xs = info.xs, xe = info.xs + info.xm;
		PetscInt  	ys = info.ys, ye = info.ys + info.ym;
		PetscInt	zs = info.zs, ze = info.zs + info.zm;
		PetscInt	mx = info.mx, my = info.my, mz = info.mz;

		DAGetGhostedCoordinates(user[bi].da, &Coor);
		DAVecGetArray(user[bi].fda, Coor, &coor);

		double buffer;

		for (k=0; k<KM; k++)
		for (j=0; j<JM; j++)
		for (i=0; i<IM; i++) {
			if (xyz_input) ;
			else if (binary_input) fread(&buffer, sizeof(double), 1, fd);
			else fscanf(fd, "%le", &buffer);

			if ( k>=zs && k<ze && j>=ys && j<ye && i>=xs && i<xe ) {
				if (xyz_input) coor[k][j][i].x = X[i]/cl;
				else coor[k][j][i].x = buffer/cl;
			}
		}

		for (k=0; k<KM; k++)
		for (j=0; j<JM; j++)
		for (i=0; i<IM; i++) {
			if (xyz_input) ;
			else if (binary_input) fread(&buffer, sizeof(double), 1, fd);
			else fscanf(fd, "%le", &buffer);

			if ( k>=zs && k<ze && j>=ys && j<ye && i>=xs && i<xe ) {
				if (xyz_input) coor[k][j][i].y = Y[j]/cl;
				else coor[k][j][i].y = buffer/cl;
			}
		}

		for (k=0; k<KM; k++)
		for (j=0; j<JM; j++)
		for (i=0; i<IM; i++) {
			if (xyz_input) ;
			else if (binary_input) fread(&buffer, sizeof(double), 1, fd);
			else fscanf(fd, "%le", &buffer);

			if ( k>=zs && k<ze && j>=ys && j<ye && i>=xs && i<xe ) {
				if (xyz_input) coor[k][j][i].z = Z[k]/cl;
				else coor[k][j][i].z = buffer/cl;
			}
		}

		DAVecRestoreArray(user[bi].fda, Coor, &coor);

		Vec	gCoor;
		DAGetCoordinates(user[bi].da, &gCoor);
		DALocalToGlobal(user[bi].fda, Coor, INSERT_VALUES, gCoor);
		DAGlobalToLocalBegin(user[bi].fda, gCoor, INSERT_VALUES, Coor);
		DAGlobalToLocalEnd(user[bi].fda, gCoor, INSERT_VALUES, Coor);

	}

	fclose(fd);

	printf("Finish Reading %s !\n", str);

	for (bi=0; bi<block_number; bi++) {
		user[bi]._this = bi;
	}
	return(0);
}

void Calc_avg_shear_stress(UserCtx *user)
{
	double N=(double)tis+1.0;
	DALocalInfo	info = user->info;
	PetscInt	xs = info.xs, xe = info.xs + info.xm;
	PetscInt  	ys = info.ys, ye = info.ys + info.ym;
	PetscInt	zs = info.zs, ze = info.zs + info.zm;
	PetscInt	mx = info.mx, my = info.my, mz = info.mz;

	PetscInt	lxs, lxe, lys, lye, lzs, lze;

	PetscInt i, j, k;
	lxs = xs; lxe = xe; lys = ys; lye = ye; lzs = zs; lze = ze;

	Cmpnts ***usum, ***csi, ***eta, ***zet;
	PetscReal ***aj, ***psum, ***nvert;
	if (lxs==0) lxs++;
	if (lxe==mx) lxe--;
	if (lys==0) lys++;
	if (lye==my) lye--;
	if (lzs==0) lzs++;
	if (lze==mz) lze--;

	char filen[256];
	PetscViewer	viewer;

	Vec P_sum;
	DACreateGlobalVector(user->da, &P_sum);
	DACreateGlobalVector(user->fda, &user->Ucat_sum);

	ti=tis;
	sprintf(filen, "su0_%06d_%1d.dat", ti, user->_this);
	PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
	VecLoadIntoVector(viewer, (user->Ucat_sum));
	PetscViewerDestroy(viewer);

	ti=tis;
	sprintf(filen, "sp_%06d_%1d.dat", ti, user->_this);
	PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
	VecLoadIntoVector(viewer, P_sum);
	PetscViewerDestroy(viewer);

	DAVecGetArray(user->fda, user->Csi, &csi);
	DAVecGetArray(user->fda, user->Eta, &eta);
	DAVecGetArray(user->fda, user->Zet, &zet);
	DAVecGetArray(user->da, user->Aj, &aj);
	DAVecGetArray(user->da, user->Nvert, &nvert);
	DAVecGetArray(user->fda, user->Ucat_sum, &usum);
	DAVecGetArray(user->da, P_sum, &psum);


	double force_skin_bottom = 0;
	double force_pressure_bottom = 0;
	double force_bottom = 0;
	double area_bottom = 0;

	double force_skin_top = 0;
	double force_pressure_top = 0;
	double force_top = 0;
	double area_top = 0;

	j=0;
	for (k=lzs; k<lze; k++)
	for (i=lxs; i<lxe; i++) {
		if (nvert[k][j+1][i] < 0.1) {
			double du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz;
			double dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz;

			dudc=0, dvdc=0, dwdc=0;

			dude=usum[k][j+1][i].x * 2.0 / N;
			dvde=usum[k][j+1][i].y * 2.0 / N;
			dwde=usum[k][j+1][i].z * 2.0 / N;

			dudz=0, dvdz=0, dwdz=0;

			double ajc = aj[k][j+1][i];
			double csi0 = csi[k][j+1][i].x, csi1 = csi[k][j+1][i].y, csi2 = csi[k][j+1][i].z;
			double eta0 = eta[k][j+1][i].x, eta1 = eta[k][j+1][i].y, eta2 = eta[k][j+1][i].z;
			double zet0 = zet[k][j+1][i].x, zet1 = zet[k][j+1][i].y, zet2 = zet[k][j+1][i].z;

			Compute_du_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc,
					dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz,
					&du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz );

			double j_area = sqrt( eta[k][j+1][i].x*eta[k][j+1][i].x + eta[k][j+1][i].y*eta[k][j+1][i].y + eta[k][j+1][i].z*eta[k][j+1][i].z );
			double ni[3], nj[3], nk[3];
			double nx, ny, nz;
			Calculate_normal(csi[k][j+1][i], eta[k][j+1][i], zet[k][j+1][i], ni, nj, nk);
			nx = nj[0]; //inward normal
			ny = nj[1]; //inward normal
			nz = nj[2]; //inward normal


			double Fp = - psum[k][j+1][i] * eta2 / N;
			double Fs = (dw_dx * nx + dw_dy * ny + dw_dz * nz) / user->ren * j_area;
			//double Fs = (du_dx * nx + du_dy * ny + du_dz * nz) / user->ren * j_area;

			force_skin_bottom += Fs;
			force_pressure_bottom += Fp;
			force_bottom += Fs + Fp;
			area_bottom += fabs(eta1);	// projected area
		}
	}

	j=my-2;
	for (k=lzs; k<lze; k++)
	for (i=lxs; i<lxe; i++) {
		if (nvert[k][j][i] < 0.1) {
			double du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz;
			double dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz;

			dudc=0, dvdc=0, dwdc=0;

			dude = -usum[k][j][i].x * 2.0 / N;
			dvde = -usum[k][j][i].y * 2.0 / N;
			dwde = -usum[k][j][i].z * 2.0 / N;

			dudz=0, dvdz=0, dwdz=0;

			double ajc = aj[k][j][i];
			double csi0 = csi[k][j][i].x, csi1 = csi[k][j][i].y, csi2 = csi[k][j][i].z;
			double eta0 = eta[k][j][i].x, eta1 = eta[k][j][i].y, eta2 = eta[k][j][i].z;
			double zet0 = zet[k][j][i].x, zet1 = zet[k][j][i].y, zet2 = zet[k][j][i].z;

			Compute_du_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc,
					dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz,
					&du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz );

			double j_area = sqrt( eta[k][j][i].x*eta[k][j][i].x + eta[k][j][i].y*eta[k][j][i].y + eta[k][j][i].z*eta[k][j][i].z );
			double ni[3], nj[3], nk[3];
			double nx, ny, nz;
			Calculate_normal(csi[k][j][i], eta[k][j][i], zet[k][j][i], ni, nj, nk);
			nx = -nj[0]; //inward normal
			ny = -nj[1]; //inward normal
			nz = -nj[2]; //inward normal


			double Fp = - psum[k][j][i] * eta2 / N;
			double Fs = (dw_dx * nx + dw_dy * ny + dw_dz * nz) / user->ren * j_area;
			//double Fs = (du_dx * nx + du_dy * ny + du_dz * nz) / user->ren * j_area;

			force_skin_top += Fs;
			force_pressure_top += Fp;
			force_top += Fs + Fp;
			area_top += fabs(eta1);	// projected area
		}
	}

	DAVecRestoreArray(user->fda, user->Csi, &csi);
	DAVecRestoreArray(user->fda, user->Eta, &eta);
	DAVecRestoreArray(user->fda, user->Zet, &zet);
	DAVecRestoreArray(user->da, user->Aj, &aj);
	DAVecRestoreArray(user->da, user->Nvert, &nvert);
	DAVecRestoreArray(user->fda, user->Ucat_sum, &usum);
	DAVecRestoreArray(user->da, P_sum, &psum);

	VecDestroy(P_sum);
	VecDestroy(user->Ucat_sum);

	printf("Top:\tarea=%f, force=%f, skin force=%f, pressure force=%f\n",
				area_top, force_top, force_skin_top, force_pressure_top);

	printf("\tstress=%f, skin stress=%f, pressure stress=%f, u*=%f, Re*=%f\n",
				force_top/area_top, force_skin_top/area_top, force_pressure_top/area_top,
				sqrt(fabs(force_top/area_top)), sqrt(fabs(force_top/area_top))*user->ren);

	printf("\n");

	printf("Bottom:\tarea=%f, force=%f, skin force=%f, pressure force=%f\n",
				area_bottom, force_bottom, force_skin_bottom, force_pressure_bottom);

	printf("\tstress=%f, skin stress=%f, pressure stress=%f, u*=%f, Re*=%f\n",
				force_bottom/area_bottom, force_skin_bottom/area_bottom, force_pressure_bottom/area_bottom,
				sqrt(fabs(force_bottom/area_bottom)), sqrt(fabs(force_bottom/area_bottom))*user->ren);
}

PetscErrorCode Lambda2(UserCtx *user)
{
	DALocalInfo	info = user->info;
	PetscInt	xs = info.xs, xe = info.xs + info.xm;
	PetscInt  	ys = info.ys, ye = info.ys + info.ym;
	PetscInt	zs = info.zs, ze = info.zs + info.zm;
	PetscInt	mx = info.mx, my = info.my, mz = info.mz;

	PetscInt	lxs, lxe, lys, lye, lzs, lze;

	PetscInt i, j, k;
	lxs = xs; lxe = xe; lys = ys; lye = ye; lzs = zs; lze = ze;

	Cmpnts ***ucat, ***csi, ***eta, ***zet;
	PetscReal ***aj, ***q, ***nvert;
	PetscReal ***wx, ***wy, ***wz;
	if (lxs==0) lxs++;
	if (lxe==mx) lxe--;
	if (lys==0) lys++;
	if (lye==my) lye--;
	if (lzs==0) lzs++;
	if (lze==mz) lze--;

	DAVecGetArray(user->fda, user->Ucat, &ucat);
	DAVecGetArray(user->fda, user->Csi, &csi);
	DAVecGetArray(user->fda, user->Eta, &eta);
	DAVecGetArray(user->fda, user->Zet, &zet);

	DAVecGetArray(user->da, user->Aj, &aj);
	DAVecGetArray(user->da, user->Nvert, &nvert);
	DAVecGetArray(user->da, user->tempValue, &q);

	//PetscReal uc, vc, wc, ue, ve, we, uz, vz, wz;

	PetscReal s11, s12, s13, s21, s22, s23, s31, s32, s33;
	PetscReal d11, d12, d13, d21, d22, d23, d31, d32, d33;

	PetscReal w11, w12, w13, w21, w22, w23, w31, w32, w33;
	//PetscReal so, wo;
	PetscReal csi1, csi2, csi3, eta1, eta2, eta3, zet1, zet2, zet3;
	for (k=lzs; k<lze; k++) {
		for (j=lys; j<lye; j++) {
			for (i=lxs; i<lxe; i++) {

				double dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz;
				double du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz;
				double csi0 = csi[k][j][i].x, csi1 = csi[k][j][i].y, csi2 = csi[k][j][i].z;
				double eta0= eta[k][j][i].x, eta1 = eta[k][j][i].y, eta2 = eta[k][j][i].z;
				double zet0 = zet[k][j][i].x, zet1 = zet[k][j][i].y, zet2 = zet[k][j][i].z;
				double ajc = aj[k][j][i];

				Compute_du_center (i, j, k, mx, my, mz, ucat, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde, &dudz, &dvdz, &dwdz);
				Compute_du_dxyz (	csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz,
										&du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz );

				double Sxx = 0.5*( du_dx + du_dx ), Sxy = 0.5*(du_dy + dv_dx), Sxz = 0.5*(du_dz + dw_dx);
				double Syx = Sxy, Syy = 0.5*(dv_dy + dv_dy),	Syz = 0.5*(dv_dz + dw_dy);
				double Szx = Sxz, Szy=Syz, Szz = 0.5*(dw_dz + dw_dz);


				w11 = 0;
				w12 = 0.5*(du_dy - dv_dx);
				w13 = 0.5*(du_dz - dw_dx);
				w21 = -w12;
				w22 = 0.;
				w23 = 0.5*(dv_dz - dw_dy);
				w31 = -w13;
				w32 = -w23;
				w33 = 0.;


				double S[3][3], W[3][3], D[3][3];

				D[0][0] = du_dx, D[0][1] = du_dy, D[0][2] = du_dz;
				D[1][0] = dv_dx, D[1][1] = dv_dy, D[1][2] = dv_dz;
				D[2][0] = dw_dx, D[2][1] = dw_dy, D[2][2] = dw_dz;

				S[0][0] = Sxx;
				S[0][1] = Sxy;
				S[0][2] = Sxz;

				S[1][0] = Syx;
				S[1][1] = Syy;
				S[1][2] = Syz;

				S[2][0] = Szx;
				S[2][1] = Szy;
				S[2][2] = Szz;

				W[0][0] = w11;
				W[0][1] = w12;
				W[0][2] = w13;
				W[1][0] = w21;
				W[1][1] = w22;
				W[1][2] = w23;
				W[2][0] = w31;
				W[2][1] = w32;
				W[2][2] = w33;

				// lambda-2
				double A[3][3], V[3][3], d[3];

				for (int row=0; row<3; row++)
				for (int col=0; col<3; col++) A[row][col]=0;

				for (int row=0; row<3; row++)
				for (int col=0; col<3; col++) {
					A[row][col] += S[row][0] * S[0][col];
					A[row][col] += S[row][1] * S[1][col];
					A[row][col] += S[row][2] * S[2][col];
				}

				for (int row=0; row<3; row++)
				for (int col=0; col<3; col++) {
					A[row][col] += W[row][0] * W[0][col];
					A[row][col] += W[row][1] * W[1][col];
					A[row][col] += W[row][2] * W[2][col];
				}

				if (nvert[k][j][i]<0.1) {
					eigen_decomposition(A, V, d);
					q[k][j][i] = d[1];
				}
				else q[k][j][i] = 1000.0;
/*
				// delta criterion
				double DD[3][3];
				for (int row=0; row<3; row++)
				for (int col=0; col<3; col++) DD[row][col]=0;

				for (int row=0; row<3; row++)
				for (int col=0; col<3; col++) {
					DD[row][col] += D[row][0] * D[0][col];
					DD[row][col] += D[row][1] * D[1][col];
					DD[row][col] += D[row][2] * D[2][col];
				}
				double tr_DD = DD[0][0] + DD[1][1] + DD[2][2];
				double det_D = D[0][0]*(D[2][2]*D[1][1]-D[2][1]*D[1][2])-D[1][0]*(D[2][2]*D[0][1]-D[2][1]*D[0][2])+D[2][0]*(D[1][2]*D[0][1]-D[1][1]*D[0][2]);

				//double Q = -0.5*tr_DD;

				double SS=0, WW=0;
				for (int row=0; row<3; row++)
				for (int col=0; col<3; col++) {
					SS+=S[row][col]*S[row][col];
					WW+=W[row][col]*W[row][col];
				}
				double Q = 0.5*(WW - SS);

				double R = - det_D;
				if (nvert[k][j][i]<0.1) {
					q[k][j][i] = pow( 0.5*R, 2. )  + pow( Q/3., 3.);
				}
				else q[k][j][i] = -10;
				if (q[k][j][i]<0) q[k][j][i]=-10;
*/
			}
		}
	}

	DAVecRestoreArray(user->fda, user->Ucat, &ucat);
	DAVecRestoreArray(user->fda, user->Csi, &csi);
	DAVecRestoreArray(user->fda, user->Eta, &eta);
	DAVecRestoreArray(user->fda, user->Zet, &zet);

	DAVecRestoreArray(user->da, user->Aj, &aj);
	DAVecRestoreArray(user->da, user->Nvert, &nvert);
	DAVecRestoreArray(user->da, user->tempValue, &q);

	return 0;
}



PetscErrorCode Vort(UserCtx *user, int dir)
{

	DALocalInfo	info = user->info;
	PetscInt	xs = info.xs, xe = info.xs + info.xm;
	PetscInt  	ys = info.ys, ye = info.ys + info.ym;
	PetscInt	zs = info.zs, ze = info.zs + info.zm;
	PetscInt	mx = info.mx, my = info.my, mz = info.mz;

	PetscInt	lxs, lxe, lys, lye, lzs, lze;

	PetscInt i, j, k;
	lxs = xs; lxe = xe; lys = ys; lye = ye; lzs = zs; lze = ze;

	Cmpnts ***ucat, ***csi, ***eta, ***zet;
	PetscReal ***aj, ***q, ***nvert;
	PetscReal ***wx, ***wy, ***wz;
	if (lxs==0) lxs++;
	if (lxe==mx) lxe--;
	if (lys==0) lys++;
	if (lye==my) lye--;
	if (lzs==0) lzs++;
	if (lze==mz) lze--;

	DAVecGetArray(user->fda, user->Ucat, &ucat);
	DAVecGetArray(user->fda, user->Csi, &csi);
	DAVecGetArray(user->fda, user->Eta, &eta);
	DAVecGetArray(user->fda, user->Zet, &zet);

	DAVecGetArray(user->da, user->Aj, &aj);
	DAVecGetArray(user->da, user->Nvert, &nvert);
	DAVecGetArray(user->da, user->tempValue, &q);
	
	//PetscReal uc, vc, wc, ue, ve, we, uz, vz, wz;

	PetscReal s11, s12, s13, s21, s22, s23, s31, s32, s33;
	PetscReal d11, d12, d13, d21, d22, d23, d31, d32, d33;

	PetscReal w11, w12, w13, w21, w22, w23, w31, w32, w33;
	PetscReal so, wo;
	PetscReal csi1, csi2, csi3, eta1, eta2, eta3, zet1, zet2, zet3;


	for (k=lzs; k<lze; k++) {
		for (j=lys; j<lye; j++) {
			for (i=lxs; i<lxe; i++) {
				double dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz;
				double du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz;
				double csi0 = csi[k][j][i].x, csi1 = csi[k][j][i].y, csi2 = csi[k][j][i].z;
				double eta0= eta[k][j][i].x, eta1 = eta[k][j][i].y, eta2 = eta[k][j][i].z;
				double zet0 = zet[k][j][i].x, zet1 = zet[k][j][i].y, zet2 = zet[k][j][i].z;
				double ajc = aj[k][j][i];

				Compute_du_center (i, j, k, mx, my, mz, ucat, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde, &dudz, &dvdz, &dwdz);
				Compute_du_dxyz (	csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz,
										&du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz );
				
				if ( nvert[k][j][i]>0.1 ) {
					q[k][j][i] = 0;
				}
				else {
					if (dir == 1) {q[k][j][i] = -(dv_dz - dw_dy);}
					else if (dir == 2) {q[k][j][i] = (du_dz - dw_dx);}
					else if(dir == 3) {q[k][j][i] = -(du_dy - dv_dx);}
				}

			}
		}
	}

	DAVecRestoreArray(user->fda, user->Ucat, &ucat);
	DAVecRestoreArray(user->fda, user->Csi, &csi);
	DAVecRestoreArray(user->fda, user->Eta, &eta);
	DAVecRestoreArray(user->fda, user->Zet, &zet);

	DAVecRestoreArray(user->da, user->Aj, &aj);
	DAVecRestoreArray(user->da, user->Nvert, &nvert);
	DAVecRestoreArray(user->da, user->tempValue, &q);

	return 0;
}


PetscErrorCode ASCIIOutPut(UserCtx *user, PetscInt ti, PetscInt component)
{

	DALocalInfo	info = user->info;
	PetscInt	xs = info.xs, xe = info.xs + info.xm;
	PetscInt  	ys = info.ys, ye = info.ys + info.ym;
	PetscInt	zs = info.zs, ze = info.zs + info.zm;
	PetscInt	mx = info.mx, my = info.my, mz = info.mz;

	PetscInt	lxs, lxe, lys, lye, lzs, lze;

	PetscInt i, j, k, nx, ny, nz;
	
	lxs = xs; lxe = xe; lys = ys; lye = ye; lzs = zs; lze = ze;

	Cmpnts ***ucat, ***csi, ***eta, ***zet, ***wCv;
	PetscReal ***aj, ***q, ***nvert;
	PetscReal ***wx, ***wy, ***wz;
	//Cmpnts ***wCv;
	
	if (lxs==0) lxs++;
	if (lxe==mx) lxe--;
	if (lys==0) lys++;
	if (lye==my) lye--;
	if (lzs==0) lzs++;
	if (lze==mz) lze--;

	DAVecGetArray(user->fda, user->Ucat, &ucat);
	DAVecGetArray(user->fda, user->WCV, &wCv);
	DAVecGetArray(user->fda, user->Csi, &csi);
	DAVecGetArray(user->fda, user->Eta, &eta);
	DAVecGetArray(user->fda, user->Zet, &zet);

	DAVecGetArray(user->da, user->Aj, &aj);
	DAVecGetArray(user->da, user->Nvert, &nvert);
	DAVecGetArray(user->da, user->tempValue, &q);
	
	//PetscReal uc, vc, wc, ue, ve, we, uz, vz, wz;

	PetscReal s11, s12, s13, s21, s22, s23, s31, s32, s33;
	PetscReal d11, d12, d13, d21, d22, d23, d31, d32, d33;

	PetscReal w11, w12, w13, w21, w22, w23, w31, w32, w33;
	PetscReal so, wo;
	PetscReal csi1, csi2, csi3, eta1, eta2, eta3, zet1, zet2, zet3;
	PetscReal uvec[3], wvec[3];
	PetscReal dcsi, deta, dzet, dv, div_wCv, div_wCv_dv;

	if(ASCII>=10){	
		if(ASCII==11 || ASCII>=20){
			if (component==5){
				FILE *fphiSim;
				fphiSim = fopen("phiInSimGrid.dat", "r"); if (!fphiSim) SETERRQ(1, "Cannot open phi field data file")
				fscanf(fphiSim, "%i", &nx);
				fscanf(fphiSim, "%i", &ny);
				fscanf(fphiSim, "%i", &nz);
				printf("ASR: nxyz=%d %d %d, lxsyszs %d %d %d",nx,ny,nz,lxe-lxs,lye-lys,lze-lzs);
				PetscReal phiSim[nz][ny][nx];
				for (k=lzs+0; k<lzs+nx; k++) {
				for (j=lys+0; j<lys+ny; j++) {
				for (i=lxs+0; i<lxs+nz; i++) {
					fscanf(fphiSim, "%le", &phiSim[k][j][i]);
				}}}
				fclose(fphiSim);
				
				for (k=lzs; k<lze; k++) {
				for (j=lys; j<lye; j++) {
				for (i=lxs; i<lxe; i++) {
					wCv[k][j][i].x = ucat[k][j][i].x*ucat[k][j][i].x + ucat[k][j][i].y*ucat[k][j][i].y + ucat[k][j][i].z*ucat[k][j][i].z; 
					wCv[k][j][i].y = 1.0;
					wCv[k][j][i].z = 1.0;
				}}}
				
				for (k=lzs; k<lze; k++) {
				for (j=lys; j<lye; j++) {
				for (i=lxs; i<lxe; i++) {
					double dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz;
					double du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz;
					double csi0 = csi[k][j][i].x, csi1 = csi[k][j][i].y, csi2 = csi[k][j][i].z;
					double eta0= eta[k][j][i].x, eta1 = eta[k][j][i].y, eta2 = eta[k][j][i].z;
					double zet0 = zet[k][j][i].x, zet1 = zet[k][j][i].y, zet2 = zet[k][j][i].z;
					double ajc = aj[k][j][i];
					Compute_du_center (i, j, k, mx, my, mz, wCv, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde, &dudz, &dvdz, &dwdz);
					Compute_du_dxyz (	csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz,
					&du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz);

					wCv[k][j][i].x = du_dx*phiSim[k][j][i];
					wCv[k][j][i].y = du_dy*phiSim[k][j][i];
					wCv[k][j][i].z = du_dz*phiSim[k][j][i];
				}}}
				
				for (k=lzs; k<lze; k++) {
				for (j=lys; j<lye; j++) {
				for (i=lxs; i<lxe; i++) {
					double dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz;
					double du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz;
					double csi0 = csi[k][j][i].x, csi1 = csi[k][j][i].y, csi2 = csi[k][j][i].z;
					double eta0= eta[k][j][i].x, eta1 = eta[k][j][i].y, eta2 = eta[k][j][i].z;
					double zet0 = zet[k][j][i].x, zet1 = zet[k][j][i].y, zet2 = zet[k][j][i].z;
					double ajc = aj[k][j][i];
					Compute_du_center (i, j, k, mx, my, mz, wCv, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde, &dudz, &dvdz, &dwdz);
					Compute_du_dxyz (	csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz,
					&du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz);
					if ( nvert[k][j][i]>0.1 ) {
						q[k][j][i] = 0;
					}
					else {
						q[k][j][i] = du_dx + dv_dy + dw_dz;
					}
				}}}
			}	
		}
		
		if(ASCII==10 || ASCII>=20){
				
			for (k=lzs; k<lze; k++) {
			for (j=lys; j<lye; j++) {
			for (i=lxs; i<lxe; i++) {
				double dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz;
				double du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz;
				double csi0 = csi[k][j][i].x, csi1 = csi[k][j][i].y, csi2 = csi[k][j][i].z;
				double eta0= eta[k][j][i].x, eta1 = eta[k][j][i].y, eta2 = eta[k][j][i].z;
				double zet0 = zet[k][j][i].x, zet1 = zet[k][j][i].y, zet2 = zet[k][j][i].z;
				double ajc = aj[k][j][i];
				Compute_du_center (i, j, k, mx, my, mz, ucat, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde, &dudz, &dvdz, &dwdz);
				Compute_du_dxyz (	csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz,
				&du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz);
				w11 = -(dv_dz - dw_dy);
				w22 = (du_dz - dw_dx);
				w33 = -(du_dy - dv_dx);

				uvec[0] = ucat[k][j][i].x;
				uvec[1] = ucat[k][j][i].y;
				uvec[2] = ucat[k][j][i].z;

				wvec[0] = w11;
				wvec[1] = w22;
				wvec[2] = w33;

				wCv[k][j][i].x = wvec[1]*uvec[2]-wvec[2]*uvec[1];
				wCv[k][j][i].y = wvec[2]*uvec[0]-wvec[0]*uvec[2];
				wCv[k][j][i].z = wvec[0]*uvec[1]-wvec[1]*uvec[0];
				
				if ( nvert[k][j][i]>0.1 ){
					q[k][j][i] = 0;
				}
				else {
					if(component==1) q[k][j][i] = wCv[k][j][i].x; 
					else if (component==2) q[k][j][i] = wCv[k][j][i].y;
					else if (component==3) q[k][j][i] = wCv[k][j][i].z;
				}
				
			}}}
			/*
			FILE *f3;
			char filen3[80];
			sprintf(filen3, "wCVx%06d.dat", ti);
			f3 = fopen(filen3, "w");
			PetscFPrintf(PETSC_COMM_WORLD, f3, "%d %d %d\n",lxe-lxs,lye-lys,lze-lzs);
			for (k=lzs; k<lze; k++) {
			for (j=lys; j<lye; j++) {
			for (i=lxs; i<lxe; i++) {
					PetscFPrintf(PETSC_COMM_WORLD, f3, "%e\n",wCv[k][j][i].x);
			}}}
			fclose(f3);

			sprintf(filen3, "wCVy%06d.dat", ti);
			f3 = fopen(filen3, "w");
			PetscFPrintf(PETSC_COMM_WORLD, f3, "%d %d %d\n",lxe-lxs,lye-lys,lze-lzs);
			for (k=lzs; k<lze; k++) {
			for (j=lys; j<lye; j++) {
			for (i=lxs; i<lxe; i++) {
					PetscFPrintf(PETSC_COMM_WORLD, f3, "%e\n",wCv[k][j][i].y);
			}}}
			fclose(f3);
			 
			sprintf(filen3, "wCVz%06d.dat", ti);
			f3 = fopen(filen3, "w");
			PetscFPrintf(PETSC_COMM_WORLD, f3, "%d %d %d\n",lxe-lxs,lye-lys,lze-lzs);
			for (k=lzs; k<lze; k++) {
			for (j=lys; j<lye; j++) {
			for (i=lxs; i<lxe; i++) {
					PetscFPrintf(PETSC_COMM_WORLD, f3, "%e\n",wCv[k][j][i].z);
			}}}
			fclose(f3);
			*/

			if(component==4){
				FILE *f2;
				char filen2[80];
				FILE *f3;
				char filen3[80];			
				sprintf(filen2, "div_wCv%06d.dat", ti);
				sprintf(filen3, "dv%06d.dat", ti);			
				f2 = fopen(filen2, "w");
				f3 = fopen(filen3, "w");			
				PetscFPrintf(PETSC_COMM_WORLD, f2, "%d\n%d\n%d\n",lxe-lxs,lye-lys,lze-lzs);
				PetscFPrintf(PETSC_COMM_WORLD, f3, "%d\n%d\n%d\n",lxe-lxs,lye-lys,lze-lzs);			
				for (k=lzs; k<lze; k++) {
				for (j=lys; j<lye; j++) {
				for (i=lxs; i<lxe; i++) {
					double dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz;
					double du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz;
					double csi0 = csi[k][j][i].x, csi1 = csi[k][j][i].y, csi2 = csi[k][j][i].z;
					double eta0= eta[k][j][i].x, eta1 = eta[k][j][i].y, eta2 = eta[k][j][i].z;
					double zet0 = zet[k][j][i].x, zet1 = zet[k][j][i].y, zet2 = zet[k][j][i].z;
					double ajc = aj[k][j][i];

					Compute_du_center (i, j, k, mx, my, mz, wCv, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde, &dudz, &dvdz, &dwdz);
					Compute_du_dxyz (	csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz,
					&du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz );
					div_wCv = du_dx + dv_dy + dw_dz;
					
					if ( nvert[k][j][i]>0.1 ) {
						q[k][j][i] = 0;
						PetscFPrintf(PETSC_COMM_WORLD, f2, "0.0\n");
					}
					else {
						PetscFPrintf(PETSC_COMM_WORLD, f2, "%e\n", div_wCv);
						q[k][j][i] = div_wCv; 
					}
					PetscFPrintf(PETSC_COMM_WORLD, f3, "%e\n", 1/ajc);										
				}}}
				fclose(f2);
				fclose(f3);
			}
		}
	}
	else{
		if(ASCII>=1){
				FILE *f2;
				char filen2[80];
				sprintf(filen2, "Ucat%06d.dat", ti);
				f2 = fopen(filen2, "w");
				PetscFPrintf(PETSC_COMM_WORLD, f2, "%d %d %d\n",lxe-lxs,lye-lys,lze-lzs);
					for (k=lzs; k<lze; k++) {
					for (j=lys; j<lye; j++) {
					for (i=lxs; i<lxe; i++) {
						if ( nvert[k][j][i]>0.1 ) {
							PetscFPrintf(PETSC_COMM_WORLD, f2, "0.0 0.0 0.0\n");
						}
						else{
							PetscFPrintf(PETSC_COMM_WORLD, f2, "%e %e %e\n",ucat[k][j][i].x, ucat[k][j][i].y, ucat[k][j][i].z );
						}
					}}}
				fclose(f2);
		}
		if(ASCII>=2){
			FILE *f2;
			char filen2[80];
			sprintf(filen2, "Vort%06d.dat", ti);
			f2 = fopen(filen2, "w");
			PetscFPrintf(PETSC_COMM_WORLD, f2, "%d %d %d\n",lxe-lxs,lye-lys,lze-lzs);
			for (k=lzs; k<lze; k++) {
			for (j=lys; j<lye; j++) {
			for (i=lxs; i<lxe; i++) {
				double dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz;
				double du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz;
				double csi0 = csi[k][j][i].x, csi1 = csi[k][j][i].y, csi2 = csi[k][j][i].z;
				double eta0= eta[k][j][i].x, eta1 = eta[k][j][i].y, eta2 = eta[k][j][i].z;
				double zet0 = zet[k][j][i].x, zet1 = zet[k][j][i].y, zet2 = zet[k][j][i].z;
				double ajc = aj[k][j][i];
				Compute_du_center (i, j, k, mx, my, mz, ucat, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde, &dudz, &dvdz, &dwdz);
				Compute_du_dxyz (	csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz,
										&du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz );
				w11 = -(dv_dz - dw_dy);
				w22 = (du_dz - dw_dx);
				w33 = -(du_dy - dv_dx);

				if ( nvert[k][j][i]>0.1 ) {
					q[k][j][i] = 0;
					PetscFPrintf(PETSC_COMM_WORLD, f2, "0.0 0.0 0.0\n");
				}
				else {
					PetscFPrintf(PETSC_COMM_WORLD, f2, "%e %e %e\n",w11, w22, w33 );
					if(component==1) q[k][j][i] = w11;
					else if (component==2) q[k][j][i] = w22;
					else if (component==3) q[k][j][i] = w33;
				}
			}}}
			fclose(f2);
		}
		if(ASCII>=3){
			FILE *f2;
			char filen2[80];
			sprintf(filen2, "Helicity%06d.dat", ti);
			f2 = fopen(filen2, "w");
			PetscFPrintf(PETSC_COMM_WORLD, f2, "%d %d %d\n",lxe-lxs,lye-lys,lze-lzs);
			for (k=lzs; k<lze; k++) {
			for (j=lys; j<lye; j++) {
			for (i=lxs; i<lxe; i++) {
				double dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz;
				double du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz;
				double csi0 = csi[k][j][i].x, csi1 = csi[k][j][i].y, csi2 = csi[k][j][i].z;
				double eta0= eta[k][j][i].x, eta1 = eta[k][j][i].y, eta2 = eta[k][j][i].z;
				double zet0 = zet[k][j][i].x, zet1 = zet[k][j][i].y, zet2 = zet[k][j][i].z;
				double ajc = aj[k][j][i];
				Compute_du_center (i, j, k, mx, my, mz, ucat, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde, &dudz, &dvdz, &dwdz);
				Compute_du_dxyz (	csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz,
										&du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz );
				w11 = -(dv_dz - dw_dy) * ucat[k][j][i].x;
				w22 = (du_dz - dw_dx) * ucat[k][j][i].y;
				w33 = -(du_dy - dv_dx) * ucat[k][j][i].z;
				if ( nvert[k][j][i]>0.1 ) {
					PetscFPrintf(PETSC_COMM_WORLD, f2, "0.0 0.0 0.0\n");
				}			
				else{
					PetscFPrintf(PETSC_COMM_WORLD, f2, "%e %e %e\n",w11, w22, w33 );
				}
			}}}
			fclose(f2);
		}
	}
	DAVecRestoreArray(user->fda, user->Ucat, &ucat);
	DAVecRestoreArray(user->fda, user->WCV, &wCv);
	DAVecRestoreArray(user->fda, user->Csi, &csi);
	DAVecRestoreArray(user->fda, user->Eta, &eta);
	DAVecRestoreArray(user->fda, user->Zet, &zet);
	DAVecRestoreArray(user->da, user->Aj, &aj);
	DAVecRestoreArray(user->da, user->Nvert, &nvert);
	DAVecRestoreArray(user->da, user->tempValue, &q);
	return 0;
}


PetscErrorCode QCriteria(UserCtx *user)
{

	DALocalInfo	info = user->info;
	PetscInt	xs = info.xs, xe = info.xs + info.xm;
	PetscInt  	ys = info.ys, ye = info.ys + info.ym;
	PetscInt	zs = info.zs, ze = info.zs + info.zm;
	PetscInt	mx = info.mx, my = info.my, mz = info.mz;

	PetscInt	lxs, lxe, lys, lye, lzs, lze;

	PetscInt i, j, k;
	lxs = xs; lxe = xe; lys = ys; lye = ye; lzs = zs; lze = ze;

	Cmpnts ***ucat, ***csi, ***eta, ***zet;
	PetscReal ***aj, ***q, ***nvert;
	PetscReal ***wx, ***wy, ***wz, ***wm, ***helicity;
	if (lxs==0) lxs++;
	if (lxe==mx) lxe--;
	if (lys==0) lys++;
	if (lye==my) lye--;
	if (lzs==0) lzs++;
	if (lze==mz) lze--;

	DAVecGetArray(user->fda, user->Ucat, &ucat);
	DAVecGetArray(user->fda, user->Csi, &csi);
	DAVecGetArray(user->fda, user->Eta, &eta);
	DAVecGetArray(user->fda, user->Zet, &zet);

	DAVecGetArray(user->da, user->Aj, &aj);
	DAVecGetArray(user->da, user->Nvert, &nvert);
	DAVecGetArray(user->da, user->tempValue, &q);
	DAVecGetArray(user->da, user->WX, &wx);
	DAVecGetArray(user->da, user->WY, &wy);
	DAVecGetArray(user->da, user->WZ, &wz);
	DAVecGetArray(user->da, user->WM, &wm);	
	DAVecGetArray(user->da, user->HELICITY, &helicity);
	
	//PetscReal uc, vc, wc, ue, ve, we, uz, vz, wz;

	PetscReal s11, s12, s13, s21, s22, s23, s31, s32, s33;
	PetscReal d11, d12, d13, d21, d22, d23, d31, d32, d33;

	PetscReal w11, w12, w13, w21, w22, w23, w31, w32, w33;
	PetscReal so, wo;
	PetscReal csi1, csi2, csi3, eta1, eta2, eta3, zet1, zet2, zet3;

	for (k=lzs; k<lze; k++) {
		for (j=lys; j<lye; j++) {
			for (i=lxs; i<lxe; i++) {
				double dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz;
				double du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz;
				double csi0 = csi[k][j][i].x, csi1 = csi[k][j][i].y, csi2 = csi[k][j][i].z;
				double eta0= eta[k][j][i].x, eta1 = eta[k][j][i].y, eta2 = eta[k][j][i].z;
				double zet0 = zet[k][j][i].x, zet1 = zet[k][j][i].y, zet2 = zet[k][j][i].z;
				double ajc = aj[k][j][i];

				Compute_du_center (i, j, k, mx, my, mz, ucat, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde, &dudz, &dvdz, &dwdz);
				Compute_du_dxyz (	csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz,
										&du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz );

				double Sxx = 0.5*( du_dx + du_dx ), Sxy = 0.5*(du_dy + dv_dx), Sxz = 0.5*(du_dz + dw_dx);
				double Syx = Sxy, Syy = 0.5*(dv_dy + dv_dy),	Syz = 0.5*(dv_dz + dw_dy);
				double Szx = Sxz, Szy=Syz, Szz = 0.5*(dw_dz + dw_dz);
				so = Sxx*Sxx + Sxy*Sxy + Sxz*Sxz + Syx*Syx + Syy*Syy + Syz*Syz + Szx*Szx + Szy*Szy + Szz*Szz;

				w11 = 0;
				w12 = 0.5*(du_dy - dv_dx);
				w13 = 0.5*(du_dz - dw_dx);
				w21 = -w12;
				w22 = 0.;
				w23 = 0.5*(dv_dz - dw_dy);
				w31 = -w13;
				w32 = -w23;
				w33 = 0.;

				wo = w11*w11 + w12*w12 + w13*w13 + w21*w21 + w22*w22 + w23*w23 + w31*w31 + w32*w32 + w33*w33;
				
				wx[k][j][i] = -(dv_dz - dw_dy);
				wy[k][j][i] = (du_dz - dw_dx);
				wz[k][j][i] = -(du_dy - dv_dx);
				wm[k][j][i] = sqrt(wx[k][j][i]*wx[k][j][i] + wy[k][j][i]*wy[k][j][i] + wz[k][j][i]*wz[k][j][i]);				
				helicity[k][j][i] = wx[k][j][i] * ucat[k][j][i].x + wy[k][j][i] * ucat[k][j][i].y + wz[k][j][i] * ucat[k][j][i].z;
/*
				so = ( d11 *  d11 + d22 * d22 + d33 * d33) + 0.5* ( (d12 + d21) * (d12 + d21) + (d13 + d31) * (d13 + d31) + (d23 + d32) * (d23 + d32) );
				wo = 0.5 * ( (d12 - d21)*(d12 - d21) + (d13 - d31) * (d13 - d31) + (d23 - d32) * (d23 - d32) );
				V19=0.5 * ( (V13 - V11)*(V13 - V11) + (V16 - V12) * (V16 - V12) + (V17 - V15) * (V17 - V15) ) - 0.5 * ( V10 *  V10 + V14 * V14 + V18 * V18) - 0.25* ( (V13 + V11) * (V13 + V11) + (V16 + V12) * (V16 + V12) + (V17 + V15) * (V17 + V15) )
*/
				if ( nvert[k][j][i]>0.1 ) q[k][j][i] = 0;
				else q[k][j][i] = (wo - so) / 2.;
			}
		}
	}

	DAVecRestoreArray(user->fda, user->Ucat, &ucat);
	DAVecRestoreArray(user->fda, user->Csi, &csi);
	DAVecRestoreArray(user->fda, user->Eta, &eta);
	DAVecRestoreArray(user->fda, user->Zet, &zet);

	DAVecRestoreArray(user->da, user->Aj, &aj);
	DAVecRestoreArray(user->da, user->Nvert, &nvert);
	DAVecRestoreArray(user->da, user->tempValue, &q);
	DAVecRestoreArray(user->da, user->WX, &wx);
	DAVecRestoreArray(user->da, user->WY, &wy);
	DAVecRestoreArray(user->da, user->WZ, &wz);
	DAVecRestoreArray(user->da, user->WM, &wm);	
	DAVecRestoreArray(user->da, user->HELICITY, &helicity);
	return 0;
}


PetscErrorCode writeJacobian(UserCtx *user){
	DALocalInfo	info = user->info;
	PetscInt	xs = info.xs, xe = info.xs + info.xm;
	PetscInt  	ys = info.ys, ye = info.ys + info.ym;
	PetscInt	zs = info.zs, ze = info.zs + info.zm;
	PetscInt	mx = info.mx, my = info.my, mz = info.mz;

	PetscInt	lxs, lxe, lys, lye, lzs, lze;

	PetscInt i, j, k;
	lxs = xs; lxe = xe; lys = ys; lye = ye; lzs = zs; lze = ze;

	PetscReal ***aj, ***q;
	if (lxs==0) lxs++;
	if (lxe==mx) lxe--;
	if (lys==0) lys++;
	if (lye==my) lye--;
	if (lzs==0) lzs++;
	if (lze==mz) lze--;

	DAVecGetArray(user->da, user->Aj, &aj);
	DAVecGetArray(user->da, user->tempValue, &q);

	for (k=lzs; k<lze; k++) {
		for (j=lys; j<lye; j++) {
			for (i=lxs; i<lxe; i++) {
				q[k][j][i] = 1/aj[k][j][i];
			}
		}
	}

	DAVecRestoreArray(user->da, user->Aj, &aj);
	DAVecRestoreArray(user->da, user->tempValue, &q);	
	return 0;
}

PetscErrorCode Velocity_Magnitude(UserCtx *user,  PetscInt component)	// store at tempValue
{
	DALocalInfo	info = user->info;
	PetscInt	xs = info.xs, xe = info.xs + info.xm;
	PetscInt  	ys = info.ys, ye = info.ys + info.ym;
	PetscInt	zs = info.zs, ze = info.zs + info.zm;
	PetscInt	mx = info.mx, my = info.my, mz = info.mz;

	PetscInt	lxs, lxe, lys, lye, lzs, lze;

	PetscInt i, j, k;
	lxs = xs; lxe = xe; lys = ys; lye = ye; lzs = zs; lze = ze;

	Cmpnts ***ucat;
	PetscReal ***q;
	if (lxs==0) lxs++;
	if (lxe==mx) lxe--;
	if (lys==0) lys++;
	if (lye==my) lye--;
	if (lzs==0) lzs++;
	if (lze==mz) lze--;

	DAVecGetArray(user->fda, user->Ucat, &ucat);
	DAVecGetArray(user->da, user->tempValue, &q);

	if(component==1){
			for (k=lzs; k<lze; k++) {
			for (j=lys; j<lye; j++) {
				for (i=lxs; i<lxe; i++) {
					q[k][j][i] = ucat[k][j][i].x;
				}
			}
		}	
		
	}
	else if(component==2){
			for (k=lzs; k<lze; k++) {
			for (j=lys; j<lye; j++) {
				for (i=lxs; i<lxe; i++) {
					q[k][j][i] = ucat[k][j][i].y;
				}
			}
		}	
		
	}
	else if(component==3){
			for (k=lzs; k<lze; k++) {
			for (j=lys; j<lye; j++) {
				for (i=lxs; i<lxe; i++) {
					q[k][j][i] = ucat[k][j][i].z;
				}
			}
		}	
		
	}
	else if(component==4){
		for (k=lzs; k<lze; k++) {
			for (j=lys; j<lye; j++) {
				for (i=lxs; i<lxe; i++) {
					q[k][j][i] = sqrt( ucat[k][j][i].x*ucat[k][j][i].x + ucat[k][j][i].y*ucat[k][j][i].y + ucat[k][j][i].z*ucat[k][j][i].z );
				}
			}
		}	
	}

	DAVecRestoreArray(user->fda, user->Ucat, &ucat);
	DAVecRestoreArray(user->da, user->tempValue, &q);

	return 0;
}


PetscErrorCode ibm_read(IBMNodes *ibm)
{
	PetscInt	rank;
	PetscInt	n_v , n_elmt ;
	PetscReal	*x_bp , *y_bp , *z_bp ;
	PetscInt	*nv1 , *nv2 , *nv3 ;
	PetscReal	*nf_x, *nf_y, *nf_z;
	PetscInt	i;
	PetscInt	n1e, n2e, n3e;
	PetscReal	dx12, dy12, dz12, dx13, dy13, dz13;
	PetscReal	t, dr;
	double xt;
	//MPI_Comm_size(PETSC_COMM_WORLD, &size);
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	if (!rank) { // root processor read in the data
		FILE *fd;
		fd = fopen("ibmdata0", "r"); if (!fd) SETERRQ(1, "Cannot open IBM node file")
		n_v =0;
		fscanf(fd, "%i", &n_v);
		fscanf(fd, "%i", &n_v);
		fscanf(fd, "%le", &xt);
		ibm->n_v = n_v;

		MPI_Bcast(&(ibm->n_v), 1, MPI_INT, 0, PETSC_COMM_WORLD);
		//    PetscPrintf(PETSC_COMM_WORLD, "nv, %d %e \n", n_v, xt);
		PetscMalloc(n_v*sizeof(PetscReal), &x_bp);
		PetscMalloc(n_v*sizeof(PetscReal), &y_bp);
		PetscMalloc(n_v*sizeof(PetscReal), &z_bp);

		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp));
		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp));
		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp));

		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp_o));
		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp_o));
		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp_o));

		PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->u));
		PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->uold));

		for (i=0; i<n_v; i++) {
			fscanf(fd, "%le %le %le %le %le %le", &x_bp[i], &y_bp[i], &z_bp[i], &t, &t, &t);
			x_bp[i] = x_bp[i] / 28.;
			y_bp[i] = y_bp[i] / 28.;
			z_bp[i] = z_bp[i] / 28.;
		}
		ibm->x_bp0 = x_bp; ibm->y_bp0 = y_bp; ibm->z_bp0 = z_bp;

		for (i=0; i<n_v; i++) {
			PetscReal temp;
			temp = ibm->y_bp0[i];
			ibm->y_bp0[i] = ibm->z_bp0[i];
			ibm->z_bp0[i] = -temp;
		}


		MPI_Bcast(ibm->x_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
		MPI_Bcast(ibm->y_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
		MPI_Bcast(ibm->z_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

		fscanf(fd, "%i\n", &n_elmt);
		ibm->n_elmt = n_elmt;
		MPI_Bcast(&(ibm->n_elmt), 1, MPI_INT, 0, PETSC_COMM_WORLD);

		PetscMalloc(n_elmt*sizeof(PetscInt), &nv1);
		PetscMalloc(n_elmt*sizeof(PetscInt), &nv2);
		PetscMalloc(n_elmt*sizeof(PetscInt), &nv3);

		PetscMalloc(n_elmt*sizeof(PetscReal), &nf_x);
		PetscMalloc(n_elmt*sizeof(PetscReal), &nf_y);
		PetscMalloc(n_elmt*sizeof(PetscReal), &nf_z);
		for (i=0; i<n_elmt; i++) {

			fscanf(fd, "%i %i %i\n", nv1+i, nv2+i, nv3+i);
			nv1[i] = nv1[i] - 1; nv2[i] = nv2[i]-1; nv3[i] = nv3[i] - 1;
			//      PetscPrintf(PETSC_COMM_WORLD, "I %d %d %d\n", nv1[i], nv2[i], nv3[i]);
		}
		ibm->nv1 = nv1; ibm->nv2 = nv2; ibm->nv3 = nv3;

		fclose(fd);

		for (i=0; i<n_elmt; i++) {
			n1e = nv1[i]; n2e =nv2[i]; n3e = nv3[i];
			dx12 = x_bp[n2e] - x_bp[n1e];
			dy12 = y_bp[n2e] - y_bp[n1e];
			dz12 = z_bp[n2e] - z_bp[n1e];

			dx13 = x_bp[n3e] - x_bp[n1e];
			dy13 = y_bp[n3e] - y_bp[n1e];
			dz13 = z_bp[n3e] - z_bp[n1e];

			nf_x[i] = dy12 * dz13 - dz12 * dy13;
			nf_y[i] = -dx12 * dz13 + dz12 * dx13;
			nf_z[i] = dx12 * dy13 - dy12 * dx13;

			dr = sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i] + nf_z[i]*nf_z[i]);

			nf_x[i] /=dr; nf_y[i]/=dr; nf_z[i]/=dr;
		}

		ibm->nf_x = nf_x; ibm->nf_y = nf_y;  ibm->nf_z = nf_z;

/*     for (i=0; i<n_elmt; i++) { */
/*       PetscPrintf(PETSC_COMM_WORLD, "%d %d %d %d\n", i, nv1[i], nv2[i], nv3[i]); */
/*     } */
		MPI_Bcast(ibm->nv1, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
		MPI_Bcast(ibm->nv2, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
		MPI_Bcast(ibm->nv3, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);

		MPI_Bcast(ibm->nf_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
		MPI_Bcast(ibm->nf_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
		MPI_Bcast(ibm->nf_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
	}
	else if (rank) {
		MPI_Bcast(&(n_v), 1, MPI_INT, 0, PETSC_COMM_WORLD);
		ibm->n_v = n_v;

		PetscMalloc(n_v*sizeof(PetscReal), &x_bp);
		PetscMalloc(n_v*sizeof(PetscReal), &y_bp);
		PetscMalloc(n_v*sizeof(PetscReal), &z_bp);

		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp));
		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp));
		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp));

		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp0));
		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp0));
		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp0));

		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp_o));
		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp_o));
		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp_o));

		PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->u));
		PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->uold));
/*     ibm->x_bp0 = x_bp;  ibm->y_bp0 = y_bp; ibm->z_bp0 = z_bp; */

		MPI_Bcast(ibm->x_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

		MPI_Bcast(ibm->y_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
		MPI_Bcast(ibm->z_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

		MPI_Bcast(&(n_elmt), 1, MPI_INT, 0, PETSC_COMM_WORLD);
		ibm->n_elmt = n_elmt;

		PetscMalloc(n_elmt*sizeof(PetscInt), &nv1);
		PetscMalloc(n_elmt*sizeof(PetscInt), &nv2);
		PetscMalloc(n_elmt*sizeof(PetscInt), &nv3);

		PetscMalloc(n_elmt*sizeof(PetscReal), &nf_x);
		PetscMalloc(n_elmt*sizeof(PetscReal), &nf_y);
		PetscMalloc(n_elmt*sizeof(PetscReal), &nf_z);

		ibm->nv1 = nv1; ibm->nv2 = nv2; ibm->nv3 = nv3;
		ibm->nf_x = nf_x; ibm->nf_y = nf_y; ibm->nf_z = nf_z;

		MPI_Bcast(ibm->nv1, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
		MPI_Bcast(ibm->nv2, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
		MPI_Bcast(ibm->nv3, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);

		MPI_Bcast(ibm->nf_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
		MPI_Bcast(ibm->nf_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
		MPI_Bcast(ibm->nf_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);

/*     MPI_Bcast(&(ibm->nv1), n_elmt, MPI_INTEGER, 0, PETSC_COMM_WORLD); */
/*     MPI_Bcast(&(ibm->nv2), n_elmt, MPI_INTEGER, 0, PETSC_COMM_WORLD); */
/*     MPI_Bcast(&(ibm->nv3), n_elmt, MPI_INTEGER, 0, PETSC_COMM_WORLD); */

/*     MPI_Bcast(&(ibm->nf_x), n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD); */
/*     MPI_Bcast(&(ibm->nf_y), n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD); */
/*     MPI_Bcast(&(ibm->nf_z), n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD); */
	}

/*   MPI_Barrier(PETSC_COMM_WORLD); */
	return(0);
}

PetscErrorCode ibm_read_ucd(IBMNodes *ibm)
{
	PetscInt	rank;
	PetscInt	n_v , n_elmt ;
	PetscReal	*x_bp , *y_bp , *z_bp ;
	PetscInt	*nv1 , *nv2 , *nv3 ;
	PetscReal	*nf_x, *nf_y, *nf_z;
	PetscInt	i;
	PetscInt	n1e, n2e, n3e;
	PetscReal	dx12, dy12, dz12, dx13, dy13, dz13;
	PetscReal	t, dr;
	PetscInt 	temp;
	double xt;
	char string[128];
	//MPI_Comm_size(PETSC_COMM_WORLD, &size);
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	if (!rank) { // root processor read in the data
		FILE *fd;
		fd = fopen("ibmdata", "r");
		if (!fd) SETERRQ(1, "Cannot open IBM node file")
		n_v =0;

		if (fd) {
			fgets(string, 128, fd);
			fgets(string, 128, fd);
			fgets(string, 128, fd);

			fscanf(fd, "%i %i %i %i %i\n", &n_v, &n_elmt, &temp, &temp, &temp);

			ibm->n_v = n_v;
			ibm->n_elmt = n_elmt;

			MPI_Bcast(&(ibm->n_v), 1, MPI_INT, 0, PETSC_COMM_WORLD);
			//    PetscPrintf(PETSC_COMM_WORLD, "nv, %d %e \n", n_v, xt);
			PetscMalloc(n_v*sizeof(PetscReal), &x_bp);
			PetscMalloc(n_v*sizeof(PetscReal), &y_bp);
			PetscMalloc(n_v*sizeof(PetscReal), &z_bp);

			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp));
			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp));
			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp));

			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp_o));
			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp_o));
			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp_o));

			PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->u));


			PetscReal cl = 1.;

			PetscOptionsGetReal(PETSC_NULL, "-chact_leng_valve", &cl, PETSC_NULL);

			for (i=0; i<n_v; i++) {
				fscanf(fd, "%i %le %le %le", &temp, &x_bp[i], &y_bp[i], &z_bp[i]);
				x_bp[i] = x_bp[i] / cl;
				y_bp[i] = y_bp[i] / cl;
				z_bp[i] = z_bp[i] / cl;

				ibm->x_bp[i] = x_bp[i];
				ibm->y_bp[i] = y_bp[i];
				ibm->z_bp[i] = z_bp[i];

				ibm->u[i].x = 0.;
				ibm->u[i].y = 0.;
				ibm->u[i].z = 0.;
			}
			ibm->x_bp0 = x_bp; ibm->y_bp0 = y_bp; ibm->z_bp0 = z_bp;

			MPI_Bcast(ibm->x_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
			MPI_Bcast(ibm->y_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
			MPI_Bcast(ibm->z_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

			MPI_Bcast(&(ibm->n_elmt), 1, MPI_INT, 0, PETSC_COMM_WORLD);

			PetscMalloc(n_elmt*sizeof(PetscInt), &nv1);
			PetscMalloc(n_elmt*sizeof(PetscInt), &nv2);
			PetscMalloc(n_elmt*sizeof(PetscInt), &nv3);

			PetscMalloc(n_elmt*sizeof(PetscReal), &nf_x);
			PetscMalloc(n_elmt*sizeof(PetscReal), &nf_y);
			PetscMalloc(n_elmt*sizeof(PetscReal), &nf_z);
			char str[20];
			for (i=0; i<n_elmt; i++) {

				fscanf(fd, "%i %i %s %i %i %i\n", &temp, &temp, str, nv1[i], nv2[i], nv3[i]);
				nv1[i] = nv1[i] - 1; nv2[i] = nv2[i]-1; nv3[i] = nv3[i] - 1;
			}
			ibm->nv1 = nv1; ibm->nv2 = nv2; ibm->nv3 = nv3;

			fclose(fd);
		}
		for (i=0; i<n_elmt; i++) {
			n1e = nv1[i]; n2e =nv2[i]; n3e = nv3[i];
			dx12 = x_bp[n2e] - x_bp[n1e];
			dy12 = y_bp[n2e] - y_bp[n1e];
			dz12 = z_bp[n2e] - z_bp[n1e];

			dx13 = x_bp[n3e] - x_bp[n1e];
			dy13 = y_bp[n3e] - y_bp[n1e];
			dz13 = z_bp[n3e] - z_bp[n1e];

			nf_x[i] = dy12 * dz13 - dz12 * dy13;
			nf_y[i] = -dx12 * dz13 + dz12 * dx13;
			nf_z[i] = dx12 * dy13 - dy12 * dx13;

			dr = sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i] + nf_z[i]*nf_z[i]);

			nf_x[i] /=dr; nf_y[i]/=dr; nf_z[i]/=dr;
		}

		ibm->nf_x = nf_x; ibm->nf_y = nf_y;  ibm->nf_z = nf_z;

		MPI_Bcast(ibm->nv1, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
		MPI_Bcast(ibm->nv2, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
		MPI_Bcast(ibm->nv3, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);

		MPI_Bcast(ibm->nf_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
		MPI_Bcast(ibm->nf_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
		MPI_Bcast(ibm->nf_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
	}
	else if (rank) {
		MPI_Bcast(&(n_v), 1, MPI_INT, 0, PETSC_COMM_WORLD);
		ibm->n_v = n_v;

		PetscMalloc(n_v*sizeof(PetscReal), &x_bp);
		PetscMalloc(n_v*sizeof(PetscReal), &y_bp);
		PetscMalloc(n_v*sizeof(PetscReal), &z_bp);

		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp));
		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp));
		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp));

		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp0));
		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp0));
		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp0));

		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp_o));
		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp_o));
		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp_o));

		PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->u));

/*     ibm->x_bp0 = x_bp;  ibm->y_bp0 = y_bp; ibm->z_bp0 = z_bp; */

		MPI_Bcast(ibm->x_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

		MPI_Bcast(ibm->y_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
		MPI_Bcast(ibm->z_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

		for (i=0; i<ibm->n_v; i++) {
			ibm->x_bp[i] = ibm->x_bp0[i];
			ibm->y_bp[i] = ibm->y_bp0[i];
			ibm->z_bp[i] = ibm->z_bp0[i];

			ibm->u[i].x = 0.;
			ibm->u[i].y = 0.;
			ibm->u[i].z = 0.;
		}
		MPI_Bcast(&(n_elmt), 1, MPI_INT, 0, PETSC_COMM_WORLD);
		ibm->n_elmt = n_elmt;

		PetscMalloc(n_elmt*sizeof(PetscInt), &nv1);
		PetscMalloc(n_elmt*sizeof(PetscInt), &nv2);
		PetscMalloc(n_elmt*sizeof(PetscInt), &nv3);

		PetscMalloc(n_elmt*sizeof(PetscReal), &nf_x);
		PetscMalloc(n_elmt*sizeof(PetscReal), &nf_y);
		PetscMalloc(n_elmt*sizeof(PetscReal), &nf_z);

		ibm->nv1 = nv1; ibm->nv2 = nv2; ibm->nv3 = nv3;
		ibm->nf_x = nf_x; ibm->nf_y = nf_y; ibm->nf_z = nf_z;

		MPI_Bcast(ibm->nv1, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
		MPI_Bcast(ibm->nv2, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
		MPI_Bcast(ibm->nv3, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);

		MPI_Bcast(ibm->nf_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
		MPI_Bcast(ibm->nf_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
		MPI_Bcast(ibm->nf_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);

/*     MPI_Bcast(&(ibm->nv1), n_elmt, MPI_INTEGER, 0, PETSC_COMM_WORLD); */
/*     MPI_Bcast(&(ibm->nv2), n_elmt, MPI_INTEGER, 0, PETSC_COMM_WORLD); */
/*     MPI_Bcast(&(ibm->nv3), n_elmt, MPI_INTEGER, 0, PETSC_COMM_WORLD); */

/*     MPI_Bcast(&(ibm->nf_x), n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD); */
/*     MPI_Bcast(&(ibm->nf_y), n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD); */
/*     MPI_Bcast(&(ibm->nf_z), n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD); */
	}

/*   MPI_Barrier(PETSC_COMM_WORLD); */
	return(0);
}


PetscErrorCode ibm_read_ucd_asr(IBMNodes *ibm, PetscInt ibi)
{
  PetscInt	rank;
  PetscInt	n_v , n_elmt ;
  PetscReal	*x_bp , *y_bp , *z_bp ;
  PetscInt	*nv1 , *nv2 , *nv3 ;
  PetscReal	*nf_x, *nf_y, *nf_z;
  PetscInt	i,ii;
  PetscInt	n1e, n2e, n3e;
  PetscReal	dx12, dy12, dz12, dx13, dy13, dz13;
  PetscReal     dr;
  //Added 4/1/06 iman
  PetscReal     *dA ;//area
  PetscReal	*nt_x, *nt_y, *nt_z;
  PetscReal	*ns_x, *ns_y, *ns_z;

  char   ss[20];
  //double xt;
  char string[128];

  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if(!rank) { // root processor read in the data
    FILE *fd;
    PetscPrintf(PETSC_COMM_SELF, "READ ibmdata\n");
    char filen[80];  
    sprintf(filen,"ibmdata%2.2d" , ibi);
 
    fd = fopen(filen, "r"); if (!fd) SETERRQ(1, "Cannot open IBM node file")
    n_v =0;

    if (fd) {
      fgets(string, 128, fd);
      fgets(string, 128, fd);
      fgets(string, 128, fd);
      
      fscanf(fd, "%i %i %i %i %i",&n_v,&n_elmt,&ii,&ii,&ii);
      PetscPrintf(PETSC_COMM_SELF, "number of nodes & elements %d %d\n",n_v, n_elmt);
      
      ibm->n_v = n_v;
      ibm->n_elmt = n_elmt;      
      
      MPI_Bcast(&(ibm->n_v), 1, MPI_INT, 0, PETSC_COMM_WORLD);
      //    PetscPrintf(PETSC_COMM_WORLD, "nv, %d %e \n", n_v, xt);
	    
	    /*
      PetscMalloc(n_v*sizeof(PetscReal), &x_bp);	// removed by seokkoo 03.04.2009
      PetscMalloc(n_v*sizeof(PetscReal), &y_bp);
      PetscMalloc(n_v*sizeof(PetscReal), &z_bp);
      */
      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp));	// added by seokkoo 03.04.2009
      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp));
      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp));
      
      x_bp = ibm->x_bp;	// seokkoo
      y_bp = ibm->y_bp;
      z_bp = ibm->z_bp;
      
      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp_o));
      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp_o));
      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp_o));

      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp0));
      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp0));
      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp0));
      
      PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->u));
      PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->uold));
      PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->urm1));

      for (i=0; i<n_v; i++) {
		fscanf(fd, "%i %le %le %le", &ii, &x_bp[i], &y_bp[i], &z_bp[i]);//, &t, &t, &t);
		
		ibm->x_bp[i] = x_bp[i];
		ibm->y_bp[i] = y_bp[i];
		ibm->z_bp[i] = z_bp[i];

		ibm->x_bp0[i] = x_bp[i];
		ibm->y_bp0[i] = y_bp[i];
		ibm->z_bp0[i] = z_bp[i];

		ibm->x_bp_o[i] = x_bp[i];
		ibm->y_bp_o[i] = y_bp[i];
		ibm->z_bp_o[i] = z_bp[i];

		ibm->u[i].x = 0.;
		ibm->u[i].y = 0.;
		ibm->u[i].z = 0.;

		ibm->uold[i].x = 0.;
		ibm->uold[i].y = 0.;
		ibm->uold[i].z = 0.;

		ibm->urm1[i].x = 0.;
		ibm->urm1[i].y = 0.;
		ibm->urm1[i].z = 0.;
      }
      i=0;
      PetscPrintf(PETSC_COMM_WORLD, "xyz_bp %le %le %le\n", x_bp[i], y_bp[i], z_bp[i]);

/*       ibm->x_bp0 = x_bp; ibm->y_bp0 = y_bp; ibm->z_bp0 = z_bp; */
/*       ibm->x_bp_o = x_bp; ibm->y_bp_o = y_bp; ibm->z_bp_o = z_bp; */

      MPI_Bcast(ibm->x_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      MPI_Bcast(ibm->y_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      MPI_Bcast(ibm->z_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

      MPI_Bcast(ibm->x_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      MPI_Bcast(ibm->y_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      MPI_Bcast(ibm->z_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      
      MPI_Bcast(ibm->x_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      MPI_Bcast(ibm->y_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      MPI_Bcast(ibm->z_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

      MPI_Bcast(&(ibm->n_elmt), 1, MPI_INT, 0, PETSC_COMM_WORLD);

      PetscMalloc(n_elmt*sizeof(PetscInt), &nv1);
      PetscMalloc(n_elmt*sizeof(PetscInt), &nv2);
      PetscMalloc(n_elmt*sizeof(PetscInt), &nv3);
      
      PetscMalloc(n_elmt*sizeof(PetscReal), &nf_x);
      PetscMalloc(n_elmt*sizeof(PetscReal), &nf_y);
      PetscMalloc(n_elmt*sizeof(PetscReal), &nf_z);
      
      // Added 4/1/06 iman
      PetscMalloc(n_elmt*sizeof(PetscReal), &dA); //Area

      PetscMalloc(n_elmt*sizeof(PetscReal), &nt_x);
      PetscMalloc(n_elmt*sizeof(PetscReal), &nt_y);
      PetscMalloc(n_elmt*sizeof(PetscReal), &nt_z);

      PetscMalloc(n_elmt*sizeof(PetscReal), &ns_x);
      PetscMalloc(n_elmt*sizeof(PetscReal), &ns_y);
      PetscMalloc(n_elmt*sizeof(PetscReal), &ns_z);
      
      // Added 6/4/06 iman
      //PetscMalloc(n_elmt*sizeof(Cmpnts), &(ibm->cent));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_x));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_y));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_z));
      // end added
      
      
	//seokkoo begin
	{	//only for rank 0
		PetscMalloc(n_elmt*sizeof(PetscInt), &ibm->_nv1);
		PetscMalloc(n_elmt*sizeof(PetscInt), &ibm->_nv2);
		PetscMalloc(n_elmt*sizeof(PetscInt), &ibm->_nv3);
		
		PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->_x_bp);
		PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->_y_bp);
		PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->_z_bp);
	}
	
	PetscMalloc(n_elmt*sizeof(PetscInt), &ibm->count);
	PetscMalloc(n_elmt*sizeof(PetscInt), &ibm->local2global_elmt);
	
	PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->shear);
	PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->mean_shear);
	PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->reynolds_stress1);
	PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->reynolds_stress2);
	PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->reynolds_stress3);
	PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->pressure);
	//seokkoo end

      for (i=0; i<n_elmt; i++) {

	fscanf(fd, "%i %i %s %i %i %i\n", &ii,&ii, &ss, nv1+i, nv2+i, nv3+i);
	nv1[i] = nv1[i] - 1; nv2[i] = nv2[i]-1; nv3[i] = nv3[i] - 1;
	      
		// seokkoo
	      ibm->_nv1[i] = nv1[i];
	      ibm->_nv2[i] = nv2[i];
	      ibm->_nv3[i] = nv3[i];
	      // seokkoo
	      ibm->_x_bp[i] = ibm->x_bp[i];
	      ibm->_y_bp[i] = ibm->y_bp[i];
	      ibm->_z_bp[i] = ibm->z_bp[i];

      }
      ibm->nv1 = nv1; ibm->nv2 = nv2; ibm->nv3 = nv3;

      i=0;
      PetscPrintf(PETSC_COMM_WORLD, "nv %d %d %d\n", nv1[i], nv2[i], nv3[i]);

      fclose(fd);
    }
      
    for (i=0; i<n_elmt; i++) {
      
      n1e = nv1[i]; n2e =nv2[i]; n3e = nv3[i];
      dx12 = x_bp[n2e] - x_bp[n1e];
      dy12 = y_bp[n2e] - y_bp[n1e];
      dz12 = z_bp[n2e] - z_bp[n1e];
      
      dx13 = x_bp[n3e] - x_bp[n1e];
      dy13 = y_bp[n3e] - y_bp[n1e];
      dz13 = z_bp[n3e] - z_bp[n1e];
      
      nf_x[i] = dy12 * dz13 - dz12 * dy13;
      nf_y[i] = -dx12 * dz13 + dz12 * dx13;
      nf_z[i] = dx12 * dy13 - dy12 * dx13;
      
      dr = sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i] + nf_z[i]*nf_z[i]);
      
      nf_x[i] /=dr; nf_y[i]/=dr; nf_z[i]/=dr;
      
      // Temp sol. 2D
/*       if (fabs(nf_x[i])<.5) */
/* 	nf_x[i]=0.; */

      // Addedd 4/2/06 iman
      if ((((1.-nf_z[i])<=1e-6 )&((-1.+nf_z[i])<1e-6))|
	  (((nf_z[i]+1.)<=1e-6 )&((-1.-nf_z[i])<1e-6))) {
	ns_x[i] = 1.;     
	ns_y[i] = 0.;     
	ns_z[i] = 0. ;
	
	nt_x[i] = 0.;
	nt_y[i] = 1.;
	nt_z[i] = 0.;
      } else {
	ns_x[i] =  nf_y[i]/ sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i]);      
	ns_y[i] = -nf_x[i]/ sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i]);     
	ns_z[i] = 0. ;
	
	nt_x[i] = -nf_x[i]*nf_z[i]/ sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i]);
	nt_y[i] = -nf_y[i]*nf_z[i]/ sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i]);
	nt_z[i] = sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i]);
      }
      
      //Added 4/1/06 iman
      dA[i] = dr/2.; 
      
      // Added 6/4/06 iman
      // Calc the center of the element
      ibm->cent_x[i]= (x_bp[n1e]+x_bp[n2e]+x_bp[n3e])/3.;
      ibm->cent_y[i]= (y_bp[n1e]+y_bp[n2e]+y_bp[n3e])/3.;
      ibm->cent_z[i]= (z_bp[n1e]+z_bp[n2e]+z_bp[n3e])/3.;	
    }
    
    
    ibm->nf_x = nf_x; ibm->nf_y = nf_y;  ibm->nf_z = nf_z;
    
    //Added 4/1/06 iman
    ibm->dA = dA;
    ibm->nt_x = nt_x; ibm->nt_y = nt_y;  ibm->nt_z = nt_z;
    ibm->ns_x = ns_x; ibm->ns_y = ns_y;  ibm->ns_z = ns_z;    
    
    MPI_Bcast(ibm->nv1, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nv2, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nv3, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
    
    MPI_Bcast(ibm->nf_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nf_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nf_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
      
    // Added 4/1/06 iman
    MPI_Bcast(ibm->dA, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
    // Added 4/2/06 iman
    MPI_Bcast(ibm->nt_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nt_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nt_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
    MPI_Bcast(ibm->ns_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->ns_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->ns_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
    // Added 6/4/06 iman
    MPI_Bcast(ibm->cent_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->cent_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->cent_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);

 /*    PetscFree(dA); */
/*     PetscFree(nf_x);PetscFree(nf_y);PetscFree(nf_z); */
/*     PetscFree(nt_x);PetscFree(nt_y);PetscFree(nt_z); */
/*     PetscFree(ns_x);PetscFree(ns_y);PetscFree(ns_z); */
/*     PetscFree(nv1);PetscFree(nv2);PetscFree(nv3); */
/*     PetscFree(x_bp);PetscFree(y_bp);PetscFree(z_bp); */
    PetscInt ti=0;
    FILE *f;
    //char filen[80];
    sprintf(filen, "post_surface%3.3d_%2.2d_nf.dat",ti,ibi);
    f = fopen(filen, "w");
	PetscFPrintf(PETSC_COMM_WORLD, f, "TITLE=\"3D TRIANGULAR SURFACE DATA\"\n");	
    PetscFPrintf(PETSC_COMM_WORLD, f, "Variables=\"x\",\"y\",\"z\",\"n_x\",\"n_y\",\"n_z\",\"nt_x\",\"nt_y\",\"nt_z\",\"ns_x\",\"ns_y\",\"ns_z\n");
    PetscFPrintf(PETSC_COMM_WORLD, f, "ZONE T='TRIANGLES', N=%d, E=%d, F=FEBLOCK, ET=TRIANGLE, VARLOCATION=([1-3]=NODAL,[4-12]=CELLCENTERED)\n", n_v, n_elmt);
    for (i=0; i<n_v; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->x_bp[i]);
    }
    for (i=0; i<n_v; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->y_bp[i]);
    }
    for (i=0; i<n_v; i++) {	
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->z_bp[i]);
    }
    for (i=0; i<n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nf_x[i]);
    }
    for (i=0; i<n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nf_y[i]);
    }
    for (i=0; i<n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nf_z[i]);
    }
    for (i=0; i<n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nt_x[i]);
    }
    for (i=0; i<n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nt_y[i]);
    }
    for (i=0; i<n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nt_z[i]);
    }
    for (i=0; i<n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->ns_x[i]);
    }
    for (i=0; i<n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->ns_y[i]);
    }
    for (i=0; i<n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->ns_z[i]);
    }
    for (i=0; i<n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%d %d %d\n", ibm->nv1[i]+1, ibm->nv2[i]+1, ibm->nv3[i]+1);
    }
    
/*     PetscFPrintf(PETSC_COMM_WORLD, f, "Variables=x,y,z\n"); */
/*     PetscFPrintf(PETSC_COMM_WORLD, f, "ZONE T='TRIANGLES', N=%d, E=%d, F=FEPOINT, ET=TRIANGLE\n", n_v, n_elmt); */
/*     for (i=0; i<n_v; i++) { */
      
/*       PetscFPrintf(PETSC_COMM_WORLD, f, "%e %e %e\n", ibm->x_bp[i], ibm->y_bp[i], ibm->z_bp[i]); */
/*     } */
/*     for (i=0; i<n_elmt; i++) { */
/*       PetscFPrintf(PETSC_COMM_WORLD, f, "%d %d %d\n", ibm->nv1[i]+1, ibm->nv2[i]+1, ibm->nv3[i]+1); */
/*     } */
    fclose(f);

  }
  else if (rank) {
    MPI_Bcast(&(n_v), 1, MPI_INT, 0, PETSC_COMM_WORLD);
    ibm->n_v = n_v;
    /*
    PetscMalloc(n_v*sizeof(PetscReal), &x_bp);	// removed by seokkoo 03.04.2009
    PetscMalloc(n_v*sizeof(PetscReal), &y_bp);
    PetscMalloc(n_v*sizeof(PetscReal), &z_bp);
    */
	
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp));
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp));
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp));
	  
	x_bp = ibm->x_bp;	// added by seokkoo 03.04.2009
	y_bp = ibm->y_bp;
	z_bp = ibm->z_bp;
	  
    
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp0));
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp0));
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp0));
    
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp_o));
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp_o));
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp_o));
    
    PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->u));
    PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->uold));
    PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->urm1));

    for (i=0; i<n_v; i++) {
      ibm->u[i].x = 0.;
      ibm->u[i].y = 0.;
      ibm->u[i].z = 0.;

      ibm->uold[i].x = 0.;
      ibm->uold[i].y = 0.;
      ibm->uold[i].z = 0.;
      
      ibm->urm1[i].x = 0.;
      ibm->urm1[i].y = 0.;
      ibm->urm1[i].z = 0.;      
    }
        
    MPI_Bcast(ibm->x_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);    
    MPI_Bcast(ibm->y_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->z_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

    MPI_Bcast(ibm->x_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->y_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->z_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

    MPI_Bcast(ibm->x_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->y_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->z_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
    MPI_Bcast(&(n_elmt), 1, MPI_INT, 0, PETSC_COMM_WORLD);
    ibm->n_elmt = n_elmt;

    PetscMalloc(n_elmt*sizeof(PetscInt), &nv1);
    PetscMalloc(n_elmt*sizeof(PetscInt), &nv2);
    PetscMalloc(n_elmt*sizeof(PetscInt), &nv3);

    PetscMalloc(n_elmt*sizeof(PetscReal), &nf_x);
    PetscMalloc(n_elmt*sizeof(PetscReal), &nf_y);
    PetscMalloc(n_elmt*sizeof(PetscReal), &nf_z);

    //Added 4/1/06 iman
    PetscMalloc(n_elmt*sizeof(PetscReal), &dA);

    //Added 4/2/06 iman
    PetscMalloc(n_elmt*sizeof(PetscReal), &nt_x);
    PetscMalloc(n_elmt*sizeof(PetscReal), &nt_y);
    PetscMalloc(n_elmt*sizeof(PetscReal), &nt_z);

    PetscMalloc(n_elmt*sizeof(PetscReal), &ns_x);
    PetscMalloc(n_elmt*sizeof(PetscReal), &ns_y);
    PetscMalloc(n_elmt*sizeof(PetscReal), &ns_z);

    ibm->nv1 = nv1; ibm->nv2 = nv2; ibm->nv3 = nv3;
    ibm->nf_x = nf_x; ibm->nf_y = nf_y; ibm->nf_z = nf_z;
    
    // Added 4/2/06 iman
    ibm->dA = dA;
    ibm->nt_x = nt_x; ibm->nt_y = nt_y;  ibm->nt_z = nt_z;
    ibm->ns_x = ns_x; ibm->ns_y = ns_y;  ibm->ns_z = ns_z;    

    // Added 6/4/06
    //PetscMalloc(n_elmt*sizeof(Cmpnts), &(ibm->cent));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_x));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_y));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_z));
    
		//seokkoo
		PetscMalloc(n_elmt*sizeof(PetscInt), &ibm->count);
		PetscMalloc(n_elmt*sizeof(PetscInt), &ibm->local2global_elmt);
		//seokkoo
		PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->shear);
		PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->mean_shear);
		PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->reynolds_stress1);
		PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->reynolds_stress2);
		PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->reynolds_stress3);
		PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->pressure);

    MPI_Bcast(ibm->nv1, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nv2, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nv3, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);

    MPI_Bcast(ibm->nf_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nf_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nf_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
    //Added 4/2/06 iman
    MPI_Bcast(ibm->dA, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);

    MPI_Bcast(ibm->nt_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nt_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nt_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
    MPI_Bcast(ibm->ns_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->ns_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->ns_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);

    MPI_Bcast(ibm->cent_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->cent_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->cent_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
  }
	PetscPrintf(PETSC_COMM_WORLD, "Read ucd file !\n");
  return(0);
}


PetscErrorCode Combine_Elmt(IBMNodes *ibm, IBMNodes *ibm0, IBMNodes *ibm1)
{

	PetscInt i;

	ibm->n_v = ibm0->n_v + ibm1->n_v;
	ibm->n_elmt = ibm0->n_elmt + ibm1->n_elmt;

	PetscInt n_v = ibm->n_v, n_elmt = ibm->n_elmt;

	for (i=0; i<ibm0->n_v; i++) {
		ibm->x_bp[i] = ibm0->x_bp[i];
		ibm->y_bp[i] = ibm0->y_bp[i];
		ibm->z_bp[i] = ibm0->z_bp[i];

		ibm->u[i] = ibm0->u[i];
		ibm->uold[i] = ibm0->uold[i];
		//    ibm->u[i].x = 0.;
/*     PetscPrintf(PETSC_COMM_WORLD, "Vel %e %e %e\n", ibm->u[i].x, ibm->u[i].y, ibm->u[i].z); */
	}
	for (i=0; i<ibm0->n_elmt; i++) {
		ibm->nv1[i] = ibm0->nv1[i];
		ibm->nv2[i] = ibm0->nv2[i];
		ibm->nv3[i] = ibm0->nv3[i];

		ibm->nf_x[i] = ibm0->nf_x[i];
		ibm->nf_y[i] = ibm0->nf_y[i];
		ibm->nf_z[i] = ibm0->nf_z[i];
	}

	for (i=ibm0->n_v; i<n_v; i++) {
		ibm->x_bp[i] = ibm1->x_bp[i-ibm0->n_v];
		ibm->y_bp[i] = ibm1->y_bp[i-ibm0->n_v];
		ibm->z_bp[i] = ibm1->z_bp[i-ibm0->n_v];
		ibm->u[i].x = 0.;
		ibm->u[i].y = 0.;
		ibm->u[i].z = 0.;
	}

	for (i=ibm0->n_elmt; i<n_elmt; i++) {
		ibm->nv1[i] = ibm1->nv1[i-ibm0->n_elmt] + ibm0->n_v;
		ibm->nv2[i] = ibm1->nv2[i-ibm0->n_elmt] + ibm0->n_v;
		ibm->nv3[i] = ibm1->nv3[i-ibm0->n_elmt] + ibm0->n_v;

		ibm->nf_x[i] = ibm1->nf_x[i-ibm0->n_elmt];
		ibm->nf_y[i] = ibm1->nf_y[i-ibm0->n_elmt];
		ibm->nf_z[i] = ibm1->nf_z[i-ibm0->n_elmt];
	}

	PetscInt rank;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
	if (!rank) {
		if (ti == ti) {
			FILE *f;
			char filen[80];
			sprintf(filen, "surface%06d.dat",ti);
			f = fopen(filen, "w");
			PetscFPrintf(PETSC_COMM_WORLD, f, "Variables=x,y,z\n");
			PetscFPrintf(PETSC_COMM_WORLD, f, "ZONE T='TRIANGLES', N=%d, E=%d, F=FEPOINT, ET=TRIANGLE\n", n_v, 1670-96);
			for (i=0; i<n_v; i++) {

	/*    ibm->x_bp[i] = ibm->x_bp0[i];
				ibm->y_bp[i] = ibm->y_bp0[i];
				ibm->z_bp[i] = ibm->z_bp0[i] + z0;*/
				PetscFPrintf(PETSC_COMM_WORLD, f, "%e %e %e\n", ibm->x_bp[i], ibm->y_bp[i], ibm->z_bp[i]);
			}
			for (i=96; i<n_elmt; i++) {
				if (fabs(ibm->nf_z[i]) > 0.5 ||
						(fabs(ibm->nf_z[i]) < 0.5 &&
						 (ibm->x_bp[ibm->nv1[i]] * ibm->x_bp[ibm->nv1[i]] +
							ibm->y_bp[ibm->nv1[i]] * ibm->y_bp[ibm->nv1[i]]) < 0.44*0.44)) {
					PetscFPrintf(PETSC_COMM_WORLD, f, "%d %d %d\n", ibm->nv1[i]+1, ibm->nv2[i]+1, ibm->nv3[i]+1);
				}
			}
			fclose(f);

			sprintf(filen, "leaflet%06d.dat",ti);
			f = fopen(filen, "w");
			PetscFPrintf(PETSC_COMM_WORLD, f, "Variables=x,y,z\n");
			PetscFPrintf(PETSC_COMM_WORLD, f, "ZONE T='TRIANGLES', N=%d, E=%d, F=FEPOINT, ET=TRIANGLE\n", n_v, 96);
			for (i=0; i<n_v; i++) {

	/*    ibm->x_bp[i] = ibm->x_bp0[i];
				ibm->y_bp[i] = ibm->y_bp0[i];
				ibm->z_bp[i] = ibm->z_bp0[i] + z0;*/
				PetscFPrintf(PETSC_COMM_WORLD, f, "%e %e %e\n", ibm->x_bp[i], ibm->y_bp[i], ibm->z_bp[i]);
			}
			for (i=0; i<96; i++) {
				if (fabs(ibm->nf_z[i]) > 0.5 ||
						(fabs(ibm->nf_z[i]) < 0.5 &&
						 (ibm->x_bp[ibm->nv1[i]] * ibm->x_bp[ibm->nv1[i]] +
							ibm->y_bp[ibm->nv1[i]] * ibm->y_bp[ibm->nv1[i]]) < 0.44*0.44)) {
					PetscFPrintf(PETSC_COMM_WORLD, f, "%d %d %d\n", ibm->nv1[i]+1, ibm->nv2[i]+1, ibm->nv3[i]+1);
				}
			}
			fclose(f);

		}
	}

	return 0;
}

PetscErrorCode Elmt_Move(IBMNodes *ibm, UserCtx *user)
{
	PetscInt n_v = ibm->n_v, n_elmt = ibm->n_elmt;

	PetscReal rcx = -0.122, rcz = -0.32, z0 = 4.52;
	rcx = -0.09450115; rcz = -0.3141615; z0 = 4.47;
	PetscReal dz;
	dz = -0.031;
	rcz = rcz-dz;
	rcx = rcx - dz * sin(10./180.*3.1415926);
	PetscReal temp;
	PetscInt i;

	PetscInt n1e, n2e, n3e;
	PetscReal dx12, dy12, dz12, dx13, dy13, dz13, dr;
	for (i=0; i<n_v; i++) {
		ibm->x_bp_o[i] = ibm->x_bp[i];
		ibm->y_bp_o[i] = ibm->y_bp[i];
		ibm->z_bp_o[i] = ibm->z_bp[i];
	}

	angle =-angle * 3.1415926/180.;
	//angle = 0;
	for (i=0; i<n_v/2; i++) {
		ibm->x_bp[i] = (ibm->x_bp0[i] -0.01- rcx) * cos(angle) - (ibm->z_bp0[i] - rcz) * sin(angle) + rcx;
		ibm->y_bp[i] = ibm->y_bp0[i];
		ibm->z_bp[i] = (ibm->x_bp0[i] -0.01- rcx) * sin(angle) + (ibm->z_bp0[i] - rcz) * cos(angle) + z0 + rcz;

	}
	rcx = -rcx;
	for (i=n_v/2; i<n_v; i++) {
		ibm->x_bp[i] = (ibm->x_bp0[i] +0.01- rcx) * cos(-angle) - (ibm->z_bp0[i] - rcz) * sin(-angle) + rcx;
		ibm->y_bp[i] = ibm->y_bp0[i];
		ibm->z_bp[i] = (ibm->x_bp0[i] +0.01- rcx) * sin(-angle) + (ibm->z_bp0[i] - rcz) * cos(-angle) + z0 + rcz;
	}

	/* Rotate 90 degree */
	for (i=0; i<n_v; i++) {
		temp = ibm->y_bp[i];
		ibm->y_bp[i] = ibm->x_bp[i];
		ibm->x_bp[i] = temp;
	}
	for (i=0; i<n_elmt; i++) {
			n1e = ibm->nv1[i]; n2e =ibm->nv2[i]; n3e =ibm->nv3[i];
			dx12 = ibm->x_bp[n2e] - ibm->x_bp[n1e];
			dy12 = ibm->y_bp[n2e] - ibm->y_bp[n1e];
			dz12 = ibm->z_bp[n2e] - ibm->z_bp[n1e];

			dx13 = ibm->x_bp[n3e] - ibm->x_bp[n1e];
			dy13 = ibm->y_bp[n3e] - ibm->y_bp[n1e];
			dz13 = ibm->z_bp[n3e] - ibm->z_bp[n1e];

			ibm->nf_x[i] = dy12 * dz13 - dz12 * dy13;
			ibm->nf_y[i] = -dx12 * dz13 + dz12 * dx13;
			ibm->nf_z[i] = dx12 * dy13 - dy12 * dx13;

			dr = sqrt(ibm->nf_x[i]*ibm->nf_x[i] +
								ibm->nf_y[i]*ibm->nf_y[i] +
								ibm->nf_z[i]*ibm->nf_z[i]);

			ibm->nf_x[i] /=dr; ibm->nf_y[i]/=dr; ibm->nf_z[i]/=dr;

/*       PetscPrintf(PETSC_COMM_WORLD, "NFZ %d %d %d %d %e\n", i, ibm->nv1[i], ibm->nv2[i], ibm->nv3[i], ibm->nf_z[i]); */
			//      PetscPrintf(PETSC_COMM_WORLD, "%le %le %le %le %le %le\n", x_bp[n1e], y_bp[n1e], ibm->x_bp0[n1e], ibm->y_bp0[n1e], x_bp[n3e], y_bp[n3e]);

	}
	if (ti>0) {
		for (i=0; i<n_v; i++) {
			//      ibm->uold[i] = ibm->u[i];

			ibm->u[i].x = (ibm->x_bp[i] - ibm->x_bp_o[i]) / user->dt;
			ibm->u[i].y = (ibm->y_bp[i] - ibm->y_bp_o[i]) / user->dt;
			ibm->u[i].z = (ibm->z_bp[i] - ibm->z_bp_o[i]) / user->dt;
		}
	}
	else {
		for (i=0; i<n_v; i++) {
			ibm->u[i].x = 0.;
			ibm->u[i].y = 0.;
			ibm->u[i].z = 0.;
		}
	}
	return 0;
}

PetscErrorCode Elmt_Move1(IBMNodes *ibm, UserCtx *user)
{
	PetscInt n_v = ibm->n_v, n_elmt = ibm->n_elmt;

	PetscReal rcx = -0.122, rcz = -0.32, z0 = 4.52;
	rcx = -0.09450115; rcz = -0.3141615; z0 = 4.47;
	PetscReal dz;
	dz = -0.031;
	rcz = rcz-dz;
	rcx = rcx - dz * sin(10./180.*3.1415926);
	PetscReal temp;
	PetscInt i;

	PetscInt n1e, n2e, n3e;
	PetscReal dx12, dy12, dz12, dx13, dy13, dz13, dr;
	for (i=0; i<n_v; i++) {
		ibm->x_bp_o[i] = ibm->x_bp[i];
		ibm->y_bp_o[i] = ibm->y_bp[i];
		ibm->z_bp_o[i] = ibm->z_bp[i];
	}

	angle =-angle * 3.1415926/180.;
	//angle = 0;
	for (i=0; i<n_v/2; i++) {
		ibm->x_bp[i] = (ibm->x_bp0[i] -0.01- rcx) * cos(angle) - (ibm->z_bp0[i] - rcz) * sin(angle) + rcx;
		ibm->y_bp[i] = ibm->y_bp0[i];
		ibm->z_bp[i] = (ibm->x_bp0[i] -0.01- rcx) * sin(angle) + (ibm->z_bp0[i] - rcz) * cos(angle) + z0 + rcz;

	}
	rcx = -rcx;
	for (i=n_v/2; i<n_v; i++) {
		ibm->x_bp[i] = (ibm->x_bp0[i] +0.01- rcx) * cos(-angle) - (ibm->z_bp0[i] - rcz) * sin(-angle) + rcx;
		ibm->y_bp[i] = ibm->y_bp0[i];
		ibm->z_bp[i] = (ibm->x_bp0[i] +0.01- rcx) * sin(-angle) + (ibm->z_bp0[i] - rcz) * cos(-angle) + z0 + rcz;
	}

	/* Rotate 90 degree */
	for (i=0; i<n_v; i++) {
		temp = ibm->y_bp[i];
		ibm->y_bp[i] = ibm->x_bp[i];
		ibm->x_bp[i] = temp;
	}
	for (i=0; i<n_elmt; i++) {
			n1e = ibm->nv1[i]; n2e =ibm->nv2[i]; n3e =ibm->nv3[i];
			dx12 = ibm->x_bp[n2e] - ibm->x_bp[n1e];
			dy12 = ibm->y_bp[n2e] - ibm->y_bp[n1e];
			dz12 = ibm->z_bp[n2e] - ibm->z_bp[n1e];

			dx13 = ibm->x_bp[n3e] - ibm->x_bp[n1e];
			dy13 = ibm->y_bp[n3e] - ibm->y_bp[n1e];
			dz13 = ibm->z_bp[n3e] - ibm->z_bp[n1e];

			ibm->nf_x[i] = dy12 * dz13 - dz12 * dy13;
			ibm->nf_y[i] = -dx12 * dz13 + dz12 * dx13;
			ibm->nf_z[i] = dx12 * dy13 - dy12 * dx13;

			dr = sqrt(ibm->nf_x[i]*ibm->nf_x[i] +
								ibm->nf_y[i]*ibm->nf_y[i] +
								ibm->nf_z[i]*ibm->nf_z[i]);

			ibm->nf_x[i] /=dr; ibm->nf_y[i]/=dr; ibm->nf_z[i]/=dr;

/*       PetscPrintf(PETSC_COMM_WORLD, "NFZ %d %d %d %d %e\n", i, ibm->nv1[i], ibm->nv2[i], ibm->nv3[i], ibm->nf_z[i]); */
			//      PetscPrintf(PETSC_COMM_WORLD, "%le %le %le %le %le %le\n", x_bp[n1e], y_bp[n1e], ibm->x_bp0[n1e], ibm->y_bp0[n1e], x_bp[n3e], y_bp[n3e]);

	}
	if (ti>0) {
		for (i=0; i<n_v; i++) {
			ibm->uold[i] = ibm->u[i];

			ibm->u[i].x = (ibm->x_bp[i] - ibm->x_bp_o[i]) / user->dt;
			ibm->u[i].y = (ibm->y_bp[i] - ibm->y_bp_o[i]) / user->dt;
			ibm->u[i].z = (ibm->z_bp[i] - ibm->z_bp_o[i]) / user->dt;
		}
	}
	else {
		for (i=0; i<n_v; i++) {
			ibm->u[i].x = 0.;
			ibm->u[i].y = 0.;
			ibm->u[i].z = 0.;
		}
	}
	return 0;
}


/*****************************************************************/

#define n 3

static double hypot2(double x, double y) {
	return sqrt(x*x+y*y);
}

// Symmetric Householder reduction to tridiagonal form.

static void tred2(double V[n][n], double d[n], double e[n]) {

//  This is derived from the Algol procedures tred2 by
//  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
//  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
//  Fortran subroutine in EISPACK.

	for (int j = 0; j < n; j++) {
		d[j] = V[n-1][j];
	}

	// Householder reduction to tridiagonal form.

	for (int i = n-1; i > 0; i--) {

		// Scale to avoid under/overflow.

		double scale = 0.0;
		double h = 0.0;
		for (int k = 0; k < i; k++) {
			scale = scale + fabs(d[k]);
		}
		if (scale == 0.0) {
			e[i] = d[i-1];
			for (int j = 0; j < i; j++) {
				d[j] = V[i-1][j];
				V[i][j] = 0.0;
				V[j][i] = 0.0;
			}
		} else {

			// Generate Householder vector.

			for (int k = 0; k < i; k++) {
				d[k] /= scale;
				h += d[k] * d[k];
			}
			double f = d[i-1];
			double g = sqrt(h);
			if (f > 0) {
				g = -g;
			}
			e[i] = scale * g;
			h = h - f * g;
			d[i-1] = f - g;
			for (int j = 0; j < i; j++) {
				e[j] = 0.0;
			}

			// Apply similarity transformation to remaining columns.

			for (int j = 0; j < i; j++) {
				f = d[j];
				V[j][i] = f;
				g = e[j] + V[j][j] * f;
				for (int k = j+1; k <= i-1; k++) {
					g += V[k][j] * d[k];
					e[k] += V[k][j] * f;
				}
				e[j] = g;
			}
			f = 0.0;
			for (int j = 0; j < i; j++) {
				e[j] /= h;
				f += e[j] * d[j];
			}
			double hh = f / (h + h);
			for (int j = 0; j < i; j++) {
				e[j] -= hh * d[j];
			}
			for (int j = 0; j < i; j++) {
				f = d[j];
				g = e[j];
				for (int k = j; k <= i-1; k++) {
					V[k][j] -= (f * e[k] + g * d[k]);
				}
				d[j] = V[i-1][j];
				V[i][j] = 0.0;
			}
		}
		d[i] = h;
	}

	// Accumulate transformations.

	for (int i = 0; i < n-1; i++) {
		V[n-1][i] = V[i][i];
		V[i][i] = 1.0;
		double h = d[i+1];
		if (h != 0.0) {
			for (int k = 0; k <= i; k++) {
				d[k] = V[k][i+1] / h;
			}
			for (int j = 0; j <= i; j++) {
				double g = 0.0;
				for (int k = 0; k <= i; k++) {
					g += V[k][i+1] * V[k][j];
				}
				for (int k = 0; k <= i; k++) {
					V[k][j] -= g * d[k];
				}
			}
		}
		for (int k = 0; k <= i; k++) {
			V[k][i+1] = 0.0;
		}
	}
	for (int j = 0; j < n; j++) {
		d[j] = V[n-1][j];
		V[n-1][j] = 0.0;
	}
	V[n-1][n-1] = 1.0;
	e[0] = 0.0;
}

// Symmetric tridiagonal QL algorithm.

static void tql2(double V[n][n], double d[n], double e[n]) {

//  This is derived from the Algol procedures tql2, by
//  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
//  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
//  Fortran subroutine in EISPACK.

	for (int i = 1; i < n; i++) {
		e[i-1] = e[i];
	}
	e[n-1] = 0.0;

	double f = 0.0;
	double tst1 = 0.0;
	double eps = pow(2.0,-52.0);
	for (int l = 0; l < n; l++) {

		// Find small subdiagonal element

		tst1 = max(tst1,fabs(d[l]) + fabs(e[l]));
		int m = l;
		while (m < n) {
			if (fabs(e[m]) <= eps*tst1) {
				break;
			}
			m++;
		}

		// If m == l, d[l] is an eigenvalue,
		// otherwise, iterate.

		if (m > l) {
			int iter = 0;
			do {
				iter = iter + 1;  // (Could check iteration count here.)

				// Compute implicit shift

				double g = d[l];
				double p = (d[l+1] - g) / (2.0 * e[l]);
				double r = hypot2(p,1.0);
				if (p < 0) {
					r = -r;
				}
				d[l] = e[l] / (p + r);
				d[l+1] = e[l] * (p + r);
				double dl1 = d[l+1];
				double h = g - d[l];
				for (int i = l+2; i < n; i++) {
					d[i] -= h;
				}
				f = f + h;

				// Implicit QL transformation.

				p = d[m];
				double c = 1.0;
				double c2 = c;
				double c3 = c;
				double el1 = e[l+1];
				double s = 0.0;
				double s2 = 0.0;
				for (int i = m-1; i >= l; i--) {
					c3 = c2;
					c2 = c;
					s2 = s;
					g = c * e[i];
					h = c * p;
					r = hypot2(p,e[i]);
					e[i+1] = s * r;
					s = e[i] / r;
					c = p / r;
					p = c * d[i] - s * g;
					d[i+1] = h + s * (c * g + s * d[i]);

					// Accumulate transformation.

					for (int k = 0; k < n; k++) {
						h = V[k][i+1];
						V[k][i+1] = s * V[k][i] + c * h;
						V[k][i] = c * V[k][i] - s * h;
					}
				}
				p = -s * s2 * c3 * el1 * e[l] / dl1;
				e[l] = s * p;
				d[l] = c * p;

				// Check for convergence.

			} while (fabs(e[l]) > eps*tst1);
		}
		d[l] = d[l] + f;
		e[l] = 0.0;
	}

	// Sort eigenvalues and corresponding vectors.

	for (int i = 0; i < n-1; i++) {
		int k = i;
		double p = d[i];
		for (int j = i+1; j < n; j++) {
			if (d[j] < p) {
				k = j;
				p = d[j];
			}
		}
		if (k != i) {
			d[k] = d[i];
			d[i] = p;
			for (int j = 0; j < n; j++) {
				p = V[j][i];
				V[j][i] = V[j][k];
				V[j][k] = p;
			}
		}
	}
}

void eigen_decomposition(double A[n][n], double V[n][n], double d[n]) {
	double e[n];
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			V[i][j] = A[i][j];
		}
	}
	tred2(V, d, e);
	tql2(V, d, e);
}



PetscErrorCode fsi_interpolation_coeff(UserCtx *user, IBMNodes *ibm, IBMInfo *ibminfo, SurfElmtInfo *elmtinfo, FSInfo *fsi)
{
  PetscBarrier(PETSC_NULL);
  PetscPrintf(PETSC_COMM_WORLD, "Closest_NearBndryPt_ToSurfElmt\n" );   
  Closest_NearBndryPt_ToSurfElmt(user, ibm, elmtinfo, fsi, 0);
  PetscBarrier(PETSC_NULL);
  Find_fsi_interp_Coeff(ibminfo, user, ibm, elmtinfo);
  PetscPrintf(PETSC_COMM_WORLD, "fsi interp coeff\n" ); 
  return(0);
}


PetscErrorCode Closest_NearBndryPt_ToSurfElmt(UserCtx *user, IBMNodes *ibm, SurfElmtInfo *elmtinfo, FSInfo *fsi, PetscInt ibi)
{

  DA da = user[ibi].da, fda = user[ibi].fda;
  DALocalInfo info = user[ibi].info;

  //DA	da = user->da, fda = user->fda;
  //DALocalInfo	info = user->info;
  PetscInt	xs = info.xs, xe = info.xs + info.xm;
  PetscInt  ys = info.ys, ye = info.ys + info.ym;
  PetscInt	zs = info.zs, ze = info.zs + info.zm;
  PetscInt	mx = info.mx, my = info.my, mz = info.mz; 
  PetscInt  lxs, lxe, lys, lye, lzs, lze;
  PetscReal     GCoorXs = 1e10, GCoorXe = -1e10, GCoorYs = 1e10, GCoorYe = -1e10, GCoorZs = 1e10, GCoorZe = -1e10; // coordinate of domain start and end position //ASR  
  PetscReal dxAtSBoundary, dxAtEBoundary, dyAtSBoundary, dyAtEBoundary, dzAtSBoundary, dzAtEBoundary;
  PetscInt	i, j, k;

  PetscInt      n_elmt = ibm->n_elmt;
  PetscInt      elmt;
  Cmpnts        ***coor,pc,cell[8];
  PetscReal     d[6];  
  PetscReal     xc,yc,zc; // tri shape surface elmt center coords
  PetscReal     x,y,z;    // near bndry pt coords
  PetscReal     dist, dmin;     // distance between surf elmt to near bndry pt

  IBMInfo       *ibminfo;
  IBMListNode   *current;
  
  PetscInt inbn,jnbn,knbn, notfound, nradius = 7;
  PetscInt zstart,zend,ystart,yend,xstart,xend;
  
  Vec Coor, Cent;
  
  //DAGetGhostedCoordinates(da, &Coor);
  DAGetCoordinates(da, &Coor);
  DAVecGetArray(fda, Coor, &coor);
  
  PetscInt	rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  lxs = xs+1; lxe = xe-1; // have to check ASR - because only one processor
  lys = ys+1; lye = ye-1;
  lzs = zs+1; lze = ze-1;
  
  for (k=lzs; k<lze; k++) {
    for (j=lys; j<lye; j++) {
      for (i=lxs; i<lxe; i++) {
		GCoorXs = min(GCoorXs,coor[k][j][i].x);
		GCoorXe = max(GCoorXe,coor[k][j][i].x);
		GCoorYs = min(GCoorYs,coor[k][j][i].y);
		GCoorYe = max(GCoorYe,coor[k][j][i].y);
		GCoorZs = min(GCoorZs,coor[k][j][i].z);
		GCoorZe = max(GCoorZe,coor[k][j][i].z);
      }
    }
  }
  printf("domain extreme after: %f %f %f %f %f %f\n", GCoorXs, GCoorXe, GCoorYs, GCoorYe, GCoorZs, GCoorZe); 

	dxAtSBoundary = coor[lzs+1][lys+1][lxs+1].x - coor[lzs][lys][lxs].x;
	dxAtEBoundary = coor[lze][lye][lxe].x - coor[lze-1][lye-1][lxe-1].x;
	dyAtSBoundary = coor[lzs+1][lys+1][lxs+1].y - coor[lzs][lys][lxs].y;
	dyAtEBoundary = coor[lze][lye][lxe].y - coor[lze-1][lye-1][lxe-1].y;
	dzAtSBoundary = coor[lzs+1][lys+1][lxs+1].z - coor[lzs][lys][lxs].z;
	dzAtEBoundary = coor[lze][lye][lxe].z - coor[lze-1][lye-1][lxe-1].z;
/*
	GCoorXs = GCoorXs + 2*dxAtSBoundary;
	GCoorXe = GCoorXe - 2*dxAtEBoundary;
	GCoorYs = GCoorYs + 2*dyAtSBoundary;
	GCoorYe = GCoorYe - 2*dyAtEBoundary;
	GCoorZs = GCoorZs + 2*dzAtSBoundary;
	GCoorZe = GCoorZe - 2*dzAtEBoundary;
*/
  PetscInt elmtCount = 0;
  PetscInt Aroundcellnum = 0;  
  for (elmt=0; elmt<n_elmt; elmt++) {
    dmin=1e10;
	pc.x = ibm->cent_x[elmt]; 
	pc.y = ibm->cent_y[elmt]; 
	pc.z = ibm->cent_z[elmt];

	if (pc.x>=GCoorXs && pc.x<GCoorXe && pc.y>=GCoorYs && pc.y<GCoorYe && pc.z>=GCoorZs && pc.z<GCoorZe) { //if inside the global domain
		elmtCount = elmtCount + 1;
//		current = user->ibmlist[ibi].head;
		//while (current) {
		for (k=lzs; k<lze; k++) 
		for (j=lys; j<lye; j++) 
		for (i=lxs; i<lxe; i++) {			
			//ibminfo = &current->ibm_intp;
			//current = current->next;	
			//i = ibminfo->ni; j= ibminfo->nj;    k = ibminfo->nk;   
			
			//if (i>=lxs && i<lxe && j>=lys && j<lye && k>=lzs && k<lze) 
			{	        
				x=coor[k][j][i].x;
				y=coor[k][j][i].y;
				z=coor[k][j][i].z;							
				dist=(x-pc.x)*(x-pc.x)+(y-pc.y)*(y-pc.y)+(z-pc.z)*(z-pc.z);
				if (dmin>dist) {
					elmtinfo[elmt].n_P=1;
					elmtinfo[elmt].rank=rank;
					dmin=dist;
					elmtinfo[elmt].Clsnbpt_i=i;
					elmtinfo[elmt].Clsnbpt_j=j;
					elmtinfo[elmt].Clsnbpt_k=k;
				}
			}
		}
    }else{
		elmtinfo[elmt].n_P=-1;
		elmtinfo[elmt].Clsnbpt_i=-1;
    	elmtinfo[elmt].Clsnbpt_j=-1;
    	elmtinfo[elmt].Clsnbpt_k=-1;
    }
	
	//PetscPrintf(PETSC_COMM_WORLD, " elmtX2  = %d %d %d %d \n", elmt, elmtinfo[elmt].Clsnbpt_i, elmtinfo[elmt].Clsnbpt_j, elmtinfo[elmt].Clsnbpt_k );   
	   
// =========================================================================================================================
	   
	if (elmtinfo[elmt].n_P>0){
		
		inbn = elmtinfo[elmt].Clsnbpt_i;
		jnbn = elmtinfo[elmt].Clsnbpt_j;
		knbn = elmtinfo[elmt].Clsnbpt_k;

		zstart = knbn-nradius; zend = knbn+nradius;
		ystart = jnbn-nradius; yend = jnbn+nradius;
		xstart = inbn-nradius; xend = inbn+nradius;

		if (zstart<lzs) zstart = lzs;
		if (ystart<lys) ystart = lys;
		if (xstart<lxs) xstart = lxs;
		if (zend>lze) zend = lze;
		if (yend>lye) yend = lye;
		if (xend>lxe) xend = lxe;

		elmtinfo[elmt].FoundAroundcell=0;

		for (k=zstart; k<zend; k++) {
		for (j=ystart; j<yend; j++) {
		for (i=xstart; i<xend; i++) {
			cell[0] = coor[k  ][j  ][i  ];
			cell[1] = coor[k  ][j  ][i+1];
			cell[2] = coor[k  ][j+1][i+1];
			cell[3] = coor[k  ][j+1][i  ];
		
			cell[4] = coor[k+1][j  ][i  ];
			cell[5] = coor[k+1][j  ][i+1];
			cell[6] = coor[k+1][j+1][i+1];
			cell[7] = coor[k+1][j+1][i  ];
		
			if(ISInsideCell(pc, cell, d)){
				elmtinfo[elmt].icell = i;
				elmtinfo[elmt].jcell = j;
				elmtinfo[elmt].kcell = k;
				elmtinfo[elmt].FoundAroundcell=1;
				// correction if pt exactly on one side of the cell
				if (fabs(d[0])<1e-6 && 
				ibm->nf_x[elmt]*(cell[1].x-cell[0].x)<0.) elmtinfo[elmt].icell = i-1;
				if (fabs(d[1])<1e-6 && 
				ibm->nf_x[elmt]*(cell[0].x-cell[1].x)<0.) elmtinfo[elmt].icell = i+1;
				if (fabs(d[2])<1e-6 && 
				ibm->nf_y[elmt]*(cell[3].y-cell[0].y)<0.) elmtinfo[elmt].jcell = j-1;
				if (fabs(d[3])<1e-6 && 
				ibm->nf_y[elmt]*(cell[0].y-cell[3].y)<0.) elmtinfo[elmt].jcell = j+1;
				if (fabs(d[4])<1e-6 && 
				ibm->nf_z[elmt]*(cell[4].z-cell[0].z)<0.) elmtinfo[elmt].kcell = k-1;
				if (fabs(d[5])<1e-6 && 
				ibm->nf_z[elmt]*(cell[0].z-cell[4].z)<0.) elmtinfo[elmt].kcell = k+1;
			}
		}}}
		
		if(elmtinfo[elmt].FoundAroundcell==0){
			printf("Inside global domain, but FoundAroundcell = %d, elmt = %d \n",elmtinfo[elmt].FoundAroundcell, elmt);
			printf("pc.x = %f, pc.y = %f , pc.z = %f of the elmt=%d\n",ibm->cent_x[elmt],ibm->cent_y[elmt],ibm->cent_z[elmt], elmt);
		}
	}
  }
  
// =========================================================================================================================	  
    PetscInt n_v=ibm->n_v;
	if (!rank) {
		FILE *f;
		char filen[80];
		sprintf(filen, "initialCheckTemp.dat");
		f = fopen(filen, "w");
		PetscFPrintf(PETSC_COMM_WORLD, f, "TITLE=\"3D TRIANGULAR SURFACE DATA\"\n");
		PetscFPrintf(PETSC_COMM_WORLD, f, "Variables=\"x\",\"y\",\"z\",\"n_p\",\"FoundAroundcell\",\"rank\"\n");
		PetscFPrintf(PETSC_COMM_WORLD, f, "ZONE T=\"TRIANGLES\", N=%d, E=%d, F=FEBLOCK, ET=TRIANGLE, VARLOCATION=([1-3]=NODAL,[4-6]=CELLCENTERED)\n", n_v, n_elmt);
		for (i=0; i<n_v; i++) {
			PetscFPrintf(PETSC_COMM_WORLD, f, "%e ", ibm->x_bp[i]);
		}PetscFPrintf(PETSC_COMM_WORLD, f, "\n");
		for (i=0; i<n_v; i++) {
			PetscFPrintf(PETSC_COMM_WORLD, f, "%e ", ibm->y_bp[i]);
		}PetscFPrintf(PETSC_COMM_WORLD, f, "\n");
		for (i=0; i<n_v; i++) {	
			PetscFPrintf(PETSC_COMM_WORLD, f, "%e ", ibm->z_bp[i]);
		}PetscFPrintf(PETSC_COMM_WORLD, f, "\n");
		for (i=0; i<n_elmt; i++) {
			PetscFPrintf(PETSC_COMM_WORLD, f, "%d ", elmtinfo[i].n_P);
		}PetscFPrintf(PETSC_COMM_WORLD, f, "\n");
		for (i=0; i<n_elmt; i++) {
			PetscFPrintf(PETSC_COMM_WORLD, f, "%d ", elmtinfo[i].FoundAroundcell);
		}PetscFPrintf(PETSC_COMM_WORLD, f, "\n");
		for (i=0; i<n_elmt; i++) {
			PetscFPrintf(PETSC_COMM_WORLD, f, "%d ", elmtinfo[i].rank);
		}PetscFPrintf(PETSC_COMM_WORLD, f, "\n");
		for (i=0; i<n_elmt; i++) {
			PetscFPrintf(PETSC_COMM_WORLD, f, "%d %d %d\n", ibm->nv1[i]+1, ibm->nv2[i]+1, ibm->nv3[i]+1);
		}
		fclose(f);
	}

  DAVecRestoreArray(fda, Coor,&coor);
  return(0);
}


PetscErrorCode Closest_NearBndryPt_ToSurfElmt_delta(UserCtx *user, IBMNodes *ibm, SurfElmtInfo *elmtinfo, FSInfo *fsi, PetscInt ibi)
{

  DA da = user[ibi].da, fda = user[ibi].fda;
  DALocalInfo info = user[ibi].info;

  //DA	da = user->da, fda = user->fda;
  //DALocalInfo	info = user->info;
  PetscInt	xs = info.xs, xe = info.xs + info.xm;
  PetscInt  ys = info.ys, ye = info.ys + info.ym;
  PetscInt	zs = info.zs, ze = info.zs + info.zm;
  PetscInt	mx = info.mx, my = info.my, mz = info.mz; 
  PetscInt  lxs, lxe, lys, lye, lzs, lze;
  PetscReal     GCoorXs = 1e10, GCoorXe = -1e10, GCoorYs = 1e10, GCoorYe = -1e10, GCoorZs = 1e10, GCoorZe = -1e10; // coordinate of domain start and end position //ASR  
  PetscReal dxAtSBoundary, dxAtEBoundary, dyAtSBoundary, dyAtEBoundary, dzAtSBoundary, dzAtEBoundary;
  PetscInt	i, j, k;

  PetscInt      n_elmt = ibm->n_elmt;
  PetscInt      elmt;
  Cmpnts        ***coor,pc,cell[8];
  PetscReal     d[6];  
  PetscReal     xc,yc,zc; // tri shape surface elmt center coords
  PetscReal     x,y,z;    // near bndry pt coords
  PetscReal     dist, dmin;     // distance between surf elmt to near bndry pt
PetscReal nfx,nfy,nfz;
  IBMInfo       *ibminfo;
  IBMListNode   *current;
  
  PetscInt inbn,jnbn,knbn, notfound, nradius = 7;
  PetscInt zstart,zend,ystart,yend,xstart,xend;
  
  Vec Coor, Cent;
  
  //DAGetGhostedCoordinates(da, &Coor);
  DAGetCoordinates(da, &Coor);
  DAVecGetArray(fda, Coor, &coor);
  
  PetscInt	rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  lxs = xs+1; lxe = xe-1; // have to check ASR - because only one processor
  lys = ys+1; lye = ye-1;
  lzs = zs+1; lze = ze-1;
  
  for (k=lzs; k<lze; k++) {
    for (j=lys; j<lye; j++) {
      for (i=lxs; i<lxe; i++) {
		GCoorXs = min(GCoorXs,coor[k][j][i].x);
		GCoorXe = max(GCoorXe,coor[k][j][i].x);
		GCoorYs = min(GCoorYs,coor[k][j][i].y);
		GCoorYe = max(GCoorYe,coor[k][j][i].y);
		GCoorZs = min(GCoorZs,coor[k][j][i].z);
		GCoorZe = max(GCoorZe,coor[k][j][i].z);
      }
    }
  }
  printf("domain extreme after: %f %f %f %f %f %f\n", GCoorXs, GCoorXe, GCoorYs, GCoorYe, GCoorZs, GCoorZe); 

	dxAtSBoundary = coor[lzs+1][lys+1][lxs+1].x - coor[lzs][lys][lxs].x;
	dxAtEBoundary = coor[lze][lye][lxe].x - coor[lze-1][lye-1][lxe-1].x;
	dyAtSBoundary = coor[lzs+1][lys+1][lxs+1].y - coor[lzs][lys][lxs].y;
	dyAtEBoundary = coor[lze][lye][lxe].y - coor[lze-1][lye-1][lxe-1].y;
	dzAtSBoundary = coor[lzs+1][lys+1][lxs+1].z - coor[lzs][lys][lxs].z;
	dzAtEBoundary = coor[lze][lye][lxe].z - coor[lze-1][lye-1][lxe-1].z;
/*
	GCoorXs = GCoorXs + 2*dxAtSBoundary;
	GCoorXe = GCoorXe - 2*dxAtEBoundary;
	GCoorYs = GCoorYs + 2*dyAtSBoundary;
	GCoorYe = GCoorYe - 2*dyAtEBoundary;
	GCoorZs = GCoorZs + 2*dzAtSBoundary;
	GCoorZe = GCoorZe - 2*dzAtEBoundary;
*/
  PetscInt elmtCount = 0;
  PetscInt Aroundcellnum = 0;  
  for (elmt=0; elmt<n_elmt; elmt++) {
    dmin=1e10;
	nfx=ibm->nf_x[elmt];
	nfy=ibm->nf_y[elmt];
	nfz=ibm->nf_z[elmt];
	//delta = 0.04;
	pc.x = delta*nfx + ibm->cent_x[elmt]; 
	pc.y = delta*nfy + ibm->cent_y[elmt]; 
	pc.z = delta*nfz + ibm->cent_z[elmt];

	if (pc.x>=GCoorXs && pc.x<GCoorXe && pc.y>=GCoorYs && pc.y<GCoorYe && pc.z>=GCoorZs && pc.z<GCoorZe) { //if inside the global domain
		elmtCount = elmtCount + 1;
//		current = user->ibmlist[ibi].head;
		//while (current) {
		for (k=lzs; k<lze; k++) 
		for (j=lys; j<lye; j++) 
		for (i=lxs; i<lxe; i++) {			
			//ibminfo = &current->ibm_intp;
			//current = current->next;	
			//i = ibminfo->ni; j= ibminfo->nj;    k = ibminfo->nk;   
			
			//if (i>=lxs && i<lxe && j>=lys && j<lye && k>=lzs && k<lze) 
			{	        
				x=coor[k][j][i].x;
				y=coor[k][j][i].y;
				z=coor[k][j][i].z;							
				dist=(x-pc.x)*(x-pc.x)+(y-pc.y)*(y-pc.y)+(z-pc.z)*(z-pc.z);
				if (dmin>dist) {
					elmtinfo[elmt].n_P=1;
					elmtinfo[elmt].rank=rank;
					dmin=dist;
					elmtinfo[elmt].Clsnbpt_i=i;
					elmtinfo[elmt].Clsnbpt_j=j;
					elmtinfo[elmt].Clsnbpt_k=k;
				}
			}
		}
    }else{
		elmtinfo[elmt].n_P=-1;
		elmtinfo[elmt].Clsnbpt_i=-1;
    	elmtinfo[elmt].Clsnbpt_j=-1;
    	elmtinfo[elmt].Clsnbpt_k=-1;
    }
	
	//PetscPrintf(PETSC_COMM_WORLD, " elmtX2  = %d %d %d %d \n", elmt, elmtinfo[elmt].Clsnbpt_i, elmtinfo[elmt].Clsnbpt_j, elmtinfo[elmt].Clsnbpt_k );   
	   
// =========================================================================================================================
	   
	if (elmtinfo[elmt].n_P>0){
		
		inbn = elmtinfo[elmt].Clsnbpt_i;
		jnbn = elmtinfo[elmt].Clsnbpt_j;
		knbn = elmtinfo[elmt].Clsnbpt_k;

		zstart = knbn-nradius; zend = knbn+nradius;
		ystart = jnbn-nradius; yend = jnbn+nradius;
		xstart = inbn-nradius; xend = inbn+nradius;

		if (zstart<lzs) zstart = lzs;
		if (ystart<lys) ystart = lys;
		if (xstart<lxs) xstart = lxs;
		if (zend>lze) zend = lze;
		if (yend>lye) yend = lye;
		if (xend>lxe) xend = lxe;

		elmtinfo[elmt].FoundAroundcell=0;

		for (k=zstart; k<zend; k++) {
		for (j=ystart; j<yend; j++) {
		for (i=xstart; i<xend; i++) {
			cell[0] = coor[k  ][j  ][i  ];
			cell[1] = coor[k  ][j  ][i+1];
			cell[2] = coor[k  ][j+1][i+1];
			cell[3] = coor[k  ][j+1][i  ];
		
			cell[4] = coor[k+1][j  ][i  ];
			cell[5] = coor[k+1][j  ][i+1];
			cell[6] = coor[k+1][j+1][i+1];
			cell[7] = coor[k+1][j+1][i  ];
		
			if(ISInsideCell(pc, cell, d)){
				elmtinfo[elmt].icell = i;
				elmtinfo[elmt].jcell = j;
				elmtinfo[elmt].kcell = k;
				elmtinfo[elmt].FoundAroundcell=1;
				// correction if pt exactly on one side of the cell
				if (fabs(d[0])<1e-6 && 
				ibm->nf_x[elmt]*(cell[1].x-cell[0].x)<0.) elmtinfo[elmt].icell = i-1;
				if (fabs(d[1])<1e-6 && 
				ibm->nf_x[elmt]*(cell[0].x-cell[1].x)<0.) elmtinfo[elmt].icell = i+1;
				if (fabs(d[2])<1e-6 && 
				ibm->nf_y[elmt]*(cell[3].y-cell[0].y)<0.) elmtinfo[elmt].jcell = j-1;
				if (fabs(d[3])<1e-6 && 
				ibm->nf_y[elmt]*(cell[0].y-cell[3].y)<0.) elmtinfo[elmt].jcell = j+1;
				if (fabs(d[4])<1e-6 && 
				ibm->nf_z[elmt]*(cell[4].z-cell[0].z)<0.) elmtinfo[elmt].kcell = k-1;
				if (fabs(d[5])<1e-6 && 
				ibm->nf_z[elmt]*(cell[0].z-cell[4].z)<0.) elmtinfo[elmt].kcell = k+1;
			}
		}}}
		
		if(elmtinfo[elmt].FoundAroundcell==0){
			PetscPrintf(PETSC_COMM_SELF,"Inside global domain, but FoundAroundcell = %d, elmt = %d \n",elmtinfo[elmt].FoundAroundcell, elmt);
			PetscPrintf(PETSC_COMM_SELF,"pc.x = %f, pc.y = %f , pc.z = %f of the elmt=%d\n",ibm->cent_x[elmt],ibm->cent_y[elmt],ibm->cent_z[elmt], elmt);
		}
	}
  }

  DAVecRestoreArray(fda, Coor,&coor);
  return(0);
}


PetscErrorCode Find_fsi_interp_Coeff_delta(IBMInfo *ibminfo, UserCtx *user, IBMNodes *ibm, SurfElmtInfo *elmtinfo)
{

/* Note:  ibminfo returns the interpolation info 
   for the fsi (fsi_intp) */

  DA	da = user->da, fda = user->fda;
  DALocalInfo	info = user->info;
  PetscInt	xs = info.xs, xe = info.xs + info.xm;
  PetscInt  ys = info.ys, ye = info.ys + info.ym;
  PetscInt	zs = info.zs, ze = info.zs + info.zm;
  PetscInt	mx = info.mx, my = info.my, mz = info.mz;
  PetscInt	i, j, k;
  PetscInt	i2, j2, k2;
  PetscInt  lxs, lxe, lys, lye, lzs, lze;

  PetscInt foundFlag;

  PetscInt      n_elmt = ibm->n_elmt;
  PetscInt      elmt, ip[8],jp[8],kp[8];
  Cmpnts        ***coor,pc[8],p, intp, intp2x;
  PetscReal	***nvert,nvertpc[8];
  PetscReal     nfx,nfy,nfz;
  Vec Coor;
  Cmpnts        ***ucat,ucatpc[8],tempV;  
  PetscInt v,count;
	lxs = xs; lxe = xe;
	lys = ys; lye = ye;
	lzs = zs; lze = ze;

	if (xs==0) lxs = xs+1; 
	if (ys==0) lys = ys+1; 
	if (zs==0) lzs = zs+1; 
	if (xe==mx) lxe = xe-1; 
	if (ye==my) lye = ye-1; 
	if (ze==mz) lze = ze-1;

  PetscInt	rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  
  DAGetCoordinates(da, &Coor);
  DAVecGetArray(fda, Coor,&coor);
  DAVecGetArray(da, user->Nvert, &nvert);
  DAVecGetArray(user->fda, user->Ucat, &ucat);
	
  for (elmt=0; elmt<n_elmt; elmt++) {
	if (elmtinfo[elmt].n_P>0 && elmtinfo[elmt].FoundAroundcell==1) {

		i = elmtinfo[elmt].icell;
		j = elmtinfo[elmt].jcell;
		k = elmtinfo[elmt].kcell;

	 //if (i>=xs && i<xe && j>=ys && j<ye && k>=zs && k<ze) {
		if (i>=lxs && i<lxe && j>=lys && j<lye && k>=lzs && k<lze) {

		  pc[0] = coor[k  ][j  ][i  ];
		  pc[1] = coor[k  ][j  ][i+1];
		  pc[2] = coor[k  ][j+1][i+1];
		  pc[3] = coor[k  ][j+1][i  ];
		  
		  pc[4] = coor[k+1][j  ][i  ];
		  pc[5] = coor[k+1][j  ][i+1];
		  pc[6] = coor[k+1][j+1][i+1];
		  pc[7] = coor[k+1][j+1][i  ];

		  nvertpc[0] = nvert[k  ][j  ][i  ];
		  nvertpc[1] = nvert[k  ][j  ][i+1];
		  nvertpc[2] = nvert[k  ][j+1][i+1];
		  nvertpc[3] = nvert[k  ][j+1][i  ];
		  
		  nvertpc[4] = nvert[k+1][j  ][i  ];
		  nvertpc[5] = nvert[k+1][j  ][i+1];
		  nvertpc[6] = nvert[k+1][j+1][i+1];
		  nvertpc[7] = nvert[k+1][j+1][i  ];

		  ucatpc[0] = ucat[k  ][j  ][i  ];
		  ucatpc[1] = ucat[k  ][j  ][i+1];
		  ucatpc[2] = ucat[k  ][j+1][i+1];
		  ucatpc[3] = ucat[k  ][j+1][i  ];
		  ucatpc[4] = ucat[k+1][j  ][i  ];
		  ucatpc[5] = ucat[k+1][j  ][i+1];
		  ucatpc[6] = ucat[k+1][j+1][i+1];
		  ucatpc[7] = ucat[k+1][j+1][i  ];
		  
		  count = 0;
		  for(v=0;v<8;v++){
			  if(ucatpc[v].x!=0 || ucatpc[v].y!=0 || ucatpc[v].z!=0){
				  count = count + 1;
				  tempV.x = tempV.x + ucatpc[v].x;
				  tempV.y = tempV.y + ucatpc[v].y;
				  tempV.z = tempV.z + ucatpc[v].z;				  
			  }
		  }
		  if (count == 0) PetscPrintf(PETSC_COMM_SELF, "Delta no point found for averaging. elm=%d\n",elmt);
		  ibminfo[elmt].cs11 = tempV.x/count;
		  ibminfo[elmt].cs22 = tempV.y/count;
		  ibminfo[elmt].cs33 = tempV.z/count;		  
		} //end of if (i>=lxs && i<lxe && j>=lys && j<lye && k>=lzs && k<lze)
	} // if(elmtinfo[elmt].n_P>0 && elmtinfo[elmt].FoundAroundcell==1)
  } // end of   for (elmt=0; elmt<n_elmt; elmt++) 
	  
  DAVecRestoreArray(fda, Coor,&coor);  
  DAVecRestoreArray(da, user->Nvert, &nvert);
  DAVecRestoreArray(user->fda, user->Ucat, &ucat);  
  return(0);
}



PetscErrorCode Find_fsi_interp_Coeff(IBMInfo *ibminfo, UserCtx *user, IBMNodes *ibm, SurfElmtInfo *elmtinfo)
{

/* Note:  ibminfo returns the interpolation info 
   for the fsi (fsi_intp) */

  DA	da = user->da, fda = user->fda;
  DALocalInfo	info = user->info;
  PetscInt	xs = info.xs, xe = info.xs + info.xm;
  PetscInt  ys = info.ys, ye = info.ys + info.ym;
  PetscInt	zs = info.zs, ze = info.zs + info.zm;
  PetscInt	mx = info.mx, my = info.my, mz = info.mz;
  PetscInt	i, j, k;
  PetscInt	i2, j2, k2;
  PetscInt  lxs, lxe, lys, lye, lzs, lze;

  PetscInt foundFlag;

  PetscInt      n_elmt = ibm->n_elmt;
  PetscInt      elmt, ip[8],jp[8],kp[8];
  Cmpnts        ***coor,pc[8],p, intp, intp2x;
  PetscReal	***nvert,nvertpc[8];
  PetscReal     nfx,nfy,nfz;
  Vec Coor;
  
	lxs = xs; lxe = xe;
	lys = ys; lye = ye;
	lzs = zs; lze = ze;

	if (xs==0) lxs = xs+1; 
	if (ys==0) lys = ys+1; 
	if (zs==0) lzs = zs+1; 
	if (xe==mx) lxe = xe-1; 
	if (ye==my) lye = ye-1; 
	if (ze==mz) lze = ze-1;

  PetscInt	rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  
  DAGetCoordinates(da, &Coor);
  DAVecGetArray(fda, Coor,&coor);
  DAVecGetArray(da, user->Nvert, &nvert);

  for (elmt=0; elmt<n_elmt; elmt++) {
	if (elmtinfo[elmt].n_P>0 && elmtinfo[elmt].FoundAroundcell==1) {
		p.x=ibm->cent_x[elmt];
		p.y=ibm->cent_y[elmt]; 
		p.z=ibm->cent_z[elmt]; 

		nfx=ibm->nf_x[elmt];
		nfy=ibm->nf_y[elmt];
		nfz=ibm->nf_z[elmt];

		i = elmtinfo[elmt].icell;
		j = elmtinfo[elmt].jcell;
		k = elmtinfo[elmt].kcell;

	 //if (i>=xs && i<xe && j>=ys && j<ye && k>=zs && k<ze) {
		if (i>=lxs && i<lxe && j>=lys && j<lye && k>=lzs && k<lze) {

		  pc[0] = coor[k  ][j  ][i  ];
		  pc[1] = coor[k  ][j  ][i+1];
		  pc[2] = coor[k  ][j+1][i+1];
		  pc[3] = coor[k  ][j+1][i  ];
		  
		  pc[4] = coor[k+1][j  ][i  ];
		  pc[5] = coor[k+1][j  ][i+1];
		  pc[6] = coor[k+1][j+1][i+1];
		  pc[7] = coor[k+1][j+1][i  ];

		  nvertpc[0] = nvert[k  ][j  ][i  ];
		  nvertpc[1] = nvert[k  ][j  ][i+1];
		  nvertpc[2] = nvert[k  ][j+1][i+1];
		  nvertpc[3] = nvert[k  ][j+1][i  ];
		  
		  nvertpc[4] = nvert[k+1][j  ][i  ];
		  nvertpc[5] = nvert[k+1][j  ][i+1];
		  nvertpc[6] = nvert[k+1][j+1][i+1];
		  nvertpc[7] = nvert[k+1][j+1][i  ];

		  kp[0]=k  ;jp[0]=j  ;ip[0]=i  ;
		  kp[1]=k  ;jp[1]=j  ;ip[1]=i+1;
		  kp[2]=k  ;jp[2]=j+1;ip[2]=i+1;
		  kp[3]=k  ;jp[3]=j+1;ip[3]=i  ;
		  kp[4]=k+1;jp[4]=j  ;ip[4]=i  ;
		  kp[5]=k+1;jp[5]=j  ;ip[5]=i+1;
		  kp[6]=k+1;jp[6]=j+1;ip[6]=i+1;
		  kp[7]=k+1;jp[7]=j+1;ip[7]=i  ;

		  fsi_InterceptionPoint(p, pc, nvertpc, nfx, nfy, nfz, ibminfo, elmt, &intp, &(elmtinfo[elmt].Need3rdPoint));	  
		  
			switch (ibminfo[elmt].imode) {
				case(0): {
					ibminfo[elmt].i1=ip[0]; ibminfo[elmt].j1 = jp[0]; ibminfo[elmt].k1 = kp[0];
					ibminfo[elmt].i2=ip[1]; ibminfo[elmt].j2 = jp[1]; ibminfo[elmt].k2 = kp[1];
					ibminfo[elmt].i3=ip[2]; ibminfo[elmt].j3 = jp[2]; ibminfo[elmt].k3 = kp[2];
					GridCellaround2ndElmt(user,ibm,elmtinfo,intp,elmt,k,j,i,&k2,&j2,&i2,&foundFlag);
					Find_fsi_2nd_interp_Coeff(foundFlag,i2,j2,k2,elmt,intp,ibminfo,user,ibm,elmtinfo);
					/* 	Find_fsi_2nd_interp_Coeff2(i2,j2,k2,elmt,intp,ibminfo,user,ibm); */
					break;
				}
				case (1): {
					ibminfo[elmt].i1=ip[0]; ibminfo[elmt].j1 = jp[0]; ibminfo[elmt].k1 = kp[0];
					ibminfo[elmt].i2=ip[2]; ibminfo[elmt].j2 = jp[2]; ibminfo[elmt].k2 = kp[2];
					ibminfo[elmt].i3=ip[3]; ibminfo[elmt].j3 = jp[3]; ibminfo[elmt].k3 = kp[3];
					GridCellaround2ndElmt(user,ibm,elmtinfo,intp,elmt,k,j,i,&k2,&j2,&i2,&foundFlag);
					Find_fsi_2nd_interp_Coeff(foundFlag,i2,j2,k2,elmt,intp,ibminfo,user,ibm,elmtinfo);
					/* 	Find_fsi_2nd_interp_Coeff2(i2,j2,k2,elmt,intp,ibminfo,user,ibm); */
					/* 	Find_fsi_2nd_interp_Coeff(i,j,k-1,elmt,intp,ibminfo,user,ibm); */
					/* 	Find_fsi_2nd_interp_Coeff2(i,j,k-1,elmt,intp,ibminfo,user,ibm); */
						break;
				}
				case (2): {
					ibminfo[elmt].i1=ip[4]; ibminfo[elmt].j1 = jp[4]; ibminfo[elmt].k1 = kp[4];
					ibminfo[elmt].i2=ip[5]; ibminfo[elmt].j2 = jp[5]; ibminfo[elmt].k2 = kp[5];
					ibminfo[elmt].i3=ip[6]; ibminfo[elmt].j3 = jp[6]; ibminfo[elmt].k3 = kp[6];
					GridCellaround2ndElmt(user,ibm,elmtinfo,intp,elmt,k,j,i,&k2,&j2,&i2,&foundFlag);
					Find_fsi_2nd_interp_Coeff(foundFlag,i2,j2,k2,elmt,intp,ibminfo,user,ibm,elmtinfo);
					/* 	Find_fsi_2nd_interp_Coeff2(i2,j2,k2,elmt,intp,ibminfo,user,ibm); */
					/* 	Find_fsi_2nd_interp_Coeff(i,j,k+1,elmt,intp,ibminfo,user,ibm); */
					/* 	Find_fsi_2nd_interp_Coeff2(i,j,k+1,elmt,intp,ibminfo,user,ibm); */
					break;
				}
				case (3): {
					ibminfo[elmt].i1=ip[4]; ibminfo[elmt].j1 = jp[4]; ibminfo[elmt].k1 = kp[4];
					ibminfo[elmt].i2=ip[6]; ibminfo[elmt].j2 = jp[6]; ibminfo[elmt].k2 = kp[6];
					ibminfo[elmt].i3=ip[7]; ibminfo[elmt].j3 = jp[7]; ibminfo[elmt].k3 = kp[7];
					GridCellaround2ndElmt(user,ibm,elmtinfo,intp,elmt,k,j,i,&k2,&j2,&i2,&foundFlag);
					Find_fsi_2nd_interp_Coeff(foundFlag,i2,j2,k2,elmt,intp,ibminfo,user,ibm,elmtinfo);
					/* 	Find_fsi_2nd_interp_Coeff2(i2,j2,k2,elmt,intp,ibminfo,user,ibm); */
					/* 	Find_fsi_2nd_interp_Coeff(i,j,k+1,elmt,intp,ibminfo,user,ibm); */
					/* 	Find_fsi_2nd_interp_Coeff2(i,j,k+1,elmt,intp,ibminfo,user,ibm); */
					break;
				}
				case (4): {
					ibminfo[elmt].i1=ip[0]; ibminfo[elmt].j1 = jp[0]; ibminfo[elmt].k1 = kp[0];
					ibminfo[elmt].i2=ip[4]; ibminfo[elmt].j2 = jp[4]; ibminfo[elmt].k2 = kp[4];
					ibminfo[elmt].i3=ip[3]; ibminfo[elmt].j3 = jp[3]; ibminfo[elmt].k3 = kp[3];
				/*	if(elmt==29473){
						PetscPrintf(PETSC_COMM_SELF, "now: imode=4; rank=%d; elmt=%d;\n",rank,elmt);
						PetscPrintf(PETSC_COMM_SELF, "x: rank=%d; i1=%d; j1=%d; k1=%d; nv1=%f;\n",rank,ibminfo[elmt].i1,ibminfo[elmt].j1,ibminfo[elmt].k1,nvertpc[0]);
						PetscPrintf(PETSC_COMM_SELF, "x: rank=%d; i2=%d; j2=%d; k2=%d; nv2=%f;\n",rank,ibminfo[elmt].i2,ibminfo[elmt].j2,ibminfo[elmt].k2,nvertpc[4]);
						PetscPrintf(PETSC_COMM_SELF, "x: rank=%d; i3=%d; j3=%d; k3=%d; nv3=%f;\n",rank,ibminfo[elmt].i3,ibminfo[elmt].j3,ibminfo[elmt].k3,nvertpc[3]);
					}*/		
					GridCellaround2ndElmt(user,ibm,elmtinfo,intp,elmt,k,j,i,&k2,&j2,&i2,&foundFlag);
					Find_fsi_2nd_interp_Coeff(foundFlag,i2,j2,k2,elmt,intp,ibminfo,user,ibm,elmtinfo);
					/* 	Find_fsi_2nd_interp_Coeff2(i2,j2,k2,elmt,intp,ibminfo,user,ibm); */
					/* 	Find_fsi_2nd_interp_Coeff(i-1,j,k,elmt,intp,ibminfo,user,ibm); */
					/* 	Find_fsi_2nd_interp_Coeff2(i-1,j,k,elmt,intp,ibminfo,user,ibm); */
					break;				
				}
				case (5): {
					ibminfo[elmt].i1=ip[3]; ibminfo[elmt].j1 = jp[3]; ibminfo[elmt].k1 = kp[3];
					ibminfo[elmt].i2=ip[4]; ibminfo[elmt].j2 = jp[4]; ibminfo[elmt].k2 = kp[4];
					ibminfo[elmt].i3=ip[7]; ibminfo[elmt].j3 = jp[7]; ibminfo[elmt].k3 = kp[7];
					GridCellaround2ndElmt(user,ibm,elmtinfo,intp,elmt,k,j,i,&k2,&j2,&i2,&foundFlag);
					Find_fsi_2nd_interp_Coeff(foundFlag,i2,j2,k2,elmt,intp,ibminfo,user,ibm,elmtinfo);
					/* 	Find_fsi_2nd_interp_Coeff2(i2,j2,k2,elmt,intp,ibminfo,user,ibm); */
					/* 	Find_fsi_2nd_interp_Coeff(i-1,j,k,elmt,intp,ibminfo,user,ibm); */
					/* 	Find_fsi_2nd_interp_Coeff2(i-1,j,k,elmt,intp,ibminfo,user,ibm); */
					break;
				}
				case (6): {
					ibminfo[elmt].i1=ip[1]; ibminfo[elmt].j1 = jp[1]; ibminfo[elmt].k1 = kp[1];
					ibminfo[elmt].i2=ip[5]; ibminfo[elmt].j2 = jp[5]; ibminfo[elmt].k2 = kp[5];
					ibminfo[elmt].i3=ip[2]; ibminfo[elmt].j3 = jp[2]; ibminfo[elmt].k3 = kp[2];
					GridCellaround2ndElmt(user,ibm,elmtinfo,intp,elmt,k,j,i,&k2,&j2,&i2,&foundFlag);
					Find_fsi_2nd_interp_Coeff(foundFlag,i2,j2,k2,elmt,intp,ibminfo,user,ibm,elmtinfo);
					/* 	Find_fsi_2nd_interp_Coeff2(i2,j2,k2,elmt,intp,ibminfo,user,ibm); */
					/* 	Find_fsi_2nd_interp_Coeff(i+1,j,k,elmt,intp,ibminfo,user,ibm); */
					/* 	Find_fsi_2nd_interp_Coeff2(i+1,j,k,elmt,intp,ibminfo,user,ibm); */
					break;
				}
				case (7): {
					ibminfo[elmt].i1=ip[2]; ibminfo[elmt].j1 = jp[2]; ibminfo[elmt].k1 = kp[2];
					ibminfo[elmt].i2=ip[6]; ibminfo[elmt].j2 = jp[6]; ibminfo[elmt].k2 = kp[6];
					ibminfo[elmt].i3=ip[5]; ibminfo[elmt].j3 = jp[5]; ibminfo[elmt].k3 = kp[5];
					GridCellaround2ndElmt(user,ibm,elmtinfo,intp,elmt,k,j,i,&k2,&j2,&i2,&foundFlag);
					Find_fsi_2nd_interp_Coeff(foundFlag,i2,j2,k2,elmt,intp,ibminfo,user,ibm,elmtinfo);
					/* 	Find_fsi_2nd_interp_Coeff2(i2,j2,k2,elmt,intp,ibminfo,user,ibm); */
					/* 	Find_fsi_2nd_interp_Coeff(i+1,j,k,elmt,intp,ibminfo,user,ibm); */
					/* 	Find_fsi_2nd_interp_Coeff2(i+1,j,k,elmt,intp,ibminfo,user,ibm); */
					break;
				}
				case (8): {
					ibminfo[elmt].i1=ip[0]; ibminfo[elmt].j1 = jp[0]; ibminfo[elmt].k1 = kp[0];
					ibminfo[elmt].i2=ip[4]; ibminfo[elmt].j2 = jp[4]; ibminfo[elmt].k2 = kp[4];
					ibminfo[elmt].i3=ip[1]; ibminfo[elmt].j3 = jp[1]; ibminfo[elmt].k3 = kp[1];
					GridCellaround2ndElmt(user,ibm,elmtinfo,intp,elmt,k,j,i,&k2,&j2,&i2,&foundFlag);
					Find_fsi_2nd_interp_Coeff(foundFlag,i2,j2,k2,elmt,intp,ibminfo,user,ibm,elmtinfo);
					/* 	Find_fsi_2nd_interp_Coeff2(i2,j2,k2,elmt,intp,ibminfo,user,ibm); */
					/* 	Find_fsi_2nd_interp_Coeff(i,j-1,k,elmt,intp,ibminfo,user,ibm); */
					/* 	Find_fsi_2nd_interp_Coeff2(i,j-1,k,elmt,intp,ibminfo,user,ibm); */
					break;
				}
				case (9): {
					ibminfo[elmt].i1=ip[1]; ibminfo[elmt].j1 = jp[1]; ibminfo[elmt].k1 = kp[1];
					ibminfo[elmt].i2=ip[4]; ibminfo[elmt].j2 = jp[4]; ibminfo[elmt].k2 = kp[4];
					ibminfo[elmt].i3=ip[5]; ibminfo[elmt].j3 = jp[5]; ibminfo[elmt].k3 = kp[5];
					GridCellaround2ndElmt(user,ibm,elmtinfo,intp,elmt,k,j,i,&k2,&j2,&i2,&foundFlag);
					Find_fsi_2nd_interp_Coeff(foundFlag,i2,j2,k2,elmt,intp,ibminfo,user,ibm,elmtinfo);
					/* 	Find_fsi_2nd_interp_Coeff2(i2,j2,k2,elmt,intp,ibminfo,user,ibm); */
					/* 	Find_fsi_2nd_interp_Coeff(i,j-1,k,elmt,intp,ibminfo,user,ibm); */
					/* 	Find_fsi_2nd_interp_Coeff2(i,j-1,k,elmt,intp,ibminfo,user,ibm); */
					break;
				}
				case (10): {
					ibminfo[elmt].i1=ip[3]; ibminfo[elmt].j1 = jp[3]; ibminfo[elmt].k1 = kp[3];
					ibminfo[elmt].i2=ip[7]; ibminfo[elmt].j2 = jp[7]; ibminfo[elmt].k2 = kp[7];
					ibminfo[elmt].i3=ip[2]; ibminfo[elmt].j3 = jp[2]; ibminfo[elmt].k3 = kp[2];
					GridCellaround2ndElmt(user,ibm,elmtinfo,intp,elmt,k,j,i,&k2,&j2,&i2,&foundFlag);
					Find_fsi_2nd_interp_Coeff(foundFlag,i2,j2,k2,elmt,intp,ibminfo,user,ibm,elmtinfo);
					/* 	Find_fsi_2nd_interp_Coeff2(i2,j2,k2,elmt,intp,ibminfo,user,ibm); */
					/* 	Find_fsi_2nd_interp_Coeff(i,j+1,k,elmt,intp,ibminfo,user,ibm); */
					/* 	Find_fsi_2nd_interp_Coeff2(i,j+1,k,elmt,intp,ibminfo,user,ibm); */
					break;
				}
				case (11): {
					ibminfo[elmt].i1=ip[2]; ibminfo[elmt].j1 = jp[2]; ibminfo[elmt].k1 = kp[2];
					ibminfo[elmt].i2=ip[7]; ibminfo[elmt].j2 = jp[7]; ibminfo[elmt].k2 = kp[7];
					ibminfo[elmt].i3=ip[6]; ibminfo[elmt].j3 = jp[6]; ibminfo[elmt].k3 = kp[6];
					GridCellaround2ndElmt(user,ibm,elmtinfo,intp,elmt,k,j,i,&k2,&j2,&i2,&foundFlag);
					Find_fsi_2nd_interp_Coeff(foundFlag,i2,j2,k2,elmt,intp,ibminfo,user,ibm,elmtinfo);
					/* 	Find_fsi_2nd_interp_Coeff2(i2,j2,k2,elmt,intp,ibminfo,user,ibm); */
					/* 	Find_fsi_2nd_interp_Coeff(i,j+1,k,elmt,intp,ibminfo,user,ibm); */
					/* 	Find_fsi_2nd_interp_Coeff2(i,j+1,k,elmt,intp,ibminfo,user,ibm); */
					break;
				}
			} // end of switch
		
			//if (ibminfo[elmt].imode<0) PetscPrintf(PETSC_COMM_SELF, "1st FSI Interpolation Coeffients Were not Found! elmt=%d imode=%d %le \n", elmt, ibminfo[elmt].imode,ibminfo[elmt].d_i);
			
		} //end of if (i>=lxs && i<lxe && j>=lys && j<lye && k>=lzs && k<lze)
	} // if(elmtinfo[elmt].n_P>0 && elmtinfo[elmt].FoundAroundcell==1)
  } // end of   for (elmt=0; elmt<n_elmt; elmt++) 
	  
  DAVecRestoreArray(fda, Coor,&coor);  
  DAVecRestoreArray(da, user->Nvert, &nvert);
  return(0);
}

PetscTruth ISInsideCell(Cmpnts p, Cmpnts cell[8], PetscReal d[6])
{
  // k direction
  distance(cell[0], cell[1], cell[2], cell[3], p, &(d[4]));
  if (d[4]<0) return(PETSC_FALSE);
  distance(cell[4], cell[7], cell[6], cell[5], p, &(d[5]));
  if (d[5]<0) return(PETSC_FALSE);

  // j direction
  distance(cell[0], cell[4], cell[5], cell[1], p, &(d[2]));
  if (d[2]<0) return(PETSC_FALSE);
  distance(cell[3], cell[2], cell[6], cell[7], p, &(d[3]));
  if (d[3]<0) return(PETSC_FALSE);

  // i direction
  distance(cell[0], cell[3], cell[7], cell[4], p, &(d[0]));
  if (d[0]<0) return(PETSC_FALSE);
  distance(cell[1], cell[5], cell[6], cell[2], p, &(d[1]));
  if (d[1]<0) return(PETSC_FALSE);
  
  return(PETSC_TRUE);
}

PetscInt ISPointInTriangle(Cmpnts p, Cmpnts p1, Cmpnts p2, Cmpnts p3, PetscReal nfx, PetscReal nfy, PetscReal nfz)
{
  PetscInt flag;
  Cpt2D pj, pj1, pj2, pj3;
  if (fabs(nfz) >= fabs(nfx) && fabs(nfz) >= fabs(nfy) ) {
    pj.x = p.x; pj.y = p.y;
    pj1.x = p1.x; pj1.y = p1.y;
    pj2.x = p2.x; pj2.y = p2.y;
    pj3.x = p3.x; pj3.y = p3.y;
  }
  else if (fabs(nfx) >= fabs(nfy) && fabs(nfx) >= fabs(nfz)) {
    pj.x = p.z; pj.y = p.y;
    pj1.x = p1.z; pj1.y = p1.y;
    pj2.x = p2.z; pj2.y = p2.y;
    pj3.x = p3.z; pj3.y = p3.y;
  }
  else {
    pj.x = p.x; pj.y = p.z;
    pj1.x = p1.x; pj1.y = p1.z;
    pj2.x = p2.x; pj2.y = p2.z;
    pj3.x = p3.x; pj3.y = p3.z;
  }
  flag = ISInsideTriangle2D(pj, pj1, pj2, pj3);
  //  if (flag > 0)
  //  PetscPrintf(PETSC_COMM_WORLD, "%d %e %e %e dddd", flag, nfx, nfy, nfz);
  return(flag);
}


PetscInt ISInsideTriangle2D(Cpt2D p, Cpt2D pa, Cpt2D pb, Cpt2D pc)
{
  // Check if point p and p3 is located on the same side of line p1p2
  PetscInt      ls;

  ls = ISSameSide2D(p, pa, pb, pc);
  //  if (flagprint) PetscPrintf(PETSC_COMM_WORLD, "aaa, %d\n", ls);
  if (ls < 0) {
    return (ls);
  }
  ls = ISSameSide2D(p, pb, pc, pa);
  //  if (flagprint) PetscPrintf(PETSC_COMM_WORLD, "bbb, %d\n", ls);
  if (ls < 0) {
    return (ls);
  }
  ls = ISSameSide2D(p, pc, pa, pb);
  //  if (flagprint) PetscPrintf(PETSC_COMM_WORLD, "ccc, %d\n", ls);
  if (ls <0) {
    return(ls);
  }
  return (ls);
}


PetscErrorCode distance(Cmpnts p1, Cmpnts p2, Cmpnts p3, Cmpnts p4, Cmpnts p, PetscReal *d)
{
  PetscReal xn1, yn1, zn1;
  PetscReal xc, yc, zc;
  
  PetscReal dx1, dy1, dz1, dx2, dy2, dz2, r;

  dx1 = p3.x - p1.x;
  dy1 = p3.y - p1.y;
  dz1 = p3.z - p1.z;
 
  dx2 = p4.x - p2.x;
  dy2 = p4.y - p2.y;
  dz2 = p4.z - p2.z;

  xn1 = dy1 * dz2 - dz1 * dy2;
  yn1 = - (dx1 * dz2 - dz1 * dx2);
  zn1 = dx1 * dy2 - dy1 * dx2;

  r = sqrt(xn1 * xn1 + yn1 * yn1 + zn1 * zn1);
  xn1 /= r; yn1 /= r; zn1 /= r;

  xc = 0.25 * (p1.x + p2.x + p3.x + p4.x);
  yc = 0.25 * (p1.y + p2.y + p3.y + p4.y);
  zc = 0.25 * (p1.z + p2.z + p3.z + p4.z);

  *d = (p.x - xc) * xn1 + (p.y - yc) * yn1 + (p.z - zc) * zn1;   
  //if (PetscAbsReal(*d)<1.e-6) *d=0.; //by ASR
  return (0);
}


PetscErrorCode triangle_intp_fsi(Cpt2D p, Cpt2D p1, Cpt2D p2, Cpt2D p3, IBMInfo *ibminfo, PetscInt number)
{
  PetscReal  x13, y13, x23, y23, xp3, yp3, a;
  x13 = p1.x - p3.x; y13 = p1.y - p3.y;
  x23 = p2.x - p3.x; y23 = p2.y - p3.y;
  xp3 = p.x - p3.x; yp3 = p.y - p3.y;
  a =x13 * y23 - x23 * y13;
  ibminfo[number].cr1 = (y23 * xp3 - x23 * yp3) / a;
  ibminfo[number].cr2 = (-y13 * xp3 + x13 * yp3) / a;
  ibminfo[number].cr3 = 1. - ibminfo[number].cr1 - ibminfo[number].cr2;
  if (ibminfo[number].cr3<0.)
    PetscPrintf(PETSC_COMM_WORLD, "SOMETHING WRONG!!!! fsi_intp Cr %d %le %le %le\n", number,ibminfo[number].cr3, ibminfo[number].cr2, ibminfo[number].cr1);
}
PetscErrorCode triangle_intp_fsi_2(Cpt2D p, Cpt2D p1, Cpt2D p2, Cpt2D p3, IBMInfo *ibminfo, PetscInt number)
{
  PetscReal  x13, y13, x23, y23, xp3, yp3, a;
  x13 = p1.x - p3.x; y13 = p1.y - p3.y;
  x23 = p2.x - p3.x; y23 = p2.y - p3.y;
  xp3 = p.x - p3.x; yp3 = p.y - p3.y;
  a =x13 * y23 - x23 * y13;
  ibminfo[number].ct11 = (y23 * xp3 - x23 * yp3) / a;
  ibminfo[number].ct22 = (-y13 * xp3 + x13 * yp3) / a;
  ibminfo[number].ct33 = 1. - ibminfo[number].ct11 - ibminfo[number].ct22;
  if (ibminfo[number].cr3<0.)
    PetscPrintf(PETSC_COMM_WORLD, "SOMETHING WRONG!!!! fsi_intp Cr %d %le %le %le\n", number,ibminfo[number].cr3, ibminfo[number].cr2, ibminfo[number].cr1);
}

PetscErrorCode triangle_intp_fsi_3(Cpt2D p, Cpt2D p1, Cpt2D p2, Cpt2D p3, IBMInfo *ibminfo, PetscInt number)
{
  PetscReal  x13, y13, x23, y23, xp3, yp3, a;
  x13 = p1.x - p3.x; y13 = p1.y - p3.y;
  x23 = p2.x - p3.x; y23 = p2.y - p3.y;
  xp3 = p.x - p3.x; yp3 = p.y - p3.y;
  a =x13 * y23 - x23 * y13;
  ibminfo[number].cr11 = (y23 * xp3 - x23 * yp3) / a;
  ibminfo[number].cr22 = (-y13 * xp3 + x13 * yp3) / a;
  ibminfo[number].cr33 = 1. - ibminfo[number].cr11 - ibminfo[number].cr22;
  if (ibminfo[number].cr3<0.)
    PetscPrintf(PETSC_COMM_WORLD, "SOMETHING WRONG!!!! fsi_intp Cr %d %le %le %le\n", number,ibminfo[number].cr3, ibminfo[number].cr2, ibminfo[number].cr1);
}

PetscErrorCode triangle_intp2_fsi(Cpt2D p, Cpt2D p1, Cpt2D p2, Cpt2D p3, IBMInfo *ibminfo, PetscInt number)
{
  PetscReal  x13, y13, x23, y23, xp3, yp3, a;
  x13 = p1.x - p3.x; y13 = p1.y - p3.y;
  x23 = p2.x - p3.x; y23 = p2.y - p3.y;
  xp3 = p.x - p3.x; yp3 = p.y - p3.y;
  a = x13 * y23 - x23 * y13;
  ibminfo[number].cs1 = (y23 * xp3 - x23 * yp3) / a;
  ibminfo[number].cs2 = (-y13 * xp3 + x13 * yp3) / a;
  ibminfo[number].cs3 = 1. - ibminfo[number].cs1 - ibminfo[number].cs2;
  if (ibminfo[number].cs3<0.)
    PetscPrintf(PETSC_COMM_WORLD, "SOMETHING WRONG!!!! fsi_intp Cs %d %le %le %le\n",number, ibminfo[number].cs3, ibminfo[number].cs2, ibminfo[number].cs1);

}

PetscErrorCode triangle_2nd_intp_fsi(Cpt2D p, Cpt2D p1, Cpt2D p2, Cpt2D p3, IBMInfo *ibminfo, PetscInt number)
{
  PetscReal  x13, y13, x23, y23, xp3, yp3, a;
  x13 = p1.x - p3.x; y13 = p1.y - p3.y;
  x23 = p2.x - p3.x; y23 = p2.y - p3.y;
  xp3 = p.x - p3.x; yp3 = p.y - p3.y;
  a =x13 * y23 - x23 * y13;
  ibminfo[number].ct1 = (y23 * xp3 - x23 * yp3) / a;
  ibminfo[number].ct2 = (-y13 * xp3 + x13 * yp3) / a;
  ibminfo[number].ct3 = 1. - ibminfo[number].ct1 - ibminfo[number].ct2;
  if (ibminfo[number].ct3<0.)
    PetscPrintf(PETSC_COMM_WORLD, "SOMETHING WRONG!!!! fsi_intp Ct %d %le %le %le\n",number,ibminfo[number].ct3, ibminfo[number].ct2, ibminfo[number].ct1);
  return(0);
}

PetscErrorCode triangle_3rd_intp_fsi(Cpt2D p, Cpt2D p1, Cpt2D p2, Cpt2D p3, IBMInfo *ibminfo, PetscInt number)
{
  PetscReal  x13, y13, x23, y23, xp3, yp3, a;
  x13 = p1.x - p3.x; y13 = p1.y - p3.y;
  x23 = p2.x - p3.x; y23 = p2.y - p3.y;
  xp3 = p.x - p3.x; yp3 = p.y - p3.y;
  a =x13 * y23 - x23 * y13;
  ibminfo[number].cs11 = (y23 * xp3 - x23 * yp3) / a;
  ibminfo[number].cs22 = (-y13 * xp3 + x13 * yp3) / a;
  ibminfo[number].cs33 = 1. - ibminfo[number].cs11 - ibminfo[number].cs22;
  if (ibminfo[number].ct33<0.)
    PetscPrintf(PETSC_COMM_WORLD, "SOMETHING WRONG!!!! fsi_intp Ct ii %d %le %le %le\n",number,ibminfo[number].cr33, ibminfo[number].cr22, ibminfo[number].cr11);
}

PetscErrorCode linear_intp(Cpt2D p, Cpt2D p1, Cpt2D p2, IBMInfo *ibminfo, PetscInt number, PetscInt nvert)
{
	Cpt2D pIn;
	PetscReal  x12, y12, slope, cLine, slopeReciprocal, cNormal, Cr;

	x12 = p1.x - p2.x; y12 = p1.y - p2.y;
	if(x12==0){
		pIn.x = p1.x;
		pIn.y = p.y;
		Cr = (pIn.y-p1.y)/(p2.y-p1.y);
	}
	else if(y12==0){
		pIn.x = p.x;
		pIn.y = p1.y;
		Cr = (pIn.x-p1.x)/(p2.x-p1.x);
	}
	else{
		slope = y12/x12;
		cLine = -p2.x*slope + p2.y;
		//% original line		//% y = slope*x + cLine
		slopeReciprocal = -1/slope;
		//% normal line		//% y = slopeReciprocal*x + cNormal;
		cNormal = p.y - slopeReciprocal*p.x;
		//% now interception point
		//% slopeReciprocal*x + cNormal =  slope*x + cLine
		//%(slopeReciprocal - slope)x = cLine - cNormal
		pIn.x = (cLine - cNormal)/(slopeReciprocal - slope);
		pIn.y = slopeReciprocal*pIn.x + cNormal;
		Cr = (pIn.x-p1.x)/(p2.x-p1.x);
	}
	
	if(isnan(Cr)){
		PetscPrintf(PETSC_COMM_SELF,"elmt=%d; cr is nan;\n",number);
	}

	if (nvert==1) {
		ibminfo[number].cr1 = 0.;
		ibminfo[number].cr2 = Cr;    
		ibminfo[number].cr3 = 1-Cr;
	} 
	else if (nvert==2) {
		ibminfo[number].cr1 = Cr;
		ibminfo[number].cr2 = 0.;    
		ibminfo[number].cr3 = 1-Cr;
	} 
	else if (nvert==3) {
		ibminfo[number].cr1 = Cr;
		ibminfo[number].cr2 = 1-Cr;    
		ibminfo[number].cr3 = 0.;
	} 
	else {
		PetscPrintf(PETSC_COMM_WORLD, "%Wrong Nvert in Linear intp!!!\n");		
	}
	return(0);  
}


PetscErrorCode linear_intp_2(Cpt2D p, Cpt2D p1, Cpt2D p2, IBMInfo *ibminfo, PetscInt number, PetscInt nvert)
{
	Cpt2D pIn;
	PetscReal  x12, y12, slope, cLine, slopeReciprocal, cNormal, Cr;

	x12 = p1.x - p2.x; y12 = p1.y - p2.y;
	if(x12==0){
		pIn.x = p1.x;
		pIn.y = p.y;
		Cr = (pIn.y-p1.y)/(p2.y-p1.y);
	}
	else if(y12==0){
		pIn.x = p.x;
		pIn.y = p1.y;
		Cr = (pIn.x-p1.x)/(p2.x-p1.x);
	}
	else{
		slope = y12/x12;
		cLine = -p2.x*slope + p2.y;
		//% original line		//% y = slope*x + cLine
		slopeReciprocal = -1/slope;
		//% normal line		//% y = slopeReciprocal*x + cNormal;
		cNormal = p.y - slopeReciprocal*p.x;
		//% now interception point
		//% slopeReciprocal*x + cNormal =  slope*x + cLine
		//%(slopeReciprocal - slope)x = cLine - cNormal
		pIn.x = (cLine - cNormal)/(slopeReciprocal - slope);
		pIn.y = slopeReciprocal*pIn.x + cNormal;
		Cr = (pIn.x-p1.x)/(p2.x-p1.x);
	}
	
	if(isnan(Cr)){
		PetscPrintf(PETSC_COMM_SELF,"elmt=%d; cr is nan;\n",number);
	}

	if (nvert==1) {
		ibminfo[number].ct11 = 0.;
		ibminfo[number].ct22 = Cr;    
		ibminfo[number].ct33 = 1-Cr;
	} 
	else if (nvert==2) {
		ibminfo[number].ct11 = Cr;
		ibminfo[number].ct22 = 0.;    
		ibminfo[number].ct33 = 1-Cr;
	} 
	else if (nvert==3) {
		ibminfo[number].ct11 = Cr;
		ibminfo[number].ct22 = 1-Cr;    
		ibminfo[number].ct33 = 0.;
	} 
	else {
		PetscPrintf(PETSC_COMM_WORLD, "%Wrong Nvert in Linear intp!!!\n");		
	}
	return(0);  
}

PetscErrorCode linear_intp_3(Cpt2D p, Cpt2D p1, Cpt2D p2, IBMInfo *ibminfo, PetscInt number, PetscInt nvert)
{
	Cpt2D pIn;
	PetscReal  x12, y12, slope, cLine, slopeReciprocal, cNormal, Cr;

	x12 = p1.x - p2.x; y12 = p1.y - p2.y;
	if(x12==0){
		pIn.x = p1.x;
		pIn.y = p.y;
		Cr = (pIn.y-p1.y)/(p2.y-p1.y);
	}
	else if(y12==0){
		pIn.x = p.x;
		pIn.y = p1.y;
		Cr = (pIn.x-p1.x)/(p2.x-p1.x);
	}
	else{
		slope = y12/x12;
		cLine = -p2.x*slope + p2.y;
		//% original line		//% y = slope*x + cLine
		slopeReciprocal = -1/slope;
		//% normal line		//% y = slopeReciprocal*x + cNormal;
		cNormal = p.y - slopeReciprocal*p.x;
		//% now interception point
		//% slopeReciprocal*x + cNormal =  slope*x + cLine
		//%(slopeReciprocal - slope)x = cLine - cNormal
		pIn.x = (cLine - cNormal)/(slopeReciprocal - slope);
		pIn.y = slopeReciprocal*pIn.x + cNormal;
		Cr = (pIn.x-p1.x)/(p2.x-p1.x);
	}
	
	if(isnan(Cr)){
		PetscPrintf(PETSC_COMM_SELF,"elmt=%d; cr is nan;\n",number);
	}

	if (nvert==1) {
		ibminfo[number].cr11 = 0.;
		ibminfo[number].cr22 = Cr;    
		ibminfo[number].cr33 = 1-Cr;
	} 
	else if (nvert==2) {
		ibminfo[number].cr11 = Cr;
		ibminfo[number].cr22 = 0.;    
		ibminfo[number].cr33 = 1-Cr;
	} 
	else if (nvert==3) {
		ibminfo[number].cr11 = Cr;
		ibminfo[number].cr22 = 1-Cr;    
		ibminfo[number].cr33 = 0.;
	} 
	else {
		PetscPrintf(PETSC_COMM_WORLD, "%Wrong Nvert in Linear intp!!!\n");		
	}
	return(0);  
}


PetscInt ISSameSide2D(Cpt2D p, Cpt2D p1, Cpt2D p2, Cpt2D p3)
     /* Check whether 2D point p is located on the same side of line p1p2
        with point p3. Returns:
        -1      different side
        1       same side (including the case when p is located
                right on the line)
        If p and p3 is located on the same side to line p1p2, then
        the (p-p1) X (p2-p1) and (p3-p1) X (p2-p1) should have the same sign
     */
{
  PetscReal t1, t2, t3;
  PetscReal     epsilon = 1.e-10;

  PetscReal A, B, C;
  A = p2.y - p1.y;
  B = -(p2.x - p1.x);
  C = (p2.x - p1.x) * p1.y - (p2.y - p1.y) * p1.x;

  t3 = fabs(A * p.x + B * p.y + C) / sqrt(A*A + B*B);

/*   if (t3<1.e-3) return(1); */
  if (t3 < 1.e-3) {
    t1 = A * p.x + B * p.y + C;
    t2 = A * p3.x + B * p3.y + C;
    //    if (flagprint) PetscPrintf(PETSC_COMM_WORLD, "%le %le %le %le %le %le\n", t1, t2, t3, A, B, C);
  }
  else {
    t1 = (p.x - p1.x) * (p2.y - p1.y) - (p.y - p1.y) * (p2.x - p1.x);
    t2 = (p3.x - p1.x) * (p2.y - p1.y) - (p3.y - p1.y) * (p2.x - p1.x);
  }

  //!!!!!!!!!!!!1 Change t1, t2 & lt !!!!!!!
  t1 = (p.x - p1.x) * (p2.y - p1.y) - (p.y - p1.y) * (p2.x - p1.x);
  t2 = (p3.x - p1.x) * (p2.y - p1.y) - (p3.y - p1.y) * (p2.x - p1.x);
  PetscReal lt;
  lt = sqrt((p1.x - p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y));
  //  if(flagprint) PetscPrintf(PETSC_COMM_WORLD, "%le %le %le %le %le %le\n", p1.x, p2.x, p3.x, p1.y, p2.y, p3.y);
  //if (fabs(t1) < epsilon) { // Point is located along the line of p1p2
  if (fabs(t1/lt) < epsilon) { // Point is located along the line of p1p2
    return(1);
  }
  // End of change !!!!!!!!!!!!!1

  if (t1 > 0) {
    if (t2 > 0) return (1); // same side
    else return(-1);  // not
  }
  else {
    if (t2 < 0) return(1); // same side
    else return(-1);
  }
}


PetscErrorCode fsi_InterceptionPoint(Cmpnts p, Cmpnts pc[8], PetscReal nvertpc[8], PetscReal nfx, PetscReal nfy, PetscReal nfz, IBMInfo *ibminfo, PetscInt number, Cmpnts *intp, PetscInt *Need3rdPoint)
{
  PetscInt 	triangles[3][12];
  Cmpnts   	p1, p2, p3;

  PetscReal	dx1, dy1, dz1, dx2, dy2, dz2, dx3, dy3, dz3, d;
  PetscReal	rx1, ry1, rz1, rx2, ry2, rz2;//, rx3, ry3, rz3;

  Cpt2D		pj1, pj2, pj3, pjp;
  PetscInt	cell, flag;

  PetscInt	i;
  Cmpnts	pint; // Interception point
  PetscReal	nfxt, nfyt, nfzt;
  
  PetscReal dxT, dyT, dzT;

  ibminfo[number].imode = -100;
  // k-plane
  triangles[0][0]  = 0; triangles[1][0]  = 1; triangles[2][0]  = 2;
  triangles[0][1]  = 0; triangles[1][1]  = 2; triangles[2][1]  = 3;
  triangles[0][2]  = 4; triangles[1][2]  = 5; triangles[2][2]  = 6;
  triangles[0][3]  = 4; triangles[1][3]  = 6; triangles[2][3]  = 7;
  // i-plane
  triangles[0][4]  = 0; triangles[1][4]  = 4; triangles[2][4]  = 3;
  triangles[0][5]  = 3; triangles[1][5]  = 4; triangles[2][5]  = 7;
  triangles[0][6]  = 1; triangles[1][6]  = 5; triangles[2][6]  = 2;
  triangles[0][7]  = 2; triangles[1][7]  = 6; triangles[2][7]  = 5;
  // j-plane
  triangles[0][8]  = 0; triangles[1][8]  = 4; triangles[2][8]  = 1;
  triangles[0][9]  = 1; triangles[1][9]  = 4; triangles[2][9]  = 5;
  triangles[0][10] = 3; triangles[1][10] = 7; triangles[2][10] = 2;
  triangles[0][11] = 2; triangles[1][11] = 7; triangles[2][11] = 6;

  for (i=0; i<12; i++) {
    p1 = pc[triangles[0][i]]; p2 = pc[triangles[1][i]], p3 = pc[triangles[2][i]];
    dx1 = p.x - p1.x; dy1 = p.y - p1.y; dz1 = p.z - p1.z;   // a1 = p - p1
    dx2 = p2.x - p1.x; dy2 = p2.y - p1.y; dz2 = p2.z - p1.z;// a2 = p2 - p1
    dx3 = p3.x - p1.x; dy3 = p3.y - p1.y; dz3 = p3.z - p1.z;// a3 = p3 - p1

    // area of the parralelogram since h=1 and V = ah = nf.(a2xa3)
    d = (nfx * (dy2 * dz3 - dz2 * dy3) - 
	     nfy * (dx2 * dz3 - dz2 * dx3) + 
	     nfz * (dx2 * dy3 - dy2 * dx3));
	 
    if (fabs(d) > 1.e-15) {
      // the distance of the point from the triangle plane d = Vol/area = a1.(a2xa3)/area
		d = -(dx1 * (dy2 * dz3 - dz2 * dy3) - 
		dy1 * (dx2 * dz3 - dz2 * dx3) + 
		dz1 * (dx2 * dy3 - dy2 * dx3)) / d;

		if (d>0) {
			pint.x = p.x + d * nfx;
			pint.y = p.y + d * nfy;
			pint.z = p.z + d * nfz;

			rx1 = p2.x - p1.x; ry1 = p2.y - p1.y; rz1 = p2.z - p1.z;
			rx2 = p3.x - p1.x; ry2 = p3.y - p1.y; rz2 = p3.z - p1.z;

			nfxt = ry1 * rz2 - rz1 * ry2;
			nfyt = -rx1 * rz2 + rz1 * rx2;
			nfzt = rx1 * ry2 - ry1 * rx2;

			flag = ISPointInTriangle(pint, p1, p2, p3, nfxt, nfyt, nfzt);
						
				if (flag >= 0) {
						cell = i;
					  // Calculate the interpolatin Coefficients

					 if ((int)(nvertpc[triangles[0][i]] +0.5) <= 1 &&
						  (int)(nvertpc[triangles[1][i]] +0.5) <= 1 &&
						  (int)(nvertpc[triangles[2][i]] +0.5) <= 1) {
							*Need3rdPoint = 0; //ASR
							if (fabs(nfxt) >= fabs(nfyt) && fabs(nfxt)>= fabs(nfzt)) {
							  pjp.x = pint.y; pjp.y = pint.z;
							  pj1.x = p1.y;   pj1.y = p1.z;
							  pj2.x = p2.y;   pj2.y = p2.z;
							  pj3.x = p3.y;   pj3.y = p3.z;
							  triangle_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							  triangle_intp2_fsi(pjp, pj1, pj2, pj3, ibminfo, number); // so far, what happens inside this function is not used.
							}
							else if (fabs(nfyt) >= fabs(nfxt) && fabs(nfyt)>= fabs(nfzt)) {
							  pjp.x = pint.x; pjp.y = pint.z;
							  pj1.x = p1.x;   pj1.y = p1.z;
							  pj2.x = p2.x;   pj2.y = p2.z;
							  pj3.x = p3.x;   pj3.y = p3.z;
							  triangle_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							  triangle_intp2_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
							else if (fabs(nfzt) >= fabs(nfyt) && fabs(nfzt)>= fabs(nfxt)) {
							  pjp.x = pint.y; pjp.y = pint.x;
							  pj1.x = p1.y;   pj1.y = p1.x;
							  pj2.x = p2.y;   pj2.y = p2.x;
							  pj3.x = p3.y;   pj3.y = p3.x;
							  triangle_intp_fsi(pjp, pj1, pj2, pj3, ibminfo,number);
							  triangle_intp2_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
					  } else  if ((int)(nvertpc[triangles[0][i]] +0.5) > 1 &&
							  (int)(nvertpc[triangles[1][i]] +0.5) <= 1 &&
							  (int)(nvertpc[triangles[2][i]] +0.5) <= 1) {
							*Need3rdPoint = 1;
							if (fabs(nfxt) >= fabs(nfyt) && fabs(nfxt)>= fabs(nfzt)) {
							  pjp.x = pint.y; pjp.y = pint.z;
							  pj1.x = p1.y;   pj1.y = p1.z;
							  pj2.x = p2.y;   pj2.y = p2.z;
							  pj3.x = p3.y;   pj3.y = p3.z;
							  linear_intp(pjp, pj2, pj3, ibminfo, number,1);
							  triangle_intp2_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
							else if (fabs(nfyt) >= fabs(nfxt) && fabs(nfyt)>= fabs(nfzt)) {
							  pjp.x = pint.x; pjp.y = pint.z;
							  pj1.x = p1.x;   pj1.y = p1.z;
							  pj2.x = p2.x;   pj2.y = p2.z;
							  pj3.x = p3.x;   pj3.y = p3.z;
							  linear_intp(pjp, pj2, pj3, ibminfo, number,1);
							  triangle_intp2_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
							else if (fabs(nfzt) >= fabs(nfyt) && fabs(nfzt)>= fabs(nfxt)) {
							  pjp.x = pint.y; pjp.y = pint.x;
							  pj1.x = p1.y;   pj1.y = p1.x;
							  pj2.x = p2.y;   pj2.y = p2.x;
							  pj3.x = p3.y;   pj3.y = p3.x;
							  linear_intp(pjp, pj2, pj3, ibminfo,number,1);
							  triangle_intp2_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
					  } else  if ((int)(nvertpc[triangles[0][i]] +0.5) <= 1 &&
							  (int)(nvertpc[triangles[1][i]] +0.5) > 1 &&
							  (int)(nvertpc[triangles[2][i]] +0.5) <= 1) {
							*Need3rdPoint = 1;
							if (fabs(nfxt) >= fabs(nfyt) && fabs(nfxt)>= fabs(nfzt)) {
							  pjp.x = pint.y; pjp.y = pint.z;
							  pj1.x = p1.y;   pj1.y = p1.z;
							  pj2.x = p2.y;   pj2.y = p2.z;
							  pj3.x = p3.y;   pj3.y = p3.z;
							  linear_intp(pjp, pj1, pj3, ibminfo, number,2);
							  triangle_intp2_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
							else if (fabs(nfyt) >= fabs(nfxt) && fabs(nfyt)>= fabs(nfzt)) {
							  pjp.x = pint.x; pjp.y = pint.z;
							  pj1.x = p1.x;   pj1.y = p1.z;
							  pj2.x = p2.x;   pj2.y = p2.z;
							  pj3.x = p3.x;   pj3.y = p3.z;
							  linear_intp(pjp, pj1, pj3, ibminfo, number,2);
							  triangle_intp2_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
							else if (fabs(nfzt) >= fabs(nfyt) && fabs(nfzt)>= fabs(nfxt)) {
							  pjp.x = pint.y; pjp.y = pint.x;
							  pj1.x = p1.y;   pj1.y = p1.x;
							  pj2.x = p2.y;   pj2.y = p2.x;
							  pj3.x = p3.y;   pj3.y = p3.x;
							  linear_intp(pjp, pj1, pj3, ibminfo,number,2);
							  triangle_intp2_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}

					  } else  if ((int)(nvertpc[triangles[0][i]] +0.5) <= 1 &&
							  (int)(nvertpc[triangles[1][i]] +0.5) <= 1 &&
							  (int)(nvertpc[triangles[2][i]] +0.5) > 1) {
							*Need3rdPoint = 1;
							if (fabs(nfxt) >= fabs(nfyt) && fabs(nfxt)>= fabs(nfzt)) {
							  pjp.x = pint.y; pjp.y = pint.z;
							  pj1.x = p1.y;   pj1.y = p1.z;
							  pj2.x = p2.y;   pj2.y = p2.z;
							  pj3.x = p3.y;   pj3.y = p3.z;
							  linear_intp(pjp, pj1, pj2, ibminfo, number,3);
							  triangle_intp2_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
							else if (fabs(nfyt) >= fabs(nfxt) && fabs(nfyt)>= fabs(nfzt)) {
							  pjp.x = pint.x; pjp.y = pint.z;
							  pj1.x = p1.x;   pj1.y = p1.z;
							  pj2.x = p2.x;   pj2.y = p2.z;
							  pj3.x = p3.x;   pj3.y = p3.z;
							  linear_intp(pjp, pj1, pj2, ibminfo, number,3);
							  triangle_intp2_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
							else if (fabs(nfzt) >= fabs(nfyt) && fabs(nfzt)>= fabs(nfxt)) {
							  pjp.x = pint.y; pjp.y = pint.x;
							  pj1.x = p1.y;   pj1.y = p1.x;
							  pj2.x = p2.y;   pj2.y = p2.x;
							  pj3.x = p3.y;   pj3.y = p3.x;
							  linear_intp(pjp, pj1, pj2, ibminfo,number,3);
							  triangle_intp2_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
					  } else  if ((int)(nvertpc[triangles[0][i]] +0.5) <= 1 &&
							  (int)(nvertpc[triangles[1][i]] +0.5) > 1 &&
							  (int)(nvertpc[triangles[2][i]] +0.5) > 1) {
							*Need3rdPoint = -1;
							ibminfo[number].cr1 = 1.;
							ibminfo[number].cr2 = 0.;
							ibminfo[number].cr3 = 0.;
							if (fabs(nfxt) >= fabs(nfyt) && fabs(nfxt)>= fabs(nfzt)) {
							  pjp.x = pint.y; pjp.y = pint.z;
							  pj1.x = p1.y;   pj1.y = p1.z;
							  pj2.x = p2.y;   pj2.y = p2.z;
							  pj3.x = p3.y;   pj3.y = p3.z;
							  triangle_intp2_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
							else if (fabs(nfyt) >= fabs(nfxt) && fabs(nfyt)>= fabs(nfzt)) {
							  pjp.x = pint.x; pjp.y = pint.z;
							  pj1.x = p1.x;   pj1.y = p1.z;
							  pj2.x = p2.x;   pj2.y = p2.z;
							  pj3.x = p3.x;   pj3.y = p3.z;
							  triangle_intp2_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
							else if (fabs(nfzt) >= fabs(nfyt) && fabs(nfzt)>= fabs(nfxt)) {
							  pjp.x = pint.y; pjp.y = pint.x;
							  pj1.x = p1.y;   pj1.y = p1.x;
							  pj2.x = p2.y;   pj2.y = p2.x;
							  pj3.x = p3.y;   pj3.y = p3.x;
							  triangle_intp2_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
					  } else  if ((int)(nvertpc[triangles[0][i]] +0.5) > 1 &&
							  (int)(nvertpc[triangles[1][i]] +0.5) <= 1 &&
							  (int)(nvertpc[triangles[2][i]] +0.5) > 1) {
							*Need3rdPoint = -1;
							ibminfo[number].cr1 = 0.;
							ibminfo[number].cr2 = 1.;
							ibminfo[number].cr3 = 0.;
							if (fabs(nfxt) >= fabs(nfyt) && fabs(nfxt)>= fabs(nfzt)) {
							  pjp.x = pint.y; pjp.y = pint.z;
							  pj1.x = p1.y;   pj1.y = p1.z;
							  pj2.x = p2.y;   pj2.y = p2.z;
							  pj3.x = p3.y;   pj3.y = p3.z;
							  triangle_intp2_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
							else if (fabs(nfyt) >= fabs(nfxt) && fabs(nfyt)>= fabs(nfzt)) {
							  pjp.x = pint.x; pjp.y = pint.z;
							  pj1.x = p1.x;   pj1.y = p1.z;
							  pj2.x = p2.x;   pj2.y = p2.z;
							  pj3.x = p3.x;   pj3.y = p3.z;
							  triangle_intp2_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
							else if (fabs(nfzt) >= fabs(nfyt) && fabs(nfzt)>= fabs(nfxt)) {
							  pjp.x = pint.y; pjp.y = pint.x;
							  pj1.x = p1.y;   pj1.y = p1.x;
							  pj2.x = p2.y;   pj2.y = p2.x;
							  pj3.x = p3.y;   pj3.y = p3.x;
							  triangle_intp2_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
					  } else  if ((int)(nvertpc[triangles[0][i]] +0.5) > 1 &&
							  (int)(nvertpc[triangles[1][i]] +0.5) > 1 &&
							  (int)(nvertpc[triangles[2][i]] +0.5) <= 1) {
							*Need3rdPoint = -1;
							ibminfo[number].cr1 = 0.;
							ibminfo[number].cr2 = 0.;
							ibminfo[number].cr3 = 1.;
							if (fabs(nfxt) >= fabs(nfyt) && fabs(nfxt)>= fabs(nfzt)) {
							  pjp.x = pint.y; pjp.y = pint.z;
							  pj1.x = p1.y;   pj1.y = p1.z;
							  pj2.x = p2.y;   pj2.y = p2.z;
							  pj3.x = p3.y;   pj3.y = p3.z;
							  triangle_intp2_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
							else if (fabs(nfyt) >= fabs(nfxt) && fabs(nfyt)>= fabs(nfzt)) {
							  pjp.x = pint.x; pjp.y = pint.z;
							  pj1.x = p1.x;   pj1.y = p1.z;
							  pj2.x = p2.x;   pj2.y = p2.z;
							  pj3.x = p3.x;   pj3.y = p3.z;
							  triangle_intp2_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
							else if (fabs(nfzt) >= fabs(nfyt) && fabs(nfzt)>= fabs(nfxt)) {
							  pjp.x = pint.y; pjp.y = pint.x;
							  pj1.x = p1.y;   pj1.y = p1.x;
							  pj2.x = p2.y;   pj2.y = p2.x;
							  pj3.x = p3.y;   pj3.y = p3.x;
							  triangle_intp2_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
					  } else {
						  *Need3rdPoint = -100;
						//PetscPrintf(PETSC_COMM_WORLD, "1st: host nodes are blanked!\n");
						//PetscPrintf(PETSC_COMM_SELF, " ASR xxv elmt = %d;  imode=%d; di=%le; nvertpc = (%d, %d, %d); \n", number, ibminfo[number].imode, d, (int)nvertpc[triangles[0][cell]],(int)nvertpc[triangles[1][cell]],(int)nvertpc[triangles[2][cell]]);
					  }
					  
					ibminfo[number].d_i = sqrt((pint.x-p.x)*(pint.x - p.x) + (pint.y-p.y) * (pint.y-p.y) + (pint.z - p.z)* (pint.z - p.z));
					ibminfo[number].imode = cell;

					/*dxT = pc[2].x - pc[1].x;
					dyT = pc[4].y - pc[1].y;
					dzT = pc[5].z - pc[1].z;
					d = d + 0.5*sqrt(dxT*dxT+dyT*dyT+dzT*dzT);*/
					d = (abs(dx2)+abs(dy2)+abs(dz2)+abs(dx3)+abs(dy3)+abs(dz3))/6;
					d = 2.1*d;
					pint.x = p.x + d * nfx;
					pint.y = p.y + d * nfy;
					pint.z = p.z + d * nfz;						

					*intp = pint;
					//if(number==elmtCheck1 || number==elmtCheck2) PetscPrintf(PETSC_COMM_SELF, " ASR xxv elmt = %d;  imode=%d; di=%le; nvertpc = (%d, %d, %d); \n", number, ibminfo[number].imode, d, (int)nvertpc[triangles[0][cell]],(int)nvertpc[triangles[1][cell]],(int)nvertpc[triangles[2][cell]]);							
				}
		} else{
				//PetscPrintf(PETSC_COMM_SELF, "Error: d is negative. elmt = %d;  imode=%d; di=%le; nvertpc = (%le, %le, %le); \n", number, ibminfo[number].imode,d, nvertpc[triangles[0][i]],nvertpc[triangles[1][i]],nvertpc[triangles[2][i]]);
		}
    } else {
		//PetscPrintf(PETSC_COMM_SELF, "Error: d(area) is very small. elmt = %d;  imode=%d; di=%le; nvertpc = (%le, %le, %le); \n", number, ibminfo[number].imode, d, nvertpc[triangles[0][i]],nvertpc[triangles[1][i]],nvertpc[triangles[2][i]]);
	}
  } // for (i=0; i<12; i++) each triangle
  
  return(0);
} //fsi_InterceptionPoint

PetscErrorCode GridCellaround2ndElmt(UserCtx *user, IBMNodes *ibm, SurfElmtInfo *elmtinfo, Cmpnts pc, PetscInt elmt, PetscInt knbn,PetscInt jnbn, PetscInt inbn, PetscInt *kin, PetscInt *jin, PetscInt *iin, PetscInt *foundFlag)
{
	DA	da = user->da, fda = user->fda;
	DALocalInfo	info = user->info;
	PetscInt	xs = info.xs, xe = info.xs + info.xm;
	PetscInt  	ys = info.ys, ye = info.ys + info.ym;
	PetscInt	zs = info.zs, ze = info.zs + info.zm;
	PetscInt	mx = info.mx, my = info.my, mz = info.mz; 
	PetscInt      lxs, lxe, lys, lye, lzs, lze;
	Vec Coor;

	lxs = xs; lxe = xe;
	lys = ys; lye = ye;
	lzs = zs; lze = ze;

	if (xs==0) lxs = xs+1;
	if (ys==0) lys = ys+1;
	if (zs==0) lzs = zs+1;

	if (xe==mx) lxe = xe-2;
	if (ye==my) lye = ye-2;
	if (ze==mz) lze = ze-2;

	PetscInt      zstart,zend,ystart,yend,xstart,xend;
	PetscInt	i, j, k, notfound;

	PetscInt      n_elmt = ibm->n_elmt;
	PetscInt      nradius=6;
	Cmpnts        ***coor,cell[8];
	PetscReal     d[6];

	DAGetCoordinates(da, &Coor);
	DAVecGetArray(fda, Coor,&coor);

	zstart=knbn-nradius; zend=knbn+nradius;
	ystart=jnbn-nradius; yend=jnbn+nradius;
	xstart=inbn-nradius; xend=inbn+nradius;

	if (zstart<lzs) zstart=lzs;
	if (ystart<lys) ystart=lys;
	if (xstart<lxs) xstart=lxs;
	if (zend>lze) zend=lze;
	if (yend>lye) yend=lye;
	if (xend>lxe) xend=lxe;

	notfound=0;
	*foundFlag = 0;

    for (k=zstart; k<zend; k++) {
      for (j=ystart; j<yend; j++) {
		for (i=xstart; i<xend; i++) {
		  cell[0] = coor[k  ][j  ][i  ];
		  cell[1] = coor[k  ][j  ][i+1];
		  cell[2] = coor[k  ][j+1][i+1];
		  cell[3] = coor[k  ][j+1][i  ];
		  
		  cell[4] = coor[k+1][j  ][i  ];
		  cell[5] = coor[k+1][j  ][i+1];
		  cell[6] = coor[k+1][j+1][i+1];
		  cell[7] = coor[k+1][j+1][i  ];
		
		  if(ISInsideCell(pc, cell, d)){
			*kin=k;
			*jin=j;
			*iin=i;

			//if (j==22)  PetscPrintf(PETSC_COMM_SELF, "%le %le\n",coor[k][j][i].y, coor[k][j+1][i].y);
			// correction if pt exactly on one side of the cell
					
			if (fabs(d[0])<1e-6 && ibm->nf_x[elmt]*(cell[1].x-cell[0].x)<0.) *iin=i-1;//elmtinfo[elmt].icell=i-1;
			if (fabs(d[1])<1e-6 && ibm->nf_x[elmt]*(cell[0].x-cell[1].x)<0.) *iin=i+1;//elmtinfo[elmt].icell=i+1;
			if (fabs(d[2])<1e-6 && ibm->nf_y[elmt]*(cell[3].y-cell[0].y)<0.) *jin=j-1;//elmtinfo[elmt].jcell=j-1;
			if (fabs(d[3])<1e-6 && ibm->nf_y[elmt]*(cell[0].y-cell[3].y)<0.) *jin=j+1;//elmtinfo[elmt].jcell=j+1;
			if (fabs(d[4])<1e-6 && ibm->nf_z[elmt]*(cell[4].z-cell[0].z)<0.) *kin=k-1;//elmtinfo[elmt].kcell=k-1;
			if (fabs(d[5])<1e-6 && ibm->nf_z[elmt]*(cell[0].z-cell[4].z)<0.) *kin=k+1;//elmtinfo[elmt].kcell=k+1;
			
			notfound=1;
			*foundFlag = 1;
			elmtinfo[elmt].FoundAround2ndCell=1;
			break;
		  }
		}
      }	
    }

    if (!notfound) {
		*iin = inbn;
		*jin = jnbn;
		*kin = knbn;
		elmtinfo[elmt].FoundAround2ndCell=0;
		PetscPrintf(PETSC_COMM_SELF, "2nd Around Cell WAS NOT FOUND! %d %d %d %d\n", elmt,inbn,jnbn,knbn);
	}
	
  DAVecRestoreArray(fda, Coor,&coor);

  return(0);
}


PetscErrorCode Find_fsi_2nd_interp_Coeff(PetscInt foundFlag, PetscInt i, PetscInt j, PetscInt k, PetscInt elmt, Cmpnts p, IBMInfo *ibminfo, UserCtx *user, IBMNodes *ibm, SurfElmtInfo *elmtinfo)
{

/* Note:  ibminfo returns the interpolation info 
   for the fsi (fsi_intp) */

  DA	da = user->da, fda = user->fda;
  DALocalInfo	info = user->info;
  PetscInt	xs = info.xs, xe = info.xs + info.xm;
  PetscInt  	ys = info.ys, ye = info.ys + info.ym;
  PetscInt	zs = info.zs, ze = info.zs + info.zm;
  PetscInt	mx = info.mx, my = info.my, mz = info.mz;
    PetscInt      lxs, lxe, lys, lye, lzs, lze;
	Vec Coor;

  PetscInt	i2, j2, k2;

  PetscInt      ip[8],jp[8],kp[8];
  Cmpnts        ***coor,pc[8];
  Cmpnts        intp, pOriginal;
  PetscReal	***nvert,nvertpc[8];
  PetscReal     nfx,nfy,nfz;
  
  	lxs = xs; lxe = xe;
	lys = ys; lye = ye;
	lzs = zs; lze = ze;

	if (xs==0) lxs = xs+1; 
	if (ys==0) lys = ys+1; 
	if (zs==0) lzs = zs+1; 
	if (xe==mx) lxe = xe-2; 
	if (ye==my) lye = ye-2; 
	if (ze==mz) lze = ze-2;

  PetscInt	rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  
  DAGetCoordinates(da, &Coor);
  DAVecGetArray(fda, Coor,&coor);
  DAVecGetArray(da, user->Nvert, &nvert);

	nfx=ibm->nf_x[elmt];
	nfy=ibm->nf_y[elmt];
	nfz=ibm->nf_z[elmt];

	pOriginal.x=ibm->cent_x[elmt]; 
	pOriginal.y=ibm->cent_y[elmt]; 
	pOriginal.z=ibm->cent_z[elmt]; 

    // normal correction for near domain bndry pts
/*     if (i==1) { */
/*       nfx=0.;//-0.0001*PetscSign(coor[k][j][i].x-coor[k][j][i+1].x); */
/*     } */
    
/*     if (i==mx-3) { */
/*       nfx=0.;//-0.0001*PetscSign(coor[k][j][i].x-coor[k][j][i-1].x); */
/*     } */


  // normal correction for near domain bndry pts
  //if (i>=xs && i<xe && j>=ys && j<ye && k>=zs && k<ze) {
  if (i>=lxs && i<lxe && j>=lys && j<lye && k>=lzs && k<lze) {

	pc[0] = coor[k  ][j  ][i  ];
	pc[1] = coor[k  ][j  ][i+1];
	pc[2] = coor[k  ][j+1][i+1];
	pc[3] = coor[k  ][j+1][i  ];

	pc[4] = coor[k+1][j  ][i  ];
	pc[5] = coor[k+1][j  ][i+1];
	pc[6] = coor[k+1][j+1][i+1];
	pc[7] = coor[k+1][j+1][i  ];

	nvertpc[0] = nvert[k  ][j  ][i  ];
	nvertpc[1] = nvert[k  ][j  ][i+1];
	nvertpc[2] = nvert[k  ][j+1][i+1];
	nvertpc[3] = nvert[k  ][j+1][i  ];

	nvertpc[4] = nvert[k+1][j  ][i  ];
	nvertpc[5] = nvert[k+1][j  ][i+1];
	nvertpc[6] = nvert[k+1][j+1][i+1];
	nvertpc[7] = nvert[k+1][j+1][i  ];


	kp[0]=k  ;jp[0]=j  ;ip[0]=i  ;
	kp[1]=k  ;jp[1]=j  ;ip[1]=i+1;
	kp[2]=k  ;jp[2]=j+1;ip[2]=i+1;
	kp[3]=k  ;jp[3]=j+1;ip[3]=i  ;
	kp[4]=k+1;jp[4]=j  ;ip[4]=i  ;
	kp[5]=k+1;jp[5]=j  ;ip[5]=i+1;
	kp[6]=k+1;jp[6]=j+1;ip[6]=i+1;
	kp[7]=k+1;jp[7]=j+1;ip[7]=i  ;

//	if(foundFlag){

		fsi_2nd_InterceptionPoint(p, pc, nvertpc, nfx, nfy, nfz, ibminfo, elmt, &intp, pOriginal, &elmtinfo[elmt].Need3rdPoint_2);
			
		switch (ibminfo[elmt].smode) {
			case(0): {
			  ibminfo[elmt].ii1=ip[0]; ibminfo[elmt].jj1 = jp[0]; ibminfo[elmt].kk1 = kp[0];
			  ibminfo[elmt].ii2=ip[1]; ibminfo[elmt].jj2 = jp[1]; ibminfo[elmt].kk2 = kp[1];
			  ibminfo[elmt].ii3=ip[2]; ibminfo[elmt].jj3 = jp[2]; ibminfo[elmt].kk3 = kp[2];
		       GridCellaround2ndElmt(user,ibm,elmtinfo,intp,elmt,k,j,i,&k2,&j2,&i2,&foundFlag); 
		       Find_fsi_3rd_interp_Coeff(i2,j2,k2,elmt,intp,ibminfo,user,ibm,elmtinfo); 
			  break;
			}
			case (1): {
			  ibminfo[elmt].ii1=ip[0]; ibminfo[elmt].jj1 = jp[0]; ibminfo[elmt].kk1 = kp[0];
			  ibminfo[elmt].ii2=ip[2]; ibminfo[elmt].jj2 = jp[2]; ibminfo[elmt].kk2 = kp[2];
			  ibminfo[elmt].ii3=ip[3]; ibminfo[elmt].jj3 = jp[3]; ibminfo[elmt].kk3 = kp[3];
		       GridCellaround2ndElmt(user,ibm,elmtinfo,intp,elmt,k,j,i,&k2,&j2,&i2,&foundFlag); 
		       Find_fsi_3rd_interp_Coeff(i2,j2,k2,elmt,intp,ibminfo,user,ibm,elmtinfo); 
			  break;
			}
			case (2): {
			  ibminfo[elmt].ii1=ip[4]; ibminfo[elmt].jj1 = jp[4]; ibminfo[elmt].kk1 = kp[4];
			  ibminfo[elmt].ii2=ip[5]; ibminfo[elmt].jj2 = jp[5]; ibminfo[elmt].kk2 = kp[5];
			  ibminfo[elmt].ii3=ip[6]; ibminfo[elmt].jj3 = jp[6]; ibminfo[elmt].kk3 = kp[6];
		       GridCellaround2ndElmt(user,ibm,elmtinfo,intp,elmt,k,j,i,&k2,&j2,&i2,&foundFlag); 
		       Find_fsi_3rd_interp_Coeff(i2,j2,k2,elmt,intp,ibminfo,user,ibm,elmtinfo); 
			  break;
			}
			case (3): {
			  ibminfo[elmt].ii1=ip[4]; ibminfo[elmt].jj1 = jp[4]; ibminfo[elmt].kk1 = kp[4];
			  ibminfo[elmt].ii2=ip[6]; ibminfo[elmt].jj2 = jp[6]; ibminfo[elmt].kk2 = kp[6];
			  ibminfo[elmt].ii3=ip[7]; ibminfo[elmt].jj3 = jp[7]; ibminfo[elmt].kk3 = kp[7];
		       GridCellaround2ndElmt(user,ibm,elmtinfo,intp,elmt,k,j,i,&k2,&j2,&i2,&foundFlag); 
		       Find_fsi_3rd_interp_Coeff(i2,j2,k2,elmt,intp,ibminfo,user,ibm,elmtinfo); 
			  break;
			}
			case (4): {
			  ibminfo[elmt].ii1=ip[0]; ibminfo[elmt].jj1 = jp[0]; ibminfo[elmt].kk1 = kp[0];
			  ibminfo[elmt].ii2=ip[4]; ibminfo[elmt].jj2 = jp[4]; ibminfo[elmt].kk2 = kp[4];
			  ibminfo[elmt].ii3=ip[3]; ibminfo[elmt].jj3 = jp[3]; ibminfo[elmt].kk3 = kp[3];
		       GridCellaround2ndElmt(user,ibm,elmtinfo,intp,elmt,k,j,i,&k2,&j2,&i2,&foundFlag); 
		       Find_fsi_3rd_interp_Coeff(i2,j2,k2,elmt,intp,ibminfo,user,ibm,elmtinfo); 
			  break;
			}
			case (5): {
			  ibminfo[elmt].ii1=ip[3]; ibminfo[elmt].jj1 = jp[3]; ibminfo[elmt].kk1 = kp[3];
			  ibminfo[elmt].ii2=ip[4]; ibminfo[elmt].jj2 = jp[4]; ibminfo[elmt].kk2 = kp[4];
			  ibminfo[elmt].ii3=ip[7]; ibminfo[elmt].jj3 = jp[7]; ibminfo[elmt].kk3 = kp[7];
		       GridCellaround2ndElmt(user,ibm,elmtinfo,intp,elmt,k,j,i,&k2,&j2,&i2,&foundFlag); 
		       Find_fsi_3rd_interp_Coeff(i2,j2,k2,elmt,intp,ibminfo,user,ibm,elmtinfo); 
			  break;
			}
			case (6): {
			  ibminfo[elmt].ii1=ip[1]; ibminfo[elmt].jj1 = jp[1]; ibminfo[elmt].kk1 = kp[1];
			  ibminfo[elmt].ii2=ip[5]; ibminfo[elmt].jj2 = jp[5]; ibminfo[elmt].kk2 = kp[5];
			  ibminfo[elmt].ii3=ip[2]; ibminfo[elmt].jj3 = jp[2]; ibminfo[elmt].kk3 = kp[2];
		       GridCellaround2ndElmt(user,ibm,elmtinfo,intp,elmt,k,j,i,&k2,&j2,&i2,&foundFlag); 
		       Find_fsi_3rd_interp_Coeff(i2,j2,k2,elmt,intp,ibminfo,user,ibm,elmtinfo); 
			  break;
			}
			case (7): {
			  ibminfo[elmt].ii1=ip[2]; ibminfo[elmt].jj1 = jp[2]; ibminfo[elmt].kk1 = kp[2];
			  ibminfo[elmt].ii2=ip[6]; ibminfo[elmt].jj2 = jp[6]; ibminfo[elmt].kk2 = kp[6];
			  ibminfo[elmt].ii3=ip[5]; ibminfo[elmt].jj3 = jp[5]; ibminfo[elmt].kk3 = kp[5];
		       GridCellaround2ndElmt(user,ibm,elmtinfo,intp,elmt,k,j,i,&k2,&j2,&i2,&foundFlag); 
		       Find_fsi_3rd_interp_Coeff(i2,j2,k2,elmt,intp,ibminfo,user,ibm,elmtinfo); 
			  break;
			}
			case (8): {
			  ibminfo[elmt].ii1=ip[0]; ibminfo[elmt].jj1 = jp[0]; ibminfo[elmt].kk1 = kp[0];
			  ibminfo[elmt].ii2=ip[4]; ibminfo[elmt].jj2 = jp[4]; ibminfo[elmt].kk2 = kp[4];
			  ibminfo[elmt].ii3=ip[1]; ibminfo[elmt].jj3 = jp[1]; ibminfo[elmt].kk3 = kp[1];
		       GridCellaround2ndElmt(user,ibm,elmtinfo,intp,elmt,k,j,i,&k2,&j2,&i2,&foundFlag); 
		       Find_fsi_3rd_interp_Coeff(i2,j2,k2,elmt,intp,ibminfo,user,ibm,elmtinfo); 
			  break;
			}
			case (9): {
			  ibminfo[elmt].ii1=ip[1]; ibminfo[elmt].jj1 = jp[1]; ibminfo[elmt].kk1 = kp[1];
			  ibminfo[elmt].ii2=ip[4]; ibminfo[elmt].jj2 = jp[4]; ibminfo[elmt].kk2 = kp[4];
			  ibminfo[elmt].ii3=ip[5]; ibminfo[elmt].jj3 = jp[5]; ibminfo[elmt].kk3 = kp[5];
		       GridCellaround2ndElmt(user,ibm,elmtinfo,intp,elmt,k,j,i,&k2,&j2,&i2,&foundFlag); 
		       Find_fsi_3rd_interp_Coeff(i2,j2,k2,elmt,intp,ibminfo,user,ibm,elmtinfo); 
			  break;
			}
			case (10): {
			  ibminfo[elmt].ii1=ip[3]; ibminfo[elmt].jj1 = jp[3]; ibminfo[elmt].kk1 = kp[3];
			  ibminfo[elmt].ii2=ip[7]; ibminfo[elmt].jj2 = jp[7]; ibminfo[elmt].kk2 = kp[7];
			  ibminfo[elmt].ii3=ip[2]; ibminfo[elmt].jj3 = jp[2]; ibminfo[elmt].kk3 = kp[2];
		       GridCellaround2ndElmt(user,ibm,elmtinfo,intp,elmt,k,j,i,&k2,&j2,&i2,&foundFlag); 
		       Find_fsi_3rd_interp_Coeff(i2,j2,k2,elmt,intp,ibminfo,user,ibm,elmtinfo); 
			  break;
			}
			case (11): {
			  ibminfo[elmt].ii1=ip[2]; ibminfo[elmt].jj1 = jp[2]; ibminfo[elmt].kk1 = kp[2];
			  ibminfo[elmt].ii2=ip[7]; ibminfo[elmt].jj2 = jp[7]; ibminfo[elmt].kk2 = kp[7];
			  ibminfo[elmt].ii3=ip[6]; ibminfo[elmt].jj3 = jp[6]; ibminfo[elmt].kk3 = kp[6];
		       GridCellaround2ndElmt(user,ibm,elmtinfo,intp,elmt,k,j,i,&k2,&j2,&i2,&foundFlag); 
		       Find_fsi_3rd_interp_Coeff(i2,j2,k2,elmt,intp,ibminfo,user,ibm,elmtinfo);
			  break;
			}
		}//end switch
//	} // if(foundFlag)
    //if (ibminfo[elmt].smode<0) PetscPrintf(PETSC_COMM_SELF, "2nd FSI Interpolation Coeffients Were not Found!!!! foundFlag=%d elmt=%d  smode=%d d_s=%le i=%d j=%d k=%d\n", foundFlag, elmt, ibminfo[elmt].smode,ibminfo[elmt].d_s,i,j,k);
    //PetscPrintf(PETSC_COMM_SELF, "FSI Interpolatoion host %d  %d  %d \n", ibminfo[elmt].i1,ibminfo[elmt].j1,ibminfo[elmt].k1);
  }

  DAVecRestoreArray(fda, Coor,&coor);  
  DAVecRestoreArray(da, user->Nvert, &nvert);
  return(0);
}

PetscErrorCode fsi_2nd_InterceptionPoint(Cmpnts p, Cmpnts pc[8], PetscReal nvertpc[8], PetscReal nfx, PetscReal nfy, PetscReal nfz, IBMInfo *ibminfo, PetscInt number, Cmpnts *intp, Cmpnts pOriginal, PetscInt *Need3rdPoint_2)
{
  PetscInt 	triangles[3][12];
  Cmpnts   	p1, p2, p3;

  PetscReal	dx1, dy1, dz1, dx2, dy2, dz2, dx3, dy3, dz3, d;
  PetscReal	rx1, ry1, rz1, rx2, ry2, rz2;//, rx3, ry3, rz3;

  Cpt2D		pj1, pj2, pj3, pjp;
  PetscInt	cell, flag;
  
  PetscInt	i;
  Cmpnts	pint; // Interception point
  PetscReal	nfxt, nfyt, nfzt;

  ibminfo[number].smode = -100;
  // k-plane
  triangles[0][0]  = 0; triangles[1][0]  = 1; triangles[2][0]  = 2;
  triangles[0][1]  = 0; triangles[1][1]  = 2; triangles[2][1]  = 3;
  triangles[0][2]  = 4; triangles[1][2]  = 5; triangles[2][2]  = 6;
  triangles[0][3]  = 4; triangles[1][3]  = 6; triangles[2][3]  = 7;
  // i-plane
  triangles[0][4]  = 0; triangles[1][4]  = 4; triangles[2][4]  = 3;
  triangles[0][5]  = 3; triangles[1][5]  = 4; triangles[2][5]  = 7;
  triangles[0][6]  = 1; triangles[1][6]  = 5; triangles[2][6]  = 2;
  triangles[0][7]  = 2; triangles[1][7]  = 6; triangles[2][7]  = 5;
  // j-plane
  triangles[0][8]  = 0; triangles[1][8]  = 4; triangles[2][8]  = 1;
  triangles[0][9]  = 1; triangles[1][9]  = 4; triangles[2][9]  = 5;
  triangles[0][10] = 3; triangles[1][10] = 7; triangles[2][10] = 2;
  triangles[0][11] = 2; triangles[1][11] = 7; triangles[2][11] = 6;

  for (i=0; i<12; i++) {
    p1 = pc[triangles[0][i]]; p2 = pc[triangles[1][i]], p3 = pc[triangles[2][i]];
    dx1 = p.x - p1.x; dy1 = p.y - p1.y; dz1 = p.z - p1.z;   // a1 = p - p1
    dx2 = p2.x - p1.x; dy2 = p2.y - p1.y; dz2 = p2.z - p1.z;// a2 = p2 - p1
    dx3 = p3.x - p1.x; dy3 = p3.y - p1.y; dz3 = p3.z - p1.z;// a3 = p3 - p1

    // area of the parralelogram since h=1 and V = ah = nf.(a2xa3)
    d = (nfx * (dy2 * dz3 - dz2 * dy3) - 
	     nfy * (dx2 * dz3 - dz2 * dx3) + 
	     nfz * (dx2 * dy3 - dy2 * dx3));
	 
    if (fabs(d) > 1.e-15) {
      // the distance of the point from the triangle plane d = Vol/area = a1.(a2xa3)/area
		d = -(dx1 * (dy2 * dz3 - dz2 * dy3) - 
		dy1 * (dx2 * dz3 - dz2 * dx3) + 
		dz1 * (dx2 * dy3 - dy2 * dx3)) / d;
		
		if (d>0) {
			pint.x = p.x + d * nfx;
			pint.y = p.y + d * nfy;
			pint.z = p.z + d * nfz;

			rx1 = p2.x - p1.x; ry1 = p2.y - p1.y; rz1 = p2.z - p1.z;
			rx2 = p3.x - p1.x; ry2 = p3.y - p1.y; rz2 = p3.z - p1.z;

			nfxt = ry1 * rz2 - rz1 * ry2;
			nfyt = -rx1 * rz2 + rz1 * rx2;
			nfzt = rx1 * ry2 - ry1 * rx2;

			flag = ISPointInTriangle(pint, p1, p2, p3, nfxt, nfyt, nfzt);
			
		if (flag >= 0) {
		  cell = i;
			
			
		  /*	  if (flagprint==1) {
			PetscPrintf(PETSC_COMM_WORLD, "%e %e %e %e \n", pint.x, pint.y, pint.z, d);
			PetscPrintf(PETSC_COMM_WORLD, "%e %e %e\n", nfx, nfy, nfz);
			PetscPrintf(PETSC_COMM_WORLD, "%e %e %e\n", p2.x, p2.y, p2.z);
			PetscPrintf(PETSC_COMM_WORLD, "%e %e %e\n", p3.x, p3.y, p3.z);
			}*/

		  // Calculate the interpolation Coefficients
		  
		/*  if (fabs(nfxt) >= fabs(nfyt) && fabs(nfxt)>= fabs(nfzt)) {
			pjp.x = pint.y; pjp.y = pint.z;
			pj1.x = p1.y;   pj1.y = p1.z;
			pj2.x = p2.y;   pj2.y = p2.z;
			pj3.x = p3.y;   pj3.y = p3.z;
			triangle_2nd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
		  }
		  else if (fabs(nfyt) >= fabs(nfxt) && fabs(nfyt)>= fabs(nfzt)) {
			pjp.x = pint.x; pjp.y = pint.z;
			pj1.x = p1.x;   pj1.y = p1.z;
			pj2.x = p2.x;   pj2.y = p2.z;
			pj3.x = p3.x;   pj3.y = p3.z;
			triangle_2nd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
		  }
		  else if (fabs(nfzt) >= fabs(nfyt) && fabs(nfzt)>= fabs(nfxt)) {
			pjp.x = pint.y; pjp.y = pint.x;
			pj1.x = p1.y;   pj1.y = p1.x;
			pj2.x = p2.y;   pj2.y = p2.x;
			pj3.x = p3.y;   pj3.y = p3.x;
			triangle_2nd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo,number);
		  }*/
		  
					if ((int)(nvertpc[triangles[0][i]] +0.5) <= 1 &&
						  (int)(nvertpc[triangles[1][i]] +0.5) <= 1 &&
						  (int)(nvertpc[triangles[2][i]] +0.5) <= 1) {
							*Need3rdPoint_2 = 0; //ASR
							if (fabs(nfxt) >= fabs(nfyt) && fabs(nfxt)>= fabs(nfzt)) {
							  pjp.x = pint.y; pjp.y = pint.z;
							  pj1.x = p1.y;   pj1.y = p1.z;
							  pj2.x = p2.y;   pj2.y = p2.z;
							  pj3.x = p3.y;   pj3.y = p3.z;
							  triangle_intp_fsi_2(pjp, pj1, pj2, pj3, ibminfo, number);
							  triangle_2nd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number); // so far, what happens inside this function is not used.
							}
							else if (fabs(nfyt) >= fabs(nfxt) && fabs(nfyt)>= fabs(nfzt)) {
							  pjp.x = pint.x; pjp.y = pint.z;
							  pj1.x = p1.x;   pj1.y = p1.z;
							  pj2.x = p2.x;   pj2.y = p2.z;
							  pj3.x = p3.x;   pj3.y = p3.z;
							  triangle_intp_fsi_2(pjp, pj1, pj2, pj3, ibminfo, number);
							  triangle_2nd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
							else if (fabs(nfzt) >= fabs(nfyt) && fabs(nfzt)>= fabs(nfxt)) {
							  pjp.x = pint.y; pjp.y = pint.x;
							  pj1.x = p1.y;   pj1.y = p1.x;
							  pj2.x = p2.y;   pj2.y = p2.x;
							  pj3.x = p3.y;   pj3.y = p3.x;
							  triangle_intp_fsi_2(pjp, pj1, pj2, pj3, ibminfo,number);
							  triangle_2nd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
					  } else  if ((int)(nvertpc[triangles[0][i]] +0.5) > 1 &&
							  (int)(nvertpc[triangles[1][i]] +0.5) <= 1 &&
							  (int)(nvertpc[triangles[2][i]] +0.5) <= 1) {
							*Need3rdPoint_2 = 1;
							if (fabs(nfxt) >= fabs(nfyt) && fabs(nfxt)>= fabs(nfzt)) {
							  pjp.x = pint.y; pjp.y = pint.z;
							  pj1.x = p1.y;   pj1.y = p1.z;
							  pj2.x = p2.y;   pj2.y = p2.z;
							  pj3.x = p3.y;   pj3.y = p3.z;
							  linear_intp_2(pjp, pj2, pj3, ibminfo, number,1);
							  triangle_2nd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
							else if (fabs(nfyt) >= fabs(nfxt) && fabs(nfyt)>= fabs(nfzt)) {
							  pjp.x = pint.x; pjp.y = pint.z;
							  pj1.x = p1.x;   pj1.y = p1.z;
							  pj2.x = p2.x;   pj2.y = p2.z;
							  pj3.x = p3.x;   pj3.y = p3.z;
							  linear_intp_2(pjp, pj2, pj3, ibminfo, number,1);
							  triangle_2nd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
							else if (fabs(nfzt) >= fabs(nfyt) && fabs(nfzt)>= fabs(nfxt)) {
							  pjp.x = pint.y; pjp.y = pint.x;
							  pj1.x = p1.y;   pj1.y = p1.x;
							  pj2.x = p2.y;   pj2.y = p2.x;
							  pj3.x = p3.y;   pj3.y = p3.x;
							  linear_intp_2(pjp, pj2, pj3, ibminfo,number,1);
							  triangle_2nd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
					  } else  if ((int)(nvertpc[triangles[0][i]] +0.5) <= 1 &&
							  (int)(nvertpc[triangles[1][i]] +0.5) > 1 &&
							  (int)(nvertpc[triangles[2][i]] +0.5) <= 1) {
							*Need3rdPoint_2 = 1;
							if (fabs(nfxt) >= fabs(nfyt) && fabs(nfxt)>= fabs(nfzt)) {
							  pjp.x = pint.y; pjp.y = pint.z;
							  pj1.x = p1.y;   pj1.y = p1.z;
							  pj2.x = p2.y;   pj2.y = p2.z;
							  pj3.x = p3.y;   pj3.y = p3.z;
							  linear_intp_2(pjp, pj1, pj3, ibminfo, number,2);
							  triangle_2nd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
							else if (fabs(nfyt) >= fabs(nfxt) && fabs(nfyt)>= fabs(nfzt)) {
							  pjp.x = pint.x; pjp.y = pint.z;
							  pj1.x = p1.x;   pj1.y = p1.z;
							  pj2.x = p2.x;   pj2.y = p2.z;
							  pj3.x = p3.x;   pj3.y = p3.z;
							  linear_intp_2(pjp, pj1, pj3, ibminfo, number,2);
							  triangle_2nd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
							else if (fabs(nfzt) >= fabs(nfyt) && fabs(nfzt)>= fabs(nfxt)) {
							  pjp.x = pint.y; pjp.y = pint.x;
							  pj1.x = p1.y;   pj1.y = p1.x;
							  pj2.x = p2.y;   pj2.y = p2.x;
							  pj3.x = p3.y;   pj3.y = p3.x;
							  linear_intp_2(pjp, pj1, pj3, ibminfo,number,2);
							  triangle_2nd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
					  } else  if ((int)(nvertpc[triangles[0][i]] +0.5) <= 1 &&
							  (int)(nvertpc[triangles[1][i]] +0.5) <= 1 &&
							  (int)(nvertpc[triangles[2][i]] +0.5) > 1) {
							*Need3rdPoint_2 = 1;
							if (fabs(nfxt) >= fabs(nfyt) && fabs(nfxt)>= fabs(nfzt)) {
							  pjp.x = pint.y; pjp.y = pint.z;
							  pj1.x = p1.y;   pj1.y = p1.z;
							  pj2.x = p2.y;   pj2.y = p2.z;
							  pj3.x = p3.y;   pj3.y = p3.z;
							  linear_intp_2(pjp, pj1, pj2, ibminfo, number,3);
							  triangle_2nd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
							else if (fabs(nfyt) >= fabs(nfxt) && fabs(nfyt)>= fabs(nfzt)) {
							  pjp.x = pint.x; pjp.y = pint.z;
							  pj1.x = p1.x;   pj1.y = p1.z;
							  pj2.x = p2.x;   pj2.y = p2.z;
							  pj3.x = p3.x;   pj3.y = p3.z;
							  linear_intp_2(pjp, pj1, pj2, ibminfo, number,3);
							  triangle_2nd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
							else if (fabs(nfzt) >= fabs(nfyt) && fabs(nfzt)>= fabs(nfxt)) {
							  pjp.x = pint.y; pjp.y = pint.x;
							  pj1.x = p1.y;   pj1.y = p1.x;
							  pj2.x = p2.y;   pj2.y = p2.x;
							  pj3.x = p3.y;   pj3.y = p3.x;
							  linear_intp_2(pjp, pj1, pj2, ibminfo,number,3);
							  triangle_2nd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
					  } else  if ((int)(nvertpc[triangles[0][i]] +0.5) <= 1 &&
							  (int)(nvertpc[triangles[1][i]] +0.5) > 1 &&
							  (int)(nvertpc[triangles[2][i]] +0.5) > 1) {
							*Need3rdPoint_2 = -1;
							ibminfo[number].ct11 = 1.;
							ibminfo[number].ct22 = 0.;
							ibminfo[number].ct33 = 0.;
							if (fabs(nfxt) >= fabs(nfyt) && fabs(nfxt)>= fabs(nfzt)) {
							  pjp.x = pint.y; pjp.y = pint.z;
							  pj1.x = p1.y;   pj1.y = p1.z;
							  pj2.x = p2.y;   pj2.y = p2.z;
							  pj3.x = p3.y;   pj3.y = p3.z;
							  triangle_2nd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
							else if (fabs(nfyt) >= fabs(nfxt) && fabs(nfyt)>= fabs(nfzt)) {
							  pjp.x = pint.x; pjp.y = pint.z;
							  pj1.x = p1.x;   pj1.y = p1.z;
							  pj2.x = p2.x;   pj2.y = p2.z;
							  pj3.x = p3.x;   pj3.y = p3.z;
							  triangle_2nd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
							else if (fabs(nfzt) >= fabs(nfyt) && fabs(nfzt)>= fabs(nfxt)) {
							  pjp.x = pint.y; pjp.y = pint.x;
							  pj1.x = p1.y;   pj1.y = p1.x;
							  pj2.x = p2.y;   pj2.y = p2.x;
							  pj3.x = p3.y;   pj3.y = p3.x;
							  triangle_2nd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
					  } else  if ((int)(nvertpc[triangles[0][i]] +0.5) > 1 &&
							  (int)(nvertpc[triangles[1][i]] +0.5) <= 1 &&
							  (int)(nvertpc[triangles[2][i]] +0.5) > 1) {
							*Need3rdPoint_2 = -1;
							ibminfo[number].ct11 = 0.;
							ibminfo[number].ct22 = 1.;
							ibminfo[number].ct33 = 0.;
							if (fabs(nfxt) >= fabs(nfyt) && fabs(nfxt)>= fabs(nfzt)) {
							  pjp.x = pint.y; pjp.y = pint.z;
							  pj1.x = p1.y;   pj1.y = p1.z;
							  pj2.x = p2.y;   pj2.y = p2.z;
							  pj3.x = p3.y;   pj3.y = p3.z;
							  triangle_2nd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
							else if (fabs(nfyt) >= fabs(nfxt) && fabs(nfyt)>= fabs(nfzt)) {
							  pjp.x = pint.x; pjp.y = pint.z;
							  pj1.x = p1.x;   pj1.y = p1.z;
							  pj2.x = p2.x;   pj2.y = p2.z;
							  pj3.x = p3.x;   pj3.y = p3.z;
							  triangle_2nd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
							else if (fabs(nfzt) >= fabs(nfyt) && fabs(nfzt)>= fabs(nfxt)) {
							  pjp.x = pint.y; pjp.y = pint.x;
							  pj1.x = p1.y;   pj1.y = p1.x;
							  pj2.x = p2.y;   pj2.y = p2.x;
							  pj3.x = p3.y;   pj3.y = p3.x;
							  triangle_2nd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
					  } else  if ((int)(nvertpc[triangles[0][i]] +0.5) > 1 &&
							  (int)(nvertpc[triangles[1][i]] +0.5) > 1 &&
							  (int)(nvertpc[triangles[2][i]] +0.5) <= 1) {
							*Need3rdPoint_2 = -1;
							ibminfo[number].ct11 = 0.;
							ibminfo[number].ct22 = 0.;
							ibminfo[number].ct33 = 1.;
							if (fabs(nfxt) >= fabs(nfyt) && fabs(nfxt)>= fabs(nfzt)) {
							  pjp.x = pint.y; pjp.y = pint.z;
							  pj1.x = p1.y;   pj1.y = p1.z;
							  pj2.x = p2.y;   pj2.y = p2.z;
							  pj3.x = p3.y;   pj3.y = p3.z;
							  triangle_2nd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
							else if (fabs(nfyt) >= fabs(nfxt) && fabs(nfyt)>= fabs(nfzt)) {
							  pjp.x = pint.x; pjp.y = pint.z;
							  pj1.x = p1.x;   pj1.y = p1.z;
							  pj2.x = p2.x;   pj2.y = p2.z;
							  pj3.x = p3.x;   pj3.y = p3.z;
							  triangle_2nd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
							else if (fabs(nfzt) >= fabs(nfyt) && fabs(nfzt)>= fabs(nfxt)) {
							  pjp.x = pint.y; pjp.y = pint.x;
							  pj1.x = p1.y;   pj1.y = p1.x;
							  pj2.x = p2.y;   pj2.y = p2.x;
							  pj3.x = p3.y;   pj3.y = p3.x;
							  triangle_2nd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
					  } else {
						*Need3rdPoint_2 = -100;
					  }
		  		  
			ibminfo[number].d_s = sqrt((pint.x-pOriginal.x)*(pint.x - pOriginal.x) + (pint.y-pOriginal.y) * (pint.y-pOriginal.y) + (pint.z - pOriginal.z)* (pint.z - pOriginal.z));
			ibminfo[number].smode = cell;
			
			d = (abs(dx2)+abs(dy2)+abs(dz2)+abs(dx3)+abs(dy3)+abs(dz3))/6;
			d = 2.1*d;
			pint.x = p.x + d * nfx;
			pint.y = p.y + d * nfy;
			pint.z = p.z + d * nfz;		

			*intp = pint;

			return (0);
		}
      }
    }
  }
  
  return(0);
} //fsi_2nd_InterceptionPoint


PetscErrorCode Find_fsi_3rd_interp_Coeff(PetscInt i, PetscInt j, PetscInt k, PetscInt elmt, Cmpnts p, IBMInfo *ibminfo, UserCtx *user, IBMNodes *ibm, SurfElmtInfo *elmtinfo)
{
  DA	da = user->da, fda = user->fda;
  DALocalInfo	info = user->info;
  PetscInt	xs = info.xs, xe = info.xs + info.xm;
  PetscInt  	ys = info.ys, ye = info.ys + info.ym;
  PetscInt	zs = info.zs, ze = info.zs + info.zm;
  PetscInt	mx = info.mx, my = info.my, mz = info.mz;
  //PetscInt	i, j, k;
  PetscInt	i2, j2, k2;
    PetscInt      lxs, lxe, lys, lye, lzs, lze;
	Vec Coor;

  //PetscInt      n_elmt = ibm->n_elmt;
  PetscInt      ip[8],jp[8],kp[8];
  Cmpnts        ***coor,pc[8];
  Cmpnts        intp,pOriginal;
  PetscReal	***nvert,nvertpc[8];
  PetscReal     nfx,nfy,nfz;
  
  	lxs = xs; lxe = xe;
	lys = ys; lye = ye;
	lzs = zs; lze = ze;

	if (xs==0) lxs = xs+1; 
	if (ys==0) lys = ys+1; 
	if (zs==0) lzs = zs+1; 
	if (xe==mx) lxe = xe-1; 
	if (ye==my) lye = ye-1; 
	if (ze==mz) lze = ze-1;

  DAGetCoordinates(da, &Coor);
  DAVecGetArray(fda, Coor,&coor);
  DAVecGetArray(da, user->Nvert, &nvert);

  nfx=ibm->nf_x[elmt];
  nfy=ibm->nf_y[elmt];
  nfz=ibm->nf_z[elmt];
  
  	pOriginal.x=ibm->cent_x[elmt]; 
	pOriginal.y=ibm->cent_y[elmt]; 
	pOriginal.z=ibm->cent_z[elmt]; 
  
    if (i>=xs && i<xe && j>=ys && j<ye && k>=zs && k<ze) {
    pc[0] = coor[k  ][j  ][i  ];
    pc[1] = coor[k  ][j  ][i+1];
    pc[2] = coor[k  ][j+1][i+1];
    pc[3] = coor[k  ][j+1][i  ];
    
    pc[4] = coor[k+1][j  ][i  ];
    pc[5] = coor[k+1][j  ][i+1];
    pc[6] = coor[k+1][j+1][i+1];
    pc[7] = coor[k+1][j+1][i  ];
	
	nvertpc[0] = nvert[k  ][j  ][i  ];
	nvertpc[1] = nvert[k  ][j  ][i+1];
	nvertpc[2] = nvert[k  ][j+1][i+1];
	nvertpc[3] = nvert[k  ][j+1][i  ];

	nvertpc[4] = nvert[k+1][j  ][i  ];
	nvertpc[5] = nvert[k+1][j  ][i+1];
	nvertpc[6] = nvert[k+1][j+1][i+1];
	nvertpc[7] = nvert[k+1][j+1][i  ];

    kp[0]=k  ;jp[0]=j  ;ip[0]=i  ;
    kp[1]=k  ;jp[1]=j  ;ip[1]=i+1;
    kp[2]=k  ;jp[2]=j+1;ip[2]=i+1;
    kp[3]=k  ;jp[3]=j+1;ip[3]=i  ;
    kp[4]=k+1;jp[4]=j  ;ip[4]=i  ;
    kp[5]=k+1;jp[5]=j  ;ip[5]=i+1;
    kp[6]=k+1;jp[6]=j+1;ip[6]=i+1;
    kp[7]=k+1;jp[7]=j+1;ip[7]=i  ;
	
	fsi_3rd_InterceptionPoint(p, pc, nvertpc, nfx, nfy, nfz, ibminfo, elmt, &intp, pOriginal, &elmtinfo[elmt].Need3rdPoint_3);    
    
    switch (ibminfo[elmt].ssmode) {
		case(0): {
		  ibminfo[elmt].ii11=ip[0]; ibminfo[elmt].jj11 = jp[0]; ibminfo[elmt].kk11 = kp[0];
		  ibminfo[elmt].ii22=ip[1]; ibminfo[elmt].jj22 = jp[1]; ibminfo[elmt].kk22 = kp[1];
		  ibminfo[elmt].ii33=ip[2]; ibminfo[elmt].jj33 = jp[2]; ibminfo[elmt].kk33 = kp[2];
		  break;
		}
		case (1): {
		  ibminfo[elmt].ii11=ip[0]; ibminfo[elmt].jj11 = jp[0]; ibminfo[elmt].kk11 = kp[0];
		  ibminfo[elmt].ii22=ip[2]; ibminfo[elmt].jj22 = jp[2]; ibminfo[elmt].kk22 = kp[2];
		  ibminfo[elmt].ii33=ip[3]; ibminfo[elmt].jj33 = jp[3]; ibminfo[elmt].kk33 = kp[3];
		  break;
		}
		case (2): {
		  ibminfo[elmt].ii11=ip[4]; ibminfo[elmt].jj11 = jp[4]; ibminfo[elmt].kk11 = kp[4];
		  ibminfo[elmt].ii22=ip[5]; ibminfo[elmt].jj22 = jp[5]; ibminfo[elmt].kk22 = kp[5];
		  ibminfo[elmt].ii33=ip[6]; ibminfo[elmt].jj33 = jp[6]; ibminfo[elmt].kk33 = kp[6];
		  break;
		}
		case (3): {
		  ibminfo[elmt].ii11=ip[4]; ibminfo[elmt].jj11 = jp[4]; ibminfo[elmt].kk11 = kp[4];
		  ibminfo[elmt].ii22=ip[6]; ibminfo[elmt].jj22 = jp[6]; ibminfo[elmt].kk22 = kp[6];
		  ibminfo[elmt].ii33=ip[7]; ibminfo[elmt].jj33 = jp[7]; ibminfo[elmt].kk33 = kp[7];
		  break;
		}
		case (4): {
		  ibminfo[elmt].ii11=ip[0]; ibminfo[elmt].jj11 = jp[0]; ibminfo[elmt].kk11 = kp[0];
		  ibminfo[elmt].ii22=ip[4]; ibminfo[elmt].jj22 = jp[4]; ibminfo[elmt].kk22 = kp[4];
		  ibminfo[elmt].ii33=ip[3]; ibminfo[elmt].jj33 = jp[3]; ibminfo[elmt].kk33 = kp[3];
		  break;
		}
		case (5): {
		  ibminfo[elmt].ii11=ip[3]; ibminfo[elmt].jj11 = jp[3]; ibminfo[elmt].kk11 = kp[3];
		  ibminfo[elmt].ii22=ip[4]; ibminfo[elmt].jj22 = jp[4]; ibminfo[elmt].kk22 = kp[4];
		  ibminfo[elmt].ii33=ip[7]; ibminfo[elmt].jj33 = jp[7]; ibminfo[elmt].kk33 = kp[7];
		  break;
		}
		case (6): {
		  ibminfo[elmt].ii11=ip[1]; ibminfo[elmt].jj11 = jp[1]; ibminfo[elmt].kk11 = kp[1];
		  ibminfo[elmt].ii22=ip[5]; ibminfo[elmt].jj22 = jp[5]; ibminfo[elmt].kk22 = kp[5];
		  ibminfo[elmt].ii33=ip[2]; ibminfo[elmt].jj33 = jp[2]; ibminfo[elmt].kk33 = kp[2];
		  break;
		}
		case (7): {
		  ibminfo[elmt].ii11=ip[2]; ibminfo[elmt].jj11 = jp[2]; ibminfo[elmt].kk11 = kp[2];
		  ibminfo[elmt].ii22=ip[6]; ibminfo[elmt].jj22 = jp[6]; ibminfo[elmt].kk22 = kp[6];
		  ibminfo[elmt].ii33=ip[5]; ibminfo[elmt].jj33 = jp[5]; ibminfo[elmt].kk33 = kp[5];
		  break;
		}
		case (8): {
		  ibminfo[elmt].ii11=ip[0]; ibminfo[elmt].jj11 = jp[0]; ibminfo[elmt].kk11 = kp[0];
		  ibminfo[elmt].ii22=ip[4]; ibminfo[elmt].jj22 = jp[4]; ibminfo[elmt].kk22 = kp[4];
		  ibminfo[elmt].ii33=ip[1]; ibminfo[elmt].jj33 = jp[1]; ibminfo[elmt].kk33 = kp[1];
		  break;
		}
		case (9): {
		  ibminfo[elmt].ii11=ip[1]; ibminfo[elmt].jj11 = jp[1]; ibminfo[elmt].kk11 = kp[1];
		  ibminfo[elmt].ii22=ip[4]; ibminfo[elmt].jj22 = jp[4]; ibminfo[elmt].kk22 = kp[4];
		  ibminfo[elmt].ii33=ip[5]; ibminfo[elmt].jj33 = jp[5]; ibminfo[elmt].kk33 = kp[5];
		  break;
		}
		case (10): {
		  ibminfo[elmt].ii11=ip[3]; ibminfo[elmt].jj11 = jp[3]; ibminfo[elmt].kk11 = kp[3];
		  ibminfo[elmt].ii22=ip[7]; ibminfo[elmt].jj22 = jp[7]; ibminfo[elmt].kk22 = kp[7];
		  ibminfo[elmt].ii33=ip[2]; ibminfo[elmt].jj33 = jp[2]; ibminfo[elmt].kk33 = kp[2];
		  break;
		}
		case (11): {
		  ibminfo[elmt].ii11=ip[2]; ibminfo[elmt].jj11 = jp[2]; ibminfo[elmt].kk11 = kp[2];
		  ibminfo[elmt].ii22=ip[7]; ibminfo[elmt].jj22 = jp[7]; ibminfo[elmt].kk22 = kp[7];
		  ibminfo[elmt].ii33=ip[6]; ibminfo[elmt].jj33 = jp[6]; ibminfo[elmt].kk33 = kp[6];
		  break;
		}
    }
    if (ibminfo[elmt].ssmode<0) PetscPrintf(PETSC_COMM_SELF, "3rd Interpolation Coeffients Were not Found!!!! %d  %d %le %d %d %d\n", elmt, ibminfo[elmt].smode,ibminfo[elmt].d_s,i,j,k);    
  }

  DAVecRestoreArray(fda, Coor,&coor);  
  DAVecRestoreArray(da, user->Nvert, &nvert);
  return(0);
}


PetscErrorCode fsi_3rd_InterceptionPoint(Cmpnts p, Cmpnts pc[8], PetscReal nvertpc[8], PetscReal nfx, PetscReal nfy, PetscReal nfz, IBMInfo *ibminfo, PetscInt number, Cmpnts *intp, Cmpnts pOriginal, PetscInt *Need3rdPoint_3)
{
  PetscInt 	triangles[3][12];
  Cmpnts   	p1, p2, p3;

  PetscReal	dx1, dy1, dz1, dx2, dy2, dz2, dx3, dy3, dz3, d;
  PetscReal	rx1, ry1, rz1, rx2, ry2, rz2;//, rx3, ry3, rz3;

  Cpt2D		pj1, pj2, pj3, pjp;
  PetscInt	cell, flag;

  PetscInt	i;
  Cmpnts	pint; // Interception point
  PetscReal	nfxt, nfyt, nfzt;

  ibminfo[number].ssmode = -100;
  // k-plane
  triangles[0][0]  = 0; triangles[1][0]  = 1; triangles[2][0]  = 2;
  triangles[0][1]  = 0; triangles[1][1]  = 2; triangles[2][1]  = 3;
  triangles[0][2]  = 4; triangles[1][2]  = 5; triangles[2][2]  = 6;
  triangles[0][3]  = 4; triangles[1][3]  = 6; triangles[2][3]  = 7;
  // i-plane
  triangles[0][4]  = 0; triangles[1][4]  = 4; triangles[2][4]  = 3;
  triangles[0][5]  = 3; triangles[1][5]  = 4; triangles[2][5]  = 7;
  triangles[0][6]  = 1; triangles[1][6]  = 5; triangles[2][6]  = 2;
  triangles[0][7]  = 2; triangles[1][7]  = 6; triangles[2][7]  = 5;
  // j-plane
  triangles[0][8]  = 0; triangles[1][8]  = 4; triangles[2][8]  = 1;
  triangles[0][9]  = 1; triangles[1][9]  = 4; triangles[2][9]  = 5;
  triangles[0][10] = 3; triangles[1][10] = 7; triangles[2][10] = 2;
  triangles[0][11] = 2; triangles[1][11] = 7; triangles[2][11] = 6;

  for (i=0; i<12; i++) {
    p1 = pc[triangles[0][i]]; p2 = pc[triangles[1][i]], p3 = pc[triangles[2][i]];

    dx1 = p.x - p1.x; dy1 = p.y - p1.y; dz1 = p.z - p1.z;   //a1=p -p1
    dx2 = p2.x - p1.x; dy2 = p2.y - p1.y; dz2 = p2.z - p1.z;//a2=p2-p1
    dx3 = p3.x - p1.x; dy3 = p3.y - p1.y; dz3 = p3.z - p1.z;//a3=p3-p1

    // area of the parralelogram since h=1 and V=ah=nf.(a2xa3)
    d = (nfx * (dy2 * dz3 - dz2 * dy3) - 
	 nfy * (dx2 * dz3 - dz2 * dx3) + 
	 nfz * (dx2 * dy3 - dy2 * dx3));
    if (fabs(d) > 1.e-10) {
      // the distance of the point from the triangle plane
      // d = Vol/area = a1.(a2xa3)/area
      d = -(dx1 * (dy2 * dz3 - dz2 * dy3) - 
	    dy1 * (dx2 * dz3 - dz2 * dx3) + 
	    dz1 * (dx2 * dy3 - dy2 * dx3)) / d;
      

      if (d>0) {
		pint.x = p.x + d * nfx;
		pint.y = p.y + d * nfy;
		pint.z = p.z + d * nfz;

		rx1 = p2.x - p1.x; ry1 = p2.y - p1.y; rz1 = p2.z - p1.z;
		rx2 = p3.x - p1.x; ry2 = p3.y - p1.y; rz2 = p3.z - p1.z;
		  
		nfxt = ry1 * rz2 - rz1 * ry2;
		nfyt = -rx1 * rz2 + rz1 * rx2;
		nfzt = rx1 * ry2 - ry1 * rx2;

		flag = ISPointInTriangle(pint, p1, p2, p3, nfxt, nfyt, nfzt);
		if (flag >= 0) {
		  cell = i;

		  /*	  if (flagprint==1) {
			PetscPrintf(PETSC_COMM_WORLD, "%e %e %e %e \n", pint.x, pint.y, pint.z, d);
			PetscPrintf(PETSC_COMM_WORLD, "%e %e %e\n", nfx, nfy, nfz);
			PetscPrintf(PETSC_COMM_WORLD, "%e %e %e\n", p2.x, p2.y, p2.z);
			PetscPrintf(PETSC_COMM_WORLD, "%e %e %e\n", p3.x, p3.y, p3.z);
			}*/

		  // Calculate the interpolatin Coefficients
		  
					if ((int)(nvertpc[triangles[0][i]] +0.5) <= 1 &&
						  (int)(nvertpc[triangles[1][i]] +0.5) <= 1 &&
						  (int)(nvertpc[triangles[2][i]] +0.5) <= 1) {
							*Need3rdPoint_3 = 0; //ASR
							if (fabs(nfxt) >= fabs(nfyt) && fabs(nfxt)>= fabs(nfzt)) {
							  pjp.x = pint.y; pjp.y = pint.z;
							  pj1.x = p1.y;   pj1.y = p1.z;
							  pj2.x = p2.y;   pj2.y = p2.z;
							  pj3.x = p3.y;   pj3.y = p3.z;
							  triangle_intp_fsi_3(pjp, pj1, pj2, pj3, ibminfo, number);
							  triangle_3rd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number); // so far, what happens inside this function is not used.
							}
							else if (fabs(nfyt) >= fabs(nfxt) && fabs(nfyt)>= fabs(nfzt)) {
							  pjp.x = pint.x; pjp.y = pint.z;
							  pj1.x = p1.x;   pj1.y = p1.z;
							  pj2.x = p2.x;   pj2.y = p2.z;
							  pj3.x = p3.x;   pj3.y = p3.z;
							  triangle_intp_fsi_3(pjp, pj1, pj2, pj3, ibminfo, number);
							  triangle_3rd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
							else if (fabs(nfzt) >= fabs(nfyt) && fabs(nfzt)>= fabs(nfxt)) {
							  pjp.x = pint.y; pjp.y = pint.x;
							  pj1.x = p1.y;   pj1.y = p1.x;
							  pj2.x = p2.y;   pj2.y = p2.x;
							  pj3.x = p3.y;   pj3.y = p3.x;
							  triangle_intp_fsi_3(pjp, pj1, pj2, pj3, ibminfo,number);
							  triangle_3rd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
					  } else  if ((int)(nvertpc[triangles[0][i]] +0.5) > 1 &&
							  (int)(nvertpc[triangles[1][i]] +0.5) <= 1 &&
							  (int)(nvertpc[triangles[2][i]] +0.5) <= 1) {
							*Need3rdPoint_3 = 1;
							if (fabs(nfxt) >= fabs(nfyt) && fabs(nfxt)>= fabs(nfzt)) {
							  pjp.x = pint.y; pjp.y = pint.z;
							  pj1.x = p1.y;   pj1.y = p1.z;
							  pj2.x = p2.y;   pj2.y = p2.z;
							  pj3.x = p3.y;   pj3.y = p3.z;
							  linear_intp_3(pjp, pj2, pj3, ibminfo, number,1);
							  triangle_3rd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
							else if (fabs(nfyt) >= fabs(nfxt) && fabs(nfyt)>= fabs(nfzt)) {
							  pjp.x = pint.x; pjp.y = pint.z;
							  pj1.x = p1.x;   pj1.y = p1.z;
							  pj2.x = p2.x;   pj2.y = p2.z;
							  pj3.x = p3.x;   pj3.y = p3.z;
							  linear_intp_3(pjp, pj2, pj3, ibminfo, number,1);
							  triangle_3rd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
							else if (fabs(nfzt) >= fabs(nfyt) && fabs(nfzt)>= fabs(nfxt)) {
							  pjp.x = pint.y; pjp.y = pint.x;
							  pj1.x = p1.y;   pj1.y = p1.x;
							  pj2.x = p2.y;   pj2.y = p2.x;
							  pj3.x = p3.y;   pj3.y = p3.x;
							  linear_intp_3(pjp, pj2, pj3, ibminfo,number,1);
							  triangle_3rd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
					  } else  if ((int)(nvertpc[triangles[0][i]] +0.5) <= 1 &&
							  (int)(nvertpc[triangles[1][i]] +0.5) > 1 &&
							  (int)(nvertpc[triangles[2][i]] +0.5) <= 1) {
							*Need3rdPoint_3 = 1;
							if (fabs(nfxt) >= fabs(nfyt) && fabs(nfxt)>= fabs(nfzt)) {
							  pjp.x = pint.y; pjp.y = pint.z;
							  pj1.x = p1.y;   pj1.y = p1.z;
							  pj2.x = p2.y;   pj2.y = p2.z;
							  pj3.x = p3.y;   pj3.y = p3.z;
							  linear_intp_3(pjp, pj1, pj3, ibminfo, number,2);
							  triangle_3rd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
							else if (fabs(nfyt) >= fabs(nfxt) && fabs(nfyt)>= fabs(nfzt)) {
							  pjp.x = pint.x; pjp.y = pint.z;
							  pj1.x = p1.x;   pj1.y = p1.z;
							  pj2.x = p2.x;   pj2.y = p2.z;
							  pj3.x = p3.x;   pj3.y = p3.z;
							  linear_intp_3(pjp, pj1, pj3, ibminfo, number,2);
							  triangle_3rd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
							else if (fabs(nfzt) >= fabs(nfyt) && fabs(nfzt)>= fabs(nfxt)) {
							  pjp.x = pint.y; pjp.y = pint.x;
							  pj1.x = p1.y;   pj1.y = p1.x;
							  pj2.x = p2.y;   pj2.y = p2.x;
							  pj3.x = p3.y;   pj3.y = p3.x;
							  linear_intp_3(pjp, pj1, pj3, ibminfo,number,2);
							  triangle_3rd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
					  } else  if ((int)(nvertpc[triangles[0][i]] +0.5) <= 1 &&
							  (int)(nvertpc[triangles[1][i]] +0.5) <= 1 &&
							  (int)(nvertpc[triangles[2][i]] +0.5) > 1) {
							*Need3rdPoint_3 = 1;
							if (fabs(nfxt) >= fabs(nfyt) && fabs(nfxt)>= fabs(nfzt)) {
							  pjp.x = pint.y; pjp.y = pint.z;
							  pj1.x = p1.y;   pj1.y = p1.z;
							  pj2.x = p2.y;   pj2.y = p2.z;
							  pj3.x = p3.y;   pj3.y = p3.z;
							  linear_intp_3(pjp, pj1, pj2, ibminfo, number,3);
							  triangle_3rd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
							else if (fabs(nfyt) >= fabs(nfxt) && fabs(nfyt)>= fabs(nfzt)) {
							  pjp.x = pint.x; pjp.y = pint.z;
							  pj1.x = p1.x;   pj1.y = p1.z;
							  pj2.x = p2.x;   pj2.y = p2.z;
							  pj3.x = p3.x;   pj3.y = p3.z;
							  linear_intp_3(pjp, pj1, pj2, ibminfo, number,3);
							  triangle_3rd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
							else if (fabs(nfzt) >= fabs(nfyt) && fabs(nfzt)>= fabs(nfxt)) {
							  pjp.x = pint.y; pjp.y = pint.x;
							  pj1.x = p1.y;   pj1.y = p1.x;
							  pj2.x = p2.y;   pj2.y = p2.x;
							  pj3.x = p3.y;   pj3.y = p3.x;
							  linear_intp_3(pjp, pj1, pj2, ibminfo,number,3);
							  triangle_3rd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
					  } else  if ((int)(nvertpc[triangles[0][i]] +0.5) <= 1 &&
							  (int)(nvertpc[triangles[1][i]] +0.5) > 1 &&
							  (int)(nvertpc[triangles[2][i]] +0.5) > 1) {
							*Need3rdPoint_3 = -1;
							ibminfo[number].cr11 = 1.;
							ibminfo[number].cr22 = 0.;
							ibminfo[number].cr33 = 0.;
							if (fabs(nfxt) >= fabs(nfyt) && fabs(nfxt)>= fabs(nfzt)) {
							  pjp.x = pint.y; pjp.y = pint.z;
							  pj1.x = p1.y;   pj1.y = p1.z;
							  pj2.x = p2.y;   pj2.y = p2.z;
							  pj3.x = p3.y;   pj3.y = p3.z;
							  triangle_3rd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
							else if (fabs(nfyt) >= fabs(nfxt) && fabs(nfyt)>= fabs(nfzt)) {
							  pjp.x = pint.x; pjp.y = pint.z;
							  pj1.x = p1.x;   pj1.y = p1.z;
							  pj2.x = p2.x;   pj2.y = p2.z;
							  pj3.x = p3.x;   pj3.y = p3.z;
							  triangle_3rd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
							else if (fabs(nfzt) >= fabs(nfyt) && fabs(nfzt)>= fabs(nfxt)) {
							  pjp.x = pint.y; pjp.y = pint.x;
							  pj1.x = p1.y;   pj1.y = p1.x;
							  pj2.x = p2.y;   pj2.y = p2.x;
							  pj3.x = p3.y;   pj3.y = p3.x;
							  triangle_3rd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
					  } else  if ((int)(nvertpc[triangles[0][i]] +0.5) > 1 &&
							  (int)(nvertpc[triangles[1][i]] +0.5) <= 1 &&
							  (int)(nvertpc[triangles[2][i]] +0.5) > 1) {
							*Need3rdPoint_3 = -1;
							ibminfo[number].cr11 = 0.;
							ibminfo[number].cr22 = 1.;
							ibminfo[number].cr33 = 0.;
							if (fabs(nfxt) >= fabs(nfyt) && fabs(nfxt)>= fabs(nfzt)) {
							  pjp.x = pint.y; pjp.y = pint.z;
							  pj1.x = p1.y;   pj1.y = p1.z;
							  pj2.x = p2.y;   pj2.y = p2.z;
							  pj3.x = p3.y;   pj3.y = p3.z;
							  triangle_3rd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
							else if (fabs(nfyt) >= fabs(nfxt) && fabs(nfyt)>= fabs(nfzt)) {
							  pjp.x = pint.x; pjp.y = pint.z;
							  pj1.x = p1.x;   pj1.y = p1.z;
							  pj2.x = p2.x;   pj2.y = p2.z;
							  pj3.x = p3.x;   pj3.y = p3.z;
							  triangle_3rd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
							else if (fabs(nfzt) >= fabs(nfyt) && fabs(nfzt)>= fabs(nfxt)) {
							  pjp.x = pint.y; pjp.y = pint.x;
							  pj1.x = p1.y;   pj1.y = p1.x;
							  pj2.x = p2.y;   pj2.y = p2.x;
							  pj3.x = p3.y;   pj3.y = p3.x;
							  triangle_3rd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
					  } else  if ((int)(nvertpc[triangles[0][i]] +0.5) > 1 &&
							  (int)(nvertpc[triangles[1][i]] +0.5) > 1 &&
							  (int)(nvertpc[triangles[2][i]] +0.5) <= 1) {
							*Need3rdPoint_3 = -1;
							ibminfo[number].cr11 = 0.;
							ibminfo[number].cr22 = 0.;
							ibminfo[number].cr33 = 1.;
							if (fabs(nfxt) >= fabs(nfyt) && fabs(nfxt)>= fabs(nfzt)) {
							  pjp.x = pint.y; pjp.y = pint.z;
							  pj1.x = p1.y;   pj1.y = p1.z;
							  pj2.x = p2.y;   pj2.y = p2.z;
							  pj3.x = p3.y;   pj3.y = p3.z;
							  triangle_3rd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
							else if (fabs(nfyt) >= fabs(nfxt) && fabs(nfyt)>= fabs(nfzt)) {
							  pjp.x = pint.x; pjp.y = pint.z;
							  pj1.x = p1.x;   pj1.y = p1.z;
							  pj2.x = p2.x;   pj2.y = p2.z;
							  pj3.x = p3.x;   pj3.y = p3.z;
							  triangle_3rd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
							else if (fabs(nfzt) >= fabs(nfyt) && fabs(nfzt)>= fabs(nfxt)) {
							  pjp.x = pint.y; pjp.y = pint.x;
							  pj1.x = p1.y;   pj1.y = p1.x;
							  pj2.x = p2.y;   pj2.y = p2.x;
							  pj3.x = p3.y;   pj3.y = p3.x;
							  triangle_3rd_intp_fsi(pjp, pj1, pj2, pj3, ibminfo, number);
							}
					  } else {
						*Need3rdPoint_3 = -100;
						PetscPrintf(PETSC_COMM_SELF, "3rd step: Last host node empty. \n Try to increase d in fsi_2nd_InterceptionPoint() \n elmt=%d, cell=%d; nv=%d, %d, %d; \n", number, cell, (int)nvertpc[triangles[0][cell]],(int)nvertpc[triangles[1][cell]],(int)nvertpc[triangles[2][cell]]);
					  }
		  ibminfo[number].d_ss = sqrt((pint.x-pOriginal.x)*(pint.x - pOriginal.x) + (pint.y-pOriginal.y) * (pint.y-pOriginal.y) + (pint.z - pOriginal.z)* (pint.z - pOriginal.z));
		  ibminfo[number].ssmode = cell;

		  *intp = pint;

		  return (0);
		}
      }
    }
  }
  return(0);
}


PetscErrorCode Calc_fsi_surf_stress2(IBMInfo *ibminfo, UserCtx *user, IBMNodes *ibm, SurfElmtInfo *elmtinfo)
{
	DA	        da = user->da, fda = user->fda;
	DALocalInfo	info = user->info;
	PetscInt	xs = info.xs, xe = info.xs + info.xm;
	PetscInt  	ys = info.ys, ye = info.ys + info.ym;
	PetscInt	zs = info.zs, ze = info.zs + info.zm;
	PetscInt      lxs, lxe, lys, lye, lzs, lze;
	PetscInt      mx = info.mx, my = info.my, mz = info.mz;


	PetscInt      n_elmt = ibm->n_elmt;
	PetscInt      elmt;
	PetscInt	ip1, ip2, ip3, jp1, jp2, jp3, kp1, kp2, kp3;
	PetscInt	ip11, ip22, ip33, jp11, jp22, jp33, kp11, kp22, kp33;
	PetscInt	i, j, k;
	
	PetscInt	i1, i2, i3, j1, j2, j3, k1, k2, k3;

	PetscReal     c1, c2, c3;
	PetscReal     cr1, cr2, cr3;
	PetscReal     cv1, cv2, cv3;
	PetscReal     cr11, cr22, cr33;
	PetscReal     cv11, cv22, cv33;
	PetscReal     cs1, cs2, cs3;
	PetscReal     cs11, cs22, cs33;

	PetscReal     sv1, sv2, sv3;

	PetscInt	iip11, iip22, iip33, jjp11, jjp22, jjp33, kkp11, kkp22, kkp33;
	PetscInt	iip1, iip2, iip3, jjp1, jjp2, jjp3, kkp1, kkp2, kkp3;
	PetscReal     ct1, ct2, ct3;
	PetscReal     ct11, ct22, ct33;
	PetscReal     ds,sd;

	PetscReal     di;
	PetscReal     nf_x, nf_y, nf_z;
	PetscReal     nt_x, nt_y, nt_z;
	PetscReal     ns_x, ns_y, ns_z;
	PetscReal     ***p;
	Cmpnts        ***ucat, uinp, pint, pOriginal;
	PetscReal	***nvert; 
	PetscReal nv1,nv2,nv3;
	PetscReal P, Tow_ws, Tow_wt, Tow_wn;
	Vec Coor;
	Cmpnts        ***coor;
	DAVecGetArray(fda, user->Ucat, &ucat);
	DAVecGetArray(da, user->P, &p);
	DAVecGetArray(da, user->Nvert, &nvert);  
	DAGetCoordinates(da, &Coor);
	DAVecGetArray(fda, Coor,&coor);
	PetscInt rank;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	lxs = xs; lxe = xe;
	lys = ys; lye = ye;
	lzs = zs; lze = ze;

	if (xs==0) lxs = xs+1;
	if (ys==0) lys = ys+1;
	if (zs==0) lzs = zs+1;

	if (xe==mx) lxe = xe-1;
	if (ye==my) lye = ye-1;
	if (ze==mz) lze = ze-1;

for (elmt=0; elmt<n_elmt; elmt++) {
		P = 0;
		Tow_ws = 0;
		Tow_wt = 0;
		Tow_wn = 0;
		pOriginal.x=ibm->cent_x[elmt]; 
		pOriginal.y=ibm->cent_y[elmt]; 
		pOriginal.z=ibm->cent_z[elmt]; 
		
		if (elmtinfo[elmt].n_P>0  && elmtinfo[elmt].FoundAroundcell==1) {
			ip1 = ibminfo[elmt].i1; jp1 = ibminfo[elmt].j1; kp1 = ibminfo[elmt].k1;
			ip2 = ibminfo[elmt].i2; jp2 = ibminfo[elmt].j2; kp2 = ibminfo[elmt].k2;
			ip3 = ibminfo[elmt].i3; jp3 = ibminfo[elmt].j3; kp3 = ibminfo[elmt].k3;

			cr1 = ibminfo[elmt].cr1; cr2 = ibminfo[elmt].cr2; cr3 = ibminfo[elmt].cr3;
			cs1 = ibminfo[elmt].cs1; cs2 = ibminfo[elmt].cs2; cs3 = ibminfo[elmt].cs3;

			ip11 = ibminfo[elmt].i11; jp11 = ibminfo[elmt].j11; kp11 = ibminfo[elmt].k11;
			ip22 = ibminfo[elmt].i22; jp22 = ibminfo[elmt].j22; kp22 = ibminfo[elmt].k22;
			ip33 = ibminfo[elmt].i33; jp33 = ibminfo[elmt].j33; kp33 = ibminfo[elmt].k33;

			cr11 = ibminfo[elmt].cr11; cr22 = ibminfo[elmt].cr22; cr33 = ibminfo[elmt].cr33;
			cs11 = ibminfo[elmt].cs11; cs22 = ibminfo[elmt].cs22; cs33 = ibminfo[elmt].cs33;

			iip1 = ibminfo[elmt].ii1; jjp1 = ibminfo[elmt].jj1; kkp1 = ibminfo[elmt].kk1;
			iip2 = ibminfo[elmt].ii2; jjp2 = ibminfo[elmt].jj2; kkp2 = ibminfo[elmt].kk2;
			iip3 = ibminfo[elmt].ii3; jjp3 = ibminfo[elmt].jj3; kkp3 = ibminfo[elmt].kk3;

			iip11 = ibminfo[elmt].ii11; jjp11 = ibminfo[elmt].jj11; kkp11 = ibminfo[elmt].kk11;
			iip22 = ibminfo[elmt].ii22; jjp22 = ibminfo[elmt].jj22; kkp22 = ibminfo[elmt].kk22;
			iip33 = ibminfo[elmt].ii33; jjp33 = ibminfo[elmt].jj33; kkp33 = ibminfo[elmt].kk33;

			ct1 = ibminfo[elmt].ct1; ct2 = ibminfo[elmt].ct2; ct3 = ibminfo[elmt].ct3;
			ct11 = ibminfo[elmt].ct11; ct22 = ibminfo[elmt].ct22; ct33 = ibminfo[elmt].ct33;

			i = elmtinfo[elmt].icell;
			j = elmtinfo[elmt].jcell;
			k = elmtinfo[elmt].kcell;

			//  if (i>=xs && i<xe && j>=ys && j<ye && k>=zs && k<ze) {
			if (i>=lxs && i<lxe && j>=lys && j<lye && k>=lzs && k<lze) {
				/*
				if (elmt==elmtCheck1 || elmt==elmtCheck2) PetscPrintf(PETSC_COMM_SELF, "rank=%d; elmt = %d; lzs=%d; lze=%d; lys=%d; lye=%d; lxs=%d; lxe=%d; \n",rank, elmt, lzs, lze, lys, lye, lxs, lxe);
				if (elmt==elmtCheck1 || elmt==elmtCheck2) PetscPrintf(PETSC_COMM_SELF, "rank=%d; elmt = %d; kp1=%d; kp2=%d; kp3=%d; jp1=%d; jp2=%d; jp3=%d; ip1=%d; ip2=%d; ip3=%d; \n",rank,elmt, kp1, kp2, kp3, jp1, jp2, jp3, ip1, ip2, ip3);	
				if (elmt==elmtCheck1 || elmt==elmtCheck2) PetscPrintf(PETSC_COMM_SELF, "rank=%d; elmt = %d; kkp1=%d; kkp2=%d; kkp3=%d; jjp1=%d; jjp2=%d; jjp3=%d; iip1=%d; iip2=%d; iip3=%d; \n", rank, elmt, kkp1, kkp2, kkp3, jjp1, jjp2, jjp3, iip1, iip2, iip3);
				if (elmt==elmtCheck1 || elmt==elmtCheck2) PetscPrintf(PETSC_COMM_SELF, "rank=%d; elmt = %d; kkp11=%d; kkp22=%d; kkp33=%d; jjp11=%d; jjp22=%d; jjp33=%d; iip11=%d; iip22=%d; iip33=%d; \n", rank, elmt, kkp11, kkp22, kkp33, jjp11, jjp22, jjp33, iip11, iip22, iip33);
				*/
				k1 = kp1; j1 = jp1; i1 = ip1; k2 = kp2; j2 = jp2; i2 = ip2; k3 = kp3; j3 = jp3; i3 = ip3;
				c1 = cr1; c2 = cr2; c3 = cr3;
				di = ibminfo[elmt].d_i;
				/*
				nv1 = nvert[k1][j1][i1];
				nv2 = nvert[k2][j2][i2];
				nv3 = nvert[k3][j3][i3];								
				*/
				//if (elmt==elmtCheck1 || elmt==elmtCheck2) PetscPrintf(PETSC_COMM_SELF, "Nv 1st rank=%d; elmt = %d; nv1=%f; nv2=%f; nv3=%f; \n", rank, elmt, nv1, nv2, nv3);
				/*
				if(elmtinfo[elmt].Need3rdPoint!=0){
					k1 = kkp1; j1 = jjp1; i1 = iip1; k2 = kkp2; j2 = jjp2; i2 = iip2; k3 = kkp3; j3 = jjp3; i3 = iip3;
					c1 = ct11; c2 = ct22; c3 = ct33;
					di = ibminfo[elmt].d_s;
					//nv1 = nvert[k1][j1][i1]; nv2 = nvert[k2][j2][i2]; nv3 = nvert[k3][j3][i3];
					//if (elmt==elmtCheck1 || elmt==elmtCheck2) PetscPrintf(PETSC_COMM_SELF, "Nv 2nd rank=%d; elmt = %d; nv1=%f; nv2=%f; nv3=%f; 3rdP = %d, 3rdP_2 = %d, smode = %d \n", rank, elmt, nv1, nv2, nv3, elmtinfo[elmt].Need3rdPoint, elmtinfo[elmt].Need3rdPoint_2, ibminfo[elmt].smode);
					if(elmtinfo[elmt].Need3rdPoint_2!=0){
						k1 = kkp11; j1 = jjp11; i1 = iip11; k2 = kkp22; j2 = jjp22; i2 = iip22; k3 = kkp33; j3 = jjp33; i3 = iip33;
						c1 = cr11; c2 = cr22; c3 = cr33;
						di = ibminfo[elmt].d_ss;
						//nv1 = nvert[k1][j1][i1]; nv2 = nvert[k2][j2][i2]; nv3 = nvert[k3][j3][i3];
						//if (elmt==elmtCheck1 || elmt==elmtCheck2) PetscPrintf(PETSC_COMM_SELF, "Nv 3rd rank=%d; elmt = %d; nv1=%f; nv2=%f; nv3=%f; 3rdP = %d, 3rdP_2 = %d, 3rdp_3 = %d, smode = %d \n", rank, elmt, nv1, nv2, nv3, elmtinfo[elmt].Need3rdPoint, elmtinfo[elmt].Need3rdPoint_2, elmtinfo[elmt].Need3rdPoint_3, ibminfo[elmt].smode);
					}
				}*/
				
				cv1 = p[k1][j1][i1];
				cv2 = p[k2][j2][i2];
				cv3 = p[k3][j3][i3];
				P = cv1 * c1 + cv2 * c2 + cv3 * c3;
				
				//if (elmt==elmtCheck1 || elmt==elmtCheck2) PetscPrintf(PETSC_COMM_SELF, "P2 rank=%d; elmt = %d; cv1=%f; cv2=%f; cv3=%f; c1=%f; c2=%f; c3=%f; \n", rank, elmt, cv1, cv2, cv3, c1, c2, c3);

				double xloc, yloc, zloc;
				xloc = ibm->x_bp[ibm->nv2[elmt]];
				yloc = ibm->y_bp[ibm->nv2[elmt]];
				zloc = ibm->z_bp[ibm->nv2[elmt]];		
				//if (elmt==elmtCheck1 || elmt==elmtCheck2) PetscPrintf(PETSC_COMM_SELF, "3rdP=%d %d x=%f y=%f z=%f\n",elmtinfo[elmt].Need3rdPoint,elmt,xloc,yloc,zloc);						
				nv1 = nvert[k1][j1][i1];
				nv2 = nvert[k2][j2][i2];
				nv3 = nvert[k3][j3][i3];
				//if (elmt==elmtCheck1 || elmt==elmtCheck2) PetscPrintf(PETSC_COMM_SELF, "Nv 3rd rank=%d; elmt = %d; nv1=%f; nv2=%f; nv3=%f; 3rdP = %d, 3rdP_2 = %d, 3rdp_3 = %d, smode = %d \n", rank, elmt, nv1, nv2, nv3, elmtinfo[elmt].Need3rdPoint, elmtinfo[elmt].Need3rdPoint_2, elmtinfo[elmt].Need3rdPoint_3, ibminfo[elmt].smode);
					
				cv1 = ucat[k1][j1][i1].x;
				cv2 = ucat[k2][j2][i2].x;
				cv3 = ucat[k3][j3][i3].x;
				uinp.x = (cv1 * c1 + cv2 * c2 + cv3 * c3 );
				//if (elmt==elmtCheck1 || elmt==elmtCheck2) PetscPrintf(PETSC_COMM_SELF, "V %d u_x= %le (%le %le %le) c1=%le c2=%le c3=%le\n",elmt,uinp.x,cv1,cv2,cv3,c1,c2,c3);			

				cv1 = ucat[k1][j1][i1].y;
				cv2 = ucat[k2][j2][i2].y;
				cv3 = ucat[k3][j3][i3].y;		
				uinp.y = (cv1 * c1 + cv2 * c2 + cv3 * c3 );
				//if (elmt==elmtCheck1 || elmt==elmtCheck2) PetscPrintf(PETSC_COMM_SELF, "V %d u_y=%le (%le %le %le) c1=%le c2=%le c3=%le\n",elmt,uinp.y,cv1,cv2,cv3,c1,c2,c3);		
			
				cv1 = ucat[k1][j1][i1].z;
				cv2 = ucat[k2][j2][i2].z;
				cv3 = ucat[k3][j3][i3].z;
				uinp.z = (cv1 * c1 + cv2 * c2 + cv3 * c3 );
				//if (elmt==elmtCheck1 || elmt==elmtCheck2) PetscPrintf(PETSC_COMM_SELF, "V %d u_z=%le (%le %le %le) c1=%le c2=%le c3=%le\n",elmt,uinp.z,cv1,cv2,cv3,c1,c2,c3);		
				//if (elmt==elmtCheck1 || elmt==elmtCheck2)  PetscPrintf(PETSC_COMM_SELF, "V di=%f\n", di);

				cv1 = coor[k1][j1][i1].x;
				cv2 = coor[k2][j2][i2].x;
				cv3 = coor[k3][j3][i3].x;
				pint.x = (cv1 * c1 + cv2 * c2 + cv3 * c3 );
				//pint.x = (cv1+cv2+cv3)/3;
				cv1 = coor[k1][j1][i1].y;
				cv2 = coor[k2][j2][i2].y;
				cv3 = coor[k3][j3][i3].y;
				pint.y = (cv1 * c1 + cv2 * c2 + cv3 * c3 );
				//pint.y = (cv1+cv2+cv3)/3;				
				cv1 = coor[k1][j1][i1].z;
				cv2 = coor[k2][j2][i2].z;
				cv3 = coor[k3][j3][i3].z;
				//pint.z = (cv1+cv2+cv3)/3;
				pint.z = (cv1 * c1 + cv2 * c2 + cv3 * c3 );
				
				di = sqrt((pint.x-pOriginal.x)*(pint.x - pOriginal.x) + (pint.y-pOriginal.y) * (pint.y-pOriginal.y) + (pint.z - pOriginal.z)* (pint.z - pOriginal.z));
				
				if(delta!=0){
					uinp.x = ibminfo[elmt].cs11;
					uinp.y = ibminfo[elmt].cs22;
					uinp.z = ibminfo[elmt].cs33;
					di = delta;
				}
				
				sv1 = ( ibm->u[ibm->nv1[elmt]].x + ibm->u[ibm->nv2[elmt]].x + ibm->u[ibm->nv3[elmt]].x)/3.;
				sv2 = ( ibm->u[ibm->nv1[elmt]].y + ibm->u[ibm->nv2[elmt]].y + ibm->u[ibm->nv3[elmt]].y)/3.;
				sv3 = ( ibm->u[ibm->nv1[elmt]].z + ibm->u[ibm->nv2[elmt]].z + ibm->u[ibm->nv3[elmt]].z)/3.;
			  
				ns_x= ibm->ns_x[elmt];
				ns_y= ibm->ns_y[elmt];	
				ns_z= ibm->ns_z[elmt];
				nt_x= ibm->nt_x[elmt];
				nt_y= ibm->nt_y[elmt];
				nt_z= ibm->nt_z[elmt];
				nf_x= ibm->nf_x[elmt];	
				nf_y= ibm->nf_y[elmt];
				nf_z= ibm->nf_z[elmt];

				if (di>1e-6) {
					Tow_ws = ((uinp.x-sv1)*ns_x + (uinp.y-sv2)*ns_y + (uinp.z-sv3)*ns_z)/di;
					Tow_wt = ((uinp.x-sv1)*nt_x + (uinp.y-sv2)*nt_y + (uinp.z-sv3)*nt_z)/di;
					Tow_wn = ((uinp.x-sv1)*nf_x + (uinp.y-sv2)*nf_y + (uinp.z-sv3)*nf_z)/di;
				}
				else {
					PetscPrintf(PETSC_COMM_SELF, "very small di %le for elmt %d",di,elmt);
					Tow_ws = 0;
					Tow_wt = 0;
					Tow_wn = 0;
				}
			}
			elmtinfo[elmt].P = P;
			elmtinfo[elmt].Tow_ws = Tow_ws;
			elmtinfo[elmt].Tow_wt = Tow_wt;
			elmtinfo[elmt].Tow_wn = Tow_wn;			
		}else{
			elmtinfo[elmt].P = 0;
			elmtinfo[elmt].Tow_ws = 0;
			elmtinfo[elmt].Tow_wt = 0;
			elmtinfo[elmt].Tow_wn = 0;
		}
		/*
		if(elmt==elmtCheck1 || elmt==elmtCheck2){
			PetscPrintf(PETSC_COMM_SELF, "ASR 1 rank=%d; elmt=%d; FoundAroundcell=%d; Tow_ws=%f; Tow_wt=%f; Tow_wn=%f; Tow_ws_sum=%f; Tow_wt_sum=%f; Tow_wn_sum=%f; di=%f; P=%f; P_sum=%f\n", rank, elmt, elmtinfo[elmt].FoundAroundcell, Tow_ws, Tow_wt, Tow_wn, elmtinfo[elmt].Tow_ws, elmtinfo[elmt].Tow_wt, elmtinfo[elmt].Tow_wn, ibminfo[elmt].d_i,P,elmtinfo[elmt].P);
			PetscPrintf(PETSC_COMM_SELF, "ASR 1 rank=%d cr1=%f; cr2=%f; cr3=%f; cv1=%f; cv2=%f; cv3=%f; sv1=%f; sv2=%f; sv3=%f; uinp.x=%f; uinp.y=%f; uinp.z=%f;\n",rank,cr1,cr2,cr3,cv1,cv2,cv3,sv1,sv2,sv3,uinp.x,uinp.y,uinp.z);
			PetscPrintf(PETSC_COMM_SELF, "ASR 1 rank=%d nsx=%f; nsy=%f; nsz=%f; ntx=%f; nty=%f; ntz=%f; nfx=%f; nfy=%f; nfz=%f;\n",rank,ns_x,ns_y,ns_z,nt_x,nt_y,nt_z,nf_x,nf_y,nf_z);
			PetscPrintf(PETSC_COMM_SELF, "rank=%d cr1=%f; cr2=%f; cr3=%f; cv1=%f; cv2=%f; cv3=%f;\n",rank,uinp.x,uinp.y,uinp.z,cv1,cv2,cv3);
		}		*/
	}
	PetscPrintf(PETSC_COMM_WORLD, " Printing IBM Data ");
	PetscInt n_v=ibm->n_v;
	  if (!rank) {
		  FILE *f;
		  char filen[80];
		  sprintf(filen, "Stress_Calc_fsi_surf_stress2_%05d.dat",ti);
		  f = fopen(filen, "w");
		  
		  PetscFPrintf(PETSC_COMM_WORLD, f, "TITLE=\"3D TRIANGULAR SURFACE DATA\"\n");
		  PetscFPrintf(PETSC_COMM_WORLD, f, "Variables=\"x\",\"y\",\"z\",\"tow_t\",\"tow_s\",\"tow_n\",\"p\",\"n_p\",\"FoundAroundcell\",\"rank\",\"nf_x\",\"nf_y\",\"nf_z\",\"nt_x\",\"nt_y\",\"nt_z\",\"ns_x\",\"ns_y\",\"ns_z\",\"di\"\n");
		  PetscFPrintf(PETSC_COMM_WORLD, f, "ZONE T=\"TRIANGLES\", N=%d, E=%d, F=FEBLOCK, ET=TRIANGLE, VARLOCATION=([1-3]=NODAL,[4-20]=CELLCENTERED)\n", n_v, n_elmt);
		  for (i=0; i<n_v; i++) {
		PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->x_bp[i]);
		  }//PetscFPrintf(PETSC_COMM_WORLD, f, "\n");
		  for (i=0; i<n_v; i++) {
		PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->y_bp[i]);
		  }//PetscFPrintf(PETSC_COMM_WORLD, f, "\n");
		  for (i=0; i<n_v; i++) {	
		PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->z_bp[i]);
		  }//PetscFPrintf(PETSC_COMM_WORLD, f, "\n");
		  for (i=0; i<n_elmt; i++) {
		PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", elmtinfo[i].Tow_wt);
		  }//PetscFPrintf(PETSC_COMM_WORLD, f, "\n");
		  for (i=0; i<n_elmt; i++) {
		PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", elmtinfo[i].Tow_ws);
		  }//PetscFPrintf(PETSC_COMM_WORLD, f, "\n");
		  for (i=0; i<n_elmt; i++) {
		PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", elmtinfo[i].Tow_wn);
		  }//PetscFPrintf(PETSC_COMM_WORLD, f, "\n");
		  for (i=0; i<n_elmt; i++) {
		PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", elmtinfo[i].P);  	
		}//PetscFPrintf(PETSC_COMM_WORLD, f, "\n");
		for (i=0; i<n_elmt; i++) {
		PetscFPrintf(PETSC_COMM_WORLD, f, "%d\n", elmtinfo[i].n_P);
		  }//PetscFPrintf(PETSC_COMM_WORLD, f, "\n");
		for (i=0; i<n_elmt; i++) {
		PetscFPrintf(PETSC_COMM_WORLD, f, "%d\n", elmtinfo[i].FoundAroundcell);
		  }//PetscFPrintf(PETSC_COMM_WORLD, f, "\n");
		for (i=0; i<n_elmt; i++) {
			PetscFPrintf(PETSC_COMM_WORLD, f, "%d\n", 0);
		  }//PetscFPrintf(PETSC_COMM_WORLD, f, "\n");
		  for (i=0; i<n_elmt; i++) {
		PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nf_x[i]);
		  }//PetscFPrintf(PETSC_COMM_WORLD, f, "\n");
		  for (i=0; i<n_elmt; i++) {
		PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nf_y[i]);
		  }//PetscFPrintf(PETSC_COMM_WORLD, f, "\n");
		  for (i=0; i<n_elmt; i++) {
		PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nf_z[i]);
		  }//PetscFPrintf(PETSC_COMM_WORLD, f, "\n");
		  for (i=0; i<n_elmt; i++) {
		PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nt_x[i]);
		  }//PetscFPrintf(PETSC_COMM_WORLD, f, "\n");
		  for (i=0; i<n_elmt; i++) {
		PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nt_y[i]);
		  }//PetscFPrintf(PETSC_COMM_WORLD, f, "\n");
		  for (i=0; i<n_elmt; i++) {
		PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nt_z[i]);
		  }//PetscFPrintf(PETSC_COMM_WORLD, f, "\n");
		  for (i=0; i<n_elmt; i++) {
		PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->ns_x[i]);
		  }//PetscFPrintf(PETSC_COMM_WORLD, f, "\n");
		  for (i=0; i<n_elmt; i++) {
		PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->ns_y[i]);
		  }//PetscFPrintf(PETSC_COMM_WORLD, f, "\n");
		  for (i=0; i<n_elmt; i++) {
		PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->ns_z[i]);
		  }//PetscFPrintf(PETSC_COMM_WORLD, f, "\n");
		  for (i=0; i<n_elmt; i++) {
		PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibminfo[i].d_i);
		  }//PetscFPrintf(PETSC_COMM_WORLD, f, "\n");
		  for (i=0; i<n_elmt; i++) {
		PetscFPrintf(PETSC_COMM_WORLD, f, "%d %d %d\n", ibm->nv1[i]+1, ibm->nv2[i]+1, ibm->nv3[i]+1);
		  }
		  fclose(f);
		  //}
	  }

	DAVecRestoreArray(fda, user->Ucat, &ucat);
	DAVecRestoreArray(da, user->P, &p);  
	DAVecRestoreArray(da, user->Nvert, &nvert);
   DAVecRestoreArray(fda, Coor,&coor);  

	return(0);
}