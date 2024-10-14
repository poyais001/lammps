// clang-format off
/* ----------------------------------------------------------------------
 *
 *                    *** Smooth Mach Dynamics ***
 *
 * This file is part of the MACHDYN package for LAMMPS.
 * Copyright (2014) Georg C. Ganzenmueller, georg.ganzenmueller@emi.fhg.de
 * Fraunhofer Ernst-Mach Institute for High-Speed Dynamics, EMI,
 * Eckerstrasse 4, D-79104 Freiburg i.Br, Germany.
 *
 * ----------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
 LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
 https://www.lammps.org/, Sandia National Laboratories
 LAMMPS development team: developers@lammps.org

 Copyright (2003) Sandia Corporation.  Under the terms of Contract
 DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
 certain rights in this software.  This software is distributed under
 the GNU General Public License.

 See the README file in the top-level LAMMPS directory.
 ------------------------------------------------------------------------- */

#include "pair_smd_tlsph.h"

#include "fix_smd_tlsph_reference_configuration.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "group.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "smd_kernels.h"
#include "smd_material_models.h"
#include "smd_math.h"
#include "update.h"

#include <cmath>
#include <cstring>
#include <Eigen/Eigen>

using namespace SMD_Kernels;
using namespace Eigen;
using namespace LAMMPS_NS;
using namespace SMD_Math;

static constexpr bool JAUMANN = false;
static constexpr double DETF_MIN = 0.002; // maximum compression deformation allow
static constexpr double DETF_MAX = 200.0; // maximum tension deformation allowed

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884197169399375105820974944
#endif

static double CalculateScale(const float degradation) {
  double start = 0.9;
  if (degradation <= start) {
    return 1.0;
  }
  if (degradation >= 1.0) {
    return 0.0;
  }
  
  return 0.5 + 0.5 * cos( M_PI * (degradation - start) / (1.0 - start) );
}

static Matrix3d CreateOrthonormalBasisFromOneVector(Vector3d sU) {
  Matrix3d P;
  Vector3d sV, sW;
  double sU_Norm;

  // Make sure that sU has a norm of one:
  sU_Norm = sU.norm();
  if (sU_Norm != 1.0) {
    sU /= sU_Norm;
  }

  if (abs(float(sU[1])) > 1.0e-15) {
    sV[0] = 0.0;
    sV[1] = - sU[2];
    sV[2] = sU[1];
  } else if (abs(float(sU[2])) > 1.0e-15) {
    sV[0] = sU[2];
    sV[1] = 0.0;
    sV[2] = -sU[0];
  } else {
    sV[0] = 0.0;
    sV[1] = 1.0;
    sV[2] = 0.0;
  }
  
  sV /= sV.norm();
  sW = sU.cross(sV);
  //sW /= sW.norm(); This can be skipped since sU and sV are orthogonal and both unitary.

  P.col(0) = sU;
  P.col(1) = sV;
  P.col(2) = sW;

  return P;
}

/* ---------------------------------------------------------------------- */

PairTlsph::PairTlsph(LAMMPS *lmp) :
  Pair(lmp) {

  onerad_dynamic = onerad_frozen = maxrad_dynamic = maxrad_frozen = nullptr;

  failureModel = nullptr;
  strengthModel = eos = nullptr;

  nmax = 0; // make sure no atom on this proc such that initial memory allocation is correct
  Fdot = Fincr = K = Kundeg = PK1 = nullptr;
  R = FincrInv = W = D = nullptr;
  detF = nullptr;
  smoothVelDifference = nullptr;
  surfaceNormal = nullptr;
  numNeighsRefConfig = nullptr;
  CauchyStress = nullptr;
  hourglass_error = nullptr;
  Lookup = nullptr;
  particle_dt = nullptr;
  vij_max = nullptr;

  updateFlag = 0;
  updateKundegFlag = 1;
  updateSurfaceNormal = 1;
  first = true;
  dtCFL = 0.0; // initialize dtCFL so it is set to safe value if extracted on zero-th timestep

  comm_forward = 22; // this pair style communicates 20 doubles to ghost atoms : PK1 tensor + F tensor + shepardWeight
  fix_tlsph_reference_configuration = nullptr;

  cut_comm = MAX(neighbor->cutneighmax, comm->cutghostuser); // cutoff radius within which ghost atoms are communicated.
}

/* ---------------------------------------------------------------------- */

PairTlsph::~PairTlsph() {

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(strengthModel);
    memory->destroy(eos);
    memory->destroy(Lookup);

    delete[] onerad_dynamic;
    delete[] onerad_frozen;
    delete[] maxrad_dynamic;
    delete[] maxrad_frozen;

    delete[] Fdot;
    delete[] Fincr;
    delete[] K;
    delete[] Kundeg;
    delete[] detF;
    delete[] PK1;
    delete[] smoothVelDifference;
    delete[] surfaceNormal;
    delete[] R;
    delete[] FincrInv;
    delete[] W;
    delete[] D;
    delete[] numNeighsRefConfig;
    delete[] CauchyStress;
    delete[] hourglass_error;
    delete[] particle_dt;
    delete[] vij_max;

    delete[] failureModel;
  }
}

/* ----------------------------------------------------------------------
 *
 * use half neighbor list to re-compute shape matrix
 *
 ---------------------------------------------------------------------- */

void PairTlsph::PreCompute() {
  tagint *mol = atom->molecule;
  double *vfrac = atom->vfrac;
  double *radius = atom->radius;
  double **x0 = atom->x0;
  double **x = atom->x;
  double **v = atom->vest; // extrapolated velocities corresponding to current positions
  double **vint = atom->v; // Velocity-Verlet algorithm velocities
  double *damage = atom->damage;
  tagint *tag = atom->tag;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double dt = update->dt;
  int jnum, jj, i, j, itype, idim;

  tagint **partner = (dynamic_cast<FixSMD_TLSPH_ReferenceConfiguration *>(modify->fix[ifix_tlsph]))->partner;
  int *npartner = (dynamic_cast<FixSMD_TLSPH_ReferenceConfiguration *>(modify->fix[ifix_tlsph]))->npartner;
  float **wfd_list = (dynamic_cast<FixSMD_TLSPH_ReferenceConfiguration *>(modify->fix[ifix_tlsph]))->wfd_list;
  float **wf_list = (dynamic_cast<FixSMD_TLSPH_ReferenceConfiguration *>(modify->fix[ifix_tlsph]))->wf_list;
  float **degradation_ij = (dynamic_cast<FixSMD_TLSPH_ReferenceConfiguration *>(modify->fix[ifix_tlsph]))->degradation_ij;
  Vector3d **partnerx0 = (dynamic_cast<FixSMD_TLSPH_ReferenceConfiguration *>(modify->fix[ifix_tlsph]))->partnerx0;
  double **partnervol = (dynamic_cast<FixSMD_TLSPH_ReferenceConfiguration *>(modify->fix[ifix_tlsph]))->partnervol;
  double r, r0, r0Sq, wf, wfd, h, irad, voli, volj, scale, shepardWeight, inverseShepardWeight;
  Vector3d dx, dx0, dx0mirror, dv, g;
  Matrix3d Ktmp, Ftmp, Fdottmp, L, U, eye;
  Vector3d vi, vj, vinti, vintj, xi, xj, x0i, x0j, dvint;
  int periodic = (domain->xperiodic || domain->yperiodic || domain->zperiodic);
  bool status;
  Matrix3d F0;
  double surfaceNormalNormi, dv_norm;

  dtCFL = 1.0e22;
  eye.setIdentity();

  for (i = 0; i < nlocal; i++) {
    vij_max[i] = 0.0;

    itype = type[i];
    /*if ((damage[i] >= 1.0) && (mol[i] >= 0)) {
      printf("deleting particle [%d] because damage = %f\n", tag[i], damage[i]);
      mol[i] = -1;
      D[i].setZero();
      Fdot[i].setZero();
      Fincr[i].setIdentity();
      smoothVelDifference[i].setZero();
      detF[i] = 1.0;
      K[i].setIdentity();

      vint[i][0] = 0.0;
      vint[i][1] = 0.0;
      vint[i][2] = 0.0;
      }*/
    
    if (setflag[itype][itype] == 1) {

      K[i].setZero();
      if (updateKundegFlag == 1) Kundeg[i].setZero();
      Fincr[i].setZero();
      Fdot[i].setZero();
      numNeighsRefConfig[i] = 0;
      smoothVelDifference[i].setZero();
      if (updateSurfaceNormal == 1) surfaceNormal[i].setZero();
      hourglass_error[i] = 0.0;

      if (mol[i] < 0) { // valid SPH particle have mol > 0
        continue;
      }

      // initialize aveage mass density
      h = 2.0 * radius[i];
      r0 = 0.0;
      spiky_kernel_and_derivative(h, r0, domain->dimension, wf, wfd);

      jnum = npartner[i];
      irad = radius[i];
      voli = vfrac[i];
      shepardWeight = wf * voli;

      // initialize Eigen data structures from LAMMPS data structures
      for (idim = 0; idim < 3; idim++) {
        xi(idim) = x[i][idim];
        x0i(idim) = x0[i][idim];
        vi(idim) = v[i][idim];
        vinti(idim) = vint[i][idim];
      }

      //Matrix3d gradAbsX;
      //gradAbsX.setZero();
      for (jj = 0; jj < jnum; jj++) {

        if (degradation_ij[i][jj] >= 1.0) {  
          volj = partnervol[i][jj];
          dx0 = partnerx0[i][jj] - x0i;
          
          if (periodic)
            domain->minimum_image(dx0(0), dx0(1), dx0(2));
          
          r0 = dx0.norm();
          if (updateKundegFlag == 1) Kundeg[i].noalias() -= volj * (wfd_list[i][jj] / r0) * dx0 * dx0.transpose();
          if (updateSurfaceNormal == 1) surfaceNormal[i].noalias() += volj * wfd_list[i][jj] * dx0;
          //printf("Link between %d and %d destroyed!\n", tag[i], partner[i][jj]);
          continue;
        }
        j = atom->map(partner[i][jj]);
        if (j < 0) { //                 // check if lost a partner without first breaking bond
          printf("Link between %d and %d destroyed without first breaking bond! Damage level in the link was: %f\n", tag[i], partner[i][jj], degradation_ij[i][jj]);
          volj = partnervol[i][jj];
          dx0 = partnerx0[i][jj] - x0i;
          
          if (periodic)
            domain->minimum_image(dx0(0), dx0(1), dx0(2));
          
          r0 = dx0.norm();
          if (updateKundegFlag == 1) Kundeg[i].noalias() -= volj * (wfd_list[i][jj] / r0) * dx0 * dx0.transpose();
          if (updateSurfaceNormal == 1) surfaceNormal[i].noalias() += volj * wfd_list[i][jj] * dx0;
          degradation_ij[i][jj] = 1.0;
          continue;
        }

        if (mol[j] < 0) { // particle has failed. do not include it for computing any property
          continue;
        }

        if (mol[i] != mol[j]) {
          continue;
        }
        /*if (failureModel[itype].integration_point_wise){ // check if the particles are fully damaged. If so, the bond is broken. This is important when the list of neighbors is updated.
          if ((damage[i] == 1) || (damage[j] == 1)) {
            degradation_ij[i][jj] = 1;
                                    partner[i][jj] = 0;
            continue;
          }
          }*/

        // initialize Eigen data structures from LAMMPS data structures
        for (idim = 0; idim < 3; idim++) {
          xj(idim) = x[j][idim];
          x0j(idim) = x0[j][idim];
          vj(idim) = v[j][idim];
          vintj(idim) = vint[j][idim];
        }

        dx0 = x0j - x0i;
        dx = xj - xi;
        r = dx.norm(); // current distance

        if (periodic)
          domain->minimum_image(dx0(0), dx0(1), dx0(2));

        r0Sq = dx0.squaredNorm();
        h = irad + radius[j];

        r0 = sqrt(r0Sq);
        volj = vfrac[j];

        // distance vectors in current and reference configuration, velocity difference
        dv = vj - vi;
        dvint = vintj - vinti;
        dv_norm = dv.norm();
        if (dv_norm > vij_max[i]) vij_max[i] = dv_norm;

        // scale the interaction according to the damage variable
        scale = CalculateScale(degradation_ij[i][jj]);
        wf = wf_list[i][jj] * scale;
        wfd = wfd_list[i][jj] * scale;
        g = (wfd / r0) * dx0;

        /* build matrices */
        Ktmp = -g * dx0.transpose();
        Fdottmp = -dv * g.transpose();
        Ftmp = -(dx - dx0) * g.transpose();

        K[i].noalias() += volj * Ktmp;
        if (updateKundegFlag == 1) Kundeg[i].noalias() -= volj * (wfd_list[i][jj] / r0) * dx0 * dx0.transpose();
        Fdot[i].noalias() += volj * Fdottmp;
        Fincr[i].noalias() += volj * Ftmp;
        shepardWeight += volj * wf;
        smoothVelDifference[i].noalias() += volj * wf * dvint;
        
        if (updateSurfaceNormal == 1) surfaceNormal[i].noalias() += volj * wfd_list[i][jj] * dx0;
        
        numNeighsRefConfig[i]++;
      } // end loop over j

      // normalize average velocity field around an integration point
      if (shepardWeight > 0.0) {
        inverseShepardWeight = 1/shepardWeight;
        smoothVelDifference[i] *= inverseShepardWeight;
      } else {
        smoothVelDifference[i].setZero();
      }

      pseudo_inverse_SVD(K[i]);
      if (updateKundegFlag == 1) {
        Matrix3d KundegINV;
        KundegINV = Kundeg[i];
        pseudo_inverse_SVD(KundegINV);
        surfaceNormal[i] = KundegINV * surfaceNormal[i];
      } else {
        if (updateSurfaceNormal == 1) surfaceNormal[i] = Kundeg[i] * surfaceNormal[i];
      }
      Fdot[i] *= K[i];
      Fincr[i] *= K[i];
      Fincr[i].noalias() += eye;
      
      if (updateKundegFlag == 1) {
      // Recalculate Kundeg to include mirror particles:
      
      surfaceNormalNormi = surfaceNormal[i].norm();
      //Vector3d sU;
      //sU = surfaceNormal[i] / surfaceNormalNormi;

      if (surfaceNormalNormi > 0.75) {
        surfaceNormal[i] /= surfaceNormalNormi;
        
        for (jj = 0; jj < jnum; jj++) {
          
          if (degradation_ij[i][jj] >= 1.0) 
          {
            volj = partnervol[i][jj];
            dx0 = partnerx0[i][jj] - x0i;
            
            if (periodic)
              domain->minimum_image(dx0(0), dx0(1), dx0(2));

            if (surfaceNormal[i].dot(dx0) > -0.5*pow(volj, 1.0/3.0)) {
              continue;
            }

            dx0mirror = dx0 - 2 * (dx0.dot(surfaceNormal[i])) * surfaceNormal[i];
            r0 = dx0.norm();
            Kundeg[i].noalias() -= volj * (wfd_list[i][jj] / r0) * dx0mirror * dx0mirror.transpose();
            continue;
          }
          j = atom->map(partner[i][jj]);
          if (j < 0) { //      // check if lost a partner without first breaking bond
            error->all(FLERR, "Bond broken not detected during PreCompute - 1!");
            continue;
          }
          
          if (mol[j] < 0) { // particle has failed. do not include it for computing any property
            continue;
          }
          
          if (mol[i] != mol[j]) {
            continue;
          }
                    
          // initialize Eigen data structures from LAMMPS data structures
          for (idim = 0; idim < 3; idim++) {
            x0j(idim) = x0[j][idim];
          }
          dx0 = x0j - x0i;
          
          if (periodic)
            domain->minimum_image(dx0(0), dx0(1), dx0(2));
          
          if (surfaceNormal[i].dot(dx0) > -0.5*pow(volj, 1.0/3.0)) {
            continue;
          }
          
          dx0mirror = dx0 - 2 * (dx0.dot(surfaceNormal[i])) * surfaceNormal[i];
          r0 = dx0.norm();
          volj = vfrac[j];
          
          Kundeg[i].noalias() -= volj * (wfd_list[i][jj] / r0) * dx0mirror * dx0mirror.transpose();
        }
        
      } else {
        surfaceNormal[i].setZero();
      }
      // END RECALCULATE Kundeg
      pseudo_inverse_SVD(Kundeg[i]);
      }

      if (JAUMANN) {
        R[i].setIdentity(); // for Jaumann stress rate, we do not need a subsequent rotation back into the reference configuration
      } else {
        status = PolDec(Fincr[i], R[i], U, false); // polar decomposition of the deformation gradient, F = R * U
        if (!status) {
          error->message(FLERR, "Polar decomposition of deformation gradient failed.\n");
          mol[i] = -1;
        } else {
          Fincr[i] = R[i] * U;
        }
      }

      detF[i] = Fincr[i].determinant();
      FincrInv[i] = Fincr[i].inverse();

      // velocity gradient
      L = Fdot[i] * FincrInv[i];

      // symmetric (D) and asymmetric (W) parts of L
      D[i] = 0.5 * (L + L.transpose());
      W[i] = 0.5 * (L - L.transpose()); // spin tensor:: need this for Jaumann rate

      // unrotated rate-of-deformation tensor d, see right side of Pronto2d, eqn.(2.1.7)
      // convention: unrotated frame is that one, where the true rotation of an integration point has been subtracted.
      // stress in the unrotated frame of reference is denoted sigma (stress seen by an observer doing rigid body rotations along with the material)
      // stress in the true frame of reference (a stationary observer) is denoted by T, "true stress"
      D[i] = (R[i].transpose() * D[i] * R[i]).eval();

      // limit strain rate
      //double limit = 1.0e-3 * Lookup[SIGNAL_VELOCITY][itype] / radius[i];
      //D[i] = LimitEigenvalues(D[i], limit);

      /*
       * make sure F stays within some limits
       */

      if ((numNeighsRefConfig[i] == 0)) {
        utils::logmesg(lmp, "deleting particle [{}] because nn = {}\n", tag[i], numNeighsRefConfig[i]);
        dtCFL = MIN(dtCFL, dt); //Keep the same (small) time step when a particule breaks.           
        mol[i] = -1;
      }
      /*if ((detF[i] < DETF_MIN) || (detF[i] > DETF_MAX) || (numNeighsRefConfig[i] == 0)) {
        utils::logmesg(lmp, "deleting particle [{}] because det(F)={}f is outside stable range"
                       " {} -- {} \n", tag[i], Fincr[i].determinant(), DETF_MIN, DETF_MAX);
        utils::logmesg(lmp,"nn = {}, damage={}\n", numNeighsRefConfig[i], damage[i]);
        mol[i] = -1;
      }*/

      if (mol[i] < 0) {
        D[i].setZero();
        Fdot[i].setZero();
        Fincr[i].setIdentity();
        smoothVelDifference[i].setZero();
        detF[i] = 1.0;
        K[i].setIdentity();
        Kundeg[i].setIdentity();

        vint[i][0] = 0.0;
        vint[i][1] = 0.0;
        vint[i][2] = 0.0;
      }
    } // end check setflag 
  } // end loop over i
  updateKundegFlag = 0;
  updateSurfaceNormal = 0;
}

/* ---------------------------------------------------------------------- */

void PairTlsph::compute(int eflag, int vflag) {

  if (atom->nmax > nmax) {
    nmax = atom->nmax;
    delete[] Fdot;
    Fdot = new Matrix3d[nmax]; // memory usage: 9 doubles
    delete[] Fincr;
    Fincr = new Matrix3d[nmax]; // memory usage: 9 doubles
    delete[] K;
    K = new Matrix3d[nmax]; // memory usage: 9 doubles
    delete[] Kundeg;
    Kundeg = new Matrix3d[nmax]; // memory usage: 9 doubles
    delete[] PK1;
    PK1 = new Matrix3d[nmax]; // memory usage: 9 doubles; total 5*9=45 doubles
    delete[] detF;
    detF = new double[nmax]; // memory usage: 1 double; total 46 doubles
    delete[] smoothVelDifference;
    smoothVelDifference = new Vector3d[nmax]; // memory usage: 3 doubles; total 49 doubles
    delete[] surfaceNormal;
    surfaceNormal = new Vector3d[nmax]; // memory usage: 3 doubles; total 49 doubles
    delete[] R;
    R = new Matrix3d[nmax]; // memory usage: 9 doubles; total 67 doubles
    delete[] FincrInv;
    FincrInv = new Matrix3d[nmax]; // memory usage: 9 doubles; total 85 doubles
    delete[] W;
    W = new Matrix3d[nmax]; // memory usage: 9 doubles; total 94 doubles
    delete[] D;
    D = new Matrix3d[nmax]; // memory usage: 9 doubles; total 103 doubles
    delete[] numNeighsRefConfig;
    numNeighsRefConfig = new int[nmax]; // memory usage: 1 int; total 108 doubles
    delete[] CauchyStress;
    CauchyStress = new Matrix3d[nmax]; // memory usage: 9 doubles; total 118 doubles
    delete[] hourglass_error;
    hourglass_error = new double[nmax];
    delete[] particle_dt;
    particle_dt = new double[nmax];
    delete[] vij_max;
    vij_max = new double[nmax];
  }

  if (first) { // return on first call, because reference connectivity lists still needs to be built. Also zero quantities which are otherwise undefined.
    first = false;

    for (int i = 0; i < atom->nlocal; i++) {
      Fincr[i].setZero();
      detF[i] = 0.0;
      smoothVelDifference[i].setZero();
      D[i].setZero();
      numNeighsRefConfig[i] = 0;
      CauchyStress[i].setZero();
      hourglass_error[i] = 0.0;
      particle_dt[i] = 0.0;
      vij_max[i] = 0.0;
    }

    return;
  }

  /*
   * calculate deformations and rate-of-deformations
   */
  PairTlsph::PreCompute();

  /*
   * calculate stresses from constitutive models
   */
  PairTlsph::AssembleStress();

  /*
   * QUANTITIES ABOVE HAVE ONLY BEEN CALCULATED FOR NLOCAL PARTICLES.
   * NEED TO DO A FORWARD COMMUNICATION TO GHOST ATOMS NOW
   */
  comm->forward_comm(this);

  /*
   * compute forces between particles
   */
  updateFlag = 0;
  ComputeForces(eflag, vflag);

  UpdateDegradation();
}

void PairTlsph::ComputeForces(int eflag, int vflag) {
  tagint *mol = atom->molecule;
  tagint *tag = atom->tag;
  double **x = atom->x;
  double **v = atom->vest;
  double **x0 = atom->x0;
  double **f = atom->f;
  double *vfrac = atom->vfrac;
  double *desph = atom->desph;
  double *rmass = atom->rmass;
  double *radius = atom->radius;
  double *damage = atom->damage;
  double *plastic_strain = atom->eff_plastic_strain;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int i, j, jj, jnum, itype, idim;
  double r, hg_mag, wf, wfd, h, r0, r0Sq, voli, volj, r_plus_h, over_r_plus_h;
  double delVdotDelR, deltaE, mu_ij, hg_err, gamma_dot_dx, delta, scale, rmassij;
  double softening_strain, shepardWeight;
  double surfaceNormalNormi;
  Vector3d fi, fj, dx0, dx, dv, f_stress, f_hg, dxp_i, dxp_j, gamma, g, gamma_i, gamma_j, x0i, x0j, wfddx;
  Vector3d xi, xj, vi, vj, f_visc, sumForces, f_spring;
  int periodic = (domain->xperiodic || domain->yperiodic || domain->zperiodic);

  tagint **partner = (dynamic_cast<FixSMD_TLSPH_ReferenceConfiguration *>(modify->fix[ifix_tlsph]))->partner;
  int *npartner = (dynamic_cast<FixSMD_TLSPH_ReferenceConfiguration *>(modify->fix[ifix_tlsph]))->npartner;
  float **wfd_list = (dynamic_cast<FixSMD_TLSPH_ReferenceConfiguration *>(modify->fix[ifix_tlsph]))->wfd_list;
  float **wf_list = (dynamic_cast<FixSMD_TLSPH_ReferenceConfiguration *>(modify->fix[ifix_tlsph]))->wf_list;
  float **degradation_ij = (dynamic_cast<FixSMD_TLSPH_ReferenceConfiguration *>(modify->fix[ifix_tlsph]))->degradation_ij;
  float **energy_per_bond = (dynamic_cast<FixSMD_TLSPH_ReferenceConfiguration *>(modify->fix[ifix_tlsph]))->energy_per_bond;
  Vector3d **partnerx0 = (dynamic_cast<FixSMD_TLSPH_ReferenceConfiguration *>(modify->fix[ifix_tlsph]))->partnerx0;
  double **partnervol = (dynamic_cast<FixSMD_TLSPH_ReferenceConfiguration *>(modify->fix[ifix_tlsph]))->partnervol;
  Matrix3d eye;
  Vector3d sU, sV, sW;
  eye.setIdentity();

  ev_init(eflag, vflag);

  /*
   * iterate over pairs of particles i, j and assign forces using PK1 stress tensor
   */

  //updateFlag = 0;
  hMin = 1.0e22;
  dtRelative = 1.0e22;

  for (i = 0; i < nlocal; i++) {

    if (mol[i] < 0) {
      continue; // Particle i is not a valid SPH particle (anymore). Skip all interactions with this particle.
    }

    itype = type[i];
    jnum = npartner[i];
    voli = vfrac[i];

    // initialize aveage mass density
    h = 2.0 * radius[i];
    r = 0.0;
    spiky_kernel_and_derivative(h, r, domain->dimension, wf, wfd);
    shepardWeight = wf * voli;

    for (idim = 0; idim < 3; idim++) {
      x0i(idim) = x0[i][idim];
      xi(idim) = x[i][idim];
      vi(idim) = v[i][idim];
    }

    for (jj = 0; jj < jnum; jj++) {
      j = atom->map(partner[i][jj]);
      if (j < 0) { //                 // check if lost a partner without first breaking bond
        error->all(FLERR, "Bond broken not detected during PreCompute - 2!");
        continue;
      }

      if (mol[j] < 0) {
        continue; // Particle j is not a valid SPH particle (anymore). Skip all interactions with this particle.
      }

      if (mol[i] != mol[j]) {
        continue;
      }

      if (type[j] != itype)
        error->all(FLERR, "particle pair is not of same type!");

      for (idim = 0; idim < 3; idim++) {
        x0j(idim) = x0[j][idim];
        xj(idim) = x[j][idim];
        vj(idim) = v[j][idim];
      }

      if (periodic)
        domain->minimum_image(dx0(0), dx0(1), dx0(2));

      // check that distance between i and j (in the reference config) is less than cutoff
      dx0 = x0j - x0i;
      r0Sq = dx0.squaredNorm();
      h = radius[i] + radius[j];
      hMin = MIN(hMin, h);
      r0 = sqrt(r0Sq);
      volj = vfrac[j];

      // distance vectors in current and reference configuration, velocity difference
      dx = xj - xi;
      dv = vj - vi;
      r = dx.norm(); // current distance

      // scale the interaction according to the damage variable
      //scale = CalculateScale(degradation_ij[i][jj], r, r0);
      scale = CalculateScale(degradation_ij[i][jj]);
      wf = wf_list[i][jj];// * scale;
      wfd = wfd_list[i][jj];// * scale;

      g = (wfd_list[i][jj] / r0) * dx0; // uncorrected kernel gradient

      /*
       * force contribution -- note that the kernel gradient correction has been absorbed into PK1
       */
      
      // What is required is to build a basis with surfaceNormal as one of the vectors:

      f_stress = -(voli * volj * scale) * (PK1[j] + PK1[i]) * (Kundeg[i] * g);

      energy_per_bond[i][jj] = f_stress.dot(dx); // THIS IS NOT THE ENERGY PER BOND, I AM USING THIS VARIABLE TO STORE THIS VALUE TEMPORARILY
      
      /*
       * artificial viscosity
       */
      over_r_plus_h = 1 / (r + 0.1 * h);
      delVdotDelR = dx.dot(dv) * over_r_plus_h; // project relative velocity onto unit particle distance vector [m/s]
      LimitDoubleMagnitude(delVdotDelR, 0.01 * Lookup[SIGNAL_VELOCITY][itype]);
      mu_ij = h * delVdotDelR * over_r_plus_h; // units: [m * m/s / m = m/s]
      wfddx = wfd * dx;
      //if (delVdotDelR < 0) { // i.e. if (dx.dot(dv) < 0) // To be consistent with the viscosity proposed by Monaghan
      //visc_magnitude = ((-Lookup[VISCOSITY_Q1][itype] * Lookup[SIGNAL_VELOCITY][itype] + Lookup[VISCOSITY_Q2][itype] * mu_ij) * mu_ij) / Lookup[REFERENCE_DENSITY][itype]; // units: m^5/(s^2 kg))
      rmassij = rmass[i] * rmass[j];
      r_plus_h = r + 1.0e-2 * h;
      f_visc = rmassij * (-Lookup[VISCOSITY_Q1][itype] * Lookup[SIGNAL_VELOCITY][itype] + Lookup[VISCOSITY_Q2][itype] * mu_ij) * mu_ij * wfddx / (r_plus_h * Lookup[REFERENCE_DENSITY][itype]); // units: kg^2 * m^5/(s^2 kg) * m^-4 = kg m / s^2 = N
        //} else {
        //f_visc = Vector3d(0.0, 0.0, 0.0);
        //}

      /*
       * hourglass deviation of particles i and j
       */

      gamma = 0.5 * (Fincr[i] + Fincr[j]) * dx0 - dx;
      hg_err = gamma.norm() / r0;
      hourglass_error[i] += volj * wf * hg_err;

      /* SPH-like hourglass formulation */

      if (MAX(plastic_strain[i], plastic_strain[j]) > 1.0e-3) {
        /*
         * viscous hourglass formulation for particles with plastic deformation
         */
        delta = gamma.dot(dx);
        if (delVdotDelR * delta < 0.0) {
          hg_err = MAX(hg_err, 0.05); // limit hg_err to avoid numerical instabilities
          hg_mag = -hg_err * Lookup[HOURGLASS_CONTROL_AMPLITUDE][itype] * Lookup[SIGNAL_VELOCITY][itype] * mu_ij
            / Lookup[REFERENCE_DENSITY][itype]; // this has units of pressure
        } else {
          hg_mag = 0.0;
        }
        f_hg = rmassij * hg_mag * wfddx / r_plus_h;

      } else {
        /*
         * stiffness hourglass formulation for particle in the elastic regime
         */

        gamma_dot_dx = gamma.dot(dx); // project hourglass error vector onto pair distance vector
        LimitDoubleMagnitude(gamma_dot_dx, 0.1 * r); // limit projected vector to avoid numerical instabilities
        delta = 0.5 * gamma_dot_dx * over_r_plus_h; // delta has dimensions of [m]
        hg_mag = Lookup[HOURGLASS_CONTROL_AMPLITUDE][itype] * delta / (r0Sq + 0.01 * h * h); // hg_mag has dimensions [m^(-1)]
        hg_mag *= -voli * volj * wf * Lookup[YOUNGS_MODULUS][itype]; // hg_mag has dimensions [J*m^(-1)] = [N]
        f_hg = (hg_mag * over_r_plus_h) * dx;
      }

      // scale hourglass force with damage
      f_hg *= (1.0 - damage[i]) * (1.0 - damage[j]);

      // sum stress, viscous, and hourglass forces
      sumForces = f_stress + f_visc + f_hg; // + f_spring;

      // energy rate -- project velocity onto force vector
      deltaE = 0.5 * sumForces.dot(dv);

      // apply forces to pair of particles
      f[i][0] += sumForces(0);
      f[i][1] += sumForces(1);
      f[i][2] += sumForces(2);
      desph[i] += deltaE;

      // tally atomistic stress tensor
      if (evflag) {
        ev_tally_xyz(i, j, nlocal, 0, 0.0, 0.0, sumForces(0), sumForces(1), sumForces(2), dx(0), dx(1), dx(2));
      }

      shepardWeight += wf * volj;

      // check if a particle has moved too much w.r.t another particle
      if (r > r0) {
        if (update_method == UPDATE_CONSTANT_THRESHOLD) {
          if (r - r0 > update_threshold) {
            updateFlag = 1;
          }
        } else if (update_method == UPDATE_PAIRWISE_RATIO) {
          if ((r - r0) / h > update_threshold) {
            updateFlag = 1;
          }
        }
      }

      if (failureModel[itype].failure_energy_release_rate) {
        energy_per_bond[i][jj] += update->dt * f_stress.dot(dv) / (voli * volj);
      }

    } // end loop over jj neighbors of i

    // avoid division by zero and overflow
    if ((shepardWeight != 0.0) && (fabs(hourglass_error[i]) < 1.0e300)) {
      hourglass_error[i] /= shepardWeight;
    }
    double deltat_1 = sqrt(2 * radius[i] * rmass[i]/ sqrt( f[i][0]*f[i][0] + f[i][1]*f[i][1] + f[i][2]*f[i][2] ));
    if (particle_dt[i] > deltat_1) {
      printf("particle_dt[%d] > deltat_1 with f = [%f %f %f]\n", tag[i], f[i][0], f[i][1], f[i][2]);
    }
    particle_dt[i] = MIN(particle_dt[i], deltat_1); // Monaghan deltat_1 
    dtCFL = MIN(dtCFL, particle_dt[i]);

  } // end loop over i

  //cout << "Here is sumf_stress.norm(): " << sumf_stress.norm() << endl;
  if (vflag_fdotr)
    virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
 assemble unrotated stress tensor using deviatoric and pressure components.
 Convert to corotational Cauchy stress, then to PK1 stress and apply
 shape matrix correction
 ------------------------------------------------------------------------- */
void PairTlsph::AssembleStress() {
  tagint *mol = atom->molecule;
  double **v = atom->vest;
  double *eff_plastic_strain = atom->eff_plastic_strain;
  double *eff_plastic_strain_rate = atom->eff_plastic_strain_rate;
  double **tlsph_stress = atom->smd_stress;
  int *type = atom->type;
  double *radius = atom->radius;
  double *damage = atom->damage;
  double *rmass = atom->rmass;
  double *vfrac = atom->vfrac;
  double *rho = atom->rho;
  double *esph = atom->esph;
  double pInitial, d_iso, pFinal, p_rate, plastic_strain_increment;
  int i, itype, idim;
  int nlocal = atom->nlocal;
  double dt = update->dt;
  double M_eff, p_wave_speed, mass_specific_energy, vol_specific_energy;
  Matrix3d sigma_rate, eye, sigmaInitial, sigmaFinal, T, T_damaged, Jaumann_rate, sigma_rate_check;
  Matrix3d d_dev, sigmaInitial_dev, sigmaFinal_dev, sigma_dev_rate, strain;
  Vector3d vi;

  eye.setIdentity();
  //dtCFL = 1.0e22;
  pFinal = 0.0;

  for (i = 0; i < nlocal; i++) {
    particle_dt[i] = 0.0;

    itype = type[i];
    if (setflag[itype][itype] == 1) {
      if (mol[i] > 0) { // only do the following if particle has not failed -- mol < 0 means particle has failed

        /*
         * initial stress state: given by the unrotateted Cauchy stress.
         * Assemble Eigen 3d matrix from stored stress state
         */
        sigmaInitial(0, 0) = tlsph_stress[i][0];
        sigmaInitial(0, 1) = tlsph_stress[i][1];
        sigmaInitial(0, 2) = tlsph_stress[i][2];
        sigmaInitial(1, 1) = tlsph_stress[i][3];
        sigmaInitial(1, 2) = tlsph_stress[i][4];
        sigmaInitial(2, 2) = tlsph_stress[i][5];
        sigmaInitial(1, 0) = sigmaInitial(0, 1);
        sigmaInitial(2, 0) = sigmaInitial(0, 2);
        sigmaInitial(2, 1) = sigmaInitial(1, 2);

        //cout << "this is sigma initial" << endl << sigmaInitial << endl;

        pInitial = sigmaInitial.trace() / 3.0; // isotropic part of initial stress
        sigmaInitial_dev = Deviator(sigmaInitial);
        d_iso = D[i].trace(); // volumetric part of stretch rate
        d_dev = Deviator(D[i]); // deviatoric part of stretch rate
        strain = 0.5 * (Fincr[i].transpose() * Fincr[i] - eye);
        mass_specific_energy = esph[i] / rmass[i]; // energy per unit mass
        rho[i] = rmass[i] / (detF[i] * vfrac[i]);
        vol_specific_energy = mass_specific_energy * rho[i]; // energy per current volume

        /*
         * pressure: compute pressure rate p_rate and final pressure pFinal
         */

        ComputePressure(i, rho[i], mass_specific_energy, vol_specific_energy, pInitial, d_iso, pFinal, p_rate);

        /*
         * material strength
         */

        //cout << "this is the strain deviator rate" << endl << d_dev << endl;
        ComputeStressDeviator(i, mass_specific_energy, sigmaInitial_dev, d_dev, sigmaFinal_dev, sigma_dev_rate, plastic_strain_increment, pFinal);
        //cout << "this is the stress deviator rate" << endl << sigma_dev_rate << endl;

        // keep a rolling average of the plastic strain rate over the last 100 or so timesteps
        eff_plastic_strain[i] += plastic_strain_increment;

        // compute a characteristic time over which to average the plastic strain
        double tav = 1000 * radius[i] / (Lookup[SIGNAL_VELOCITY][itype]);
        eff_plastic_strain_rate[i] -= eff_plastic_strain_rate[i] * dt / tav;
        eff_plastic_strain_rate[i] += plastic_strain_increment / tav;
        eff_plastic_strain_rate[i] = MAX(0.0, eff_plastic_strain_rate[i]);

        /*
         *  assemble total stress from pressure and deviatoric stress
         */
        sigmaFinal = pFinal * eye + sigmaFinal_dev; // this is the stress that is kept

        if (JAUMANN) {
          /*
           * sigma is already the co-rotated Cauchy stress.
           * The stress rate, however, needs to be made objective.
           */

          if (dt > 1.0e-16) {
            sigma_rate = (1.0 / dt) * (sigmaFinal - sigmaInitial);
          } else {
            sigma_rate.setZero();
          }

          Jaumann_rate = sigma_rate + W[i] * sigmaInitial + sigmaInitial * W[i].transpose();
          sigmaFinal = sigmaInitial + dt * Jaumann_rate;
          T = sigmaFinal;
        } else {
          /*
           * sigma is the unrotated stress.
           * need to do forward rotation of the unrotated stress sigma to the current configuration
           */
          T = R[i] * sigmaFinal * R[i].transpose();
        }

        /*
         * store unrotated stress in atom vector
         * symmetry is exploited
         */
        tlsph_stress[i][0] = sigmaFinal(0, 0);
        tlsph_stress[i][1] = sigmaFinal(0, 1);
        tlsph_stress[i][2] = sigmaFinal(0, 2);
        tlsph_stress[i][3] = sigmaFinal(1, 1);
        tlsph_stress[i][4] = sigmaFinal(1, 2);
        tlsph_stress[i][5] = sigmaFinal(2, 2);

        /*
         *  Damage due to failure criteria.
         */

        if (failureModel[itype].integration_point_wise) {
          ComputeDamage(i, strain, T, T_damaged, plastic_strain_increment);
          T = T_damaged;
        }

        // store rotated, "true" Cauchy stress
        CauchyStress[i] = T;

        /*
         * We have the corotational Cauchy stress.
         * Convert to PK1. Note that reference configuration used for computing the forces is linked via
         * the incremental deformation gradient, not the full deformation gradient.
         */
        PK1[i] = detF[i] * T * FincrInv[i].transpose();

        /*
         * pre-multiply stress tensor with shape matrix to save computation in force loop
         */
        //PK1[i] = PK1[i] * K[i];

        /*
         * compute stable time step according to Pronto 2d
         */

        Matrix3d deltaSigma;
        deltaSigma = sigmaFinal - sigmaInitial;
        p_rate = deltaSigma.trace() / (3.0 * dt + 1.0e-16);
        sigma_dev_rate = Deviator(deltaSigma) / (dt + 1.0e-16);

        double K_eff, mu_eff;
        effective_longitudinal_modulus(itype, dt, d_iso, p_rate, d_dev, sigma_dev_rate, damage[i], K_eff, mu_eff, M_eff);
        p_wave_speed = sqrt(M_eff / rho[i]);

        if (mol[i] < 0) {
          error->one(FLERR, "this should not happen");
        }

        for (idim = 0; idim < 3; idim++) {
          vi(idim) = v[i][idim];
        }
        //double max_damage = max(0.0001, 1 - damage[i]);
        particle_dt[i] = 2.0 * radius[i] / (p_wave_speed + vij_max[i]); //* max(0.0001, 1 - damage[i] * vi.norm()*dt/radius[i]);
        dtCFL = MIN(dtCFL, particle_dt[i]);

      } else { // end if mol > 0
        PK1[i].setZero();
        K[i].setIdentity();
        Kundeg[i].setIdentity();
        CauchyStress[i].setZero();
        sigma_rate.setZero();
        tlsph_stress[i][0] = 0.0;
        tlsph_stress[i][1] = 0.0;
        tlsph_stress[i][2] = 0.0;
        tlsph_stress[i][3] = 0.0;
        tlsph_stress[i][4] = 0.0;
        tlsph_stress[i][5] = 0.0;
      } // end  if mol > 0
    } // end setflag
  } // end for
}

/* ----------------------------------------------------------------------
 allocate all arrays
 ------------------------------------------------------------------------- */

void PairTlsph::allocate() {
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(strengthModel, n + 1, "pair:strengthmodel");
  memory->create(eos, n + 1, "pair:eosmodel");
  failureModel = new failure_types[n + 1];
  memory->create(Lookup, MAX_KEY_VALUE, n + 1, "pair:LookupTable");

  memory->create(cutsq, n + 1, n + 1, "pair:cutsq"); // always needs to be allocated, even with granular neighborlist

  onerad_dynamic = new double[n + 1];
  onerad_frozen = new double[n + 1];
  maxrad_dynamic = new double[n + 1];
  maxrad_frozen = new double[n + 1];

}

/* ----------------------------------------------------------------------
 global settings
 ------------------------------------------------------------------------- */

void PairTlsph::settings(int narg, char **arg) {

  if (comm->me == 0)
    utils::logmesg(lmp,"\n>>========>>========>>========>>========>>========>>========>>========>>========>>========>>========>>========>>========\n"
                   "TLSPH settings\n");

  /*
   * default value for update_threshold for updates of reference configuration:
   * The maximum relative displacement which is tracked by the construction of LAMMPS' neighborlists
   * is the following.
   */

  cut_comm = MAX(neighbor->cutneighmax, comm->cutghostuser); // cutoff radius within which ghost atoms are communicated.
  update_threshold = cut_comm;
  update_method = UPDATE_NONE;

  int iarg = 0;

  while (true) {

    if (iarg >= narg) {
      break;
    }

    if (strcmp(arg[iarg], "*UPDATE_CONSTANT") == 0) {
      iarg++;
      if (iarg == narg) {
        error->all(FLERR, "expected number following *UPDATE_CONSTANT keyword");
      }

      update_method = UPDATE_CONSTANT_THRESHOLD;
      update_threshold = utils::numeric(FLERR, arg[iarg],false,lmp);

    } else if (strcmp(arg[iarg], "*UPDATE_PAIRWISE") == 0) {
      iarg++;
      if (iarg == narg) {
        error->all(FLERR, "expected number following *UPDATE_PAIRWISE keyword");
      }

      update_method = UPDATE_PAIRWISE_RATIO;
      update_threshold = utils::numeric(FLERR, arg[iarg],false,lmp);

    } else {
      error->all(FLERR, "Illegal keyword for smd/integrate_tlsph: {}\n", arg[iarg]);
    }
    iarg++;
  }

  if ((update_threshold > cut_comm) && (update_method == UPDATE_CONSTANT_THRESHOLD)) {
    if (comm->me == 0) {
      utils::logmesg(lmp, "\n                ***** WARNING ***\n");
      utils::logmesg(lmp, "requested reference configuration update threshold is {} length units\n", update_threshold);
      utils::logmesg(lmp, "This value exceeds the maximum value {} beyond which TLSPH displacements can be tracked at current settings.\n",cut_comm);
      utils::logmesg(lmp, "Expect loss of neighbors!\n");
    }
  }

  if (comm->me == 0) {
    if (update_method == UPDATE_CONSTANT_THRESHOLD) {
      utils::logmesg(lmp, "... will update reference configuration if magnitude of relative displacement exceeds {} length units\n",
             update_threshold);
    } else if (update_method == UPDATE_PAIRWISE_RATIO) {
      utils::logmesg(lmp, "... will update reference configuration if ratio pairwise distance / smoothing length  exceeds {}\n",
             update_threshold);
    } else if (update_method == UPDATE_NONE) {
      utils::logmesg(lmp, "... will never update reference configuration\n");
    }
    utils::logmesg(lmp,">>========>>========>>========>>========>>========>>========>>========>>========>>========>>========>>========>>========\n");
  }
}

/* ----------------------------------------------------------------------
 set coeffs for one or more type pairs
 ------------------------------------------------------------------------- */

void PairTlsph::coeff(int narg, char **arg) {
  int ioffset, iarg, iNextKwd, itype;
  std::string s, t;

  if (narg < 3)
    error->all(FLERR, "number of arguments for pair tlsph is too small!");

  if (!allocated)
    allocate();

  /*
   * check that TLSPH parameters are given only in i,i form
   */
  if (utils::inumeric(FLERR, arg[0], false, lmp) != utils::inumeric(FLERR, arg[1], false, lmp))
    error->all(FLERR, "TLSPH coefficients can only be specified between particles of same type!");

  itype = utils::inumeric(FLERR, arg[0],false,lmp);

// set all eos, strength and failure models to inactive by default
  eos[itype] = EOS_NONE;
  strengthModel[itype] = STRENGTH_NONE;

  if (comm->me == 0)
    utils::logmesg(lmp,"\n>>========>>========>>========>>========>>========>>========>>========>>========>>========>>========>>========>>========\n"
                   "SMD / TLSPH PROPERTIES OF PARTICLE TYPE {}:\n", itype);

  /*
   * read parameters which are common -- regardless of material / eos model
   */

  ioffset = 2;
  if (strcmp(arg[ioffset], "*COMMON") != 0)
    error->all(FLERR, "common keyword missing!");

  t = std::string("*");
  iNextKwd = -1;
  for (iarg = ioffset + 1; iarg < narg; iarg++) {
    s = std::string(arg[iarg]);
    if (s.compare(0, t.length(), t) == 0) {
      iNextKwd = iarg;
      break;
    }
  }

  if (iNextKwd < 0) error->all(FLERR, "no *KEYWORD terminates *COMMON");
  if (iNextKwd - ioffset != 7 + 1)
    error->all(FLERR, "expected 7 arguments following *COMMON but got {}\n", iNextKwd - ioffset - 1);

  Lookup[REFERENCE_DENSITY][itype] = utils::numeric(FLERR, arg[ioffset + 1],false,lmp);
  Lookup[YOUNGS_MODULUS][itype] = utils::numeric(FLERR, arg[ioffset + 2],false,lmp);
  Lookup[POISSON_RATIO][itype] = utils::numeric(FLERR, arg[ioffset + 3],false,lmp);
  Lookup[VISCOSITY_Q1][itype] = utils::numeric(FLERR, arg[ioffset + 4],false,lmp);
  Lookup[VISCOSITY_Q2][itype] = utils::numeric(FLERR, arg[ioffset + 5],false,lmp);
  Lookup[HOURGLASS_CONTROL_AMPLITUDE][itype] = utils::numeric(FLERR, arg[ioffset + 6],false,lmp);
  Lookup[HEAT_CAPACITY][itype] = utils::numeric(FLERR, arg[ioffset + 7],false,lmp);

  Lookup[LAME_LAMBDA][itype] = Lookup[YOUNGS_MODULUS][itype] * Lookup[POISSON_RATIO][itype]
    / ((1.0 + Lookup[POISSON_RATIO][itype]) * (1.0 - 2.0 * Lookup[POISSON_RATIO][itype]));
  Lookup[SHEAR_MODULUS][itype] = Lookup[YOUNGS_MODULUS][itype] / (2.0 * (1.0 + Lookup[POISSON_RATIO][itype]));
  Lookup[M_MODULUS][itype] = Lookup[LAME_LAMBDA][itype] + 2.0 * Lookup[SHEAR_MODULUS][itype];
  Lookup[SIGNAL_VELOCITY][itype] = sqrt(
    (Lookup[LAME_LAMBDA][itype] + 2.0 * Lookup[SHEAR_MODULUS][itype]) / Lookup[REFERENCE_DENSITY][itype]);
  Lookup[BULK_MODULUS][itype] = Lookup[LAME_LAMBDA][itype] + 2.0 * Lookup[SHEAR_MODULUS][itype] / 3.0;

  if (comm->me == 0) {
    utils::logmesg(lmp, "\nmaterial unspecific properties for SMD/TLSPH definition of particle type {}:\n", itype);
    utils::logmesg(lmp, "{:60} : {}\n", "reference density", Lookup[REFERENCE_DENSITY][itype]);
    utils::logmesg(lmp, "{:60} : {}\n", "Young's modulus", Lookup[YOUNGS_MODULUS][itype]);
    utils::logmesg(lmp, "{:60} : {}\n", "Poisson ratio", Lookup[POISSON_RATIO][itype]);
    utils::logmesg(lmp, "{:60} : {}\n", "linear viscosity coefficient", Lookup[VISCOSITY_Q1][itype]);
    utils::logmesg(lmp, "{:60} : {}\n", "quadratic viscosity coefficient", Lookup[VISCOSITY_Q2][itype]);
    utils::logmesg(lmp, "{:60} : {}\n", "hourglass control coefficient", Lookup[HOURGLASS_CONTROL_AMPLITUDE][itype]);
    utils::logmesg(lmp, "{:60} : {}\n", "heat capacity [energy / (mass * temperature)]", Lookup[HEAT_CAPACITY][itype]);
    utils::logmesg(lmp, "{:60} : {}\n", "Lame constant lambda", Lookup[LAME_LAMBDA][itype]);
    utils::logmesg(lmp, "{:60} : {}\n", "shear modulus", Lookup[SHEAR_MODULUS][itype]);
    utils::logmesg(lmp, "{:60} : {}\n", "bulk modulus", Lookup[BULK_MODULUS][itype]);
    utils::logmesg(lmp, "{:60} : {}\n", "signal velocity", Lookup[SIGNAL_VELOCITY][itype]);
  }

  /*
   * read following material cards
   */

  eos[itype] = EOS_NONE;
  strengthModel[itype] = STRENGTH_NONE;

  while (true) {
    if (strcmp(arg[iNextKwd], "*END") == 0) {
      if (comm->me == 0)
        utils::logmesg(lmp,"found *END keyword"
                       "\n>>========>>========>>========>>========>>========>>========>>========>>========>>========>>========>>========>>========\n\n");
      break;
    }

    /*
     * Linear Elasticity model based on deformation gradient
     */
    ioffset = iNextKwd;
    if (strcmp(arg[ioffset], "*LINEAR_DEFGRAD") == 0) {
      strengthModel[itype] = LINEAR_DEFGRAD;

      if (comm->me == 0) utils::logmesg(lmp, "reading *LINEAR_DEFGRAD\n");

      t = std::string("*");
      iNextKwd = -1;
      for (iarg = ioffset + 1; iarg < narg; iarg++) {
        s = std::string(arg[iarg]);
        if (s.compare(0, t.length(), t) == 0) {
          iNextKwd = iarg;
          break;
        }
      }

      if (iNextKwd < 0)
        error->all(FLERR, "no *KEYWORD terminates *LINEAR_DEFGRAD");

      if (iNextKwd - ioffset != 1)
        error->all(FLERR, "expected 0 arguments following *LINEAR_DEFGRAD but got {}\n", iNextKwd - ioffset - 1);

      if (comm->me == 0) utils::logmesg(lmp, "\nLinear Elasticity model based on deformation gradient\n");

    } else if (strcmp(arg[ioffset], "*STRENGTH_LINEAR") == 0) {

      /*
       * Linear Elasticity strength only model based on strain rate
       */

      strengthModel[itype] = STRENGTH_LINEAR;
      if (comm->me == 0) utils::logmesg(lmp,"reading *STRENGTH_LINEAR\n");

      t = std::string("*");
      iNextKwd = -1;
      for (iarg = ioffset + 1; iarg < narg; iarg++) {
        s = std::string(arg[iarg]);
        if (s.compare(0, t.length(), t) == 0) {
          iNextKwd = iarg;
          break;
        }
      }

      if (iNextKwd < 0)
        error->all(FLERR, "no *KEYWORD terminates *STRENGTH_LINEAR");

      if (iNextKwd - ioffset != 1)
        error->all(FLERR, "expected 0 arguments following *STRENGTH_LINEAR but got {}\n", iNextKwd - ioffset - 1);

      if (comm->me == 0) utils::logmesg(lmp, "Linear Elasticity strength based on strain rate\n");

    } // end Linear Elasticity strength only model based on strain rate

    else if (strcmp(arg[ioffset], "*STRENGTH_LINEAR_PLASTIC") == 0) {

      /*
       * Linear Elastic / perfectly plastic strength only model based on strain rate
       */

      strengthModel[itype] = STRENGTH_LINEAR_PLASTIC;
      if (comm->me == 0) utils::logmesg(lmp,"reading *STRENGTH_LINEAR_PLASTIC\n");

      t = std::string("*");
      iNextKwd = -1;
      for (iarg = ioffset + 1; iarg < narg; iarg++) {
        s = std::string(arg[iarg]);
        if (s.compare(0, t.length(), t) == 0) {
          iNextKwd = iarg;
          break;
        }
      }

      if (iNextKwd < 0)
        error->all(FLERR, "no *KEYWORD terminates *STRENGTH_LINEAR_PLASTIC");

      if (iNextKwd - ioffset != 2 + 1)
        error->all(FLERR, "expected 2 arguments following *STRENGTH_LINEAR_PLASTIC but got {}\n", iNextKwd - ioffset - 1);

      Lookup[YIELD_STRESS][itype] = utils::numeric(FLERR, arg[ioffset + 1],false,lmp);
      Lookup[HARDENING_PARAMETER][itype] = utils::numeric(FLERR, arg[ioffset + 2],false,lmp);

      if (comm->me == 0) {
        utils::logmesg(lmp, "Linear elastic / perfectly plastic strength based on strain rate");
        utils::logmesg(lmp, "{:60} : {}\n", "Young's modulus", Lookup[YOUNGS_MODULUS][itype]);
        utils::logmesg(lmp, "{:60} : {}\n", "Poisson ratio", Lookup[POISSON_RATIO][itype]);
        utils::logmesg(lmp, "{:60} : {}\n", "shear modulus", Lookup[SHEAR_MODULUS][itype]);
        utils::logmesg(lmp, "{:60} : {}\n", "constant yield stress", Lookup[YIELD_STRESS][itype]);
        utils::logmesg(lmp, "{:60} : {}\n", "constant hardening parameter", Lookup[HARDENING_PARAMETER][itype]);
      }
    } // end Linear Elastic / perfectly plastic strength only model based on strain rate

    else if (strcmp(arg[ioffset], "*JOHNSON_COOK") == 0) {

      /*
       * JOHNSON - COOK
       */

      strengthModel[itype] = STRENGTH_JOHNSON_COOK;
      if (comm->me == 0) utils::logmesg(lmp, "reading *JOHNSON_COOK\n");

      t = std::string("*");
      iNextKwd = -1;
      for (iarg = ioffset + 1; iarg < narg; iarg++) {
        s = std::string(arg[iarg]);
        if (s.compare(0, t.length(), t) == 0) {
          iNextKwd = iarg;
          break;
        }
      }

      if (iNextKwd < 0)
        error->all(FLERR, "no *KEYWORD terminates *JOHNSON_COOK");

      if (iNextKwd - ioffset != 8 + 1)
        error->all(FLERR, "expected 8 arguments following *JOHNSON_COOK but got {}\n", iNextKwd - ioffset - 1);

      Lookup[JC_A][itype] = utils::numeric(FLERR, arg[ioffset + 1],false,lmp);
      Lookup[JC_B][itype] = utils::numeric(FLERR, arg[ioffset + 2],false,lmp);
      Lookup[JC_a][itype] = utils::numeric(FLERR, arg[ioffset + 3],false,lmp);
      Lookup[JC_C][itype] = utils::numeric(FLERR, arg[ioffset + 4],false,lmp);
      Lookup[JC_epdot0][itype] = utils::numeric(FLERR, arg[ioffset + 5],false,lmp);
      Lookup[JC_T0][itype] = utils::numeric(FLERR, arg[ioffset + 6],false,lmp);
      Lookup[JC_Tmelt][itype] = utils::numeric(FLERR, arg[ioffset + 7],false,lmp);
      Lookup[JC_M][itype] = utils::numeric(FLERR, arg[ioffset + 8],false,lmp);

      if (comm->me == 0) {
        utils::logmesg(lmp, "Johnson Cook material strength model\n");
        utils::logmesg(lmp, "{:60} : {}\n", "A: initial yield stress", Lookup[JC_A][itype]);
        utils::logmesg(lmp, "{:60} : {}\n", "B : proportionality factor for plastic strain dependency", Lookup[JC_B][itype]);
        utils::logmesg(lmp, "{:60} : {}\n", "a : exponent for plastic strain dependency", Lookup[JC_a][itype]);
        utils::logmesg(lmp, "{:60} : {}\n", "C : proportionality factor for logarithmic plastic strain rate dependency",Lookup[JC_C][itype]);
        utils::logmesg(lmp, "{:60} : {}\n", "epdot0 : dimensionality factor for plastic strain rate dependency", Lookup[JC_epdot0][itype]);
        utils::logmesg(lmp, "{:60} : {}\n", "T0 : reference (room) temperature", Lookup[JC_T0][itype]);
        utils::logmesg(lmp, "{:60} : {}\n", "Tmelt : melting temperature", Lookup[JC_Tmelt][itype]);
        utils::logmesg(lmp, "{:60} : {}\n", "M : exponent for temperature dependency", Lookup[JC_M][itype]);
      }

    } else if (strcmp(arg[ioffset], "*LUDWICK_HOLLOMON") == 0) {

      /*
       * LUDWICK - HOLLOMON
       */

      strengthModel[itype] = STRENGTH_LUDWICK_HOLLOMON;
      if (comm->me == 0) utils::logmesg(lmp, "reading *LUDWICK_HOLLOMON\n");

      t = std::string("*");
      iNextKwd = -1;
      for (iarg = ioffset + 1; iarg < narg; iarg++) {
        s = std::string(arg[iarg]);
        if (s.compare(0, t.length(), t) == 0) {
          iNextKwd = iarg;
          break;
        }
      }

      if (iNextKwd < 0) 
        error->all(FLERR, "no *KEYWORD terminates *LUDWICK_HOLLOMON");

      if (iNextKwd - ioffset != 3 + 1) 
        error->all(FLERR, "expected 3 arguments following *LUDWICK_HOLLOMON but got {}\n", iNextKwd - ioffset - 1);

      Lookup[LH_A][itype] = utils::numeric(FLERR, arg[ioffset + 1],false,lmp);
      Lookup[LH_B][itype] = utils::numeric(FLERR, arg[ioffset + 2],false,lmp);
      Lookup[LH_n][itype] = utils::numeric(FLERR, arg[ioffset + 3],false,lmp);

      if (comm->me == 0) {
        utils::logmesg(lmp, "Ludwick-Hollomon material strength model\n");
        utils::logmesg(lmp, "{:60} : {}\n", "A: initial yield stress", Lookup[LH_A][itype]);
        utils::logmesg(lmp, "{:60} : {}\n", "B : proportionality factor for plastic strain dependency", Lookup[LH_B][itype]);
        utils::logmesg(lmp, "{:60} : {}\n", "n : exponent for plastic strain dependency", Lookup[LH_n][itype]);
      }

    } else if (strcmp(arg[ioffset], "*SWIFT") == 0) {

      /*
       * SWIFT
       */

      strengthModel[itype] = STRENGTH_SWIFT;
      if (comm->me == 0) utils::logmesg(lmp, "reading *SWIFT\n");

      t = std::string("*");
      iNextKwd = -1;
      for (iarg = ioffset + 1; iarg < narg; iarg++) {
        s = std::string(arg[iarg]);
        if (s.compare(0, t.length(), t) == 0) {
          iNextKwd = iarg;
          break;
        }
      }

      if (iNextKwd < 0) 
        error->all(FLERR, "no *KEYWORD terminates *SWIFT");

      if (iNextKwd - ioffset != 4 + 1) 
        error->all(FLERR, "expected 4 arguments following *SWIFT but got {}\n", iNextKwd - ioffset - 1);

      Lookup[SWIFT_A][itype] = utils::numeric(FLERR, arg[ioffset + 1],false,lmp);
      Lookup[SWIFT_B][itype] = utils::numeric(FLERR, arg[ioffset + 2],false,lmp);
      Lookup[SWIFT_n][itype] = utils::numeric(FLERR, arg[ioffset + 3],false,lmp);
      Lookup[SWIFT_eps0][itype] = utils::numeric(FLERR, arg[ioffset + 4],false,lmp);

      if (Lookup[SWIFT_eps0][itype] < 0.0)
        error->all(FLERR, "the 4th argument following *SWIFT should be positive");

      if (comm->me == 0) {
        utils::logmesg(lmp, "Swift material strength model\n");
        utils::logmesg(lmp, "{:60} : {}\n", "A: initial yield stress", Lookup[SWIFT_A][itype]);
        utils::logmesg(lmp, "{:60} : {}\n", "B : proportionality factor for plastic strain dependency", Lookup[SWIFT_B][itype]);
        utils::logmesg(lmp, "{:60} : {}\n", "n : exponent for plastic strain dependency", Lookup[SWIFT_n][itype]);
        utils::logmesg(lmp, "{:60} : {}\n", "eps0 : initial plastic strain", Lookup[SWIFT_eps0][itype]);
      }

    } else if (strcmp(arg[ioffset], "*EOS_NONE") == 0) {

      /*
       * no eos
       */

      eos[itype] = EOS_NONE;
      if (comm->me == 0) utils::logmesg(lmp, "reading *EOS_NONE\n");

      t = std::string("*");
      iNextKwd = -1;
      for (iarg = ioffset + 1; iarg < narg; iarg++) {
        s = std::string(arg[iarg]);
        if (s.compare(0, t.length(), t) == 0) {
          iNextKwd = iarg;
          break;
        }
      }

      if (iNextKwd < 0)
        error->all(FLERR, "no *KEYWORD terminates *EOS_NONE");

      if (iNextKwd - ioffset != 1)
        error->all(FLERR, "expected 0 arguments following *EOS_NONE but got {}\n", iNextKwd - ioffset - 1);

      if (comm->me == 0) utils::logmesg(lmp, "\nno EOS selected\n");

    } else if (strcmp(arg[ioffset], "*EOS_LINEAR") == 0) {

      /*
       * linear eos
       */

      eos[itype] = EOS_LINEAR;
      if (comm->me == 0) utils::logmesg(lmp, "reading *EOS_LINEAR\n");

      t = std::string("*");
      iNextKwd = -1;
      for (iarg = ioffset + 1; iarg < narg; iarg++) {
        s = std::string(arg[iarg]);
        if (s.compare(0, t.length(), t) == 0) {
          iNextKwd = iarg;
          break;
        }
      }

      if (iNextKwd < 0)
        error->all(FLERR, "no *KEYWORD terminates *EOS_LINEAR");

      if (iNextKwd - ioffset != 1)
        error->all(FLERR, "expected 0 arguments following *EOS_LINEAR but got {}\n", iNextKwd - ioffset - 1);

      if (comm->me == 0) {
        utils::logmesg(lmp, "\nlinear EOS based on strain rate\n");
        utils::logmesg(lmp, "{:60} : {}\n", "bulk modulus", Lookup[BULK_MODULUS][itype]);
      }
    } // end linear eos
    else if (strcmp(arg[ioffset], "*EOS_SHOCK") == 0) {

      /*
       * shock eos
       */

      eos[itype] = EOS_SHOCK;
      if (comm->me == 0) utils::logmesg(lmp, "reading *EOS_SHOCK\n");

      t = std::string("*");
      iNextKwd = -1;
      for (iarg = ioffset + 1; iarg < narg; iarg++) {
        s = std::string(arg[iarg]);
        if (s.compare(0, t.length(), t) == 0) {
          iNextKwd = iarg;
          break;
        }
      }

      if (iNextKwd < 0)
        error->all(FLERR, "no *KEYWORD terminates *EOS_SHOCK");

      if (iNextKwd - ioffset != 3 + 1)
        error->all(FLERR, "expected 3 arguments (c0, S, Gamma) following *EOS_SHOCK but got {}\n", iNextKwd - ioffset - 1);

      Lookup[EOS_SHOCK_C0][itype] = utils::numeric(FLERR, arg[ioffset + 1],false,lmp);
      Lookup[EOS_SHOCK_S][itype] = utils::numeric(FLERR, arg[ioffset + 2],false,lmp);
      Lookup[EOS_SHOCK_GAMMA][itype] = utils::numeric(FLERR, arg[ioffset + 3],false,lmp);
      if (comm->me == 0) {
        utils::logmesg(lmp, "\nshock EOS based on strain rate\n");
        utils::logmesg(lmp, "{:60} : {}\n", "reference speed of sound", Lookup[EOS_SHOCK_C0][itype]);
        utils::logmesg(lmp, "{:60} : {}\n", "Hugoniot parameter S", Lookup[EOS_SHOCK_S][itype]);
        utils::logmesg(lmp, "{:60} : {}\n", "Grueneisen Gamma", Lookup[EOS_SHOCK_GAMMA][itype]);
      }
    } // end shock eos

    else if (strcmp(arg[ioffset], "*EOS_POLYNOMIAL") == 0) {
      /*
       * polynomial eos
       */

      eos[itype] = EOS_POLYNOMIAL;
      if (comm->me == 0) utils::logmesg(lmp, "reading *EOS_POLYNOMIAL\n");

      t = std::string("*");
      iNextKwd = -1;
      for (iarg = ioffset + 1; iarg < narg; iarg++) {
        s = std::string(arg[iarg]);
        if (s.compare(0, t.length(), t) == 0) {
          iNextKwd = iarg;
          break;
        }
      }

      if (iNextKwd < 0)
        error->all(FLERR, "no *KEYWORD terminates *EOS_POLYNOMIAL");

      if (iNextKwd - ioffset != 7 + 1)
        error->all(FLERR, "expected 7 arguments following *EOS_POLYNOMIAL but got {}\n", iNextKwd - ioffset - 1);

      Lookup[EOS_POLYNOMIAL_C0][itype] = utils::numeric(FLERR, arg[ioffset + 1],false,lmp);
      Lookup[EOS_POLYNOMIAL_C1][itype] = utils::numeric(FLERR, arg[ioffset + 2],false,lmp);
      Lookup[EOS_POLYNOMIAL_C2][itype] = utils::numeric(FLERR, arg[ioffset + 3],false,lmp);
      Lookup[EOS_POLYNOMIAL_C3][itype] = utils::numeric(FLERR, arg[ioffset + 4],false,lmp);
      Lookup[EOS_POLYNOMIAL_C4][itype] = utils::numeric(FLERR, arg[ioffset + 5],false,lmp);
      Lookup[EOS_POLYNOMIAL_C5][itype] = utils::numeric(FLERR, arg[ioffset + 6],false,lmp);
      Lookup[EOS_POLYNOMIAL_C6][itype] = utils::numeric(FLERR, arg[ioffset + 7],false,lmp);
      if (comm->me == 0) {
        utils::logmesg(lmp, "\npolynomial EOS based on strain rate\n");
        utils::logmesg(lmp, "{:60} : {}\n", "parameter c0", Lookup[EOS_POLYNOMIAL_C0][itype]);
        utils::logmesg(lmp, "{:60} : {}\n", "parameter c1", Lookup[EOS_POLYNOMIAL_C1][itype]);
        utils::logmesg(lmp, "{:60} : {}\n", "parameter c2", Lookup[EOS_POLYNOMIAL_C2][itype]);
        utils::logmesg(lmp, "{:60} : {}\n", "parameter c3", Lookup[EOS_POLYNOMIAL_C3][itype]);
        utils::logmesg(lmp, "{:60} : {}\n", "parameter c4", Lookup[EOS_POLYNOMIAL_C4][itype]);
        utils::logmesg(lmp, "{:60} : {}\n", "parameter c5", Lookup[EOS_POLYNOMIAL_C5][itype]);
        utils::logmesg(lmp, "{:60} : {}\n", "parameter c6", Lookup[EOS_POLYNOMIAL_C6][itype]);
      }
    } // end polynomial eos

    else if (strcmp(arg[ioffset], "*FAILURE_MAX_PLASTIC_STRAIN") == 0) {

      /*
       * maximum plastic strain failure criterion
       */

      if (comm->me == 0) utils::logmesg(lmp, "reading *FAILURE_MAX_PLASTIC_SRTRAIN\n");

      t = std::string("*");
      iNextKwd = -1;
      for (iarg = ioffset + 1; iarg < narg; iarg++) {
        s = std::string(arg[iarg]);
        if (s.compare(0, t.length(), t) == 0) {
          iNextKwd = iarg;
          break;
        }
      }

      if (iNextKwd < 0) error->all(FLERR, "no *KEYWORD terminates *FAILURE_MAX_PLASTIC_STRAIN");
      if (iNextKwd - ioffset != 1 + 1)
        error->all(FLERR, "expected 1 arguments following *FAILURE_MAX_PLASTIC_STRAIN but got {}\n", iNextKwd - ioffset - 1);

      failureModel[itype].failure_max_plastic_strain = true;
      failureModel[itype].integration_point_wise = true;
      failureModel[itype].failure_none = false;
      Lookup[FAILURE_MAX_PLASTIC_STRAIN_THRESHOLD][itype] = utils::numeric(FLERR, arg[ioffset + 1],false,lmp);

      if (comm->me == 0) {
        utils::logmesg(lmp, "\nmaximum plastic strain failure criterion\n");
        utils::logmesg(lmp, "{:60} : {}\n", "failure occurs when plastic strain reaches limit",
                       Lookup[FAILURE_MAX_PLASTIC_STRAIN_THRESHOLD][itype]);
      }
    } // end maximum plastic strain failure criterion
    else if (strcmp(arg[ioffset], "*FAILURE_MAX_PAIRWISE_STRAIN") == 0) {

      /*
       * failure criterion based on maximum strain between a pair of TLSPH particles.
       */

      if (comm->me == 0) utils::logmesg(lmp, "reading *FAILURE_MAX_PAIRWISE_STRAIN\n");

      if (update_method != UPDATE_NONE) {
        error->all(FLERR, "cannot use *FAILURE_MAX_PAIRWISE_STRAIN with updated Total-Lagrangian formalism");
      }

      t = std::string("*");
      iNextKwd = -1;
      for (iarg = ioffset + 1; iarg < narg; iarg++) {
        s = std::string(arg[iarg]);
        if (s.compare(0, t.length(), t) == 0) {
          iNextKwd = iarg;
          break;
        }
      }

      if (iNextKwd < 0)
        error->all(FLERR, "no *KEYWORD terminates *FAILURE_MAX_PAIRWISE_STRAIN");

      if (iNextKwd - ioffset != 1 + 1)
        error->all(FLERR, "expected 1 arguments following *FAILURE_MAX_PAIRWISE_STRAIN but got {}\n", iNextKwd - ioffset - 1);

      failureModel[itype].failure_max_pairwise_strain = true;
      failureModel[itype].integration_point_wise = false;
      failureModel[itype].failure_none = false;
      Lookup[FAILURE_MAX_PAIRWISE_STRAIN_THRESHOLD][itype] = utils::numeric(FLERR, arg[ioffset + 1],false,lmp);

      if (comm->me == 0) {
        utils::logmesg(lmp, "\nmaximum pairwise strain failure criterion\n");
        utils::logmesg(lmp, "{:60} : {}\n", "failure occurs when pairwise strain reaches limit",
               Lookup[FAILURE_MAX_PAIRWISE_STRAIN_THRESHOLD][itype]);
      }
    } // end pair based maximum strain failure criterion
    else if (strcmp(arg[ioffset], "*FAILURE_MAX_PRINCIPAL_STRAIN") == 0) {
      error->all(FLERR, "this failure model is currently unsupported");

      /*
       * maximum principal strain failure criterion
       */
      if (comm->me == 0) utils::logmesg(lmp, "reading *FAILURE_MAX_PRINCIPAL_STRAIN\n");

      t = std::string("*");
      iNextKwd = -1;
      for (iarg = ioffset + 1; iarg < narg; iarg++) {
        s = std::string(arg[iarg]);
        if (s.compare(0, t.length(), t) == 0) {
          iNextKwd = iarg;
          break;
        }
      }

      if (iNextKwd < 0)
        error->all(FLERR, "no *KEYWORD terminates *FAILURE_MAX_PRINCIPAL_STRAIN");

      if (iNextKwd - ioffset != 1 + 1)
        error->all(FLERR, "expected 1 arguments following *FAILURE_MAX_PRINCIPAL_STRAIN but got {}\n", iNextKwd - ioffset - 1);

      failureModel[itype].failure_max_principal_strain = true;
      failureModel[itype].integration_point_wise = true;
      failureModel[itype].failure_none = false;
      Lookup[FAILURE_MAX_PRINCIPAL_STRAIN_THRESHOLD][itype] = utils::numeric(FLERR, arg[ioffset + 1],false,lmp);

      if (comm->me == 0) {
        utils::logmesg(lmp, "\nmaximum principal strain failure criterion\n");
        utils::logmesg(lmp, "{:60} : {}\n", "failure occurs when principal strain reaches limit",
               Lookup[FAILURE_MAX_PRINCIPAL_STRAIN_THRESHOLD][itype]);
      }
    } // end maximum principal strain failure criterion
    else if (strcmp(arg[ioffset], "*FAILURE_JOHNSON_COOK") == 0) {
      //error->all(FLERR, "this failure model is currently unsupported");
      if (comm->me == 0) utils::logmesg(lmp, "reading *FAILURE_JOHNSON_COOK\n");

      t = std::string("*");
      iNextKwd = -1;
      for (iarg = ioffset + 1; iarg < narg; iarg++) {
        s = std::string(arg[iarg]);
        if (s.compare(0, t.length(), t) == 0) {
          iNextKwd = iarg;
          break;
        }
      }

      if (iNextKwd < 0)
        error->all(FLERR, "no *KEYWORD terminates *FAILURE_JOHNSON_COOK");

      if (iNextKwd - ioffset != 5 + 1)
        error->all(FLERR, "expected 5 arguments following *FAILURE_JOHNSON_COOK but got {}\n", iNextKwd - ioffset - 1);

      failureModel[itype].failure_johnson_cook = true;
      failureModel[itype].integration_point_wise = true;
      failureModel[itype].failure_none = false;

      Lookup[FAILURE_JC_D1][itype] = utils::numeric(FLERR, arg[ioffset + 1],false,lmp);
      Lookup[FAILURE_JC_D2][itype] = utils::numeric(FLERR, arg[ioffset + 2],false,lmp);
      Lookup[FAILURE_JC_D3][itype] = utils::numeric(FLERR, arg[ioffset + 3],false,lmp);
      Lookup[FAILURE_JC_D4][itype] = utils::numeric(FLERR, arg[ioffset + 4],false,lmp);
      Lookup[FAILURE_JC_EPDOT0][itype] = utils::numeric(FLERR, arg[ioffset + 5],false,lmp);

      if (comm->me == 0) {
        utils::logmesg(lmp, "\nJohnson-Cook failure criterion\n");
        utils::logmesg(lmp, "{:60} : {}\n", "parameter d1", Lookup[FAILURE_JC_D1][itype]);
        utils::logmesg(lmp, "{:60} : {}\n", "parameter d2", Lookup[FAILURE_JC_D2][itype]);
        utils::logmesg(lmp, "{:60} : {}\n", "parameter d3", Lookup[FAILURE_JC_D3][itype]);
        utils::logmesg(lmp, "{:60} : {}\n", "parameter d4", Lookup[FAILURE_JC_D4][itype]);
        utils::logmesg(lmp, "{:60} : {}\n", "reference plastic strain rate", Lookup[FAILURE_JC_EPDOT0][itype]);
      }

    } else if (strcmp(arg[ioffset], "*FAILURE_MAX_PRINCIPAL_STRESS") == 0) {
      error->all(FLERR, "this failure model is currently unsupported");

      /*
       * maximum principal stress failure criterion
       */

      if (comm->me == 0) utils::logmesg(lmp, "reading *FAILURE_MAX_PRINCIPAL_STRESS\n");

      t = std::string("*");
      iNextKwd = -1;
      for (iarg = ioffset + 1; iarg < narg; iarg++) {
        s = std::string(arg[iarg]);
        if (s.compare(0, t.length(), t) == 0) {
          iNextKwd = iarg;
          break;
        }
      }

      if (iNextKwd < 0)
        error->all(FLERR, "no *KEYWORD terminates *FAILURE_MAX_PRINCIPAL_STRESS");

      if (iNextKwd - ioffset != 1 + 1)
        error->all(FLERR, "expected 1 arguments following *FAILURE_MAX_PRINCIPAL_STRESS but got {}\n", iNextKwd - ioffset - 1);

      failureModel[itype].failure_max_principal_stress = true;
      failureModel[itype].integration_point_wise = true;
      failureModel[itype].failure_none = false;
      Lookup[FAILURE_MAX_PRINCIPAL_STRESS_THRESHOLD][itype] = utils::numeric(FLERR, arg[ioffset + 1],false,lmp);

      if (comm->me == 0) {
        utils::logmesg(lmp, "\nmaximum principal stress failure criterion\n");
        utils::logmesg(lmp, "{:60} : {}\n", "failure occurs when principal stress reaches limit",
               Lookup[FAILURE_MAX_PRINCIPAL_STRESS_THRESHOLD][itype]);
      }
    } // end maximum principal stress failure criterion

    else if (strcmp(arg[ioffset], "*FAILURE_ENERGY_RELEASE_RATE") == 0) {
      if (comm->me == 0) utils::logmesg(lmp, "reading *FAILURE_ENERGY_RELEASE_RATE\n");

      t = std::string("*");
      iNextKwd = -1;
      for (iarg = ioffset + 1; iarg < narg; iarg++) {
        s = std::string(arg[iarg]);
        if (s.compare(0, t.length(), t) == 0) {
          iNextKwd = iarg;
          break;
        }
      }

      if (iNextKwd < 0)
        error->all(FLERR, "no *KEYWORD terminates *FAILURE_ENERGY_RELEASE_RATE");

      if (iNextKwd - ioffset != 1 + 1)
        error->all(FLERR, "expected 1 arguments following *FAILURE_ENERGY_RELEASE_RATE but got {}\n", iNextKwd - ioffset - 1);

      failureModel[itype].failure_energy_release_rate = true;
      failureModel[itype].failure_none = false;
      Lookup[CRITICAL_ENERGY_RELEASE_RATE][itype] = utils::numeric(FLERR, arg[ioffset + 1],false,lmp);

      if (comm->me == 0) {
        utils::logmesg(lmp,"\ncritical energy release rate failure criterion\n");
        utils::logmesg(lmp, "{:60} : {}\n", "failure occurs when energy release rate reaches limit",
               Lookup[CRITICAL_ENERGY_RELEASE_RATE][itype]);
      }
    } // end energy release rate failure criterion

    else if (strcmp(arg[ioffset], "*GURSON_TVERGAARD_NEEDLEMAN") == 0) {

      /*
       * GURSON - TVERGAARD - NEEDLEMAN Model
       */
      
      if (comm->me == 0) utils::logmesg(lmp, "reading *GURSON_TVERGAARD_NEEDLEMAN\n");

      t = std::string("*");
      iNextKwd = -1;
      for (iarg = ioffset + 1; iarg < narg; iarg++) {
        s = std::string(arg[iarg]);
        if (s.compare(0, t.length(), t) == 0) {
          iNextKwd = iarg;
          break;
        }
      }

      if (iNextKwd < 0)
        error->all(FLERR, "no *KEYWORD terminates *GURSON_TVERGAARD_NEEDLEMAN");

      if (iNextKwd - ioffset != 3 + 1)
        error->all(FLERR, "expected 3 arguments following *GURSON_TVERGAARD_NEEDLEMAN but got {}\n", iNextKwd - ioffset - 1);

      failureModel[itype].failure_gtn = true;
      failureModel[itype].integration_point_wise = true;
      failureModel[itype].failure_none = false;
      
      Lookup[GTN_Q1][itype] = utils::numeric(FLERR, arg[ioffset + 1],false,lmp);
      Lookup[GTN_Q2][itype] = utils::numeric(FLERR, arg[ioffset + 2],false,lmp);
      Lookup[GTN_AN][itype] = utils::numeric(FLERR, arg[ioffset + 3],false,lmp);

      if (comm->me == 0) {
        utils::logmesg(lmp, "\nGurson-Tvergaard-Needleman failure criterion\n");
        utils::logmesg(lmp, "{:60} : {}\n", "Q1: ", Lookup[GTN_Q1][itype]);
        utils::logmesg(lmp, "{:60} : {}\n", "Q2: ", Lookup[GTN_Q2][itype]);
        utils::logmesg(lmp, "{:60} : {}\n", "An: ", Lookup[GTN_AN][itype]);
      }

    } 

    else error->all(FLERR, "unknown *KEYWORD: {}", arg[ioffset]);
  }
  setflag[itype][itype] = 1;
}

/* ----------------------------------------------------------------------
 init for one type pair i,j and corresponding j,i
 ------------------------------------------------------------------------- */

double PairTlsph::init_one(int i, int j) {

  if (!allocated)
    allocate();

  if (setflag[i][j] == 0)
    error->all(FLERR, "All pair coeffs are not set");

  if (force->newton == 1)
    error->all(FLERR, "Pair style tlsph requires newton off");

// cutoff = sum of max I,J radii for
// dynamic/dynamic & dynamic/frozen interactions, but not frozen/frozen

  double cutoff = maxrad_dynamic[i] + maxrad_dynamic[j];
  cutoff = MAX(cutoff, maxrad_frozen[i] + maxrad_dynamic[j]);
  cutoff = MAX(cutoff, maxrad_dynamic[i] + maxrad_frozen[j]);
  return cutoff;
}

/* ----------------------------------------------------------------------
 init specific to this pair style
 ------------------------------------------------------------------------- */

void PairTlsph::init_style() {
  int i;

  if (force->newton_pair == 1) {
    error->all(FLERR, "Pair style tlsph requires newton pair off");
  }

// request a granular neighbor list
  neighbor->add_request(this, NeighConst::REQ_SIZE);

// set maxrad_dynamic and maxrad_frozen for each type
// include future Fix pour particles as dynamic

  for (i = 1; i <= atom->ntypes; i++)
    onerad_dynamic[i] = onerad_frozen[i] = 0.0;

  double *radius = atom->radius;
  int *type = atom->type;
  int nlocal = atom->nlocal;

  for (i = 0; i < nlocal; i++)
    onerad_dynamic[type[i]] = MAX(onerad_dynamic[type[i]], radius[i]);

  MPI_Allreduce(&onerad_dynamic[1], &maxrad_dynamic[1], atom->ntypes, MPI_DOUBLE, MPI_MAX, world);
  MPI_Allreduce(&onerad_frozen[1], &maxrad_frozen[1], atom->ntypes, MPI_DOUBLE, MPI_MAX, world);

// if first init, create Fix needed for storing reference configuration neighbors

  int igroup = group->find("tlsph");
  if (igroup == -1)
    error->all(FLERR, "Pair style tlsph requires its particles to be part of a group named tlsph. This group does not exist.");

  if (fix_tlsph_reference_configuration == nullptr) {
    auto fixarg = new char*[3];
    fixarg[0] = (char *) "SMD_TLSPH_NEIGHBORS";
    fixarg[1] = (char *) "tlsph";
    fixarg[2] = (char *) "SMD_TLSPH_NEIGHBORS";
    modify->add_fix(3, fixarg);
    delete[] fixarg;
    fix_tlsph_reference_configuration = dynamic_cast<FixSMD_TLSPH_ReferenceConfiguration *>(modify->fix[modify->nfix - 1]);
    fix_tlsph_reference_configuration->pair = this;
  }

// find associated SMD_TLSPH_NEIGHBORS fix that must exist
// could have changed locations in fix list since created

  ifix_tlsph = -1;
  for (int i = 0; i < modify->nfix; i++)
    if (strcmp(modify->fix[i]->style, "SMD_TLSPH_NEIGHBORS") == 0)
      ifix_tlsph = i;
  if (ifix_tlsph == -1)
    error->all(FLERR, "Fix SMD_TLSPH_NEIGHBORS does not exist");

}

/* ----------------------------------------------------------------------
 neighbor callback to inform pair style of neighbor list to use
 optional granular history list
 ------------------------------------------------------------------------- */

void PairTlsph::init_list(int id, class NeighList *ptr) {
  if (id == 0) list = ptr;
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based arrays
 ------------------------------------------------------------------------- */

double PairTlsph::memory_usage() {
  return 118.0 * nmax * sizeof(double);
}

/* ----------------------------------------------------------------------
 extract method to provide access to this class' data structures
 ------------------------------------------------------------------------- */

void *PairTlsph::extract(const char *str, int &/*i*/) {
  if (strcmp(str, "smd/tlsph/Fincr_ptr") == 0) {
    return (void *) Fincr;
  } else if (strcmp(str, "smd/tlsph/detF_ptr") == 0) {
    return (void *) detF;
  } else if (strcmp(str, "smd/tlsph/PK1_ptr") == 0) {
    return (void *) PK1;
  } else if (strcmp(str, "smd/tlsph/Kundeg_ptr") == 0) {
    return (void *) Kundeg;
  } else if (strcmp(str, "smd/tlsph/smoothVel_ptr") == 0) {
    return (void *) smoothVelDifference;
  } else if (strcmp(str, "smd/tlsph/surfaceNormal_ptr") == 0) {
    return (void *) surfaceNormal;
  } else if (strcmp(str, "smd/tlsph/numNeighsRefConfig_ptr") == 0) {
    return (void *) numNeighsRefConfig;
  } else if (strcmp(str, "smd/tlsph/stressTensor_ptr") == 0) {
    return (void *) CauchyStress;
  } else if (strcmp(str, "smd/tlsph/updateFlag_ptr") == 0) {
    return (void *) &updateFlag;
  } else if (strcmp(str, "smd/tlsph/strain_rate_ptr") == 0) {
    return (void *) D;
  } else if (strcmp(str, "smd/tlsph/hMin_ptr") == 0) {
    return (void *) &hMin;
  } else if (strcmp(str, "smd/tlsph/dtCFL_ptr") == 0) {
    return (void *) &dtCFL;
  } else if (strcmp(str, "smd/tlsph/dtRelative_ptr") == 0) {
    return (void *) &dtRelative;
  } else if (strcmp(str, "smd/tlsph/hourglass_error_ptr") == 0) {
    return (void *) hourglass_error;
  } else if (strcmp(str, "smd/tlsph/particle_dt_ptr") == 0) {
    return (void *) particle_dt;
  } else if (strcmp(str, "smd/tlsph/rotation_ptr") == 0) {
    return (void *) R;
  }

  return nullptr;
}

/* ---------------------------------------------------------------------- */

int PairTlsph::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int * /*pbc*/) {
  int i, j, m;
  tagint *mol = atom->molecule;
  double *damage = atom->damage;
  double *eff_plastic_strain = atom->eff_plastic_strain;
  double *eff_plastic_strain_rate = atom->eff_plastic_strain_rate;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = PK1[j](0, 0); // PK1 is not symmetric
    buf[m++] = PK1[j](0, 1);
    buf[m++] = PK1[j](0, 2);
    buf[m++] = PK1[j](1, 0);
    buf[m++] = PK1[j](1, 1);
    buf[m++] = PK1[j](1, 2);
    buf[m++] = PK1[j](2, 0);
    buf[m++] = PK1[j](2, 1);
    buf[m++] = PK1[j](2, 2); // 9

    buf[m++] = Fincr[j](0, 0); // Fincr is not symmetric
    buf[m++] = Fincr[j](0, 1);
    buf[m++] = Fincr[j](0, 2);
    buf[m++] = Fincr[j](1, 0);
    buf[m++] = Fincr[j](1, 1);
    buf[m++] = Fincr[j](1, 2);
    buf[m++] = Fincr[j](2, 0);
    buf[m++] = Fincr[j](2, 1);
    buf[m++] = Fincr[j](2, 2); // 9 + 9 = 18

    buf[m++] = mol[j]; //19
    buf[m++] = damage[j]; //20
    buf[m++] = eff_plastic_strain[j]; //21
    buf[m++] = eff_plastic_strain_rate[j]; //22

  }
  return m;
}

/* ---------------------------------------------------------------------- */

void PairTlsph::unpack_forward_comm(int n, int first, double *buf) {
  int i, m, last;
  tagint *mol = atom->molecule;
  double *damage = atom->damage;
  double *eff_plastic_strain = atom->eff_plastic_strain;
  double *eff_plastic_strain_rate = atom->eff_plastic_strain_rate;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {

    PK1[i](0, 0) = buf[m++]; // PK1 is not symmetric
    PK1[i](0, 1) = buf[m++];
    PK1[i](0, 2) = buf[m++];
    PK1[i](1, 0) = buf[m++];
    PK1[i](1, 1) = buf[m++];
    PK1[i](1, 2) = buf[m++];
    PK1[i](2, 0) = buf[m++];
    PK1[i](2, 1) = buf[m++];
    PK1[i](2, 2) = buf[m++];

    Fincr[i](0, 0) = buf[m++];
    Fincr[i](0, 1) = buf[m++];
    Fincr[i](0, 2) = buf[m++];
    Fincr[i](1, 0) = buf[m++];
    Fincr[i](1, 1) = buf[m++];
    Fincr[i](1, 2) = buf[m++];
    Fincr[i](2, 0) = buf[m++];
    Fincr[i](2, 1) = buf[m++];
    Fincr[i](2, 2) = buf[m++];

    mol[i] = static_cast<int>(buf[m++]);
    damage[i] = buf[m++];
    eff_plastic_strain[i] = buf[m++]; //22
    eff_plastic_strain_rate[i] = buf[m++]; //23
  }
}

/* ----------------------------------------------------------------------
 compute effective P-wave speed
 determined by longitudinal modulus
 ------------------------------------------------------------------------- */

void PairTlsph::effective_longitudinal_modulus(const int itype, const double dt, const double d_iso, const double p_rate,
                                               const Matrix3d& d_dev, const Matrix3d& sigma_dev_rate, const double /*damage*/, double &K_eff, double &mu_eff, double &M_eff) {
  double M0; // initial longitudinal modulus
  double shear_rate_sq;

  M0 = Lookup[M_MODULUS][itype];

  if (dt * d_iso > 1.0e-6) {
    K_eff = p_rate / d_iso;
    if (K_eff < 0.0) { // it is possible for K_eff to become negative due to strain softening
      K_eff = Lookup[BULK_MODULUS][itype];
    }
  } else {
    K_eff = Lookup[BULK_MODULUS][itype];
  }

  //if (K_eff < Lookup[BULK_MODULUS][itype]) printf("K_eff = %f\n", K_eff);

  if (domain->dimension == 3) {
    // Calculate 2 mu by looking at ratio shear stress / shear strain. Use numerical softening to avoid divide-by-zero.
    mu_eff = 0.5 * (sigma_dev_rate(0, 1) + sigma_dev_rate(0, 2) + sigma_dev_rate(1, 2) ) / (d_dev(0, 1) + d_dev(0, 2) + d_dev(1, 2) + 1.0e-16);
    // mu_eff = 0.5
    //    * (sigma_dev_rate(0, 1) / (d_dev(0, 1) + 1.0e-16) + sigma_dev_rate(0, 2) / (d_dev(0, 2) + 1.0e-16)
    //        + sigma_dev_rate(1, 2) / (d_dev(1, 2) + 1.0e-16)); //This gives a mu_eff up to three times higher than what it should be.
    //double mut = 0.5*max(max((sigma_dev_rate(0, 1) / (d_dev(0, 1) + 1.0e-16)), sigma_dev_rate(0, 2) / (d_dev(0, 2) + 1.0e-16)), sigma_dev_rate(1, 2) / (d_dev(1, 2) + 1.0e-16));
    //if (mu_eff > 1.1*mut) {
    //  printf("mu_eff = %f, mut = %f\n", mu_eff, mut);
    //  printf("sigma_dev_rate(0, 1) / d_dev(0, 1) = %f\n", (sigma_dev_rate(0, 1) / (d_dev(0, 1) + 1.0e-16)));
    //  printf("sigma_dev_rate(0, 2) / d_dev(0, 2) = %f\n", (sigma_dev_rate(0, 2) / (d_dev(0, 2) + 1.0e-16)));
    //  printf("sigma_dev_rate(1, 2) / d_dev(1, 2) = %f\n", (sigma_dev_rate(1, 2) / (d_dev(1, 2) + 1.0e-16)));
    //}
    // Calculate magnitude of deviatoric strain rate. This is used for deciding if shear modulus should be computed from current rate or be taken as the initial value.
    shear_rate_sq = d_dev(0, 1) * d_dev(0, 1) + d_dev(0, 2) * d_dev(0, 2) + d_dev(1, 2) * d_dev(1, 2);
  } else {
    mu_eff = 0.5 * (sigma_dev_rate(0, 1) / (d_dev(0, 1) + 1.0e-16));
    shear_rate_sq = d_dev(0, 1) * d_dev(0, 1);
  }

  if (dt * dt * shear_rate_sq < 1.0e-8) {
    mu_eff = Lookup[SHEAR_MODULUS][itype];
  }

  if (mu_eff < Lookup[SHEAR_MODULUS][itype]) { // it is possible for mu_eff to become negative due to strain softening
    mu_eff = Lookup[SHEAR_MODULUS][itype];
  }

  if (mu_eff < 0.0) {
    error->one(FLERR, "mu_eff = {}, tau={}, gamma={}", mu_eff, sigma_dev_rate(0, 1), d_dev(0, 1));

  }

  M_eff = (K_eff + 4.0 * mu_eff / 3.0); // effective dilational modulus, see Pronto 2d eqn 3.4.8

  if (M_eff < M0) { // do not allow effective dilatational modulus to decrease beyond its initial value
    M_eff = M0;
  }
}

/* ----------------------------------------------------------------------
 compute pressure. Called from AssembleStress().
 ------------------------------------------------------------------------- */
void PairTlsph::ComputePressure(const int i, const double rho, const double mass_specific_energy, const double vol_specific_energy,
                                const double pInitial, const double d_iso, double &pFinal, double &p_rate) {
  int *type = atom->type;
  double dt = update->dt;
  double *damage = atom->damage;
  int itype;

  itype = type[i];

  switch (eos[itype]) {
  case EOS_LINEAR:
    if ((failureModel[itype].integration_point_wise == true) && (damage[i] > 0.0)){
      LinearEOSwithDamage(rho, Lookup[REFERENCE_DENSITY][itype], Lookup[BULK_MODULUS][itype], pInitial, dt, pFinal, p_rate, damage[i]);
    } else {
      LinearEOS(Lookup[BULK_MODULUS][itype], pInitial, d_iso, dt, pFinal, p_rate);
    }
    break;
  case EOS_NONE:
    pFinal = 0.0;
    p_rate = 0.0;
    break;
  case EOS_SHOCK:
//  rho,  rho0,  e,  e0,  c0,  S,  Gamma,  pInitial,  dt,  &pFinal,  &p_rate);
    ShockEOS(rho, Lookup[REFERENCE_DENSITY][itype], mass_specific_energy, 0.0, Lookup[EOS_SHOCK_C0][itype],
             Lookup[EOS_SHOCK_S][itype], Lookup[EOS_SHOCK_GAMMA][itype], pInitial, dt, pFinal, p_rate, damage[i]);
    break;
  case EOS_POLYNOMIAL:
    polynomialEOS(rho, Lookup[REFERENCE_DENSITY][itype], vol_specific_energy, Lookup[EOS_POLYNOMIAL_C0][itype],
                  Lookup[EOS_POLYNOMIAL_C1][itype], Lookup[EOS_POLYNOMIAL_C2][itype], Lookup[EOS_POLYNOMIAL_C3][itype],
                  Lookup[EOS_POLYNOMIAL_C4][itype], Lookup[EOS_POLYNOMIAL_C5][itype], Lookup[EOS_POLYNOMIAL_C6][itype], pInitial, dt,
                  pFinal, p_rate, damage[i]);

    break;
  default:
    error->one(FLERR, "unknown EOS.");
    break;
  }
}

/* ----------------------------------------------------------------------
 Compute stress deviator. Called from AssembleStress().
 ------------------------------------------------------------------------- */
void PairTlsph::ComputeStressDeviator(const int i, const double mass_specific_energy, const Matrix3d& sigmaInitial_dev, const Matrix3d& d_dev, 
                                      Matrix3d &sigmaFinal_dev, Matrix3d &sigma_dev_rate, double &plastic_strain_increment, const double pFinal) {
  double *eff_plastic_strain = atom->eff_plastic_strain;
  double *eff_plastic_strain_rate = atom->eff_plastic_strain_rate;
  int *type = atom->type;
  double *rmass = atom->rmass;
  double *esph = atom->esph;
  double dt = update->dt;
  double yieldStress;
  int itype;
  double *damage = atom->damage;

  plastic_strain_increment = 0.0;
  itype = type[i];

  switch (strengthModel[itype]) {
  case STRENGTH_LINEAR:

    sigma_dev_rate = 2.0 * Lookup[SHEAR_MODULUS][itype] * d_dev;
    sigmaFinal_dev = sigmaInitial_dev + dt * sigma_dev_rate;

    break;
  case LINEAR_DEFGRAD:
//LinearStrengthDefgrad(Lookup[LAME_LAMBDA][itype], Lookup[SHEAR_MODULUS][itype], Fincr[i], &sigmaFinal_dev);
//eff_plastic_strain[i] = 0.0;
//p_rate = pInitial - sigmaFinal_dev.trace() / 3.0;
//sigma_dev_rate = sigmaInitial_dev - Deviator(sigmaFinal_dev);
    error->one(FLERR, "LINEAR_DEFGRAD is only for debugging purposes and currently deactivated.");
    R[i].setIdentity();
    break;
  case STRENGTH_LINEAR_PLASTIC:
    yieldStress = Lookup[YIELD_STRESS][itype] + Lookup[HARDENING_PARAMETER][itype] * eff_plastic_strain[i];
    if (failureModel[itype].failure_gtn)
      GTNStrength(Lookup[SHEAR_MODULUS][itype], Lookup[GTN_Q1][itype], Lookup[GTN_Q2][itype],
                  dt, damage[i], sigmaInitial_dev, d_dev, pFinal, yieldStress,
                  sigmaFinal_dev, sigma_dev_rate, plastic_strain_increment);
    else 
      LinearPlasticStrength(Lookup[SHEAR_MODULUS][itype], yieldStress, sigmaInitial_dev, d_dev, dt, 
                            sigmaFinal_dev, sigma_dev_rate, plastic_strain_increment, damage[i]);
    break;
  case STRENGTH_LUDWICK_HOLLOMON:
    yieldStress = Lookup[YIELD_STRESS][itype] + Lookup[HARDENING_PARAMETER][itype] * pow(eff_plastic_strain[i], Lookup[LH_n][itype]);
    if (failureModel[itype].failure_gtn)
      GTNStrength(Lookup[SHEAR_MODULUS][itype], Lookup[GTN_Q1][itype], Lookup[GTN_Q2][itype],
                  dt, damage[i], sigmaInitial_dev, d_dev, pFinal, yieldStress,
                  sigmaFinal_dev, sigma_dev_rate, plastic_strain_increment);
    else
      LinearPlasticStrength(Lookup[SHEAR_MODULUS][itype], yieldStress, sigmaInitial_dev, d_dev, dt, 
                            sigmaFinal_dev, sigma_dev_rate, plastic_strain_increment, damage[i]);
    break;
  case STRENGTH_SWIFT:
    yieldStress = Lookup[YIELD_STRESS][itype] + Lookup[HARDENING_PARAMETER][itype] * pow(eff_plastic_strain[i] + Lookup[SWIFT_eps0][itype], Lookup[SWIFT_n][itype]);
    if (failureModel[itype].failure_gtn)
      GTNStrength(Lookup[SHEAR_MODULUS][itype], Lookup[GTN_Q1][itype], Lookup[GTN_Q2][itype],
                  dt, damage[i], sigmaInitial_dev, d_dev, pFinal, yieldStress,
                  sigmaFinal_dev, sigma_dev_rate, plastic_strain_increment);
    else 
      LinearPlasticStrength(Lookup[SHEAR_MODULUS][itype], yieldStress, sigmaInitial_dev, d_dev, dt, 
                            sigmaFinal_dev, sigma_dev_rate, plastic_strain_increment, damage[i]);
    break;
  case STRENGTH_JOHNSON_COOK:
    JohnsonCookStrength(Lookup[SHEAR_MODULUS][itype], Lookup[HEAT_CAPACITY][itype], mass_specific_energy, Lookup[JC_A][itype],
                        Lookup[JC_B][itype], Lookup[JC_a][itype], Lookup[JC_C][itype], Lookup[JC_epdot0][itype], Lookup[JC_T0][itype],
                        Lookup[JC_Tmelt][itype], Lookup[JC_M][itype], dt, eff_plastic_strain[i], eff_plastic_strain_rate[i],
                        sigmaInitial_dev, d_dev, sigmaFinal_dev, sigma_dev_rate, plastic_strain_increment, damage[i]);
    break;
  case STRENGTH_NONE:
    sigmaFinal_dev.setZero();
    sigma_dev_rate.setZero();
    break;
  default:
    error->one(FLERR, "unknown strength model.");
    break;
  }

}

/* ----------------------------------------------------------------------
 Compute damage. Called from AssembleStress().
 ------------------------------------------------------------------------- */
void PairTlsph::ComputeDamage(const int i, const Matrix3d& strain, const Matrix3d& stress, Matrix3d &stress_damaged, double plastic_strain_increment) {
  double *eff_plastic_strain = atom->eff_plastic_strain;
  double *eff_plastic_strain_rate = atom->eff_plastic_strain_rate;
  double *radius = atom->radius;
  double *damage = atom->damage;
  int *type = atom->type;
  int itype = type[i];
  double jc_failure_strain;
  Matrix3d eye, stress_deviator;

  eye.setIdentity();
  stress_deviator = Deviator(stress);
  double pressure = -stress.trace() / 3.0;

  //// First apply damage to integration point (to stay consistent throughout the loop):
  //if (pressure > 0.0) { // compression: particle can carry compressive load but reduced shear
  //  stress_damaged = -pressure * eye + (1.0 - damage[i]) * Deviator(stress);
        //} else { // tension: particle has reduced tensile and shear load bearing capability
  //  stress_damaged = (1.0 - damage[i]) * (-pressure * eye + Deviator(stress));
        //}

  stress_damaged = stress;
  // Then calculate updated damage onset value:

  if (failureModel[itype].failure_max_principal_stress) {
    error->one(FLERR, "not yet implemented");
    /*
     * maximum stress failure criterion:
     */
    IsotropicMaxStressDamage(stress, Lookup[FAILURE_MAX_PRINCIPAL_STRESS_THRESHOLD][itype]);
  } else if (failureModel[itype].failure_max_principal_strain) {
    error->one(FLERR, "not yet implemented");
    /*
     * maximum strain failure criterion:
     */
    IsotropicMaxStrainDamage(strain, Lookup[FAILURE_MAX_PRINCIPAL_STRAIN_THRESHOLD][itype]);
  } else if (failureModel[itype].failure_max_plastic_strain) {
    if (eff_plastic_strain[i] >= Lookup[FAILURE_MAX_PLASTIC_STRAIN_THRESHOLD][itype]) {
      damage[i] = 1.0;
    }
  } else if (failureModel[itype].failure_johnson_cook) {
    damage[i] += JohnsonCookDamageIncrement(pressure, stress_deviator, Lookup[FAILURE_JC_D1][itype],
                                            Lookup[FAILURE_JC_D2][itype], Lookup[FAILURE_JC_D3][itype], Lookup[FAILURE_JC_D4][itype],
                                            Lookup[FAILURE_JC_EPDOT0][itype], eff_plastic_strain_rate[i], plastic_strain_increment);
  } else if (failureModel[itype].failure_gtn) {
    /*
     * Gurson - Tvergaard - Needleman damage evolution model:
     */
    double vol_change_rate = Fdot[i].determinant();
    damage[i] += Lookup[GTN_AN][itype] * plastic_strain_increment + (1.0 - damage[i]) * vol_change_rate;
  }

  damage[i] = MIN(damage[i], 1.0);
  //damage[i] = MIN(damage[i], 0.99);

}

void PairTlsph::UpdateDegradation() {
  tagint *mol = atom->molecule;
  tagint *tag = atom->tag;
  double **x = atom->x;
  double **v = atom->vest;
  double **x0 = atom->x0;
  double **f = atom->f;
  double *vfrac = atom->vfrac;
  double *radius = atom->radius;
  double *damage = atom->damage;
  double **vint = atom->v;
  double *plastic_strain = atom->eff_plastic_strain;
  double *eff_plastic_strain_rate = atom->eff_plastic_strain_rate;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int i, j, jj, jnum, itype, idim;
  double r, h, r0, r0Sq;
  double strain1d, strain1d_max, softening_strain;
  Vector3d dx0, dx, dv, x0i, x0j;
  Vector3d xi, xj;
  int periodic = (domain->xperiodic || domain->yperiodic || domain->zperiodic);

  char str[128];
  tagint **partner = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->partner;
  int *npartner = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->npartner;
  float **degradation_ij = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->degradation_ij;
  float **energy_per_bond = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->energy_per_bond;
  Vector3d **partnerx0 = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->partnerx0;
  double **partnervol = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->partnervol;

  for (i = 0; i < nlocal; i++) {

    if (mol[i] < 0) {
      continue; // Particle i is not a valid SPH particle (anymore). Skip all interactions with this particle.
    }

    itype = type[i];
    
    if (failureModel[itype].failure_none) { // Do not update degradation if no failure mode is activated for the mol.
      continue;
    }

    itype = type[i];
    jnum = npartner[i];

    // initialize aveage mass density
    h = 2.0 * radius[i];
    r = 0.0;

    if (failureModel[itype].failure_max_pairwise_strain) {
      for (idim = 0; idim < 3; idim++) {
        x0i(idim) = x0[i][idim];
        xi(idim) = x[i][idim];
      }
    }

    int numNeighbors = 0;
    
    for (jj = 0; jj < jnum; jj++) {
      if (degradation_ij[i][jj] >= 1.0)
        continue;
      j = atom->map(partner[i][jj]);
      if (j < 0) { //      // check if lost a partner without first breaking bond
        error->all(FLERR, "Bond broken not detected during PreCompute -3!");
        continue;
      }

      if (mol[j] < 0) {
        continue; // Particle j is not a valid SPH particle (anymore). Skip all interactions with this particle.
      }

      if (mol[i] != mol[j]) {
        continue;
      }

      if (type[j] != itype) {
        sprintf(str, "particle pair is not of same type!");
        error->all(FLERR, str);
      }

      if (failureModel[itype].failure_max_pairwise_strain) {
        for (idim = 0; idim < 3; idim++) {
          x0j(idim) = x0[j][idim];
          xj(idim) = x[j][idim];
        }

        if (periodic)
          domain->minimum_image(dx0(0), dx0(1), dx0(2));

        // check that distance between i and j (in the reference config) is less than cutoff
        dx0 = x0j - x0i;
        r0Sq = dx0.squaredNorm();
        h = radius[i] + radius[j];
        r0 = sqrt(r0Sq);

        // distance vectors in current and reference configuration, velocity difference
        dx = xj - xi;
        r = dx.norm(); // current distance

        strain1d = (r - r0) / r0;
        strain1d_max = Lookup[FAILURE_MAX_PAIRWISE_STRAIN_THRESHOLD][itype];
        softening_strain = 2.0 * strain1d_max;

        if (strain1d > strain1d_max) {
          degradation_ij[i][jj] = std::max(degradation_ij[i][jj], float((strain1d - strain1d_max) / softening_strain));
          if (degradation_ij[i][jj] >= 0.99) {
            printf("Link between %d and %d destroyed.\n", tag[i], partner[i][jj]);
            std::cout << "Here is dx0:" << std::endl << dx0 << std::endl;
            degradation_ij[i][jj] = 0.99;
          }
          //degradation_ij[i][jj] = (strain1d - strain1d_max) / softening_strain;
        } else {
          //degradation_ij[i][jj] = 0.0;
        }
      }

      if (failureModel[itype].failure_energy_release_rate) {
        
        h = radius[i] + radius[j];
        double Vic = (2.0 / 3.0) * h * h * h * h; // interaction volume for 2d plane strain
        double critical_energy_per_bond = Lookup[CRITICAL_ENERGY_RELEASE_RATE][itype] / (2.0 * Vic);

        if (energy_per_bond[i][jj] > critical_energy_per_bond) {
          degradation_ij[i][jj] = 1.0;
        }
      }

      if (failureModel[itype].integration_point_wise) {
        degradation_ij[i][jj] = 1 - (1 - damage[i]) * (1 - damage[j]);
        if (degradation_ij[i][jj] >= 1.0) { // delete interaction if fully damaged
          printf("Link between %d and %d destroyed due to complete degradation.\n", tag[i], partner[i][jj]);
          degradation_ij[i][jj] = 1.0;
        }
        /*
        if ((damage[i] == 1.0) && (damage[j]==1.0)) {
          strain1d = (r - r0) / r0;

          if (energy_per_bond[i][jj] > 0.0) { // f_stress.dot(dx) > 0.0 if the link i-jj is in tension, only then we apply degradation
            if (strain1d > 0.0) {
        
        // check if damage_onset is already defined
        softening_strain = 0.01;
        
        degradation_ij[i][jj] += 0.5 * (eff_plastic_strain_rate[i] + eff_plastic_strain_rate[j]) * update->dt / softening_strain;
        
            }
          }
            
            
          if (degradation_ij[i][jj] >= 1.0) { // delete interaction if fully damaged
            printf("Link between %d and %d destroyed due to complete degradation.\n", tag[i], tag[jj]);
            degradation_ij[i][jj] = 1.0;
          }
          }*/
      }
      
      if (degradation_ij[i][jj] < 1.0) {
        numNeighbors += 1;
      }
    } // end loop over jj neighbors of i
    
    if (numNeighbors == 0) {
      //printf("Deleting particle [%d] because damage = %f\n", tag[i], damage[i]);
      //dtCFL = MIN(dtCFL, update->dt);
      //mol[i] = -1;
      vint[i][0] = 0.0;
      vint[i][1] = 0.0;
      vint[i][2] = 0.0;
      f[i][0] = 0.0;
      f[i][1] = 0.0;
      f[i][2] = 0.0;
      smoothVelDifference[i].setZero();
    }

  } // end loop over i
}


void PairTlsph:: AdjustStressForZeroForceBC(const Matrix3d sigma, const Vector3d sU, Matrix3d &sigmaBC) {
  Vector3d sV, sW, sigman;
  Matrix3d P;
  //cout << "Creating mirror particle i=" << tag[i] << " and j=" << tag[j] << endl;
  
  P = CreateOrthonormalBasisFromOneVector(sU);

  sigmaBC = P.transpose() * sigma * P; // We transform sigmaBC to the surface basis
  
  
  // sigmaBC.[1, 0, 0] == [0 0 0] if there is no stress on the boundary!
  sigmaBC.col(0).setZero();
  sigmaBC.row(0).setZero();
  
  sigmaBC = P * sigmaBC * P.transpose();

  // Check if sigmaBC * surfaceNormalNormi = 0:
  sigman = sigmaBC * sU;
  if (sigman.norm() > 1.0e-5){
    std::cout << "Here is sigman :" << std::endl << sigman << std::endl;
    std::cout << "Here is P.transpose() * sigmaBC * P :" << std::endl << P.transpose() * sigmaBC * P << std::endl;
    std::cout << "Here is P.transpose() * sU :" << std::endl << P.transpose() * sU << std::endl;
    std::cout << "Here is P :" << std::endl << P << std::endl;
  }
}


Vector3d PairTlsph::ComputeFstress(const int i, const int j, const int jj, const double surfaceNormalNormi, const Vector3d dx0, const double r0, const Vector3d g, const Matrix3d sigmaBC_i, const double scale, const double strain1d) {
  double *vfrac = atom->vfrac;
  float **wfd_list = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->wfd_list;
  double voli, volj;
  Matrix3d sigmaInternalBC;
  Vector3d fij, eij, compForce, shearForce;

  voli = vfrac[i];
  volj = vfrac[j];

  eij = dx0/r0;

  if (strain1d > 0.0) {
    // The bond ij is in tension, the whole contribution needs to be scaled
    if (scale == 0.0) {
      fij = PK1[i] * Kundeg[i] * g;
    } else {
      fij = (scale * PK1[j] + PK1[i]) * Kundeg[i] * g; 
    }
  } else {
    fij = PK1[j] * Kundeg[i] * g;
    
    if (scale < 1.0) {
      // The bond ij is in compression, only shear forces need to be scaled:
      
      Matrix3d P = CreateOrthonormalBasisFromOneVector(dx0);
      Vector3d sU = P.col(0);
      Vector3d sV = P.col(1);
      Vector3d sW = P.col(2);
      
      compForce = fij.dot(sU) * sU;
      shearForce = ( fij.dot(sV) * sV + fij.dot(sW) * sW)*scale;
      if (abs(compForce.dot(shearForce)) > 1.0e-10) {
        printf("compForce and shearForce should be orthogonal.");
        std::cout << "Here is P:" << std::endl << P << std::endl;
        std::cout << "sU.sV = " << sU.dot(sV) << std::endl;
        std::cout <<"sU.sW = " << sU.dot(sW) << std::endl;
        std::cout <<"sV.sW = " << sV.dot(sW) << std::endl;
        std::cout << "Here is compForce: " << std::endl << compForce << std::endl;
        std::cout << "Here is shearForce: " << std::endl << shearForce << std::endl;
        std::cout << "compForce dot shearForce =  " << compForce.dot(shearForce) << std::endl;
        //error->all(FLERR, "Error. compForce and shearForce should be orthogonal.");
      }
      fij = compForce + shearForce;
    }
    fij += PK1[i]*Kundeg[i]*g;
    
  }
  
  
  if ((surfaceNormalNormi > 0.5) && (surfaceNormal[i].dot(dx0) <= -0.5*pow(volj, 1.0/3.0))) {
    // i is a surface particle, and j is in the bulk.
    Vector3d dx0mirror = dx0 - 2 * (dx0.dot(surfaceNormal[i])) * surfaceNormal[i];
    //fij += (PK1[i] - sigmaBC_i) * Kundeg[i] * (wfd_list[i][jj] / r0) * dx0mirror; // Contribution of the virtual mirror particle (is not affected by damage!)
    fij += PK1[i] * Kundeg[i] * (wfd_list[i][jj] / r0) * dx0mirror; // Contribution of the virtual mirror particle (is not affected by damage!)
  }
  return -voli * volj * fij;
  
}
