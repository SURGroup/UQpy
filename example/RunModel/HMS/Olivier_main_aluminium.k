*KEYWORD
*TITLE
Material (aluminium) with micro-voids
$===============================================================================
$ Author: Audrey Olivier
$ Last updated: April 3, 2020
$ This specific file sets all variables that are the same for all models
$ Units throughout are ms, mg, mm (stress is in 1e-3 MPa, force in 1e-3 N)
$-------------------------------------------------------------------------------
$===============================================================================
$                        *** CONTROL SECTION ***
$===============================================================================
$*CONTROL_TERMINATION
$$      end                 min
$      5e-4               5e-10
*CONTROL_TIMESTEP
$   dtinit    tssfac      isdo    tslimt     dt2ms      lctm     erode     ms1st
     0.000       0.6
$   dt2msf   dt2mslc     imscl
     0.000         0         0
*CONTROL_ENERGY
$     hgen      rwen    slnten     rylen
         2         2         2         2
$*CONTROL_PARALLEL
$        24
*CONTROL_MPP_DECOMPOSITION_DISTRIBUTE_ALE_ELEMENTS
         1
*CONTROL_ALE
$#     dct      nadv      meth      afac      bfac      cfac      dfac      efac
         0         1         1      -1.0
$#   start       end     aafac     vfact      prit       ebc      pref   nsidebc
     0.000     0.000     0.000     0.000     0.000         0     0.000         0
$===============================================================================
$ [ ] OUTPUT CONTROLS
$===============================================================================
*DATABASE_BINARY_D3PLOT
$       dt      lcdt      beam     npltc    psetid
    1.0e-4
$    ioopt
         0
*DATABASE_EXTENT_BINARY
$ Set neiph to 0 or 6?
$    neiph     neips    maxint    strflg    sigflg    epsflg    rltflg    engflg
         6
$   cmpflg    ieverp    beamip     dcomp      shge     stssz    n3thdt   ialemat
         0         0         0         0         0         0         0         0
$  nintsld   pkp_sen      sclp    unused     msscl     therm
         0         0  1.000000
$===============================================================================
$ Define box and element sets for ALE outputs
*DEFINE_BOX
$    BOXID       XMN       XMX       YMN       YMX       ZMN       ZMX
         1       0.0     0.155      0.00     0.105      0.00     0.055
*DATABASE_ALE_MAT
$    DTOUT    BOXLOW     BOXUP
      1e-6         1         1
*DATABASE_GLSTAT
*SET_SOLID_GENERAL
        99
   SALECPT         1         1        31         1        21         1        11
*DATABASE_ELOUT
$    DTOUT    BINARY
      5e-6         1
*DATABASE_HISTORY_SOLID_SET
        99
*DATABASE_SPCFORC
      5e-6         1
*DATABASE_FSI
      5e-6         1
        25        25         0 
*DATABASE_BNDOUT   
      5e-6         1  
$===============================================================================
$                        *** EULERIAN MODEL SECTION ***
$===============================================================================
*ALE_STRUCTURED_MESH
$    mshid       pid      nbid      ebid      ityp    nparts
         1         1    200001    200001         0         0
$     nptx      npty      nptz      nid0     lcsid
      3001      3002      3003
$ 1 element dx=0.005 before refinement
*ALE_STRUCTURED_MESH_CONTROL_POINTS
      3001         0         1     1.000     0.000     0.000         0
                   1                0.00
                  32               0.155
*ALE_STRUCTURED_MESH_CONTROL_POINTS
      3002         0         1     1.000     0.000     0.000         0
                   1                0.00
                  22               0.105
*ALE_STRUCTURED_MESH_CONTROL_POINTS
      3003         0         1     1.000     0.000     0.000         0
                   1                0.00
                  12               0.055
*ALE_STRUCTURED_MESH_REFINE
         1         4         4         4
$-------------------------------------------------------------------------------
*ALE_MULTI-MATERIAL_GROUP
$      PID      TYPE
$ First one is for the voids inside material
        11         1
$ Second one is the material
        10         1
$ Third one is for the void outside of material
        12         1
*PART
base metal
$      PID     SECID       MID     EOSID       HID
        10        20        10         0        10
*PART
voids
$      PID     SECID       MID     EOSID       HID
        11        20        11         0        10
*PART
void outside
$      PID     SECID       MID     EOSID       HID
        12        20        11         0        10
*SECTION_SOLID
$    SECID      FORM
        20        11
$ Create a part_list with all ale materials
$*SET_PART_LIST
$        24
$        10        11        12
$ Create a MM group list, used when doing FSI, contains material and voids
*SET_MULTI-MATERIAL_GROUP_LIST
        30
         1         2
$-------------------------------------------------------------------------------
$ Define material: aluminium
$ E = 70 GPa, Bulk modulus 69 GPa, poisson ratio 0.33, density 2.70 g/cm3
$ K between 400 and 550 MPa, n between 0.2 and 0.3
*MAT_POWER_LAW_PLASTICITY
$      MID        RO         E        PR         K         N       SRC       SRP
        10       2.7     70e+6      0.33    450e+3       0.2         0         0
$ Define material: vacuum
*MAT_VACUUM
$      MID        RO
        11    1.0e-6
*HOURGLASS
        10         1    1.0e-6
$===============================================================================
$                        *** SHELL SECTIONS AROUND THE MATERIAL ***
$===============================================================================
*INCLUDE
shells_refinement[4].k
*SET_PART_LIST
        25
         7         9         5
$===============================================================================
*PART
S5 = rigid plate at xmax
$      pid     secid       mid     eosid      hgid      grav    adpopt      tmid
         5         5         5
*SECTION_SHELL
$    SECID    ELFORM      SHRF       NIP     PROPT   QR/IRID     ICOMP
         5
$       T1        T2        T3        T4      NLOC
    0.0005    0.0005    0.0005    0.0005
*MAT_RIGID 
$      MID        RO         E        PR  MADYMO_N MADY_COUP  MADYMO_M
         5       2.7    350e+6      0.33
$      CMO      CON1      CON2 
         1         5         7
$ E is 5 times the one of the steel material
$ CMO=1 for constraints
$ CON1: constrained y and z displacements
$ CON2: constrained x, y, z rotations
$      LCO         =             card 3 = must include but may be left blank
   
$===============================================================================     
*PART
S7 = rigid plate at ymax
$      pid     secid       mid     eosid      hgid      grav    adpopt      tmid
         7         7         7
*SECTION_SHELL
$    SECID    ELFORM      SHRF       NIP     PROPT   QR/IRID     ICOMP
         7
$       T1        T2        T3        T4      NLOC
    0.0005    0.0005    0.0005    0.0005
*MAT_RIGID 
$      MID        RO         E        PR  MADYMO_N MADY_COUP  MADYMO_M
         7       2.7    350e+6      0.33
$      CMO      CON1      CON2 
         1         6         7
$      LCO         =             card 3 = must include but may be left blank
         
$===============================================================================     
*PART
S9 = rigid plate at zmax
$      pid     secid       mid     eosid      hgid      grav    adpopt      tmid
         9         9         9
*SECTION_SHELL
$    SECID    ELFORM      SHRF       NIP     PROPT   QR/IRID     ICOMP
         9
$       T1        T2        T3        T4      NLOC
    0.0005    0.0005    0.0005    0.0005
*MAT_RIGID 
$      MID        RO         E        PR  MADYMO_N MADY_COUP  MADYMO_M
         9       2.7    350e+6      0.33
$      CMO      CON1      CON2 
         1         4         7
$      LCO         =             card 3 = must include but may be left blank
         
$===============================================================================
$                        *** FLUID STRUCTURE INTERACTION ***
$===============================================================================
*CONSTRAINED_LAGRANGE_IN_SOLID
$    slave    master     sstyp     mstyp     nquad     ctype     direc     mcoup
        25         1         0         1         3         4         1       -30
$    start       end      pfac      fric    frcmin      norm   normtyp      damp
       0.0       0.0       1.0       0.0       0.1         0         0      0.00
$       CQ      HMIN      HMAX     ILEAK LEAKSTIFF VLK_PLCID     NVENT    IBLOCK
       0.0       0.0       0.0         0       0.1         0         0         0
$   IBOXID   IPENCHK   INTFORC   IALESOF    LAGMUL    PFACMM
$         0         0         0         0         0         0
$ set nquad=6 is refinement is 2 (2 to 1 mapping ALE to shell)
$ DIREC = 1 coupling in compression and tension
$ MCOUP = -30 couples for the material+voids
$===============================================================================
$                        *** BOUNDARY CONDITIONS ***
$===============================================================================
$ Symmetry plane at xmin
*SET_NODE_GENERAL
$      SID
         1
$   OPTION     MSHID      XMIN      XMAX      YMIN      YMAX      ZMIN      ZMAX
   SALEFAC         1         1
*BOUNDARY_SPC_SET 
$      SID       CID         x         y         z        rx        ry        rz
         1                   1                                       1         1
*SET_NODE_GENERAL
$      SID
         2
$   OPTION     MSHID      XMIN      XMAX      YMIN      YMAX      ZMIN      ZMAX
   SALEFAC         1                             1
*BOUNDARY_SPC_SET 
$      SID       CID         x         y         z        rx        ry        rz
         2                             1                   1                   1
*SET_NODE_GENERAL
$      SID
         3
$   OPTION     MSHID      XMIN      XMAX      YMIN      YMAX      ZMIN      ZMAX
   SALEFAC         1                                                 1
*BOUNDARY_SPC_SET 
$      SID       CID         x         y         z        rx        ry        rz
         3                                       1         1         1         
$===============================================================================
$$ Define curves needed for setting BCs: 1 is ramp up, 2 is just linear
$*DEFINE_CURVE
$$     LCID      SIDR       SFA       SFO
$         1         0         1         1
$                 0.0                 0.0
$                1e-4                 1.0
$                5e-4                 1.0
$=============================================================================== 
$ Additional inputs are added from the script
$
*INITIAL_VOLUME_FRACTION_GEOMETRY
$ Base filling = base material
$      sid     idtyp     bammg    ntrace
         1         1         2
$ Filling action 1: create voids
$     type   fillopt     fammg
$       xi        yi        zi        ri
         6         0         1
<x[0]><y[0]><z[0]><R[0]>
         6         0         1
<x[1]><y[1]><z[1]><R[1]>
         6         0         1
<x[2]><y[2]><z[2]><R[2]>
$ Filling action 2: create void outside a BOX
         5         1         3
  0.000000  0.000000  0.000000  0.100000  0.100000  0.050000
$ Define curves needed for setting BCs
*DEFINE_CURVE
$     LCID      SIDR       SFA       SFO
         2         0         1         1
                 0.0                 1.0
            0.000450                 1.0
*INCLUDE
init_vel_ale_3comp.k
*INITIAL_VELOCITY_RIGID_BODY
         5100.000000
*INITIAL_VELOCITY_RIGID_BODY
         7  0.000000-20.000000
*INITIAL_VELOCITY_RIGID_BODY
         9  0.000000  0.000000-10.000000
*BOUNDARY_PRESCRIBED_MOTION_RIGID
$NID|NSID|PID    DOF       VAD      LCID        SF
         5         1         0         2100.000000
         7         2         0         2-20.000000
         9         3         0         2-10.000000
*CONTROL_TERMINATION
$      end         min
  0.000450               5e-10
*END