*KEYWORD
$ Units throughout are N, g, cm
*TITLE
$# title
Metal RVE with 2 voids. Michael D. Shields. 3/20/19
$*CONTROL_PARALLEL
$        24
*CONTROL_MPP_DECOMPOSITION_DISTRIBUTE_ALE_ELEMENTS
*CONTROL_TERMINATION
$#  endtim    endcyc     dtmin    endeng    endmas
      5e-4               5e-10
*CONTROL_TIMESTEP
$#  dtinit    tssfac      isdo    tslimt     dt2ms      lctm     erode     ms1st
     0.000  0.600000
$#  dt2msf   dt2mslc     imscl
     0.000         0         0
*CONTROL_HOURGLASS
$#     ihq        qh
         1 1.0000E-5
$ I don't think hourglass control is necessary - MDS
*CONTROL_ENERGY
$#    hgen      rwen    slnten     rylen
         2         1         2         2
$ I don't think energy control is necessary - MDS
$*DATABASE_ELOUT
$#      dt    binary      lcur     ioopt
$      1e-5
$ Output quantities for each element every 1e-5 sec.
$*DATABASE_GLSTAT
$#      dt    binary      lcur     ioopt
$ 1.0000E-8
$*DATABASE_NODOUT
$#      dt    binary      lcur     ioopt      dthf     binhf
$ 1.0000E-8
*DATABASE_BINARY_D3PLOT
$#      dt      lcdt      beam     npltc    psetid
      1e-4
$#   ioopt
         0
$*DATABASE_BINARY_D3THDT
$#      dt      lcdt      beam     npltc    psetid
$      1e-5
$*DATABASE_EXTENT_BINARY
$#   neiph     neips    maxint    strflg    sigflg    epsflg    rltflg    engflg
$         0         0         0         0         0         0         0         0
$#  cmpflg    ieverp    beamip     dcomp      shge     stssz    n3thdt   ialemat
$         0         0         0         0         0         0         0         1
$# nintsld   pkp_sen      sclp    unused     msscl     therm
$         0         0  1.000000
$ I haven't really looked at the database keywords - they are controlling output
*CONTROL_ALE
$#     dct      nadv      meth      afac      bfac      cfac      dfac      efac
         0         1         1 -1.000000
$#   start       end     aafac     vfact      prit       ebc      pref   nsidebc
     0.000     0.000     0.000     0.000     0.000         0     0.000         0
*NODE
  199997   0.0000000e+00   0.0000000e+00   0.0000000e+00
  199998   0.0000000e+00   0.0000000e+00   0.0000000e+00
  199999   0.1000000e+00   0.0000000e+00   0.0000000e+00
  200000   0.0000000e+00   0.1000000e+00   0.0000000e+00
*ALE_STRUCTURED_MESH
$    mshid       pid      nbid      ebid      ityp    nparts
         1         9    200001    200001         0         0
$     nptx      npty      nptz      nid0     lcsid
      3001      3002      3003    199997       890
*DEFINE_COORDINATE_NODES
$      cid      nid1      nid2      nid3
       890    199998    199999    200000
*ALE_STRUCTURED_MESH_CONTROL_POINTS
      3001         0         1     1.000     0.000     0.000         0
                   1                0.00
                  21                0.10
*ALE_STRUCTURED_MESH_CONTROL_POINTS
      3002         0         1     1.000     0.000     0.000         0
                   1                0.00
                  11                0.05
*ALE_STRUCTURED_MESH_CONTROL_POINTS
      3003         0         1     1.000     0.000     0.000         0
                   1                0.00
                  11                0.05
*ALE_STRUCTURED_MESH_REFINE
         1         1         1         1
*ALE_MULTI-MATERIAL_GROUP
$      PID      TYPE
        10         1
        11         1
*DEFINE_CURVE
$     LCID      SIDR       SFA       SFO
         1         0         1         1
                 0.0                 0.0
                1e-4               100.0
                5e-4               100.0
$ Ramp up velocity to 100 cm/s (1m/s) over 1e-5 seconds.
*PART
$# title
base metal
$#     PID     SECID       MID     EOSID       HID
        10        10        10         0        10
$ This PART will be used for the base metal.
*PART
$# title
voids
$#     PID     SECID       MID     EOSID       HID
        11        10        11         0        10
$ This PART will be used for the voids.
*SECTION_SOLID
$   SECID        FORM
        10        11
$ Element FORM 11 is the ALE solid element that may be assigned multi-materials
$ Element FORM 12 is the ALE solid element that may be assigned only a single material
$
*MAT_POWER_LAW_PLASTICITY
$      MID        RO         E        PR         K         N       SRC       SRP
        10      7.85    200e+6      0.33    525e+3       0.2         0         0
$ Steel with no rate effects
*MAT_VACUUM
$      MID        RO
        11    1.0e-6
*HOURGLASS
        10         1    1.0e-6
*INITIAL_VOLUME_FRACTION_GEOMETRY 
$      sid     idtyp     bammg    ntrace
         9         1         1
$     type   fillopt     fammg
         6                   2
$       x0        y0        z0        x1        y1        z1
<x0><y0><z0><R0>
$     type   fillopt     fammg
         6                   2
$       x0        y0        z0        x1        y1        z1
<x1><y1><z1><R1>
*SET_NODE_GENERAL
$      SID
         1
$   OPTION     MSHID        E2        E3        E4        E5        E6        E7
   SALEFAC         1         1
$ x=0 symmetry plane
*SET_NODE_GENERAL
$      SID
         2
$   OPTION     MSHID        E2        E3        E4        E5        E6        E7
   SALEFAC         1                   1
$ x=0.1 velocity
*SET_NODE_GENERAL
$      SID
         3
$   OPTION     MSHID        E2        E3        E4        E5        E6        E7
   SALEFAC         1                             1
$ y=0 rollers
*SET_NODE_GENERAL
$      SID
         4
$   OPTION     MSHID        E2        E3        E4        E5        E6        E7
   SALEFAC         1                                       1
$ y=0.1 rollers
*SET_NODE_GENERAL
$      SID
         5
$   OPTION     MSHID        E2        E3        E4        E5        E6        E7
   SALEFAC         1                                                 1
$ z=0 rollers
*SET_NODE_GENERAL
$      SID
         6
$   OPTION     MSHID        E2        E3        E4        E5        E6        E7
   SALEFAC         1                                                           1
$ z=0.1 rollers
$*SET_SOLID_GENERAL
$     SID
$        7
$  OPTION      MSHID
$   SALECPT         1         1        21         1        11         1        11
*BOUNDARY_SPC_SET 
$      SID       CID         x         y         z        rx        ry        rz
         1                   1                                       1         1
$ x=0 symmetry plane
*BOUNDARY_SPC_SET
$      SID       CID         x         y         z        rx        ry        rz
         3                             1
$ y=0 roller
*BOUNDARY_SPC_SET
$      SID       CID         x         y         z        rx        ry        rz
         4                             1
$ y=0.1 roller
*BOUNDARY_SPC_SET
$      SID       CID         x         y         z        rx        ry        rz
         5                                       1
$ z=0 roller
*BOUNDARY_SPC_SET
$      SID       CID         x         y         z        rx        ry        rz
         6                                       1
$ z=0.1 roller
*BOUNDARY_PRESCRIBED_MOTION_SET
$      SID       DOF       VAD      LCID        SF
         2         1         0         1         1
*DEFINE_BOX
$    BOXID       XMN       XMX       YMN       YMX       ZMN       ZMX
         1       0.0       0.1       0.0      0.05       0.0      0.05
$*DATABASE_ALE
$    DTOUT     SETID
$      1e-6         7
*DATABASE_ALE_MAT
$    DTOUT    BOXLOW     BOXUP
      1e-6         1         1
*END
