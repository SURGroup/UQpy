# OpenSees
# OpenSees Primer
#
# Units: kN, m, sec

# Create ModelBuilder with 3 dimensions and 6 DOF/node
model BasicBuilder -ndm 3 -ndf 6

##################################
#     SIMPLE 3D 2STOREY          #
#          ____________          #
#         /|     |     /|        #
#        /_|_____|____/ |        #
#       /|/|     |   /|/|        #
#      /_|_|_____|_ / | |        #
#     | /| | |   | | /| |        #
#     |/_|_=_|___=_|/ | =   y    #
#     |  |   |     |  |  z /     #
#     |  =   |     |  =  |/      #
#     |(0,0) |     |     |--->x  #
#     =      =    =              #
##################################

source columnsdimensions.tcl

set z1   3.00000000000000
set z2   [expr $z1+3]
set z3   [expr $z2+3]
set z4   [expr $z3+3]
set z5   [expr $z4+3]
set z6   [expr $z5+3] 
set ztot [expr $z1+$z2+$z3+$z4+$z5+$z6]

set volume [expr (4*$h1*$b1+2*$h2*$b2+4*$h3*$b3+2*$h4*$b4+4*$h5*$b5+2*$h6*$b6)*$z6+6*(72*0.18)+6*($hx*$bx*(18-4*$h1-2*$h2-4*$h3-2*$h4-4*$h4-2*$h5))+6*($hy*$by*(24-4*$b1-2*$b2-4*$b1-2*$b2-4*$b3-2*$b4-4*$b5-2*$b6))]

set fileId1 [open volume w 0600]
puts $fileId1 $volume;

# Create nodes
#     tag     X               Y                Z
node	1	[expr $h1/2]	[expr $b1/2]	0
node	2	[expr 6-$h1/2]	[expr $b1/2]	0
node	3	[expr $h2/2]	[expr 6-$b2/2]	0
node	4	[expr 6-$h2/2]	[expr 6-$b2/2]	0
node	5	[expr $h1/2]	[expr 12-$b1/2]	0
node	6	[expr 6-$h1/2]	[expr 12-$b1/2]	0

node	101	[expr $h1/2]	[expr $b1/2]	$z1
node	102	[expr 6-$h1/2]	[expr $b1/2]	$z1
node	103	[expr $h2/2]	[expr 6-$b2/2]	$z1
node	104	[expr 6-$h2/2]	[expr 6-$b2/2]	$z1
node	105	[expr $h1/2]	[expr 12-$b1/2]	$z1
node	106	[expr 6-$h1/2]	[expr 12-$b1/2]	$z1

node	201	[expr $h1/2]	[expr $b1/2]	$z2
node	202	[expr 6-$h1/2]	[expr $b1/2]	$z2
node	203	[expr $h2/2]	[expr 6-$b2/2]	$z2
node	204	[expr 6-$h2/2]	[expr 6-$b2/2]	$z2
node	205	[expr $h1/2]	[expr 12-$b1/2]	$z2
node	206	[expr 6-$h1/2]	[expr 12-$b1/2]	$z2

node	301	[expr $h3/2]	[expr $b3/2]	$z3
node	302	[expr 6-$h3/2]	[expr $b3/2]	$z3
node	303	[expr $h4/2]	[expr 6-$b4/2]	$z3
node	304	[expr 6-$h4/2]	[expr 6-$b4/2]	$z3
node	305	[expr $h3/2]	[expr 12-$b3/2]	$z3
node	306	[expr 6-$h3/2]	[expr 12-$b3/2]	$z3

node	401	[expr $h3/2]	[expr $b3/2]	$z4
node	402	[expr 6-$h3/2]	[expr $b3/2]	$z4
node	403	[expr $h4/2]	[expr 6-$b4/2]	$z4
node	404	[expr 6-$h4/2]	[expr 6-$b4/2]	$z4
node	405	[expr $h3/2]	[expr 12-$b3/2]	$z4
node	406	[expr 6-$h3/2]	[expr 12-$b3/2]	$z4

node	501	[expr $h5/2]	[expr $b5/2]	$z5
node	502	[expr 6-$h5/2]	[expr $b5/2]	$z5
node	503	[expr $h6/2]	[expr 6-$b6/2]	$z5
node	504	[expr 6-$h6/2]	[expr 6-$b6/2]	$z5
node	505	[expr $h5/2]	[expr 12-$b5/2]	$z5
node	506	[expr 6-$h5/2]	[expr 12-$b5/2]	$z5

node	601	[expr $h5/2]	[expr $b5/2]	$z6
node	602	[expr 6-$h5/2]	[expr $b5/2]	$z6
node	603	[expr $h6/2]	[expr 6-$b6/2]	$z6
node	604	[expr 6-$h6/2]	[expr 6-$b6/2]	$z6
node	605	[expr $h5/2]	[expr 12-$b5/2]	$z6
node	606	[expr 6-$h5/2]	[expr 12-$b5/2]	$z6

# Create nodes for rigid offsets
#    tag       X                  Y                Z
node	1011	[expr $h1]	[expr $bx/2]	[expr $z1]
node	1012	[expr $by/2]	[expr $b1]	[expr $z1]
node	1021	[expr 6-$h1]	[expr $bx/2]	[expr $z1]
node	1022	[expr 6-$by/2]	[expr $b1]	[expr $z1]
node	1031	[expr $h2]	[expr 6-$bx/2]	[expr $z1]
node	1032	[expr $by/2]	6	        [expr $z1]
node	1033	[expr $by/2]	[expr 6-$b2]	[expr $z1]
node	1041	[expr 6-$h2]	[expr 6-$bx/2]	[expr $z1]
node	1042	[expr 6-$by/2]	6	        [expr $z1]
node	1043	[expr 6-$by/2]	[expr 6-$b2]	[expr $z1]
node	1051	[expr $h1]	[expr 12-$bx/2]	[expr $z1]
node	1052	[expr $by/2]	[expr 12-$b1]	[expr $z1]
node	1061	[expr 6-$h1]	[expr 12-$bx/2]	[expr $z1]
node	1062	[expr 6-$by/2]	[expr 12-$b1]	[expr $z1]

node	2011	[expr $h1]	[expr $bx/2]	[expr $z2]
node	2012	[expr $by/2]	[expr $b1]	[expr $z2]
node	2021	[expr 6-$h1]	[expr $bx/2]	[expr $z2]
node	2022	[expr 6-$by/2]	[expr $b1]	[expr $z2]
node	2031	[expr $h2]	[expr 6-$bx/2]	[expr $z2]
node	2032	[expr $by/2]	6	        [expr $z2]
node	2033	[expr $by/2]	[expr 6-$b2]	[expr $z2]
node	2041	[expr 6-$h2]	[expr 6-$bx/2]	[expr $z2]
node	2042	[expr 6-$by/2]	6	        [expr $z2]
node	2043	[expr 6-$by/2]	[expr 6-$b2]	[expr $z2]
node	2051	[expr $h1]	[expr 12-$bx/2]	[expr $z2]
node	2052	[expr $by/2]	[expr 12-$b1]	[expr $z2]
node	2061	[expr 6-$h1]	[expr 12-$bx/2]	[expr $z2]
node	2062	[expr 6-$by/2]	[expr 12-$b1]	[expr $z2]

node	3011	[expr $h3]	[expr $bx/2]	[expr $z3]
node	3012	[expr $by/2]	[expr $b3]	[expr $z3]
node	3021	[expr 6-$h3]	[expr $bx/2]	[expr $z3]
node	3022	[expr 6-$by/2]	[expr $b3]	[expr $z3]
node	3031	[expr $h4]	[expr 6-$bx/2]	[expr $z3]
node	3032	[expr $by/2]	6	        [expr $z3]
node	3033	[expr $by/2]	[expr 6-$b4]	[expr $z3]
node	3041	[expr 6-$h4]	[expr 6-$bx/2]	[expr $z3]
node	3042	[expr 6-$by/2]	6	        [expr $z3]
node	3043	[expr 6-$by/2]	[expr 6-$b4]	[expr $z3]
node	3051	[expr $h3]	[expr 12-$bx/2]	[expr $z3]
node	3052	[expr $by/2]	[expr 12-$b3]	[expr $z3]
node	3061	[expr 6-$h3]	[expr 12-$bx/2]	[expr $z3]
node	3062	[expr 6-$by/2]	[expr 12-$b3]	[expr $z3]

node	4011	[expr $h3]	[expr $bx/2]	[expr $z4]
node	4012	[expr $by/2]	[expr $b3]	[expr $z4]
node	4021	[expr 6-$h3]	[expr $bx/2]	[expr $z4]
node	4022	[expr 6-$by/2]	[expr $b3]	[expr $z4]
node	4031	[expr $h4]	[expr 6-$bx/2]	[expr $z4]
node	4032	[expr $by/2]	6	        [expr $z4]
node	4033	[expr $by/2]	[expr 6-$b4]	[expr $z4]
node	4041	[expr 6-$h4]	[expr 6-$bx/2]	[expr $z4]
node	4042	[expr 6-$by/2]	6	        [expr $z4]
node	4043	[expr 6-$by/2]	[expr 6-$b4]	[expr $z4]
node	4051	[expr $h3]	[expr 12-$bx/2]	[expr $z4]
node	4052	[expr $by/2]	[expr 12-$b3]	[expr $z4]
node	4061	[expr 6-$h3]	[expr 12-$bx/2]	[expr $z4]
node	4062	[expr 6-$by/2]	[expr 12-$b3]	[expr $z4]

node	5011	[expr $h5]	[expr $bx/2]	[expr $z5]
node	5012	[expr $by/2]	[expr $b5]	[expr $z5]
node	5021	[expr 6-$h5]	[expr $bx/2]	[expr $z5]
node	5022	[expr 6-$by/2]	[expr $b5]	[expr $z5]
node	5031	[expr $h6]	[expr 6-$bx/2]	[expr $z5]
node	5032	[expr $by/2]	6	        [expr $z5]
node	5033	[expr $by/2]	[expr 6-$b6]	[expr $z5]
node	5041	[expr 6-$h6]	[expr 6-$bx/2]	[expr $z5]
node	5042	[expr 6-$by/2]	6	        [expr $z5]
node	5043	[expr 6-$by/2]	[expr 6-$b6]	[expr $z5]
node	5051	[expr $h5]	[expr 12-$bx/2]	[expr $z5]
node	5052	[expr $by/2]	[expr 12-$b5]	[expr $z5]
node	5061	[expr 6-$h5]	[expr 12-$bx/2]	[expr $z5]
node	5062	[expr 6-$by/2]	[expr 12-$b5]	[expr $z5]

node	6011	[expr $h5]	[expr $bx/2]	[expr $z6]
node	6012	[expr $by/2]	[expr $b5]	[expr $z6]
node	6021	[expr 6-$h5]	[expr $bx/2]	[expr $z6]
node	6022	[expr 6-$by/2]	[expr $b5]	[expr $z6]
node	6031	[expr $h6]	[expr 6-$bx/2]	[expr $z6]
node	6032	[expr $by/2]	6	        [expr $z6]
node	6033	[expr $by/2]	[expr 6-$b6]	[expr $z6]
node	6041	[expr 6-$h6]	[expr 6-$bx/2]	[expr $z6]
node	6042	[expr 6-$by/2]	6	        [expr $z6]
node	6043	[expr 6-$by/2]	[expr 6-$b6]	[expr $z6]
node	6051	[expr $h5]	[expr 12-$bx/2]	[expr $z6]
node	6052	[expr $by/2]	[expr 12-$b5]	[expr $z6]
node	6061	[expr 6-$h5]	[expr 12-$bx/2]	[expr $z6]
node	6062	[expr 6-$by/2]	[expr 12-$b5]	[expr $z6]

# Master nodes for rigid diaphragm
#      tag     X    Y    Z
node  10001   3    6     $z1
node  20001   3    6     $z2
node  30001   3    6     $z3
node  40001   3    6     $z4
node  50001   3    6     $z5
node  60001   3    6     $z6

#node   10001   4    2.5   4
#node   10002   4    7.5   7
#node   20001   4    2.5   4
#node   20002   4    7.5   7

# Set base constraints
#   tag DX DY DZ RX RY RZ
fix   1   1  1  1  1  1  1
fix   2   1  1  1  1  1  1
fix   3   1  1  1  1  1  1
fix   4   1  1  1  1  1  1
fix   5   1  1  1  1  1  1
fix   6   1  1  1  1  1  1

# Define rigid diaphragm multi-point constraints
#               normalDir    master     slaves
rigidDiaphragm	3 10001	101 102	103 104	105 106
rigidDiaphragm	3 20001	201 202	203 204	205 206
rigidDiaphragm	3 30001	301 302	303 304	305 306
rigidDiaphragm	3 40001	401 402	403 404	405 406
rigidDiaphragm	3 50001	501 502	503 504	505 506
rigidDiaphragm	3 60001	601 602	603 604	605 606
#
# Constraints for rigid diaphragm master nodes
#    tag    DX DY DZ RX RY RZ
fix  10001  0  0  1  1  1  0
fix  20001  0  0  1  1  1  0
fix  30001  0  0  1  1  1  0
fix  40001  0  0  1  1  1  0
fix  50001  0  0  1  1  1  0
fix  60001  0  0  1  1  1  0

# Define materials for nonlinear columns
# --------------------------------------

source import_variables.tcl

# CONCRETE
# Core concrete (confined)
##set fc 20000.0
set fcu1 [expr  0.8*$fc1]
set fcu2 [expr  0.8*$fc2]
set fcu3 [expr  0.8*$fc3]
set fcu4 [expr  0.8*$fc4]
set fcu5 [expr  0.8*$fc5]
set fcu6 [expr  0.8*$fc6]
#                           tag  f'c    epsc0    f'cu      epscu
uniaxialMaterial Concrete01  1   -$fc1     -0.002   -$fcu1      -0.0035
uniaxialMaterial Concrete01  2   -$fc2     -0.002   -$fcu2      -0.0035
uniaxialMaterial Concrete01  3   -$fc3     -0.002   -$fcu3      -0.0035
uniaxialMaterial Concrete01  4   -$fc4     -0.002   -$fcu4      -0.0035
uniaxialMaterial Concrete01  5   -$fc5     -0.002   -$fcu5      -0.0035
uniaxialMaterial Concrete01  6   -$fc6     -0.002   -$fcu6      -0.0035


# Cover concrete (unconfined)
set fcun1 [expr  0.8*$fc1]
set fcun2 [expr  0.8*$fc2]
set fcun3 [expr  0.8*$fc3]
set fcun4 [expr  0.8*$fc4]
set fcun5 [expr  0.8*$fc5]
set fcun6 [expr  0.8*$fc6]
set fcuun1 [expr  0.8*$fcun1]
set fcuun2 [expr  0.8*$fcun2]
set fcuun3 [expr  0.8*$fcun3]
set fcuun4 [expr  0.8*$fcun4]
set fcuun5 [expr  0.8*$fcun5]
set fcuun6 [expr  0.8*$fcun6]
uniaxialMaterial Concrete01  7  -$fcun1  -0.002   -$fcuun1      -0.0035
uniaxialMaterial Concrete01  8  -$fcun2  -0.002   -$fcuun2      -0.0035
uniaxialMaterial Concrete01  9  -$fcun3  -0.002   -$fcuun3      -0.0035
uniaxialMaterial Concrete01  10  -$fcun4  -0.002   -$fcuun4      -0.0035
uniaxialMaterial Concrete01  11  -$fcun5  -0.002   -$fcuun5      -0.0035
uniaxialMaterial Concrete01  12  -$fcun6  -0.002   -$fcuun6      -0.0035

# STEEL
# Reinforcing steel
##set fy   600000
##set Es   220000000
#                        tag  fy      E          b
uniaxialMaterial Steel01 13    $fy1     $Es1        0.01
uniaxialMaterial Steel01 14    $fy2     $Es2        0.01
uniaxialMaterial Steel01 15    $fy3     $Es3        0.01
uniaxialMaterial Steel01 16    $fy4     $Es4        0.01
uniaxialMaterial Steel01 17    $fy5     $Es5        0.01
uniaxialMaterial Steel01 18    $fy6     $Es6        0.01

# Source in a procedure for generating an RC fiber section
source RCsection.tcl

# Call the procedure to generate the column section
set f1 [expr $reinf1/8]; # area of Ö ekastote bar
set f2 [expr $reinf2/8]; # area of Ö ekastote bar
set f3 [expr $reinf3/8]; # area of Ö ekastote bar
set f4 [expr $reinf4/8]; # area of Ö ekastote bar
set f5 [expr $reinf5/8]; # area of Ö ekastote bar
set f6 [expr $reinf6/8]; # area of Ö ekastote bar
#set f4 [expr 1.1385243E-03/8]; # area of Ö ekastote bar


# Concrete elastic stiffness
set E1 [expr 9500000*pow((($fc1/1000)+8),1./3.)];
set E2 [expr 9500000*pow((($fc2/1000)+8),1./3.)];
set E3 [expr 9500000*pow((($fc3/1000)+8),1./3.)];
set E4 [expr 9500000*pow((($fc4/1000)+8),1./3.)];
set E5 [expr 9500000*pow((($fc5/1000)+8),1./3.)];
set E6 [expr 9500000*pow((($fc6/1000)+8),1./3.)];


# Column torsional stiffness
set v 0.3;
set G1 [expr $E1/(2*(1+$v))];
set G2 [expr $E2/(2*(1+$v))];
set G3 [expr $E3/(2*(1+$v))];
set G4 [expr $E4/(2*(1+$v))];
set G5 [expr $E5/(2*(1+$v))];
set G6 [expr $E6/(2*(1+$v))];

set l [expr 0.30/0.6];
set c [expr 1-0.63*$l+0.052*pow($l,5)];
set J [expr $c*1/3*0.6*pow(0.30,3)];

if {$b1 <= $h1} {
 set l1 [expr $b1/$h1];
 set c1 [expr 1-0.63*$l1+0.052*pow($l1,5)];
 set J1 [expr $c1*1/3*$h1*pow($b1,3)];
} else {
 set l1 [expr $h1/$b1];
 set c1 [expr 1-0.63*$l1+0.052*pow($l1,5)];
 set J1 [expr $c1*1/3*$b1*pow($h1,3)];
};
set area1 [expr $h1*$b1];
set Izz1 [expr $h1*pow($b1,3)/12];
set Iyy1 [expr $b1*pow($h1,3)/12];

if {$b2 <= $h2} {
 set l2 [expr $b2/$h2];
 set c2 [expr 1-0.63*$l2+0.052*pow($l2,5)];
 set J2 [expr $c2*1/3*$h2*pow($b2,3)];
} else {
 set l2 [expr $h2/$b2];
 set c2 [expr 1-0.63*$l2+0.052*pow($l2,5)];
 set J2 [expr $c2*1/3*$b2*pow($h2,3)];
};
set area2 [expr $h2*$b2];
set Izz2 [expr $h2*pow($b2,3)/12];
set Iyy2 [expr $b2*pow($h2,3)/12];

if {$b3 <= $h3} {
 set l3 [expr $b3/$h3];
 set c3 [expr 1-0.63*$l3+0.052*pow($l3,5)];
 set J3 [expr $c3*1/3*$h3*pow($b3,3)];
} else {
 set l3 [expr $h3/$b3];
 set c3 [expr 1-0.63*$l3+0.052*pow($l3,5)];
 set J3 [expr $c3*1/33*$b3*pow($h3,3)];
};

set area3 [expr $h3*$b3];
set Izz3 [expr $h3*pow($b3,3)/12];
set Iyy3 [expr $b3*pow($h3,3)/12];

if {$b4 <= $h4} {
 set l4 [expr $b4/$h4];
 set c4 [expr 1-0.63*$l4+0.052*pow($l4,5)];
 set J4 [expr $c4*1/3*$h4*pow($b4,3)];
} else {
 set l4 [expr $h4/$b4];
 set c4 [expr 1-0.63*$l4+0.052*pow($l4,5)];
 set J4 [expr $c4*1/3*$b4*pow($h4,3)];
};

set area4 [expr $h4*$b4];
set Izz4 [expr $h4*pow($b4,3)/12];
set Iyy4 [expr $b4*pow($h4,3)/12];

if {$b5 <= $h5} {
 set l5 [expr $b5/$h5];
 set c5 [expr 1-0.63*$l5+0.052*pow($l5,5)];
 set J5 [expr $c5*1/3*$h5*pow($b5,3)];
} else {
 set l5 [expr $h5/$b5];
 set c5 [expr 1-0.63*$l5+0.052*pow($l5,5)];
 set J5 [expr $c5*1/3*$b5*pow($h5,3)];
};

set area5 [expr $h5*$b5];
set Izz5 [expr $h5*pow($b5,3)/12];
set Iyy5 [expr $b5*pow($h5,3)/12];

if {$b6 <= $h6} {
 set l6 [expr $b6/$h6];
 set c6 [expr 1-0.63*$l6+0.052*pow($l6,5)];
 set J6 [expr $c6*1/3*$h6*pow($b6,3)];
} else {
 set l6 [expr $h6/$b6];
 set c6 [expr 1-0.63*$l6+0.052*pow($l6,5)];
 set J6 [expr $c6*1/3*$b6*pow($h6,3)];
};

set area6 [expr $h6*$b6];
set Izz6 [expr $h6*pow($b6,3)/12];
set Iyy6 [expr $b6*pow($h6,3)/12];

if {$bx <= $hx} {
 set l7 [expr $bx/$hx];
 set c7 [expr 1-0.63*$l7+0.052*pow($l7,5)];
 set J7 [expr $c7*1/3*$hx*pow($bx,3)];
} else {
 set l7 [expr $hx/$bx];
 set c7 [expr 1-0.63*$l7+0.052*pow($l7,5)];
 set J7 [expr $c7*1/3*$bx*pow($hx,3)];
};

set area7 [expr $hx*$bx];
set Izz7 [expr $hx*pow($bx,3)/12];
set Iyy7 [expr $bx*pow($hx,3)/12];

if {$by <= $hy} {
 set l8 [expr $by/$hy];
 set c8 [expr 1-0.63*$l8+0.052*pow($l8,5)];
 set J8 [expr $c8*1/3*$hy*pow($by,3)];
} else {
 set l8 [expr $hy/$by];
 set c8 [expr 1-0.63*$l8+0.052*pow($l8,5)];
 set J8 [expr $c8*1/3*$by*pow($hy,3)];
};

set area8 [expr $hy*$by];
set Izz8 [expr $hy*pow($by,3)/12];
set Iyy8 [expr $by*pow($hy,3)/12];
set GJ1 [expr $G1*$J];
set GJ2 [expr $G2*$J];
set GJ3 [expr $G3*$J];
set GJ4 [expr $G4*$J];
set GJ5 [expr $G5*$J];
set GJ6 [expr $G6*$J];

# Linear elastic torsion for the column
#uniaxialMaterial Elastic 11 $GJ
uniaxialMaterial Elastic 111 [expr $G1*$J1]
uniaxialMaterial Elastic 211 [expr $G2*$J2]
uniaxialMaterial Elastic 311 [expr $G3*$J3]
uniaxialMaterial Elastic 411 [expr $G4*$J4]
uniaxialMaterial Elastic 511 [expr $G5*$J5]
uniaxialMaterial Elastic 611 [expr $G6*$J6]


# Column torsional stiffness
set v 0.3;
set G1 [expr $E1/(2*(1+$v))];
set G2 [expr $E2/(2*(1+$v))];
set G3 [expr $E3/(2*(1+$v))];
set G4 [expr $E4/(2*(1+$v))];
set G5 [expr $E5/(2*(1+$v))];
set G6 [expr $E6/(2*(1+$v))];

uniaxialMaterial Elastic 21 [expr $G1*$J1]
uniaxialMaterial Elastic 22 [expr $G2*$J2]
uniaxialMaterial Elastic 23 [expr $G3*$J3]
uniaxialMaterial Elastic 24 [expr $G4*$J4]
uniaxialMaterial Elastic 25 [expr $G5*$J5]
uniaxialMaterial Elastic 26 [expr $G6*$J6]

#         id  h  b   cover core  cover steel nBars area    nfCoreY nfCoreZ nfCoverY nfCoverZ
RCsection  21 $h1 $b1  0.025    1     7     13     3    $f1       8       8       10       10
RCsection  22 $h2 $b2  0.025    2     8     14     3    $f2       8       8       10       10
RCsection  23 $h3 $b3  0.025    3     9     15     3    $f3       8       8       10       10
RCsection  24 $h4 $b4  0.025    4     10    16     3    $f4       8       8       10       10
RCsection  25 $h5 $b5  0.025    5     11    17     3    $f5       8       8       10       10
RCsection  26 $h6 $b6  0.025    6     12    18     3    $f6       8       8       10       10
#RCsection  4 $b4 $d4  0.025    1     7     13     3    $f4       8       8       10       10



# Attach torsion to the RC column section
#                 tag uniTag uniCode       secTag
section Aggregator 6     111      T        -section 21
section Aggregator 7     211      T        -section 22
section Aggregator 8     311      T        -section 23
section Aggregator 9     411      T        -section 24
section Aggregator 10    511      T        -section 25
section Aggregator 11    611      T        -section 26

#

#_________________________________________________________________
#
#set colSec 2
# Define column elements
# ----------------------
#set PDelta "ON"
set PDelta "OFF"
# Geometric transformation for columns
if {$PDelta == "ON"} {
   #                           tag  vecxz
   geomTransf LinearWithPDelta  1   0 -1 0
} else {
   geomTransf Linear  1   0 -1 0
   geomTransf Linear  2   0 -1 0
   geomTransf Linear  3   0 -1 0
   geomTransf Linear  4   0 -1 0
   geomTransf Linear  5   0 -1 0
   geomTransf Linear  6   0 -1 0
}
# Number of column integration points (sections)
set np 4
# Create the nonlinear column elements
#                           tag ndI ndJ nPts   secID  transf
element	nonlinearBeamColumn	1	1	101	$np	6	1
element	nonlinearBeamColumn	2	2	102	$np	6	1
element	nonlinearBeamColumn	3	3	103	$np	7	1
element	nonlinearBeamColumn	4	4	104	$np	7	1
element	nonlinearBeamColumn	5	5	105	$np	6	1
element	nonlinearBeamColumn	6	6	106	$np	6	1

element	nonlinearBeamColumn	7	101	201	$np	6	2
element	nonlinearBeamColumn	8	102	202	$np	6	2
element	nonlinearBeamColumn	9	103	203	$np	7	2
element	nonlinearBeamColumn	10	104	204	$np	7	2
element	nonlinearBeamColumn	11	105	205	$np	6	2
element	nonlinearBeamColumn	12	106	206	$np	6	2

element	nonlinearBeamColumn	13	201	301	$np	8	3
element	nonlinearBeamColumn	14	202	302	$np	8	3
element	nonlinearBeamColumn	15	203	303	$np	9	3
element	nonlinearBeamColumn	16	204	304	$np	9	3
element	nonlinearBeamColumn	17	205	305	$np	8	3
element	nonlinearBeamColumn	18	206	306	$np	8	3

element	nonlinearBeamColumn	19	301	401	$np	8	4
element	nonlinearBeamColumn	20	302	402	$np	8	4
element	nonlinearBeamColumn	21	303	403	$np	9	4
element	nonlinearBeamColumn	22	304	404	$np	9	4
element	nonlinearBeamColumn	23	305	405	$np	8	4
element	nonlinearBeamColumn	24	306	406	$np	8	4

element	nonlinearBeamColumn	25	401	501	$np	10	5
element	nonlinearBeamColumn	26	402	502	$np	10	5
element	nonlinearBeamColumn	27	403	503	$np	11	5
element	nonlinearBeamColumn	28	404	504	$np	11	5
element	nonlinearBeamColumn	29	405	505	$np	10	5
element	nonlinearBeamColumn	30	406	506	$np	10	5

element	nonlinearBeamColumn	31	501	601	$np	10	6
element	nonlinearBeamColumn	32	502	602	$np	10	6
element	nonlinearBeamColumn	33	503	603	$np	11	6
element	nonlinearBeamColumn	34	504	604	$np	11	6
element	nonlinearBeamColumn	35	505	605	$np	10	6
element	nonlinearBeamColumn	36	506	606	$np	10	6

# Define beam elements
# --------------------
# Define material properties for elastic beams
# Using beam depth of 24 and width of 18
# --------------------------------------------
set Abeam1 [expr $bx*$hx];
# "Cracked" second moments of area
set Ibeamyy1 [expr 0.6*$bx*pow($hx,3)/12];
set Ibeamzz1 [expr 0.6*$hx*pow($bx,3)/12];
#
set Abeam2 [expr $by*$hy];
# "Cracked" second moments of area
set Ibeamyy2 [expr 0.6*$by*pow($hy,3)/12];
set Ibeamzz2 [expr 0.6*$hy*pow($by,3)/12];

# Define elastic section for beams
#               tag  E    A      Iz     0  Iy     G   J
section Elastic  221  $E1 $Abeam1 $Ibeamzz1 $Ibeamyy1 $G1 $J7
section Elastic  222  $E1 $Abeam2 $Ibeamzz2 $Ibeamyy2 $G1 $J8
section Elastic  223  $E2 $Abeam1 $Ibeamzz1 $Ibeamyy1 $G2 $J7
section Elastic  224  $E2 $Abeam2 $Ibeamzz2 $Ibeamyy2 $G2 $J8
section Elastic  225  $E3 $Abeam1 $Ibeamzz1 $Ibeamyy1 $G3 $J7
section Elastic  226  $E3 $Abeam2 $Ibeamzz2 $Ibeamyy2 $G3 $J8
section Elastic  227  $E4 $Abeam1 $Ibeamzz1 $Ibeamyy1 $G4 $J7
section Elastic  228  $E4 $Abeam2 $Ibeamzz2 $Ibeamyy2 $G4 $J8
section Elastic  229  $E5 $Abeam1 $Ibeamzz1 $Ibeamyy1 $G5 $J7
section Elastic  230  $E5 $Abeam2 $Ibeamzz2 $Ibeamyy2 $G5 $J8
section Elastic  231  $E6 $Abeam1 $Ibeamzz1 $Ibeamyy1 $G6 $J7
section Elastic  232  $E6 $Abeam2 $Ibeamzz2 $Ibeamyy2 $G6 $J8
set beamSec1 221
set beamSec2 222
set beamSec3 223
set beamSec4 224
set beamSec5 225
set beamSec6 226
set beamSec7 227
set beamSec8 228
set beamSec9 229
set beamSec10 230
set beamSec11 231
set beamSec12 232
# Geometric transformation for beams
#                tag  vecxz
geomTransf Linear 7   0  0 -1
geomTransf Linear 8   0  0 -1
geomTransf Linear 9   0  0 -1
geomTransf Linear 10   0  0 -1
geomTransf Linear 11   0  0 -1
geomTransf Linear 12   0  0 -1
# Number of beam integration points (sections)
set np 3
# Create the beam elements
#                           tag ndI ndJ  nPts    secID   transf
element	nonlinearBeamColumn	37	1011	1021	$np	$beamSec1	7
element	nonlinearBeamColumn	38	1031	1041	$np	$beamSec1	7
element	nonlinearBeamColumn	39	1051	1061	$np	$beamSec1	7
element	nonlinearBeamColumn	40	1012	1033	$np	$beamSec2	7
element	nonlinearBeamColumn	41	1032	1052	$np	$beamSec2	7
element	nonlinearBeamColumn	42	1022	1043	$np	$beamSec2	7
element	nonlinearBeamColumn	43	1042	1062	$np	$beamSec2	7

element	nonlinearBeamColumn	44	2011	2021	$np	$beamSec3	8
element	nonlinearBeamColumn	45	2031	2041	$np	$beamSec3	8
element	nonlinearBeamColumn	46	2051	2061	$np	$beamSec3	8
element	nonlinearBeamColumn	47	2012	2033	$np	$beamSec4	8
element	nonlinearBeamColumn	48	2032	2052	$np	$beamSec4	8
element	nonlinearBeamColumn	49	2022	2043	$np	$beamSec4	8
element	nonlinearBeamColumn	50	2042	2062	$np	$beamSec4	8

element	nonlinearBeamColumn	51	3011	3021	$np	$beamSec5	9
element	nonlinearBeamColumn	52	3031	3041	$np	$beamSec5	9
element	nonlinearBeamColumn	53	3051	3061	$np	$beamSec5	9
element	nonlinearBeamColumn	54	3012	3033	$np	$beamSec6	9
element	nonlinearBeamColumn	55	3032	3052	$np	$beamSec6	9
element	nonlinearBeamColumn	56	3022	3043	$np	$beamSec6	9
element	nonlinearBeamColumn	57	3042	3062	$np	$beamSec6	9

element	nonlinearBeamColumn	58	4011	4021	$np	$beamSec7	10
element	nonlinearBeamColumn	59	4031	4041	$np	$beamSec7	10
element	nonlinearBeamColumn	60	4051	4061	$np	$beamSec7	10
element	nonlinearBeamColumn	61	4012	4033	$np	$beamSec8	10
element	nonlinearBeamColumn	62	4032	4052	$np	$beamSec8	10
element	nonlinearBeamColumn	63	4022	4043	$np	$beamSec8	10
element	nonlinearBeamColumn	64	4042	4062	$np	$beamSec8	10

element	nonlinearBeamColumn	65	5011	5021	$np	$beamSec9	11
element	nonlinearBeamColumn	66	5031	5041	$np	$beamSec9	11
element	nonlinearBeamColumn	67	5051	5061	$np	$beamSec9	11
element	nonlinearBeamColumn	68	5012	5033	$np	$beamSec10	11
element	nonlinearBeamColumn	69	5032	5052	$np	$beamSec10	11
element	nonlinearBeamColumn	70	5022	5043	$np	$beamSec10	11
element	nonlinearBeamColumn	71	5042	5062	$np	$beamSec10	11

element	nonlinearBeamColumn	72	6011	6021	$np	$beamSec11	12
element	nonlinearBeamColumn	73	6031	6041	$np	$beamSec11	12
element	nonlinearBeamColumn	74	6051	6061	$np	$beamSec11	12
element	nonlinearBeamColumn	75	6012	6033	$np	$beamSec12	12
element	nonlinearBeamColumn	76	6032	6052	$np	$beamSec12	12
element	nonlinearBeamColumn	77	6022	6043	$np	$beamSec12	12
element	nonlinearBeamColumn	78	6042	6062	$np	$beamSec12	12


section Elastic  33  $E1 1. 1. 1. $G1 1.
section Elastic  34  $E2 1. 1. 1. $G2 1.
section Elastic  35  $E3 1. 1. 1. $G3 1.
section Elastic  36  $E4 1. 1. 1. $G4 1.
section Elastic  37  $E5 1. 1. 1. $G5 1.
section Elastic  38  $E6 1. 1. 1. $G6 1.

#Create Rigid Offsets							
element	nonlinearBeamColumn	79	101	1011	$np	33	7
element	nonlinearBeamColumn	80	101	1012	$np	33	7
element	nonlinearBeamColumn	81	102	1021	$np	33	7
element	nonlinearBeamColumn	82	102	1022	$np	33	7
element	nonlinearBeamColumn	83	103	1031	$np	33	7
element	nonlinearBeamColumn	84	103	1032	$np	33	7
element	nonlinearBeamColumn	85	103	1033	$np	33	7
element	nonlinearBeamColumn	86	104	1041	$np	33	7
element	nonlinearBeamColumn	87	104	1042	$np	33	7
element	nonlinearBeamColumn	88	104	1043	$np	33	7
element	nonlinearBeamColumn	89	105	1051	$np	33	7
element	nonlinearBeamColumn	90	105	1052	$np	33	7
element	nonlinearBeamColumn	91	106	1061	$np	33	7
element	nonlinearBeamColumn	92	106	1062	$np	33	7

element	nonlinearBeamColumn	93	201	2011	$np	34	8
element	nonlinearBeamColumn	94	201	2012	$np	34	8
element	nonlinearBeamColumn	95	202	2021	$np	34	8
element	nonlinearBeamColumn	96	202	2022	$np	34	8
element	nonlinearBeamColumn	97	203	2031	$np	34	8
element	nonlinearBeamColumn	98	203	2032	$np	34	8
element	nonlinearBeamColumn	99	203	2033	$np	34	8
element	nonlinearBeamColumn	100	204	2041	$np	34	8
element	nonlinearBeamColumn	101	204	2042	$np	34	8
element	nonlinearBeamColumn	102	204	2043	$np	34	8
element	nonlinearBeamColumn	103	205	2051	$np	34	8
element	nonlinearBeamColumn	104	205	2052	$np	34	8
element	nonlinearBeamColumn	105	206	2061	$np	34	8
element	nonlinearBeamColumn	106	206	2062	$np	34	8

element	nonlinearBeamColumn	107	301	3011	$np	35	9
element	nonlinearBeamColumn	108	301	3012	$np	35	9
element	nonlinearBeamColumn	109	302	3021	$np	35	9
element	nonlinearBeamColumn	110	302	3022	$np	35	9
element	nonlinearBeamColumn	111	303	3031	$np	35	9
element	nonlinearBeamColumn	112	303	3032	$np	35	9
element	nonlinearBeamColumn	113	303	3033	$np	35	9
element	nonlinearBeamColumn	114	304	3041	$np	35	9
element	nonlinearBeamColumn	115	304	3042	$np	35	9
element	nonlinearBeamColumn	116	304	3043	$np	35	9
element	nonlinearBeamColumn	117	305	3051	$np	35	9
element	nonlinearBeamColumn	118	305	3052	$np	35	9
element	nonlinearBeamColumn	119	306	3061	$np	33	9
element	nonlinearBeamColumn	120	306	3062	$np	35	9

element	nonlinearBeamColumn	121	401	4011	$np	36	10
element	nonlinearBeamColumn	122	401	4012	$np	36	10
element	nonlinearBeamColumn	123	402	4021	$np	36	10
element	nonlinearBeamColumn	124	402	4022	$np	36	10
element	nonlinearBeamColumn	125	403	4031	$np	36	10
element	nonlinearBeamColumn	126	403	4032	$np	36	10
element	nonlinearBeamColumn	127	403	4033	$np	36	10
element	nonlinearBeamColumn	128	404	4041	$np	36	10
element	nonlinearBeamColumn	129	404	4042	$np	36	10
element	nonlinearBeamColumn	130	404	4043	$np	36	10
element	nonlinearBeamColumn	131	405	4051	$np	36	10
element	nonlinearBeamColumn	132	405	4052	$np	36	10
element	nonlinearBeamColumn	133	406	4061	$np	36	10
element	nonlinearBeamColumn	134	406	4062	$np	36	10

element	nonlinearBeamColumn	135	501	5011	$np	37	11
element	nonlinearBeamColumn	136	501	5012	$np	37	11
element	nonlinearBeamColumn	137	502	5021	$np	37	11
element	nonlinearBeamColumn	138	502	5022	$np	37	11
element	nonlinearBeamColumn	139	503	5031	$np	37	11
element	nonlinearBeamColumn	140	503	5032	$np	37	11
element	nonlinearBeamColumn	141	503	5033	$np	37	11
element	nonlinearBeamColumn	142	504	5041	$np	37	11
element	nonlinearBeamColumn	143	504	5042	$np	37	11
element	nonlinearBeamColumn	144	504	5043	$np	37	11
element	nonlinearBeamColumn	145	505	5051	$np	37	11
element	nonlinearBeamColumn	146	505	5052	$np	37	11
element	nonlinearBeamColumn	147	506	5061	$np	37	11
element	nonlinearBeamColumn	148	506	5062	$np	37	11

element	nonlinearBeamColumn	149	601	6011	$np	38	12
element	nonlinearBeamColumn	150	601	6012	$np	38	12
element	nonlinearBeamColumn	151	602	6021	$np	38	12
element	nonlinearBeamColumn	152	602	6022	$np	38	12
element	nonlinearBeamColumn	153	603	6031	$np	38	12
element	nonlinearBeamColumn	154	603	6032	$np	38	12
element	nonlinearBeamColumn	155	603	6033	$np	38	12
element	nonlinearBeamColumn	156	604	6041	$np	38	12
element	nonlinearBeamColumn	157	604	6042	$np	38	12
element	nonlinearBeamColumn	158	604	6043	$np	38	12
element	nonlinearBeamColumn	159	605	6051	$np	38	12
element	nonlinearBeamColumn	160	605	6052	$np	38	12
element	nonlinearBeamColumn	161	606	6061	$np	38	12
element	nonlinearBeamColumn	162	606	6062	$np	38	12


# Define gravity loads
# --------------------
# Gravity load applied at each corner node
# 10% of column capacity
#  lumped at master nodes
set qf 2.0;
set gf 1.5;
set bf [expr 24*0.18];
set lod [expr ($bf+$gf)+0.3*$qf]; 
set p1 [expr $lod*8.3538*0.25];
set p2 [expr $lod*6.5885*0.25];
set p3 [expr $lod*14.4693*0.25];

#  lumped at master nodes
set g 9.81;            # Gravitational constant
set m [expr $lod*72/$g];
set mtot [expr 6*$m];
# Rotary inertia of floor about master node
#set i11 [expr $m1*(32*32+100)/12.0];
#set i22 [expr $m2*(64+100)/12.0];
#set i   [expr $i11+$i22+32*10*pow((pow((16-13.6),2.)+pow((15-13),2.)),0.5)+8*10*pow((pow((4-13.6),2.)+pow((5-13),2.)),0.5)];
set i [expr $m*(36+144)/12.0]
# Set mass at the master nodes
#    tag MX MY MZ RX RY RZ
mass  10001  $m $m  0  0  0 $i
mass  20001  $m $m  0  0  0 $i
mass  30001  $m $m  0  0  0 $i
mass  40001  $m $m  0  0  0 $i
mass  50001  $m $m  0  0  0 $i
mass  60001  $m $m  0  0  0 $i


# Define gravity loads
pattern Plain 1 Linear {                                           

eleLoad     -ele 37      -type -beamUniform   -[expr $p1]    0
eleLoad     -ele 38      -type -beamUniform   -[expr 2*$p3]    0
eleLoad     -ele 39      -type -beamUniform   -[expr $p1]    0
eleLoad     -ele 40      -type -beamUniform   -[expr $p2]    0
eleLoad     -ele 41      -type -beamUniform   -[expr $p2]    0
eleLoad     -ele 42      -type -beamUniform   -[expr $p2]    0
eleLoad     -ele 43      -type -beamUniform   -[expr $p2]    0

eleLoad     -ele 44      -type -beamUniform   -[expr $p1]    0
eleLoad     -ele 45      -type -beamUniform   -[expr 2*$p3]    0
eleLoad     -ele 46      -type -beamUniform   -[expr $p1]    0
eleLoad     -ele 47      -type -beamUniform   -[expr $p2]    0
eleLoad     -ele 48      -type -beamUniform   -[expr $p2]    0
eleLoad     -ele 49      -type -beamUniform   -[expr $p2]    0
eleLoad     -ele 50      -type -beamUniform   -[expr $p2]    0

eleLoad     -ele 51      -type -beamUniform   -[expr $p1]    0
eleLoad     -ele 52      -type -beamUniform   -[expr 2*$p3]    0
eleLoad     -ele 53      -type -beamUniform   -[expr $p1]    0
eleLoad     -ele 54      -type -beamUniform   -[expr $p2]    0
eleLoad     -ele 55      -type -beamUniform   -[expr $p2]    0
eleLoad     -ele 56      -type -beamUniform   -[expr $p2]    0
eleLoad     -ele 57      -type -beamUniform   -[expr $p2]    0

eleLoad     -ele 58      -type -beamUniform   -[expr $p1]    0
eleLoad     -ele 59      -type -beamUniform   -[expr 2*$p3]    0
eleLoad     -ele 60      -type -beamUniform   -[expr $p1]    0
eleLoad     -ele 61      -type -beamUniform   -[expr $p2]    0
eleLoad     -ele 62      -type -beamUniform   -[expr $p2]    0
eleLoad     -ele 63      -type -beamUniform   -[expr $p2]    0
eleLoad     -ele 64      -type -beamUniform   -[expr $p2]    0

eleLoad     -ele 65      -type -beamUniform   -[expr $p1]    0
eleLoad     -ele 66      -type -beamUniform   -[expr 2*$p3]    0
eleLoad     -ele 67      -type -beamUniform   -[expr $p1]    0
eleLoad     -ele 68      -type -beamUniform   -[expr $p2]    0
eleLoad     -ele 69      -type -beamUniform   -[expr $p2]    0
eleLoad     -ele 70      -type -beamUniform   -[expr $p2]    0
eleLoad     -ele 71      -type -beamUniform   -[expr $p2]    0

eleLoad     -ele 72      -type -beamUniform   -[expr $p1]    0
eleLoad     -ele 73      -type -beamUniform   -[expr 2*$p3]    0
eleLoad     -ele 74      -type -beamUniform   -[expr $p1]    0
eleLoad     -ele 75      -type -beamUniform   -[expr $p2]    0
eleLoad     -ele 76      -type -beamUniform   -[expr $p2]    0
eleLoad     -ele 77      -type -beamUniform   -[expr $p2]    0
eleLoad     -ele 78      -type -beamUniform   -[expr $p2]    0
                                                                                  
}
# -----------------------
# End of model generation
# -----------------------
# ------------------------------
# Start of analysis generation
# ------------------------------
# Create the system of equation, a sparse solver with partial pivoting
system BandGeneral
# Create the constraint handler, the transformation method
constraints Transformation
# Create the DOF numberer, the reverse Cuthill-McKee algorithm
numberer RCM
# Create the convergence test, the norm of the residual with a tolerance of
# 1e-12 and a max number of iterations of 10
#test NormDispIncr 1.0e-6  200 0
test NormUnbalance 1.0e-4  50 5
# Create the solution algorithm, a Newton-Raphson algorithm
algorithm Newton
# Create the integration scheme, the LoadControl scheme using steps of 0.1
integrator LoadControl 0.1
# Create the analysis object
analysis Static
# initialize in case we need to do an initial stiffness iteration
initialize
# ------------------------------
# End of analysis generation
# ------------------------------

analyze 5


# Set the gravity loads to be constant & reset the time in the domain
loadConst -time 0.0
# Define lateral loads
# --------------------

set sa10 2000.;
set V1x [expr ($sa10*$z1)/($ztot)];             # Reference lateral load
set V2x [expr ($sa10*$z2)/($ztot)];
set V3x [expr ($sa10*$z3)/($ztot)];
set V4x [expr ($sa10*$z4)/($ztot)];
set V5x [expr ($sa10*$z5)/($ztot)];
set V6x [expr ($sa10*$z6)/($ztot)];
set V1y [expr 0.0*$V1x];
set V2y [expr 0.0*$V2x];
set V3y [expr 0.0*$V3x];
set V4y [expr 0.0*$V4x];
set V5y [expr 0.0*$V5x];
set V6y [expr 0.0*$V6x];

puts stdout Orizodiafortia
puts stdout $V1x;
puts stdout $V2x;
puts stdout $V3x;
puts stdout $V4x;
puts stdout $V5x;
puts stdout $V6x;

# Set lateral load pattern with a Linear TimeSeries
pattern Plain 6 "Linear" {
        load 10001 $V1x $V1y 0.0 0.0 0.0 0.0
        load 20001 $V2x $V2y 0.0 0.0 0.0 0.0
        load 30001 $V3x $V3y 0.0 0.0 0.0 0.0
        load 40001 $V4x $V4y 0.0 0.0 0.0 0.0
        load 50001 $V5x $V5y 0.0 0.0 0.0 0.0
        load 60001 $V6x $V6y 0.0 0.0 0.0 0.0
}

#recorder Drift dr-1-101x.out 1 101 1 3
#recorder Drift dr-1-101y.out 1 101 2 3
#recorder Drift dr-1-601x.out 1 601 1 3
#recorder Drift dr-1-601y.out 1 601 2 3
#recorder Drift dr-101-201x.out 101 201 1 3
#recorder Drift dr-101-201y.out 101 201 2 3
#recorder Drift dr-201-301x.out 201 301 1 3
#recorder Drift dr-201-301y.out 201 301 2 3
#recorder Drift dr-301-401x.out 301 401 1 3
#recorder Drift dr-301-401y.out 301 401 2 3
#recorder Drift dr-401-501x.out 401 501 1 3
#recorder Drift dr-401-501y.out 401 501 2 3
#recorder Drift dr-501-601x.out 501 601 1 3
#recorder Drift dr-501-601y.out 501 601 2 3

recorder Node -file node20001.out -time -node 60001 -dof 1 disp

# ------------------------------
# Finally perform the analysis
# ------------------------------

# Set some parameters
set dU50 0.2;
set dU [expr $dU50/5];         # Displacement increment

set dU2 0.5;
set maxU [expr $dU2*1.5];

set GravSteps 5;
integrator LoadControl 1.0 4 0.02 2.0
integrator LoadControl [expr 1./$GravSteps] 1 [expr 1./$GravSteps] [expr 1./$GravSteps]

set maxU 0.5;          # Max displacement
set controlDisp 0.0;
set ok 0;
set k2 0;

while {$controlDisp < $maxU && $ok == 0} {
    set k2 [expr $k2+1]
    set ok [analyze 1]
    set controlDisp [nodeDisp 60001 1]
    if {$ok != 0} {
        puts "... trying an initial tangent iteration"
        test NormDispIncr 1.0e-8  150 5
        algorithm ModifiedNewton -initial
        set ok [analyze 1]
        test NormDispIncr 1.0e-8  150 5
        algorithm Newton
    }
}

set fileId [open ides w 0600]
puts $fileId $k2;
close $fileId
puts stdout k2--------------;
puts stdout $k2;

if {$ok != 0} {
    puts "Pushover analysis FAILED"
	set fileId [open fail w 0600]
	puts $fileId 1;
	close $fileId
} else {
    puts "Pushover analysis completed SUCCESSFULLY.
                            WELL DONE!!!!"
	set fileId [open fail w 0600]
	puts $fileId 0;
	close $fileId
}

