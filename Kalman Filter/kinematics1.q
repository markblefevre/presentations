/ create_stm: Create a State Transition Model for a projectile in 2D
/ [d]elta [t]ime is the change in time for each time step
create_stm:{[dt]
    4 4#@[16#0.;*[5;til 4],2 7;:;#[4;1.],#[2;dt]]
    };

/ create_cim: Create a Control-Input Model for a projectile in 2D
/   The CIM accounts for the effect of acceleration on position and veolcity
create_cim:{[dt]
    4 2#@[8#0.;0 3 4 7;:;raze 2#'(0.5*dt*dt;dt)]
    };

/ create_initial_state: Create an initial state
/ [t]heta, [v]elocity
/ theta is the angle at which projectile is fired in degrees
/ velocity is the initial velocity at which projectile is fired
create_initial_state:{[t;v]
    pi:acos -1;
    r:%[t;180]*pi;
    4 1#@[4#0.;2 3;:;(v*cos r;v*sin r)]
    };

theta:45;
v0:300;
g:-9.8;
tof:(2*v0*sin acos[-1]*theta%180)%neg g;
nsteps:1000;
dt:tof%nsteps;
/dt:0.1;
stm:create_stm[dt]; 
cim:create_cim[dt];
is:create_initial_state[45;v0];
results:{flip `x`y`vx`vy!flip x} raze each 
    nsteps {[stm;cim;x] (stm$x)+cim$2 1#0 -9.8}[stm;cim;]\ is;
results:`t xcols update t:0+dt*til 1+nsteps from results;
/ Kalman Filter Matrices/Helper Functions
create_Q:{[dt;v] G:4 1#raze 2#'(0.5*dt*dt;dt); Q:*[v;v]*G$flip G};
create_P:{4 4#16#x,4#0.};
create_H:{2 4#rotate[-2;8#x,4#0.]};
create_R:{2 2#4#x,2#0.};
create_I:{4 4#16#1.,4#0.};
Q:create_Q[dt;1.];
P0:create_P[10.];
H:create_H[1.];
ra:5.0;
R:create_R[ra*ra];
I:create_I[];
u:2 1#0.,g;

/ Predict
predState:{[A;x;B;u] (A$x)+B$u};
predErrCov:{[A;P;Q] Q+(A$P)$flip A};
predict:{[A;x;B;u;P;Q] (predState[A;x;B;u];predErrCov[A;P;Q])};
/ Create a projection (only x (state variables) and P (error covariance matrix) change each iteration)
predict_p:predict[stm;;cim;u;;Q];
/predState[stm;is;cim;2 1#0 -9.8]
/predErrCov[stm;P;Q]

/ Correct
computeKalmanGain:{[P;H;R] (P$flip H)$inv R+(H$P)$flip H};
updateEstimate:{[prevxhat;K;z;H] prevxhat+K$z-H$prevxhat};
updateErrCov:{[I;K;H;P] (I-K$H)$P};
correct:{[prevP;H;R;prevxhat;Z;I]
    K:computeKalmanGain[prevP;H;R];
    xhat:updateEstimate[prevxhat;K;Z;H];
    P:updateErrCov[I;K;H;prevP];
    (K;xhat;P)
    };
/ Create a projection (only prevP (a priori error covariance estimate),
/     prevxhat (a priori state estimate), Z (new measurement) change each iteration)
correct_p:correct[;H;R;;;I];

kf:{[x;P;Z]
    aposteriori:correct_p[;;Z] . reverse apriori:predict_p . (x;P);
    apriori,aposteriori
    };

/ Create noisy measurements
/ Use Nick Psaris' bm implementation for normal variates
\cd C:\\Users\\Mark\\Documents\\Presentations\\Kalman Filter
\l stat.q
update vxn:vx+sqrt[R[0;0]]*.stat.bm nsteps?1f,vyn:vy+sqrt[R[1;1]]*.stat.bm nsteps?1f from `results where i>0;

history:();
x:is;
x:4 1#raze 0 100f,-2#create_initial_state[55;1.75*v0];
x:create_initial_state[55;1.75*v0];
P:P0;
measurements:select vxn,vyn from results where i>0;
while[0<count measurements;
    Z:2 1#value measurements[0];
    history:history,enlist res:kf[x;P;Z];
    measurements:1_measurements;
    x:res 3; P:res 4;
    ];
flip `x`y`vx`vy!flip {raze x} each history[;3]


/ Some q/kdb+ code for Excel presentation
headers:$[`;("theta (d)";"theta (r)";"v0";"g";"v0x";"v0y";"tof";"nsteps";"dt")]!
    (theta;acos[-1]*theta%180;v0;g;first create_initial_state[theta;v0] 2;first create_initial_state[theta;v0] 3;tof;nsteps;dt);
results;
predictions:flip `x`y`vx`vy!flip {raze x} each history[;0];
corrections:flip `x`y`vx`vy!flip {raze x} each history[;3];
kalman_gains:flip `x`y`vx`vy!flip {raze x[;0]} each history[;2];
raze create_initial_state[55;1.75*v0]