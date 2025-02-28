system "l stat.q";
/ For reproducibility, set random seed
system "S -314159";

/ Constants
g:9.81;  / gravity m/s^2

/ Time steps
dt:0.1;
/ [b]egin; [e]nd; [s]tep
arange:{[b;e;s] b+s*til "j"$-[e;b]%s};
times:arange[0;10;dt];

/ Initial conditions
x0:0; y0:0; vx0:50; vy0:50; / pos in m; vel in m/s

/ Actual trajectory
atraj:{[g;x0;y0;vx0;vy0;t]
  x:x0+vx0*t;               / x pos
  y:-[y0+vy0*t;0.5*g*t*t];  / y pos y0+vy0*t-0.5*g*t^2
  vx:vx0;                   / x vel (no air resistance)
  vy:vy0-g*t;               / y vel (gravity but no air res)
  (x;y;vx;vy)};
act_traj:atraj[g;x0;y0;vx0;vy0;];

/ Simulate actual trajectory
true_traj:act_traj[times];  / use the projection

/ Simulate measurements
pos_meas_noise:1.0;  / standard dev of position measurements
vel_meas_noise:0.5;  / standard dev of velocity measurements

/ Use Nick Psaris Box Muller algorithm for 
/ generating pairs of independent, standard normally distributed
/ random variables from uniformly distributed random variables
meas_traj:true_traj+
  *[(pos_meas_noise;vel_meas_noise)0 0 1 1;]  / scale std dev
  (4 0N)#.stat.bm*[4;count times]?1f;         / measurement errors

id:{(2#x)#1,x#0};  / Identity matrix from qphrasebook
/ Kalman Filter
/ Initialization
/ State vector [x, y, vx, vy]
x_est:(4;count times)#0f;
P:"f"$id 4;       / State Covariance matrix
R:*[(pos_meas_noise;vel_meas_noise)0 0 1 1;]
  "f"$id 4;       / Measurement Noise Covariance
Q:0.005*"f"$id 4; / Process Noise Covariance
F:"f"$id 4;       / State Transition Matrix
F:(@[F 0;2;+;dt];@[F 1;3;+;dt];F 2;F 3);
H:"f"$id 4;       / Measurement Matrix (pos and vel)
G:(4;4;count times)#0f;

/ KF Prediction Step
// Predict next state using prev state and the F, state transition matrix
// Predict state uncertainty (update P, covariance matrix using Q, process noise
predictState:{[F;ps;P;Q]
  ns:F$ps;                / predicted next state
  nP:$[F;$[P;flip F]]+Q;  / predicted next P
  (ns;nP)
  };

// Kalman Gain (K) weight of measurement vs prediction
// done in the update step
computeKalmanGain:{[H;pP;R]
  S:$[H;$[pP;flip H]]+R;   / intermediate matrix for inversion
  K:$[pP;$[flip H;inv S]]  / inv uses LU decomposition
  };

/ KF Update Step
// Update corrects predicted state with new measurement
// Kalman Gain (K) adjusts predicted state based on the measurement residual
// Update state estimate and uncertainty (covariance)
updateStateEstimate:{[H;pP;R;ms;ps]
  id:{(2#x)#1,x#0};
  I:"f"$id 4;    / Identity Matrix
  K:computeKalmanGain[H;pP;R];

  y:ms-$[H;ps];  / measurement residuals
  es:ps+$[K;y];  / new state estimat

  P:$[I-$[K;H];pP];
  (K;es;P)
  };

idx:0;
res:();
ps:(); pP:();
es:(); cP:();
while[idx<-1+count times;
  idx+:1;
  res:predictState[F;x_est[;idx-1];P;Q];
  ps:res 0; pP:P:res 1;
  res:updateStateEstimate[H;pP;R;meas_traj[;idx];ps];
  G[;;idx]:res 0; x_est[;idx]:res 1; P:res 2;
  ];

// Create table(s) for visualization
tms:flip ![enlist `time;enlist times];
est:flip ![`x`y`xv`yv;x_est];
true:flip `truex`truey`truexv`trueyv!true_traj;
meas:flip `measx`measy`measxv`measyv!meas_traj;
kgains:flip `gainx`gainy`gainxv`gainyv!{G[x;x;::]} each til 4;
data:(,')over(tms;est;true;meas;kgains);
save `:./data.csv;

