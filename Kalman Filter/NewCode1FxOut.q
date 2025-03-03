/system "dir *.csv"
tab1:1!("DF";enlist csv) 0: `:EURUSD.csv;
tab2:1!("DF";enlist csv) 0: `:USDJPY.csv;
fxrates:tab1 lj tab2;
/ Calculate the spread
spread:value (-/) each fxrates;
/ Need correlation for pairs trading to work well
value[tab1][last cols tab1] cor value[tab2][last cols tab2];

/ Kalman Filter
/ Initialization
x_est:(2;count fxrates)#0f;
P:"f"$1000*id 2;     / State Covariance matrix
R:"f"$1 1#5;         / Measurement Noise Covariance
Q:"f"$2 2#.001;      / Process Noise Covariance
F:"f"$2 2#1 1 0 1;   / State Transition Matrix
H:"f"$1 2#1 0;       / Measurement Matrix (spread)
G:(2;1;count fxrates)#0f;

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
  I:"f"$id 2;    / Identity Matrix
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
while[idx<-1+count fxrates;
  idx+:1;
  show idx;
  res:predictState[F;x_est[;idx-1];P;Q];
  ps:res 0; pP:P:res 1;
  res:updateStateEstimate[H;pP;R;spread[idx];ps];
  G[;;idx]:res 0; x_est[;idx]:res 1; P:res 2;
  ];

fx_out:![0;fxrates],'flip `spread`velocity!x_est;
save `:fx_out.csv;
