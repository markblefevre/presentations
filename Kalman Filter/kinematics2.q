/ create_stm: Create a state transition model for a projectile in 2D
/ [d]elta [t]ime is the change in time for each time step
create_stm:{[dt]
    4 4#@[16#0.;*[5;til 4],2 7;:;#[4;1.],#[2;dt]]
    };

/ create_cim: Create a control-input model for a projectile in 2D
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

stm:create_stm[0.1];
cim:create_cim[0.1];
is:create_initial_state[45;30];
results1:{flip `x`y!flip x[;til 2]} raze each 
    1000 {[stm;cim;x] (stm$x)+cim$2 1#0 -9.8}[stm;cim;]\ is;
