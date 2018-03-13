param T := 3 ;
param rf = 0.03 ;
param R = 3 ; 
param  u{1..R} ;
param r{1..R,1..T} ;
param p{1..T };
var x{1..R};
var v0 ;
var z ;
var v ;
var d{1..T};
var y{1..T} ;


maximize fun: v + rf * v0 ;

subject to
 c1: sum{i in 1..R}x[i]= v0;
 
 c2: sum{i in 1..R}x[i]*u[i]=v ;
 
 c3{t in 1..T}: sum{i in 1..R}r[i,t]*x[i]=y[t] ;
 
 c4: sum{t in 1..T}d[t]*p[t]=1 ;
 
 c5{t in 1..T}: d[t] >= rf*v0 - y[t];
 
 c6{t in 1..T}: d[t] >= 0 ;
 
 c7: v0 <= 10000 ;
 
 c8{i in 1..R}: x[i]>=0 ;
 


data;
 param u := 1 0.1 2 0.1 3 0.2 ;
 
 param p := 1 0.34  2 0.33 3 0.33 ;
 
 param r: 1  2  3 :=
      1  0.1 0.1 0.1
      2  0.1 0.1 0.1
      3  0.2 0.2 0.2 ;

