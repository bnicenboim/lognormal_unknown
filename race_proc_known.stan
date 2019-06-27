functions {
  // Stable approximation of the cdf that works better close to 0 and 1
  real lognormal_approx_cdf(real x, real mu, real sigma){
    real x_n = (log(x)-mu)/sigma;
    return inv_logit(0.07056 * x_n^3 + 1.5976* x_n);
  }
}
data {
  int N;
  vector<lower=0>[N] RT;
  vector[N] X;
  vector[N] trial;
  int winner_proc[N];
}
parameters {
 real<lower=0> sigma;
  real Int_proc;
  real<lower= Int_proc> Int_to;
  real slope;
  real<lower=0> r;
}
transformed parameters{
  vector[N] mu_proc= Int_proc + X* slope;
  vector[N] mu_to = Int_to  - r* (trial/1000);
  vector[N] P_proc;
    //  given:
    // X ~ lognormal(mu_proc, sigma)
    // Y ~ lognormal(mu_to, sigma)
    // then:
    // X/Y ~ lognormal(mu_proc - mu_to, sqrt(2.0)* sigma)
    // X/Y < 1 => X was faster and was selected
    // P(X < Y) = lognormal_cdf(1, mu_proc - mu_to, sqrt(2.0)* sigma)
  for(n in 1:N)
     P_proc[n] = lognormal_approx_cdf(1.0,  mu_proc[n] - mu_to[n], sqrt(2.0)*sigma);
  //P_proc[n] = lognormal_cdf(1.0,  mu_proc[n] - mu_to[n], sqrt(2.0)*sigma);
}
model {
  // Quite informative priors:
  Int_proc~ normal(6,1);
  Int_to ~ normal(log(1000),1);
  slope ~ normal(.1,.3);
  r ~ lognormal(log(.5),.2);
  sigma ~ normal(.5,.5);
  // Likelihood:
  for (n in 1:N){
    if(winner_proc[n]){
      target +=  lognormal_lpdf(RT[n]|mu_proc[n],sigma);
      target +=lognormal_lccdf(RT[n]|mu_to[n],sigma);
    } else {
      target += lognormal_lpdf(RT[n]|mu_to[n],sigma);
      target +=lognormal_lccdf(RT[n]|mu_proc[n],sigma); 
    }
  }
}
generated quantities {
  real P_proc_avg = mean(P_proc);
  real RT_proc[N];
  real RT_to[N];
  real RT_gen[N];
  int winner_proc_gen[N];
  for(n in 1:N){
  RT_proc[n] = lognormal_rng(mu_proc[n],sigma);
  RT_to[n] = lognormal_rng(mu_to[n],sigma);
  // Generate fake data
  if(RT_proc[n] < RT_to[n]){
      winner_proc_gen[n] = 1;
      RT_gen[n] = RT_proc[n];
    } else {
      winner_proc_gen[n] = 0;
      RT_gen[n] = RT_to[n];
    }
  }
}
