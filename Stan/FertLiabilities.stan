data {
  int<lower=1> N;
  int<lower=1> N_animals;

  // Binary outcome
  int<lower=0, upper=1> BinPheno[N];

  // IDs
  int<lower=1, upper=N_animals> dam_id[N];
  int<lower=1, upper=N_animals> sire_id[N];

  // Year
  int<lower=1> K_year;
  int<lower=1, upper=K_year> year_id[N];

  // PE and residual
  int<lower=1> n_mpe;
  int<lower=1> n_e_proxy;

  int<lower=1, upper=n_mpe>     mpe_id[N];
  int<lower=1, upper=n_e_proxy> e_proxy_id[N];

  // Amatrix
  matrix[N_animals, N_animals] A;
}

transformed data {
  matrix[N_animals, N_animals] A_jit = A;
  for (i in 1:N_animals) A_jit[i,i] += 1e-9;  // jitter
  matrix[N_animals, N_animals] L_A = cholesky_decompose(A_jit);
}

parameters {
  // Intercepts
  real alpha_female;
  real alpha_male;

  // Years
  vector[K_year] z_year_female;
  vector[K_year] z_year_male;
  real<lower=0> sigma_year_female;
  real<lower=0> sigma_year_male;

  // bivar
  matrix[N_animals, 2] Z_a;
  cholesky_factor_corr[2] L_Ga;   // gen correlation
  vector<lower=0>[2] tau_a;       // female, male

  // PPE
  vector[n_mpe] z_mpe;
  real<lower=0> sigma_mpe;

  // residual
  matrix[2, n_e_proxy] Z_e;
  cholesky_factor_corr[2] L_Omega_e;
  vector<lower=0>[2] tau_e;
}

transformed parameters {
  // Year RE
  vector[K_year] year_female = sigma_year_female * z_year_female;
  vector[K_year] year_male   = sigma_year_male   * z_year_male;

  // bivar
  matrix[2,2] L_Sigma_a = diag_pre_multiply(tau_a, L_Ga);
  matrix[N_animals, 2] a_animal = L_A * Z_a * L_Sigma_a';
  vector[N_animals] a_female = a_animal[,1];
  vector[N_animals] a_male   = a_animal[,2];

  // PE
  vector[n_mpe] a_mpe = sigma_mpe * z_mpe;

  // Correlated res
  matrix[2, n_e_proxy] E = diag_pre_multiply(tau_e, L_Omega_e) * Z_e;
  vector[n_e_proxy] a_e_proxy_female = (E[1])';
  vector[n_e_proxy] a_e_proxy_male   = (E[2])';

  // Model
  vector[N] female_lin;
  vector[N] male_lin;
  vector<lower=0,upper=1>[N] p;

  for (i in 1:N) {
    female_lin[i] =
      alpha_female +
      year_female[ year_id[i] ] +
      a_female[   dam_id[i] ] +
      a_e_proxy_female[ e_proxy_id[i] ];

    male_lin[i] =
      alpha_male +
      year_male[ year_id[i] ] +
      a_male[ sire_id[i] ] +
      a_mpe[  mpe_id[i] ] +
      a_e_proxy_male[ e_proxy_id[i] ];

    // Probit-links
    p[i] = Phi_approx(female_lin[i]) * Phi_approx(male_lin[i]);
  }
}

model {
  // Priors
  alpha_female ~ normal(0, 0.7);
  alpha_male   ~ normal(0, 0.7);

  // Year RE scales
  sigma_year_female ~ normal(0, 1);
  sigma_year_male   ~ normal(0, 1);
  z_year_female ~ std_normal();
  z_year_male   ~ std_normal();

  // Bivar
  tau_a[1] ~ normal(0, 1);    // female
  tau_a[2] ~ normal(0, 1);    // male
  L_Ga     ~ lkj_corr_cholesky(2);
  to_vector(Z_a) ~ std_normal();

  // PE
  sigma_mpe ~ normal(0, 1);
  z_mpe ~ std_normal();

  // residual
  tau_e      ~ normal(0, 1);
  L_Omega_e  ~ lkj_corr_cholesky(2);
  to_vector(Z_e) ~ std_normal();

  // Bernoulli
  BinPheno ~ bernoulli(p);
}

generated quantities {
  vector[N] log_lik;
  int BinPheno_rep[N];

  // Genetic corr
  matrix[2,2] Omega_a = multiply_lower_tri_self_transpose(L_Ga); // gencorr matrix
  matrix[2,2] Ga      = multiply_lower_tri_self_transpose(diag_pre_multiply(tau_a, L_Ga));
  real rho_A          = Omega_a[1,2];
  real CovA_fm        = Ga[1,2];

  real Va_female = Ga[1,1];
  real Va_male   = Ga[2,2];

  real Vp_female_liab = Va_female
                        + square(tau_e[1])
                        + square(sigma_year_female)
                        + 1;
  real Vp_male_liab   = Va_male
                        + square(sigma_mpe)
                        + square(tau_e[2])
                        + square(sigma_year_male)
                        + 1;

  real h2_female_liab = Va_female / Vp_female_liab;
  real h2_male_liab   = Va_male   / Vp_male_liab;

  matrix[2,2] Omega_e = multiply_lower_tri_self_transpose(L_Omega_e);
  real rho_e = Omega_e[1,2];

  // log-lik
  for (i in 1:N) {
    log_lik[i]     = bernoulli_lpmf(BinPheno[i] | p[i]);
    BinPheno_rep[i] = bernoulli_rng(p[i]);
  }
}
