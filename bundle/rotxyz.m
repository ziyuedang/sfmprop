function rot = rotxyz( omega, phi, kappa )
rot = [cos(kappa)*cos(phi)  cos(kappa)*sin(phi)*sin(omega)+sin(kappa)*cos(omega) (((-cos(kappa))*sin(phi)*cos(omega))+(sin(kappa)*sin(omega)))
    -sin(kappa)*cos(phi)     ((-sin(kappa))*sin(phi)*sin(omega))+(cos(kappa)*cos(omega)) sin(kappa)*sin(phi)*cos(omega)+cos(kappa)*sin(omega)
     sin(phi)              -cos(phi)*sin(omega)                                                     cos(phi)*cos(omega)];

end