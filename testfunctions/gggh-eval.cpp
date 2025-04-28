/* This program reads input line by line in the format:
 *     beta2,theta_t
 *
 * and prints out the corresponding GoSam amplitudes in the format:
 *     weight,virtual,virtual single pole,virtual double pole
 *
 * The weight is the phase-space density in the angular variable.
 */

// pt_j = pf*sin_theta_h

#define _POSIX_C_SOURCE 200809L
#define _USE_MATH_DEFINES

#include <sys/mman.h>
#include <sys/wait.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#define countof(array) (sizeof(array)/sizeof(*(array)))
#define M_PI 3.14159265358979323846

extern "C" {
    void olp_option_(const char* line,int* stat);
    void olp_start_(const char* contract_file_name,int* ierr,int* stage,int* rndseed);
    void olp_evalsubprocess_(int* label, double* momenta, double* mu, double* parameters, double* res);
    void olp_evalsubprocess2_(int* label, double* momenta, double* mu, double* parameters, double* res, double* acc);
    void olp_evalbornprocess_(int* label, double* momenta, double* parameters, double* res);
};

static double sqr(double x) { return x*x; }
//static double range(double a, double b, double x) { return a + x*(b - a); }

double kallen(double a, double sqrt_b, double sqrt_c) {
    return (a-sqr(sqrt_b+sqrt_c))*(a-sqr(sqrt_b-sqrt_c));
}

double sp(double a[4], double b[4]) {
    return a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3];
}

void gosam_gg_1l(double *amp, double qg1[4], double qg2[4], double qj1[4], double qH[4], double mt, double mh, double muren) {
    int label = 0;
    double momenta[50];
    double params[10];
    double res[3] = {NAN, NAN, NAN};
    double accuracy = NAN;
    memcpy(&momenta[5*0], &qg1[0], 4*sizeof(double)); momenta[5*0+4] = 0;
    memcpy(&momenta[5*1], &qg2[0], 4*sizeof(double)); momenta[5*1+4] = 0;
    memcpy(&momenta[5*2], &qj1[0], 4*sizeof(double)); momenta[5*2+4] = 0;
    memcpy(&momenta[5*3], &qH[0],  4*sizeof(double)); momenta[5*3+4] = mh;
    olp_evalsubprocess2_(&label, momenta, &muren, params, res, &accuracy);
    // If the precision after the gosam rescue system is less than 3 digits, discard the point
    if (accuracy < 1e-3) {
        amp[0] = res[2]; // eps^0
        amp[1] = res[1]; // eps^-1
        amp[2] = res[0]; // eps^-2
    }
    else {
        amp[0] = amp[1] = amp[2] = NAN;
    }
}

void usage(const char *prog) {
    fprintf(stderr, "Usage: %s [options ...]\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -B\n");
    fprintf(stderr, "     input and output in raw binary float64 sequences\n");
}

int main(int argc, char *argv[]) {
    bool binary_io = false;
    for (int opt; (opt = getopt(argc, argv, "hB")) != -1; ) {
        switch (opt) {
            case 'B': binary_io = true; break;
            default: usage(argv[0]); return 1;
        }
    }
    if (argc - optind != 0) {
        usage(argv[0]);
        return 1;
    }
    // Multiply by:
    // o  1/gHT^2 (removes top Yukawa)
    // o  1/(8*pi^2)^2 (corrects loop factor)
    double scaling = 1./(-2.2720136689528667)*1./(-2.2720136689528667)*1./(8.*M_PI*M_PI)*1./(8.*M_PI*M_PI);
    // Init OLP
    int err = 0;
    int parstage = 1;
    int rndseed = 1 + (rand() % 100);
    //olp_option_("gs=1", &err); // set to 1 on runcard
    //if (err != 1) { fprintf(stderr, "olp_option() error %d\n", err); exit(1); }
    olp_option_("nlo_prefactors=0", &err);
    if (err != 1) { fprintf(stderr, "olp_option() error %d\n", err); exit(1); }
    olp_option_("mH=0.7223151185146152", &err); // sqrt(12/23)*top_mass
    if (err != 1) { fprintf(stderr, "olp_option() error %d\n", err); exit(1); }
    olp_option_("mT=1", &err);
    if (err != 1) { fprintf(stderr, "olp_option() error %d\n", err); exit(1); }
    // Prevent the pollution of stdout
    FILE *fout = fdopen(dup(1), "wb");
    fclose(stdout);
    olp_start_("dummy", &err, &parstage, &rndseed);
    //if (err != 1) { fprintf(stderr, "olp_start() error %d\n", err); exit(1); }
    const double top_mass = 1;
    const double higgs_mass = 0.7223151185146152; // sqrt(12/23)*top_mass
    const double top_mass2 = sqr(top_mass);
    const double higgs_mass2 = sqr(higgs_mass);
    for (;;) {
        double beta2 = 0, theta_h = 0;
        if (binary_io) {
            double params[2];
            if (fread(&params[0], sizeof(double), 2, stdin) != 2) break;
            beta2 = params[0];
            theta_h = params[1];
        } else {
            if (scanf(" %lf,%lf", &beta2, &theta_h) != 2) break;
        }
        double cos_theta_h = cos(theta_h), sin_theta_h = sin(theta_h);
        double s = higgs_mass2/(1-beta2);
        double sqrt_s = sqrt(s);
        double sqrt_lambda = s - higgs_mass2; // sqrt(kallen(s,higgs_mass,0))
        double pf = sqrt_lambda/2./sqrt_s;
        double q1[4] = {sqrt_s/2, 0, 0, sqrt_s/2};
        double q2[4] = {sqrt_s/2, 0, 0, -sqrt_s/2};
        double qj1[4] = {pf, 0, pf*sin_theta_h, pf*cos_theta_h};
        double qH[4] = {sqrt_s - pf, 0, -pf*sin_theta_h, -pf*cos_theta_h}; // sqrt_s - pf == sqrt(higgs_mass2 + pf*pf)
        double muren = sqrt_s/2;
        if (0) {
            fprintf(stderr, "q1={%.16e,%.16e,%.16e,%.16e}\n", q1[0], q1[1], q1[2], q1[3]);
            fprintf(stderr, "q2={%.16e,%.16e,%.16e,%.16e}\n", q2[0], q2[1], q2[2], q2[3]);
            fprintf(stderr, "j1={%.16e,%.16e,%.16e,%.16e}\n", qj1[0], qj1[1], qj1[2], qj1[3]);
            fprintf(stderr, "H= {%.16e,%.16e,%.16e,%.16e}\n", qH[0], qH[1], qH[2], qH[3]);
            fprintf(stderr, "mt=%.16e mh=%.16e muren=%.16e\n", top_mass, higgs_mass, muren);
        }
        double amp[4] = {0};
        gosam_gg_1l(&amp[1], q1, q2, qj1, qH, top_mass, higgs_mass, muren);
        amp[0] = sin_theta_h*beta2/(16*M_PI); // d Phi_2 / d theta_H
        amp[1] *= scaling;
        amp[2] *= scaling;
        amp[3] *= scaling;
        if (binary_io) {
            if (fwrite(&amp[0], sizeof(amp), 1, fout) != 1) {
                fprintf(stderr, "fwrite() error\n");
                exit(1);
            }
        } else {
            fprintf(fout, "%.16e,%.16e,%.16e,%.16e\n", amp[0], amp[1], amp[2], amp[3]);
        }
        fflush(fout);
    }
    return 0;
}
