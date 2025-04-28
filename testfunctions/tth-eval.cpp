/* This program reads input line by line in the format:
 *     beta2,frac_stt,theta_h,theta_t,phi_t
 *
 * and prints out the corresponding GoSam amplitudes in the format:
 *     weight,tree,virtual,virtual single pole,virtual double pole,virtual coulomb
 *
 * The weight is the phase-space density in the angular variables.
 */

#define _POSIX_C_SOURCE 200809L

#include <sys/mman.h>
#include <sys/wait.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <string.h>
#include <algorithm>
#include <limits.h>

#define countof(array) (sizeof(array)/sizeof(*(array)))

extern "C" {
    void olp_option_(const char* line,int* stat);
    void olp_start_(const char* contract_file_name,int* ierr,int* stage,int* rndseed);
    void olp_evalsubprocess_(int* label, double* momenta, double* mu, double* parameters, double* res);
    void olp_evalsubprocess2_(int* label, double* momenta, double* mu, double* parameters, double* res, double* acc);
    void olp_evalbornprocess_(int* label, double* momenta, double* parameters, double* res);
};

static double sqr(double x) { return x*x; }
static double pow2(double x) { return x*x; }
static double pow4(double x) { return pow2(pow2(x)); }
static double range(double a, double b, double x) { return a + x*(b - a); }

double kallen(double a, double sqrt_b, double sqrt_c) {
    return (a-sqr(sqrt_b+sqrt_c))*(a-sqr(sqrt_b-sqrt_c));
}

void transform(double x[4], const double m[4][4]) {
    double y0 = m[0][0]*x[0] + m[0][1]*x[1] + m[0][2]*x[2] + m[0][3]*x[3];
    double y1 = m[1][0]*x[0] + m[1][1]*x[1] + m[1][2]*x[2] + m[1][3]*x[3];
    double y2 = m[2][0]*x[0] + m[2][1]*x[1] + m[2][2]*x[2] + m[2][3]*x[3];
    double y3 = m[3][0]*x[0] + m[3][1]*x[1] + m[3][2]*x[2] + m[3][3]*x[3];
    x[0] = y0;
    x[1] = y1;
    x[2] = y2;
    x[3] = y3;
}

void transform(double x[3], const double m[3][3]) {
    double y0 = m[0][0]*x[0] + m[0][1]*x[1] + m[0][2]*x[2];
    double y1 = m[1][0]*x[0] + m[1][1]*x[1] + m[1][2]*x[2];
    double y2 = m[2][0]*x[0] + m[2][1]*x[1] + m[2][2]*x[2];
    x[0] = y0;
    x[1] = y1;
    x[2] = y2;
}

double sp(double a[4], double b[4]) {
    return a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3];
}

void gosam_qq_0l(double *amp, double qq1[4], double qq2[4], double qt1[4], double qt2[4], double qH[4], double mt, double mh, double muren) {
    int label = 0;
    double momenta[50];
    double params[10];
    double res = NAN;
    memcpy(&momenta[5*0], &qq1[0], 4*sizeof(double)); momenta[5*0+4] = 0;
    memcpy(&momenta[5*1], &qq2[0], 4*sizeof(double)); momenta[5*1+4] = 0;
    memcpy(&momenta[5*2], &qH[0],  4*sizeof(double)); momenta[5*2+4] = mh;
    memcpy(&momenta[5*3], &qt1[0], 4*sizeof(double)); momenta[5*3+4] = mt;
    memcpy(&momenta[5*4], &qt2[0], 4*sizeof(double)); momenta[5*3+4] = mt;
    params[0] = 1;
    olp_evalbornprocess_(&label, momenta, params, &res);
    amp[0] = res; // 0l
    amp[1] = amp[2] = amp[3] = amp[4] = 0; // 1l
}

void gosam_qq_1l(double *amp, double qq1[4], double qq2[4], double qt1[4], double qt2[4], double qH[4], double mt, double mh, double muren) {
    int label = 0;
    double momenta[50];
    double params[10];
    double res[5] = {NAN, NAN, NAN, NAN, NAN};
    double accuracy = NAN;
    memcpy(&momenta[5*0], &qq1[0], 4*sizeof(double)); momenta[5*0+4] = 0;
    memcpy(&momenta[5*1], &qq2[0], 4*sizeof(double)); momenta[5*1+4] = 0;
    memcpy(&momenta[5*2], &qH[0],  4*sizeof(double)); momenta[5*2+4] = mh;
    memcpy(&momenta[5*3], &qt1[0], 4*sizeof(double)); momenta[5*3+4] = mt;
    memcpy(&momenta[5*4], &qt2[0], 4*sizeof(double)); momenta[5*3+4] = mt;
    params[0] = 1;
    olp_evalsubprocess2_(&label, momenta, &muren, params, res, &accuracy);
    // If the precision after the gosam rescue system is less than 3 digits, discard the point
    if (accuracy < 1e-5) {
        amp[0] = res[3]; // tree
        amp[1] = res[2]; // 1l eps^0
        amp[2] = res[1]; // 1l eps^-1
        amp[3] = res[0]; // 1l eps^-2
        amp[4] = res[4]; // tree * T3 T4
    } else {
        amp[0] = amp[1] = amp[2] = amp[3] = amp[4] = NAN;
    }
}

void gosam_gg_0l(double *amp, double qg1[4], double qg2[4], double qt1[4], double qt2[4], double qH[4], double mt, double mh, double muren) {
    int label = 2;
    double momenta[50];
    double params[10];
    double res = NAN;
    memcpy(&momenta[5*0], &qg1[0], 4*sizeof(double)); momenta[5*0+4] = 0;
    memcpy(&momenta[5*1], &qg2[0], 4*sizeof(double)); momenta[5*1+4] = 0;
    memcpy(&momenta[5*2], &qH[0],  4*sizeof(double)); momenta[5*2+4] = mh;
    memcpy(&momenta[5*3], &qt1[0], 4*sizeof(double)); momenta[5*3+4] = mt;
    memcpy(&momenta[5*4], &qt2[0], 4*sizeof(double)); momenta[5*3+4] = mt;
    params[0] = 1;
    olp_evalbornprocess_(&label, momenta, params, &res);
    amp[0] = res; // 0l
    amp[1] = amp[2] = amp[3] = amp[4] = 0; // 1l
}

void gosam_gg_1l(double *amp, double qg1[4], double qg2[4], double qt1[4], double qt2[4], double qH[4], double mt, double mh, double muren) {
    int label = 2;
    double momenta[50];
    double params[10];
    double res[5] = {NAN, NAN, NAN, NAN, NAN};
    double accuracy = NAN;
    memcpy(&momenta[5*0], &qg1[0], 4*sizeof(double)); momenta[5*0+4] = 0;
    memcpy(&momenta[5*1], &qg2[0], 4*sizeof(double)); momenta[5*1+4] = 0;
    memcpy(&momenta[5*2], &qH[0],  4*sizeof(double)); momenta[5*2+4] = mh;
    memcpy(&momenta[5*3], &qt1[0], 4*sizeof(double)); momenta[5*3+4] = mt;
    memcpy(&momenta[5*4], &qt2[0], 4*sizeof(double)); momenta[5*3+4] = mt;
    olp_evalsubprocess2_(&label, momenta, &muren, params, res, &accuracy);
    // If the precision after the gosam rescue system is less than 3 digits, discard the point
    if (accuracy < 1e-5) {
        amp[0] = res[3]; // tree
        amp[1] = res[2]; // 1l eps^0
        amp[2] = res[1]; // 1l eps^-1
        amp[3] = res[0]; // 1l eps^-2
        amp[4] = res[4]; // tree * T3 T4
    }
    else {
        amp[0] = amp[1] = amp[2] = amp[3] = amp[4] = NAN;
    }
}

void usage(const char *prog) {
    fprintf(stderr, "Usage: %s (qq|gg)-(0l|1l) [options ...]\n", prog);
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
    if (argc - optind != 1) {
        usage(argv[0]);
        return 1;
    }
    double scaling = 0;
    void (*ampfun)(double *amp, double q1[4], double q2[4], double qt1[4], double qt2[4], double qH[4], double mt, double mh, double muren) = NULL;
    if (strcmp(argv[optind], "qq-0l") == 0) { ampfun = gosam_qq_0l; scaling = 1.1314975879758687e6; }
    else if (strcmp(argv[optind], "qq-1l") == 0) { ampfun = gosam_qq_1l; scaling = 1.1314975879758687e6; }
    else if (strcmp(argv[optind], "gg-0l") == 0) { ampfun = gosam_gg_0l; scaling = 1.1314975879758687e6*128./9.; }
    else if (strcmp(argv[optind], "gg-1l") == 0) { ampfun = gosam_gg_1l; scaling = 1.1314975879758687e6*128./9.; }
    else {
        fprintf(stderr, "Unknown amplitude: '%s'\n", argv[optind]);
        return 1;
    }
    // Init OLP
    int err = 0;
    int parstage = 1;
    int rndseed = 1 + (rand() % 100);
    olp_option_("gs=1", &err);
    if (err != 1) { fprintf(stderr, "olp_option() error %d\n", err); exit(1); }
    olp_option_("nlo_prefactors=0", &err);
    if (err != 1) { fprintf(stderr, "olp_option() error %d\n", err); exit(1); }
    olp_option_("mH=0.7223151185146152", &err);
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
        double beta2 = 0, frac_stt = 0, theta_h = 0, theta_t = 0, phi_t = 0;
        if (binary_io) {
            double params[5];
            if (fread(&params[0], sizeof(double), 5, stdin) != 5) break;
            beta2 = params[0];
            frac_stt = params[1];
            theta_h = params[2];
            theta_t = params[3];
            phi_t = params[4];
        } else {
            if (scanf(" %lf,%lf,%lf,%lf,%lf", &beta2, &frac_stt, &theta_h, &theta_t, &phi_t) != 5) break;
        }
        double cos_theta_h = cos(theta_h), sin_theta_h = sin(theta_h);
        double cos_theta_t = cos(theta_t), sin_theta_t = sin(theta_t);
        double cos_phi_t = cos(phi_t), sin_phi_t = sin(phi_t);
        double s = sqr(2*top_mass + higgs_mass)/(1-beta2);
        double sqrt_s = sqrt(s);
        double stt = range(4*top_mass2, sqr(sqrt_s - higgs_mass), frac_stt);
        double sqrt_stt = sqrt(stt);
        // Momenta in the ttbar system
        double pt = sqrt(stt - 4*top_mass2)/2;
        double qt1[4] = {sqrt_stt/2, pt*sin_theta_t*cos_phi_t, pt*sin_theta_t*sin_phi_t, pt*cos_theta_t};
        double qt2[4] = {qt1[0], -qt1[1], -qt1[2], -qt1[3]};
        // ttbar -> cms transformation
        double kallen_stt_s_mh2 = kallen(stt, sqrt_s, higgs_mass);
        double qh2 = kallen_stt_s_mh2/4/s;
        double qh = sqrt(qh2);
        double Eh = sqrt(qh2 + higgs_mass2);
        double invrot[4][4] = {
            {1, 0, 0, 0},
            {0, -cos_theta_h, 0, -sin_theta_h},
            {0, 0, 1, 0},
            {0, sin_theta_h, 0, -cos_theta_h}
        };
        double invboost[4][4] = {
            {sqrt(1+qh2/stt), 0, 0, qh/sqrt_stt},
            {0, 1, 0, 0},
            {0, 0, 1, 0},
            {qh/sqrt_stt, 0, 0, sqrt(1+qh2/stt)}
        };
        // Momenta in CMS
        double q1[4] = {sqrt_s/2, 0, 0, sqrt_s/2};
        double q2[4] = {sqrt_s/2, 0, 0, -sqrt_s/2};
        double qH[4] = {Eh, qh*sin_theta_h, 0, qh*cos_theta_h};
        transform(qt1, invboost);
        transform(qt2, invboost);
        transform(qt1, invrot);
        transform(qt2, invrot);
        double muren = sqrt_s/2;
        //double beta_stt = sqrt(1-4*top_mass2/stt);
        double beta_stt = 2*pt/sqrt_stt;
        if (0) {
            fprintf(stderr, "q1={%.16e,%.16e,%.16e,%.16e}\n", q1[0], q1[1], q1[2], q1[3]);
            fprintf(stderr, "q2={%.16e,%.16e,%.16e,%.16e}\n", q2[0], q2[1], q2[2], q2[3]);
            fprintf(stderr, "H= {%.16e,%.16e,%.16e,%.16e}\n", qH[0], qH[1], qH[2], qH[3]);
            fprintf(stderr, "t1={%.16e,%.16e,%.16e,%.16e}\n", qt1[0], qt1[1], qt1[2], qt1[3]);
            fprintf(stderr, "t2={%.16e,%.16e,%.16e,%.16e}\n", qt2[0], qt2[1], qt2[2], qt2[3]);
            fprintf(stderr, "mt=%.16e mh=%.16e muren=%.16e\n", top_mass, higgs_mass, muren);
        }
        double amp[6] = {0};
        ampfun(&amp[1], q1, q2, qt1, qt2, qH, top_mass, higgs_mass, muren);
        double kallen_stt_mt2_mt2 = stt*(stt-4*top_mass2);
        amp[0] = (pow2(sqrt_s - higgs_mass) - 4*top_mass2)*sqrt(kallen_stt_mt2_mt2*kallen_stt_s_mh2)/(1024*pow4(M_PI)*s*stt)*sin_theta_h*sin_theta_t;
        amp[1] *= scaling;
        amp[2] *= scaling;
        amp[3] *= scaling;
        amp[4] *= scaling;
        amp[5] *= -scaling*sqr(M_PI)/beta_stt;
        if (binary_io) {
            if (fwrite(&amp[0], sizeof(amp), 1, fout) != 1) {
                fprintf(stderr, "fwrite() error\n");
                exit(1);
            }
        } else {
            fprintf(fout, "%.16e,%.16e,%.16e,%.16e,%.16e,%.16e\n", amp[0], amp[1], amp[2], amp[3], amp[4], amp[5]);
        }
        fflush(fout);
    }
    return 0;
}
