#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;


double errrel(const VectorXd& x, const VectorXd& x_exact) //funzione che calcola errore relativo
{
    double norm_x_exact = x_exact.norm();
    return (x - x_exact).norm() / norm_x_exact;
}

int main()
{
    //definizione dei sistemi lineari da risolvere
    Matrix2d A1, A2, A3;
    Vector2d b1, b2, b3;

    A1 << 5.547001962252291e-01, -3.770900990025203e-02,
        8.320502943378437e-01, -9.992887623566787e-01;
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;

    A2 << 5.547001962252291e-01, -5.540607316466765e-01,
        8.320502943378437e-01, -8.324762492991313e-01;
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;

    A3 << 5.547001962252291e-01, -5.547001955851905e-01,
        8.320502943378437e-01, -8.320502947645361e-01;
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;

    VectorXd x_es(2); //vettore soluzione esatta dei sistemi lineari
    x_es << -1.0e+0, -1.0e+00;

    VectorXd x1_palu, x2_palu, x3_palu; //risoluzione sistema con metodo PALU
    x1_palu = A1.partialPivLu().solve(b1);
    x2_palu = A2.partialPivLu().solve(b2);
    x3_palu = A3.partialPivLu().solve(b3);


    VectorXd x1_qr, x2_qr, x3_qr; //risoluzione sistema con metodo QR
    x1_qr = A1.householderQr().solve(b1);
    x2_qr = A2.householderQr().solve(b2);
    x3_qr = A3.householderQr().solve(b3);

    // calcolo errori relativi
    double rel_err1_palu = errrel(x1_palu, x_es); //calcolo errori relativi metodo PALU
    double rel_err2_palu = errrel(x2_palu, x_es);
    double rel_err3_palu = errrel(x3_palu, x_es);

    double rel_err1_qr = errrel(x1_qr, x_es); //calcolo errori relativi metodo QR
    double rel_err2_qr = errrel(x2_qr, x_es);
    double rel_err3_qr = errrel(x3_qr, x_es);

    // Stampo risultati
    cout << "Decomposizione PALU:" << endl;
    cout << "\t" << "Errore relativo del sistema 1: " << rel_err1_palu << endl;
    cout <<"\t" << "Errore relativo del sistema 2: " << rel_err2_palu << endl;
    cout <<"\t" << "Errore relativo del sistema 3: " << rel_err3_palu << endl;

    cout <<"Decomposizione QR:" << endl;
    cout <<"\t" << "Errore relativo del sistema 1: " << rel_err1_qr << endl;
    cout << "\t" <<"Errore relativo del sistema 2: " << rel_err2_qr << endl;
    cout << "\t" <<"Errore relativo del sistema 3: " << rel_err3_qr << endl;

    return 0;
}
