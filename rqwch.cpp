#include <iostream>
#include <fstream>
#include <complex>
#include <vector>
#include <cmath>
#include "Eigen/Dense"
#include "Eigen/Sparse"

using namespace std;
using namespace Eigen;

using cxd = complex<double>;
using SparseCMatrix = SparseMatrix<cxd>;
using CVector = VectorXcd;

// Operador de Hadamard
Matrix2cd coin_operator() {
    Matrix2cd H;
    double inv_sqrt2 = 1.0 / sqrt(2.0);
    H << inv_sqrt2, inv_sqrt2,
         inv_sqrt2, -inv_sqrt2;
    return H;
}

// Operador de desplazamiento
SparseCMatrix shift_operator(int size) {
    int dim = 2 * size;
    vector<Triplet<cxd>> triplets;

    for (int pos = 1; pos < size - 1; ++pos) {
        int up = 2 * pos;
        int down = 2 * pos + 1;

        // ↑  + 1
        triplets.emplace_back(up + 2, up, 1.0);

        // ↓  - 1
        triplets.emplace_back(down - 2, down, 1.0);
    }

    SparseCMatrix S(dim, dim);
    S.setFromTriplets(triplets.begin(), triplets.end());
    return S;
}

// Operador moneda ⊗ identidad
SparseCMatrix coin_tensor_identity(const Matrix2cd &coin, int size) {
    int dim = 2 * size;
    vector<Triplet<cxd>> triplets;

    for (int pos = 0; pos < size; ++pos) {
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                int row = 2 * pos + i;
                int col = 2 * pos + j;
                triplets.emplace_back(row, col, coin(i, j));
            }
        }
    }

    SparseCMatrix result(dim, dim);
    result.setFromTriplets(triplets.begin(), triplets.end());
    return result;
}

// Estado inicial (estado equilibrado)
CVector initial_state(int size, int center) {
    CVector psi = CVector::Zero(2 * size);
    psi(2 * center) = 1.0/sqrt(2.0);                 
    psi(2 * center + 1) = cxd(0.0, 1.0/sqrt(2.0));    
    return psi;
}

// Calcular distribución de probabilidad
VectorXd probability_distribution(const CVector &psi, int size) {
    VectorXd probs(size);
    for (int i = 0; i < size; ++i) {
        probs(i) = norm(psi(2 * i)) + norm(psi(2 * i + 1));
    }
    return probs;
}

void save_distribution_to_file(const VectorXd &probs, int t, int center) {
    ofstream file("probabilidad_t_" + to_string(t) + ".txt");
    for (int i = 0; i < probs.size(); ++i) {
            int x = i - center;
            file << x << " " << probs(i) << "\n";
    }
    file.close();
}

int main() {
    const vector<int> steps = {10, 100, 1000};

    for (int t : steps) {
        const int size = 2 * t + 1; 
        const int center = t;

        CVector psi = initial_state(size, center);
        Matrix2cd H = coin_operator();

        SparseCMatrix C = coin_tensor_identity(H, size);
        SparseCMatrix S = shift_operator(size);

        for (int i = 0; i < t; ++i) {
            psi = S * (C * psi);
        }

        VectorXd probs = probability_distribution(psi, size);
        save_distribution_to_file(probs, t, center);
        cout << "Guardado: probabilidad_t_" << t << ".txt\n";
    }

    return 0;
}
