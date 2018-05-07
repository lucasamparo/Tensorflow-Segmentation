#include <bits/stdc++.h>

using namespace std;

int main(int argc, char ** argv) {
    if (argc <= 2)
        return 1;
    string method = argv[1], source = argv[2];
    ifstream genuine_file(("matchs/genuine_"+method+"_"+source+".txt").c_str());
    std::vector<double> genuine_scores;
    double a, max_val = 0;
    int id1 = 0, id2 = 0, gen_discarded = 0;
    while (genuine_file >> id1 >> id2 >> a) {
        if (!a) {
            gen_discarded++;
            continue;
        }
        genuine_scores.push_back(a);
        max_val = max(a, max_val);
    }
    ifstream impostor_file(("matchs/impostor_"+method+"_"+source+".txt").c_str());
    std::vector<double> impostor_scores;
    int imp_discarded = 0;
    while (impostor_file >> id1 >> id2 >> a) {
        if (!a) {
            imp_discarded++;
            continue;
        }
        impostor_scores.push_back(a);
        max_val = max(a, max_val);
    }
    cout << gen_discarded << " " << imp_discarded << endl;
    sort(genuine_scores.begin(), genuine_scores.end());
    sort(impostor_scores.begin(), impostor_scores.end());
    int imp = 0, gen = 0;
    const int gen_size = genuine_scores.size();
    const int imp_size = impostor_scores.size();
    ofstream frr(("frr_"+method+"_"+source+".txt").c_str()), far(("far_"+method+"_"+source+".txt").c_str());
    double step = max_val/100.0;
    for (double score = 0; score < max_val; score += step) {
        while (gen < gen_size && genuine_scores[gen] <= score)
            gen++;
        while (imp < imp_size && impostor_scores[imp] <= score)
            imp++;
        frr << (double) gen / gen_size << endl;
        far << 1.0 - ((double) imp / imp_size) << endl;
    }
}
