
#ifndef LIBGRAPE_LITE_READER_H
#define LIBGRAPE_LITE_READER_H
#include "string"
#include "vector"
#include "map"
#include <unordered_map>

using namespace std;

class Reader {
 public:
  static  void read_word_vector(string folder, unordered_map<string,vector<double>> &word_embeddings);
  static void calculate_word_vector(const unordered_map<string,vector<double>> &word_embeddings, const string& word,
                                    vector<double> &v_g_word_vector);
};

#endif  // LIBGRAPE_LITE_READER_H
