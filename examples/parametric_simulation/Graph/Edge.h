//
// Created by rsltgy on 18/12/2020.
//

#ifndef LIBGRAPE_LITE_EDGE_H
#define LIBGRAPE_LITE_EDGE_H
#include <string>
using namespace std;

class Edge {

 public:
  Edge(){}
  Edge(const int &to, const string &data): to_(to), data_(data){}
 private:
  inline const string &data() const { return data_; }
  int to_; string data_;


};

#endif  // LIBGRAPE_LITE_EDGE_H
