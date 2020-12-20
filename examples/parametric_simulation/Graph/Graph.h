//
// Created by rsltgy on 18/12/2020.
//

#ifndef LIBGRAPE_LITE_GRAPH_H
#define LIBGRAPE_LITE_GRAPH_H
#include <string>
#include "Edge.h"
#include <vector>

using namespace std;

class Graph {

 public:
  Graph() {}
  //Load graph edge and node files from specified location
  void load_from_file(const string &location);
  //clear out existence graph
  ~Graph(){ out_edges_.clear(); in_edges_.clear(); nodes_.clear();}
  inline const vector<string> & nodes() const { return nodes_; }
  inline const vector<vector<Edge>> & out_edges() const { return out_edges_; }
  inline const vector<vector<Edge>> & in_edges() const { return in_edges_; }
  inline const vector<vector<int>> & descendants() const { return descendants_; }
  inline const vector<vector<pair<int,string>>> & paths() const { return paths_; }
  inline const int & number_of_nodes() const { return number_of_nodes_; }
 private:
  int number_of_nodes_;
  vector<vector<Edge>> out_edges_; // Each node has a vector of outgoing edges
  vector<vector<Edge>> in_edges_;  // Each node has a vector of incoming edges
  vector<string> nodes_;              // Each node has a data, we assume nodes starts from 0 till number_of_nodes - 1
  vector<vector<int>> descendants_;       // Each node has descendants as a vector
  vector<vector<pair<int,string>>> paths_;             // all paths from each node, starting node,end node and edge strings along the path
  void read_nodes(Graph &g,const string &location);
  void read_paths(Graph &g,const string &location);

};

#endif  // LIBGRAPE_LITE_GRAPH_H
