//
// Created by rsltgy on 18/12/2020.
//

#include "Graph.h"

//
// Created by rsltgy on 25/11/2020.
//

#include "Graph.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include "Edge.h"

using namespace std;

void Graph::read_nodes(Graph &g,const string &location) {

  // Open file if exists, otherwise exit quickly
  ifstream  n_file; n_file.open(location);
  if(!n_file) { cout << "unable to read file " << location << endl; quick_exit(0); }
  int id; string str,label;
  // Read each line of the .v file and store in nodes data structure.
  while(getline( n_file,str)){
    istringstream ss(str);
    ss >> id >> label;
    string temp;
    while(ss >> temp){
      label += " " + temp;
    }

    g.nodes_.push_back(label);
  }

  g.number_of_nodes_ = g.nodes_.size();
  n_file.close();
}

void Graph::read_paths(Graph &g,const string &location,int k){

  ifstream  d_file; d_file.open(location);
  if(!d_file) { cout << "unable to read file " << location << endl; quick_exit(0); }
  int from, to; string str,path_string,temp_string;
  // Read each line of the path file and store in descendants data structure.
  while(getline( d_file,str)){
    istringstream ss(str);
    ss >> from >> to >> path_string;   // Read id and following descendant id and path string
    while(ss >> temp_string)           // if there is more edge label, read them
      path_string += " " + temp_string;
    g.descendants_[from].push_back(to);
    if(g.paths_[from].size() >= k) continue;
    g.paths_[from].push_back(make_pair(to,path_string));
  }
  d_file.close();

}

void Graph::load_from_file(const string &location,int k){

  // Read nodes of the graph from specified location
  read_nodes(*this,location + "g.v");

  ifstream  e_file; e_file.open(location + "g.e");
  if(!e_file) { cout << "unable to read file " << location << endl; quick_exit(0); }
  int from, to; string data,str;

  // initialise the graph structure, each node has an empty edge vector
  for(int i = 0; i < this->number_of_nodes_ ; i++){
    this->out_edges_.push_back(vector<Edge>());
    this->in_edges_.push_back(vector<Edge>());
    this->paths_.push_back(vector<pair<int,string>>());
    this->descendants_.push_back(vector<int>());
  }

  // Read each line of the .e file and store in nodes data structure.
  while(getline( e_file,str)){
    istringstream ss(str);
    ss >> from >> to >> data;
    Edge e_out(to,data);
    Edge e_in(from,data);
    this->out_edges_[from].push_back(e_out);
    this->in_edges_[to].push_back(e_in);
  }
  e_file.close();

  read_paths(*this, location + "path.txt",k);

}
