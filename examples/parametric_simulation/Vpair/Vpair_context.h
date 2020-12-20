//
// Created by rsltgy on 19/12/2020.
//

#ifndef LIBGRAPE_LITE_VPAIR_CONTEXT_H
#define LIBGRAPE_LITE_VPAIR_CONTEXT_H


#include <grape/grape.h>
#include <string>
#include <boost/algorithm/string.hpp>
#include <vector>
#include "../Util/Reader.h"
#include "../Graph/Graph.h"

using namespace std;

namespace grape{
/**
* @brief Context for the parallel version of Vpair.
*
* @tparam FRAG_T
*/
template <typename FRAG_T>
class VpairContext : public VertexDataContext<FRAG_T, double> {
 public:
  using oid_t = typename FRAG_T::oid_t;
  using vid_t = typename FRAG_T::vid_t;

  explicit VpairContext(const FRAG_T& fragment)
      : VertexDataContext<FRAG_T, double>(fragment, true),
        partial_result(this->data()) {}


  void read_paths(const string &location, vector<vector<pair<int,string>>> &g_paths_, vector<vector<int>> &g_descendants_){
    ifstream  d_file; d_file.open(location);
    if(!d_file) { cout << "unable to read file " << location << endl; exit(0); }
    int from, to; string str,path_string,temp_string;
    int max_vertex_number = 0;

    while(getline( d_file,str)){
      istringstream ss(str);
      ss >> from >> to >> path_string;   // Read id and following descendant id and path string
      if(from >= to){
        if(from >= max_vertex_number)
          max_vertex_number = from;
      }else{
        if(to >= max_vertex_number)
          max_vertex_number = to;
      }
    }
    d_file.close();
    for(int i = 0; i <= max_vertex_number; i++){
      g_paths.push_back(vector<pair<int,string>>());
      g_descendants.push_back(vector<int>());
    }
    d_file.open(location);
    // Read each line of the path file and store in descendants data structure.
    while(getline( d_file,str)){
      istringstream ss(str);
      ss >> from >> to >> path_string;   // Read id and following descendant id and path string
      while(ss >> temp_string)           // if there is more edge label, read them
        path_string += " " + temp_string;

      g_descendants_[from].push_back(to);
      if(g_paths_[from].size() >= 100) continue;
      g_paths_[from].push_back(make_pair(to,path_string));
    }
    d_file.close();
  }


  void Init(ParallelMessageManager& messages, string path_of_word_embeddings_,
            string gd_evfile_, string g_pathfile_, string gd_pathfile_,oid_t u_, oid_t v_,double sigma_,double delta_) {
    auto& frag = this->fragment();
    // Read word embeddings
    Reader::read_word_vector(path_of_word_embeddings_,word_embeddings);
    // Read Graph GD
    this->GD.load_from_file(gd_evfile_);
    this->read_paths(g_pathfile_,g_paths,g_descendants);
    this->u = u_;
    this->sigma = sigma_;
    this->delta = delta_;

    curr_modified.Init(frag.Vertices());
    next_modified.Init(frag.Vertices());

#ifdef PROFILING
    preprocess_time = 0;
    exec_time = 0;
    postprocess_time = 0;
#endif
  }


  void Output(std::ostream& os) override {


#ifdef PROFILING
    VLOG(2) << "preprocess_time: " << preprocess_time << "s.";
    VLOG(2) << "exec_time: " << exec_time << "s.";
    VLOG(2) << "postprocess_time: " << postprocess_time << "s.";
#endif
  }

  //unordered_map<string,vector<double>> &word_embeddings;
  oid_t u;
  typename FRAG_T::template vertex_array_t<double>& partial_result;
  unordered_map<string,vector<double>> word_embeddings;
  vector<vector<pair<int,string>>> g_paths;
  vector<vector<int>> g_descendants;
  vector<pair<int,int>> C;
  vector<int> match_set;
  unordered_map<std::pair<int,int>, pair<bool, vector<std::pair<int,int>>>, boost::hash<std::pair<int,int>>> cache;
  unordered_map<int,vector<int>> ecache_u, ecache_v;
  Graph GD;
  double sigma;
  double delta;
  DenseVertexSet<vid_t> curr_modified, next_modified;

#ifdef PROFILING
  double preprocess_time = 0;
  double exec_time = 0;
  double postprocess_time = 0;
#endif

};

}





#endif  // LIBGRAPE_LITE_VPAIR_CONTEXT_H
