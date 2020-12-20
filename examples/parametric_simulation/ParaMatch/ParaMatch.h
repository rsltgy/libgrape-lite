//
// Created by rsltgy on 18/12/2020.
//

#ifndef LIBGRAPE_LITE_PARAMATCH_H
#define LIBGRAPE_LITE_PARAMATCH_H
#include <unordered_map>
#include <vector>
#include <iostream>
#include "../Graph/Graph.h"
#include "grape/grape.h"
#include <algorithm>
#include "cmath"
#include "../Util/Reader.h"
#include "grape/grape.h"
#include <boost/functional/hash.hpp>
#include <boost/algorithm/string.hpp>


using namespace std;

template <typename FRAG_T>
class ParaMatch {
  using oid_t = typename FRAG_T::oid_t;
  using vertex_t = typename FRAG_T::vertex_t;
   public:
  /*static bool match_pair(const Graph &GD,const FRAG_T& frag,
                         vector<vector<pair<int,string>>> &g_paths,vector<vector<int>> &g_descendants,
                         const int &u,const int &v,const double &sigma, const double &delta,
                         unordered_map<std::pair<int,int>, pair<bool, vector<std::pair<int,int>>>, pair_hash> &cache,
                         unordered_map<string,vector<double>> &word_embedding, unordered_map<int,vector<int>> &ecache_u,unordered_map<int,vector<int>> &ecache_v);
  static bool match(const Graph &GD, const FRAG_T& frag,
                    vector<vector<pair<int,string>>> &g_paths,vector<vector<int>> &g_descendants,,
                    unordered_map<string,vector<double>> &word_embedding, const int &u, const int &v,
                    const double &sigma, const double &delta);*/

  inline double cosine_similarity(const vector<double> &A, const vector<double> &B ){

    if (A.empty() || B.empty())
      return 0;

    double dot = 0.0, denom_a = 0.0, denom_b = 0.0 ;
    for (unsigned int i = 0; i < A.size(); ++i) {
      dot += A[i] * B[i] ;
      denom_a += A[i] * A[i] ;
      denom_b += B[i] * B[i] ;
    }
    return dot / (sqrt(denom_a) * sqrt(denom_b)) ;
  }
  inline double calculate_path_similarity(const Graph &GD,
                                          vector<vector<pair<int,string>>> &g_paths,
                                          unordered_map<string,vector<double>> &word_embeddings, const int &u, const int &u_prime,const int &v, const int &v_prime){

    // Check the path between u and u_prime: if there is a path take the labels, else return 0
    vector<pair<int,string>> paths_of_u = GD.paths()[u];
    string path_string_u;
    bool return_if_not_found_a_match = true;
    for(pair<int,string> path : paths_of_u){
      if(path.first == u_prime){
        path_string_u = path.second;
        return_if_not_found_a_match = false;
        break;
      }
    }
    if (return_if_not_found_a_match) return 0;

    vector<pair<int,string>> paths_of_v = g_paths[v];
    string path_string_v;
    return_if_not_found_a_match = true;
    for(pair<int,string> path : paths_of_v){
      if(path.first == v_prime){
        path_string_v = path.second;
        return_if_not_found_a_match = false;
        break;
      }
    }
    if (return_if_not_found_a_match) return 0;


    vector<double> u_word_vector;
    Reader::calculate_word_vector(word_embeddings,path_string_u,u_word_vector);
    vector<double> v_word_vector;
    Reader::calculate_word_vector(word_embeddings,path_string_v,v_word_vector);

    if(!u_word_vector.empty() && !v_word_vector.empty())
      return cosine_similarity(u_word_vector,v_word_vector);
    else
      return 0;
  }


  bool match_pair(const Graph &GD,const FRAG_T& frag,
                             vector<vector<pair<int,string>>> &g_paths,vector<vector<int>> &g_descendants,
                             const int &u,const int &v,const double &sigma, const double &delta,
                             unordered_map<std::pair<int,int>, pair<bool, vector<std::pair<int,int>>>, boost::hash<std::pair<int,int>>> &cache,
                             unordered_map<string,vector<double>> &word_embedding, unordered_map<int,vector<int>> &ecache_u,unordered_map<int,vector<int>> &ecache_v){

    vector<double> u_word_vector;
    Reader::calculate_word_vector(word_embedding,GD.nodes()[u],u_word_vector);

    vertex_t frag_vert;
    //cout << u << " --> " << v << " " << endl;
    if (!frag.GetVertex(v,frag_vert)) return false; // if the vertex not in the fragment
    string v_data;
    v_data = frag.GetData((vertex_t)frag_vert);

    vector<double> v_word_vector;
    boost::trim_right(v_data);
    Reader::calculate_word_vector(word_embedding,v_data,v_word_vector);

    double score = cosine_similarity(u_word_vector,v_word_vector);


    if(score <= sigma){                         // if the label of node u and v do not match then return false
      cache[make_pair(u,v)].first = false;
      cache[make_pair(u,v)].second.clear();
      return false;
    }

    if(GD.out_edges()[u].empty()){              // if node u is a leaf then return true
      cache[make_pair(u,v)].first = true;
      cache[make_pair(u,v)].second.clear();
      return true;
    }
    if(ecache_u.find(u) == ecache_u.end()){
      ecache_u[u] = GD.descendants()[u];
    }

    if(ecache_v.find(v) == ecache_v.end()){
      ecache_v[v] = g_descendants[v];
    }

    cache[make_pair(u,v)].first = true;
    double sum = 0;
    bool match = false;

    vector<int> v_u_k = ecache_u[u];
    vector<int> v_v_k = ecache_v[v];

    vector<pair<int,int>> matched_pairs;

    //Find all matching pairs of descendants of u and v;

    for(int node_u : v_u_k){
      u_word_vector.clear();
      Reader::calculate_word_vector(word_embedding,GD.nodes()[node_u],u_word_vector);
      for(int node_v : v_v_k){
        vertex_t frag_vert;
        if (!frag.GetVertex(node_v,frag_vert)) return false; // if the vertex not in the fragment
        v_data = frag.GetData((vertex_t)frag_vert);
        //cout << frag.fid() << " " << GD.nodes()[node_u] << " " << v_data << " " << node_u << " " << node_v << endl;
        boost::trim_right(v_data);
        v_word_vector.clear();
        Reader::calculate_word_vector(word_embedding,v_data,v_word_vector);

        score = cosine_similarity(u_word_vector,v_word_vector);
        if(score >= sigma){
          //cout << score << " " <<  frag.fid() << " " <<  node_u << " " << node_v << endl;
          matched_pairs.push_back(make_pair(node_u,node_v));
        }
      }
    }

    for(pair<int,int> u_prime_and_v_prime : matched_pairs){
      if(cache.find(u_prime_and_v_prime) != cache.end()){
        match = cache[u_prime_and_v_prime].first;
      }
      else{
        match = ParaMatch::match_pair(GD,frag,g_paths,g_descendants,u_prime_and_v_prime.first,u_prime_and_v_prime.second,sigma,delta,cache,word_embedding,ecache_u,ecache_v);
      }
      if(match){
        sum += calculate_path_similarity(GD,g_paths,word_embedding, u, u_prime_and_v_prime.first, v, u_prime_and_v_prime.second);
        cache[make_pair(u,v)].second.push_back(u_prime_and_v_prime);
        //cout << "sum is " << sum << " " << u_prime_and_v_prime.first << " " <<  u_prime_and_v_prime.second << endl;
        if (sum >= delta){
          cache[make_pair(u,v)].first = true;
          return true;
        }
        //break;
      }

    }

    // If we want all entity pairs with all descendants do not return in the loop.
    /*if (sum >= delta){
        return true;
    }*/

    cache[make_pair(u,v)].first = false;
    cache[make_pair(u,v)].second.clear();


    for(auto u_p_v_p : cache){
      auto it = find_if(u_p_v_p.second.second.begin(),u_p_v_p.second.second.end(),
                        [&u,&v](std::pair<int,int> el){
                          return u == el.first && v == el.second;
                        });

      if(it != u_p_v_p.second.second.end()){
        u_p_v_p.second.second.clear();
        u_p_v_p.second.first = false;
        ParaMatch::match_pair(GD,frag,g_paths,g_descendants,u_p_v_p.first.first,u_p_v_p.first.second,sigma,delta,cache,word_embedding,ecache_u,ecache_v);
      }
    }
    return false;

  };


  bool match(const Graph &GD, const FRAG_T& frag,
                        vector<vector<pair<int,string>>> &g_paths,vector<vector<int>> &g_descendants,
                        unordered_map<string,vector<double>> &word_embedding,  const int &u, const int &v, const double &sigma,
                        const double &delta) {

    unordered_map<std::pair<int,int> , pair<bool, vector<std::pair<int,int> >>, boost::hash<std::pair<int,int>>> cache;
    unordered_map<int,vector<int>> ecache_u, ecache_v;
    return ParaMatch::match_pair(GD,frag,g_paths,g_descendants,u,v,sigma,delta,cache,word_embedding,ecache_u,ecache_v);

  };







};






#endif  // LIBGRAPE_LITE_PARAMATCH_H
