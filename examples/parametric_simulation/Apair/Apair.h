//
// Created by rsltgy on 20/12/2020.
//

#ifndef LIBGRAPE_LITE_APAIR_H
#define LIBGRAPE_LITE_APAIR_H


#include <grape/grape.h>
#include "../ParaMatch/ParaMatch.h"
#include "Apair_context.h"
#include "grape/utils/vertex_array.h"
#include "../KDTree/KDTree.h"

using namespace std;


namespace grape{

template <typename FRAG_T>
class Apair : public ParallelAppBase<FRAG_T, ApairContext<FRAG_T>>,
public ParallelEngine {


// specialize the templated worker.
INSTALL_PARALLEL_WORKER(Apair<FRAG_T>, ApairContext<FRAG_T>, FRAG_T)
using vertex_t = typename fragment_t::vertex_t;
void PEval(const fragment_t& frag, context_t& ctx,
           message_manager_t& messages) {

  auto &match_set = ctx.match_set;
  auto &C = ctx.C;
  auto &GD = ctx.GD;
  auto &u = ctx.u;
  auto &cache = ctx.cache;
  auto &ecache_u = ctx.ecache_u;
  auto &ecache_v = ctx.ecache_v;
  auto &g_paths = ctx.g_paths;
  auto &g_descendants = ctx.g_descendants;
  auto &word_embeddings = ctx.word_embeddings;
  auto &sigma = ctx.sigma;
  auto &delta = ctx.delta;

  pointVec points;
  auto inner_vert = frag.InnerVertices();
  for (auto v : inner_vert ) {
    string v_str = "";//frag.GetData(v);
    vector<double> v_g_word_vector;
    Reader::calculate_word_vector(ctx.word_embeddings,v_str,v_g_word_vector);
    if(!v_g_word_vector.empty()){
      auto oid = frag.Gid2Oid(frag.Vertex2Gid(v));
      point_t t = make_pair(oid,v_g_word_vector);
      points.push_back(t);
    }
  }

  KDTree tree(points);

  for(int u_t = 0; u_t < GD.number_of_nodes(); u_t++){
    if(!GD.nodes()[u_t].empty()){
      vector<double> u_t_word_vector;
      Reader::calculate_word_vector(word_embeddings,GD.nodes()[u_t],u_t_word_vector);
      if(!u_t_word_vector.empty()) {
        point_t t = make_pair(u_t,u_t_word_vector);
        auto NNs = tree.neighborhood_points(t, sigma);
        for(auto returned_match : NNs){
          //vertex_t v(returned_match.first);
          //auto es = frag.GetOutgoingAdjList(v);
          C.push_back(make_pair(u_t,make_pair(returned_match.first,5)));
        }
      }
    }

  }

  // We do not need the tree anymore, clear out.
  points.clear();
  points.shrink_to_fit();


  sort(C.begin(),C.end(),[](const pair<int,pair<int,int>> &a, const pair<int,pair<int,int>> b){
    return a.second.second < b.second.second;
  });


  bool match = false;
  for(auto v : C){
    auto it = ctx.cache.find(make_pair(u,v.first));
    if(it != ctx.cache.end() && ctx.cache[make_pair(u,v.first)].first )
      ctx.match_set.push_back(v.first);
    else{
      ParaMatch<FRAG_T> p;
      match = p.match_pair(GD,frag,g_paths,g_descendants,u,v.first,sigma,delta,cache,word_embeddings,ecache_u,ecache_v);
      if(match)
        match_set.push_back(v.first);
    }

  }

}




void IncEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {



}

};



}


#endif  // LIBGRAPE_LITE_APAIR_H
