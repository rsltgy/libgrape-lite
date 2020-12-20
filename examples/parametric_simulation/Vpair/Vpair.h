//
// Created by rsltgy on 19/12/2020.
//

#ifndef LIBGRAPE_LITE_VPAIR_H
#define LIBGRAPE_LITE_VPAIR_H

#include <grape/grape.h>
#include "../ParaMatch/ParaMatch.h"
#include "Vpair_context.h"
#include "grape/utils/vertex_array.h"

using namespace std;


namespace grape{

template <typename FRAG_T>
class Vpair : public ParallelAppBase<FRAG_T, VpairContext<FRAG_T>>,
public ParallelEngine {


// specialize the templated worker.
INSTALL_PARALLEL_WORKER(Vpair<FRAG_T>, VpairContext<FRAG_T>, FRAG_T)
using vertex_t = typename fragment_t::vertex_t;
void PEval(const fragment_t& frag, context_t& ctx,
           message_manager_t& messages) {

  /*auto inner_vert = frag.InnerVertices();
  for (auto v : inner_vert ) {
    auto oid = frag.Gid2Oid(frag.Vertex2Gid(v));
    std::cout << frag.fid()  << " " << oid << " " << frag.GetData(v) << std::endl;
  }*/

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
  double &sigma = ctx.sigma;
  double &delta = ctx.delta;
  ParaMatch<FRAG_T> p;

  string u_label = GD.nodes()[u];
  vector<double> u_t_word_vector;
  Reader::calculate_word_vector(ctx.word_embeddings,u_label,u_t_word_vector);

  auto inner_vert = frag.InnerVertices();
  for (auto v : inner_vert ) {
    //auto oid = frag.Gid2Oid(frag.Vertex2Gid(v));
    //std::cout << frag.fid()  << " " << oid << " " << frag.GetData(v) << std::endl;
    string v_str = frag.GetData(v);
    //cout << GD.nodes()[u] << " " << v_str << endl;
    vector<double> v_g_word_vector;
    boost::trim_right(v_str);
    Reader::calculate_word_vector(ctx.word_embeddings,v_str,v_g_word_vector);

    double score = p.cosine_similarity(u_t_word_vector,v_g_word_vector);

    if(score >= sigma ) {
      auto oid = frag.Gid2Oid(frag.Vertex2Gid(v));
      auto es = frag.GetOutgoingAdjList(v);
      //std::cout << frag.fid()  << " " << oid << " " << frag.GetData(v) << std::endl;
      C.push_back(make_pair(oid, es.Size()));
    }

  }

  sort(C.begin(),C.end(),[](const pair<int,int> &a, const pair<int,int>b){
    return a.second < b.second;
  });

  bool match = false;
  for(auto v : C){
    //cout << frag.fid() << " --> " << u << " " << v.first << endl;
    auto it = ctx.cache.find(make_pair(u,v.first));
    if(it != ctx.cache.end()){
        if(ctx.cache[make_pair(u,v.first)].first){
          ctx.match_set.push_back(v.first);
        }
    }
    else{
      int v_vertex = v.first;
      match = p.match_pair(GD,frag,g_paths,g_descendants,u,v_vertex,sigma,delta,cache,word_embeddings,ecache_u,ecache_v);
      if(match)
        match_set.push_back(v.first);
    }

  }


  for(auto matched_vertices : match_set){
    cout << frag.fid() << " " <<  matched_vertices << endl;
  }


}




    void IncEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {



}

};



}


#endif  // LIBGRAPE_LITE_VPAIR_H