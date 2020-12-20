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
  auto &sigma = ctx.sigma;
  auto &delta = ctx.delta;
  /*vertex_t u1(0);
  unsigned int a = frag.Vertex2Gid(u1);
  if (a <= 100000000)
    cout << frag.fid() <<  frag.GetData((vertex_t)a) << endl;
  else
    cout << frag.fid() << " could not find " << endl;*/

  string u_label = GD.nodes()[u];

  auto inner_vert = frag.InnerVertices();
  for (auto v : inner_vert ) {
    vector<double> u_t_word_vector;
    Reader::calculate_word_vector(ctx.word_embeddings,GD.nodes()[u],u_t_word_vector);

    string v_str = frag.GetData(v);

    vector<double> v_g_word_vector;
    Reader::calculate_word_vector(ctx.word_embeddings,v_str,v_g_word_vector);

    if(!u_t_word_vector.empty() && !v_g_word_vector.empty() ) {
      auto oid = frag.Gid2Oid(frag.Vertex2Gid(v));
      auto es = frag.GetOutgoingAdjList(v);
      C.push_back(make_pair(oid, es.Size()));
    }

  }

  sort(C.begin(),C.end(),[](const pair<int,int> &a, const pair<int,int>b){
    return a.second < b.second;
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


#endif  // LIBGRAPE_LITE_VPAIR_H
