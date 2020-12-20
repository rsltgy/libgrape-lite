//
// Created by rsltgy on 11/12/2020.
//

#ifndef LIBGRAPE_LITE_SPAIR_H
#define LIBGRAPE_LITE_SPAIR_H

#include <grape/grape.h>
#include "../ParaMatch/ParaMatch.h"
#include "Spair_context.h"
#include "grape/utils/vertex_array.h"

using namespace std;


namespace grape{

template <typename FRAG_T>
class Spair : public ParallelAppBase<FRAG_T, SpairContext<FRAG_T>>,
              public ParallelEngine {


  // specialize the templated worker.
 INSTALL_PARALLEL_WORKER(Spair<FRAG_T>, SpairContext<FRAG_T>, FRAG_T)
  using vertex_t = typename fragment_t::vertex_t;
  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {

    /*auto inner_vert = frag.InnerVertices();
    for (auto v : inner_vert ) {
      auto oid = frag.Gid2Oid(frag.Vertex2Gid(v));
      std::cout << frag.fid()  << " " << oid << " " << frag.GetData(v) << std::endl;
    }*/


    /*vertex_t u1(0);
    unsigned int a = frag.Vertex2Gid(u1);
    if (a <= 100000000)
      cout << frag.fid() <<  frag.GetData((vertex_t)a) << endl;
    else
      cout << frag.fid() << " could not find " << endl;*/


    int u = ctx.u;
    int v = ctx.v;
    double  sigma = 0.8;
    double delta = 0.2;
    ParaMatch<FRAG_T> p;
    bool res = p.match(ctx.GD,frag,ctx.g_paths,ctx.g_descendants,ctx.word_embeddings,u,v,sigma,delta);
    cout << frag.fid() << " " << res  << endl;

/*
    vertex_t u1;
    frag.GetInnerVertex(ctx.u, u1);

    vertex_t v1;
    frag.GetInnerVertex(ctx.v, v1);


    auto es = frag.GetOutgoingAdjList(u1);
    for (auto& e : es) {
      cout << e.get_data() << endl;
      cout << e.get_neighbor().GetValue() << endl;
    }
*/




  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {



  }

};



}


#endif  // LIBGRAPE_LITE_SPAIR_H
