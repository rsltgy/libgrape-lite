//
// Created by rsltgy on 11/12/2020.
//

#ifndef LIBGRAPE_LITE_SPAIR_H
#define LIBGRAPE_LITE_SPAIR_H
#include <grape/grape.h>

#include "spair_context.h"



namespace grape{

template <typename FRAG_T>
class Spair : public ParallelAppBase<FRAG_T, SpairContext<FRAG_T>>,
public ParallelEngine {

  // specialize the templated worker.
  INSTALL_PARALLEL_WORKER(Spair<FRAG_T>, SpairContext<FRAG_T>, FRAG_T)
  using vertex_t = typename fragment_t::vertex_t;

  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {

    std::cout << "here hello "<< std::endl;

    /*auto inner_vertices = frag.InnerVertices();
    for(grape::Vertex<unsigned int> i:inner_vertices){
      std::cout << frag.fnum() << " " << i.GetValue() << std::endl;
    }*/

  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {}

};



}


#endif  // LIBGRAPE_LITE_SPAIR_H

