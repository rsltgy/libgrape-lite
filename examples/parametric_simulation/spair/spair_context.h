//
// Created by rsltgy on 11/12/2020.
//

#ifndef LIBGRAPE_LITE_SPAIR_CONTEXT_H
#define LIBGRAPE_LITE_SPAIR_CONTEXT_H
#include <grape/grape.h>



namespace grape{
/**
* @brief Context for the parallel version of spair.
*
* @tparam FRAG_T
*/
template <typename FRAG_T>
class SpairContext : public VertexDataContext<FRAG_T, double> {
 public:
  using oid_t = typename FRAG_T::oid_t;
  using vid_t = typename FRAG_T::vid_t;

  explicit SpairContext(const FRAG_T& fragment)
      : VertexDataContext<FRAG_T, double>(fragment, true),
        partial_result(this->data()) {}

  void Init(ParallelMessageManager& messages, oid_t source_id2) {
    auto& frag = this->fragment();

    this->source_id = source_id2;
    partial_result.SetValue(std::numeric_limits<double>::max());
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

  oid_t source_id;
  typename FRAG_T::template vertex_array_t<double>& partial_result;

  DenseVertexSet<vid_t> curr_modified, next_modified;

#ifdef PROFILING
  double preprocess_time = 0;
  double exec_time = 0;
  double postprocess_time = 0;
#endif

};

}


#endif  // LIBGRAPE_LITE_SPAIR_CONTEXT_H
