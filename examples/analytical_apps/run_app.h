/** Copyright 2020 Alibaba Group Holding Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef EXAMPLES_ANALYTICAL_APPS_RUN_APP_H_
#define EXAMPLES_ANALYTICAL_APPS_RUN_APP_H_

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <grape/fragment/immutable_edgecut_fragment.h>
#include <grape/fragment/loader.h>
#include <grape/grape.h>
#include <grape/util.h>

#include <gflags/gflags.h>
#include <gflags/gflags_declare.h>
#include <glog/logging.h>

#ifdef GRANULA
#include "thirdparty/atlarge-research-granula/granula.hpp"
#endif

#include "../parametric_simulation/Spair/Spair.h"
#include "bfs/bfs.h"
#include "bfs/bfs_auto.h"
#include "cdlp/cdlp.h"
#include "cdlp/cdlp_auto.h"
#include "examples/parametric_simulation/Spair/Spair.h"
#include "examples/parametric_simulation/Vpair/Vpair.h"
#include "examples/parametric_simulation/Apair/Apair.h"
#include "flags.h"
#include "lcc/lcc.h"
#include "lcc/lcc_auto.h"
#include "pagerank/pagerank.h"
#include "pagerank/pagerank_auto.h"
#include "pagerank/pagerank_local.h"
#include "pagerank/pagerank_local_parallel.h"
#include "pagerank/pagerank_parallel.h"
#include "sssp/sssp.h"
#include "sssp/sssp_auto.h"
#include "timer.h"
#include "wcc/wcc.h"
#include "wcc/wcc_auto.h"

using namespace std;

namespace grape {

void Init() {
  if (FLAGS_out_prefix.empty()) {
    LOG(FATAL) << "Please assign an output prefix.";
  }
  if (FLAGS_deserialize && FLAGS_serialization_prefix.empty()) {
    LOG(FATAL) << "Please assign a serialization prefix.";
  } else if (FLAGS_vfile.empty() || FLAGS_efile.empty()) {
    LOG(FATAL) << "Please assign input vertex/edge files.";
  }

  if (access(FLAGS_out_prefix.c_str(), 0) != 0) {
    mkdir(FLAGS_out_prefix.c_str(), 0777);
  }

  InitMPIComm();
  CommSpec comm_spec;
  comm_spec.Init(MPI_COMM_WORLD);
  if (comm_spec.worker_id() == kCoordinatorRank) {
    VLOG(1) << "Workers of libgrape-lite initialized.";
  }
}

void Finalize() {
  FinalizeMPIComm();
  VLOG(1) << "Workers finalized.";
}

template <typename FRAG_T, typename APP_T, typename... Args>
void CreateAndQuery(const CommSpec& comm_spec, const string efile,
                    const string& vfile, const string& out_prefix,
                    int fnum, const ParallelEngineSpec& spec, Args... args) {
  timer_next("load graph");
  LoadGraphSpec graph_spec = DefaultLoadGraphSpec();
  graph_spec.set_directed(FLAGS_directed);
  graph_spec.set_rebalance(FLAGS_rebalance, FLAGS_rebalance_vertex_factor);
  if (FLAGS_deserialize) {
    graph_spec.set_deserialize(true, FLAGS_serialization_prefix);
  } else if (FLAGS_serialize) {
    graph_spec.set_serialize(true, FLAGS_serialization_prefix);
  }
  shared_ptr<FRAG_T> fragment;
  if (FLAGS_segmented_partition) {
    fragment = LoadGraph<FRAG_T, SegmentedPartitioner<typename FRAG_T::oid_t>>(
        efile, vfile, comm_spec, graph_spec);
  } else {
    fragment = LoadGraph<FRAG_T, HashPartitioner<typename FRAG_T::oid_t>>(
        efile, vfile, comm_spec, graph_spec);
  }
  auto app = make_shared<APP_T>();
  timer_next("load application");
  auto worker = APP_T::CreateWorker(app, fragment);
  worker->Init(comm_spec, spec);
  timer_next("run algorithm");
  worker->Query(forward<Args>(args)...);
  timer_next("print output");

  ofstream ostream;
  string output_path =
      grape::GetResultFilename(out_prefix, fragment->fid());
  ostream.open(output_path);
  worker->Output(ostream);
  ostream.close();
  worker->Finalize();
  timer_end();
  VLOG(1) << "Worker-" << comm_spec.worker_id() << " finished: " << output_path;
}

template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T>
void Run() {
  CommSpec comm_spec;
  comm_spec.Init(MPI_COMM_WORLD);

  bool is_coordinator = comm_spec.worker_id() == kCoordinatorRank;
  timer_start(is_coordinator);
#ifdef GRANULA
  string job_id = FLAGS_jobid;
  granula::startMonitorProcess(getpid());
  granula::operation grapeJob("grape", "Id.Unique", "Job", "Id.Unique");
  granula::operation loadGraph("grape", "Id.Unique", "LoadGraph", "Id.Unique");
  if (comm_spec.worker_id() == kCoordinatorRank) {
    cout << grapeJob.getOperationInfo("StartTime", grapeJob.getEpoch())
              << endl;
    cout << loadGraph.getOperationInfo("StartTime", loadGraph.getEpoch())
              << endl;
  }

  granula::linkNode(job_id);
  granula::linkProcess(getpid(), job_id);
#endif

#ifdef GRANULA
  if (comm_spec.worker_id() == kCoordinatorRank) {
    cout << loadGraph.getOperationInfo("EndTime", loadGraph.getEpoch())
              << endl;
  }
#endif
  // FIXME: no barrier apps. more manager? or use a dynamic-cast.
  string efile = FLAGS_efile;
  string vfile = FLAGS_vfile;
  string path_we = FLAGS_path_we;
  string gd_evfile = FLAGS_gd_evfile;
  string g_pathfile = FLAGS_g_pathfile;
  string gd_pathfile = FLAGS_gd_pathfile;

  string out_prefix = FLAGS_out_prefix;
  auto spec = DefaultParallelEngineSpec();
  if (FLAGS_app_concurrency != -1) {
    spec.thread_num = FLAGS_app_concurrency;
  } else {
    spec = MultiProcessSpec(comm_spec, false);
  }
  int fnum = comm_spec.fnum();
  string name = FLAGS_application;
  if (name.find("sssp") != string::npos) {
    using GraphType = ImmutableEdgecutFragment<OID_T, VID_T, VDATA_T, double>;
    if (name == "sssp_auto") {
      using AppType = SSSPAuto<GraphType>;
      CreateAndQuery<GraphType, AppType, OID_T>(
          comm_spec, efile, vfile, out_prefix, fnum, spec, FLAGS_sssp_source);
    } else if (name == "sssp") {
      using AppType = SSSP<GraphType>;
      CreateAndQuery<GraphType, AppType, OID_T>(
          comm_spec, efile, vfile, out_prefix, fnum, spec, FLAGS_sssp_source);
    } else {
      LOG(FATAL) << "No avaiable application named [" << name << "].";
    }
  } else {
    if (name == "bfs_auto") {
      using GraphType = ImmutableEdgecutFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                                 LoadStrategy::kOnlyOut>;
      using AppType = BFSAuto<GraphType>;
      CreateAndQuery<GraphType, AppType, OID_T>(
          comm_spec, efile, vfile, out_prefix, fnum, spec, FLAGS_bfs_source);
    } else if (name == "bfs") {
      using GraphType = ImmutableEdgecutFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                                 LoadStrategy::kOnlyOut>;
      using AppType = BFS<GraphType>;
      CreateAndQuery<GraphType, AppType, OID_T>(
          comm_spec, efile, vfile, out_prefix, fnum, spec, FLAGS_bfs_source);
    } else if (name == "pagerank_local") {
      using GraphType = ImmutableEdgecutFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                                 LoadStrategy::kOnlyOut>;
      using AppType = PageRankLocal<GraphType>;
      CreateAndQuery<GraphType, AppType, double, int>(comm_spec, efile, vfile,
                                                      out_prefix, fnum, spec,
                                                      FLAGS_pr_d, FLAGS_pr_mr);
    } else if (name == "pagerank_local_parallel") {
      using GraphType = ImmutableEdgecutFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                                 LoadStrategy::kBothOutIn>;
      using AppType = PageRankLocalParallel<GraphType>;
      CreateAndQuery<GraphType, AppType, double, int>(comm_spec, efile, vfile,
                                                      out_prefix, fnum, spec,
                                                      FLAGS_pr_d, FLAGS_pr_mr);
    } else if (name == "pagerank_auto") {
      using GraphType = ImmutableEdgecutFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                                 LoadStrategy::kBothOutIn>;
      using AppType = PageRankAuto<GraphType>;
      CreateAndQuery<GraphType, AppType, double, int>(comm_spec, efile, vfile,
                                                      out_prefix, fnum, spec,
                                                      FLAGS_pr_d, FLAGS_pr_mr);
    } else if (name == "pagerank") {
      using GraphType = ImmutableEdgecutFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                                 LoadStrategy::kOnlyOut>;
      using AppType = PageRank<GraphType>;
      CreateAndQuery<GraphType, AppType, double, int>(comm_spec, efile, vfile,
                                                      out_prefix, fnum, spec,
                                                      FLAGS_pr_d, FLAGS_pr_mr);
    } else if (name == "pagerank_parallel") {
      using GraphType = ImmutableEdgecutFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                                 LoadStrategy::kBothOutIn>;
      using AppType = PageRankParallel<GraphType>;
      CreateAndQuery<GraphType, AppType, double, int>(comm_spec, efile, vfile,
                                                      out_prefix, fnum, spec,
                                                      FLAGS_pr_d, FLAGS_pr_mr);
    } else if (name == "cdlp_auto") {
      using GraphType = ImmutableEdgecutFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                                 LoadStrategy::kBothOutIn>;
      using AppType = CDLPAuto<GraphType>;
      CreateAndQuery<GraphType, AppType, int>(
          comm_spec, efile, vfile, out_prefix, fnum, spec, FLAGS_cdlp_mr);
    } else if (name == "cdlp") {
      using GraphType = ImmutableEdgecutFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                                 LoadStrategy::kOnlyOut>;
      using AppType = CDLP<GraphType>;
      CreateAndQuery<GraphType, AppType, int>(
          comm_spec, efile, vfile, out_prefix, fnum, spec, FLAGS_cdlp_mr);
    } else if (name == "wcc_auto") {
      using GraphType = ImmutableEdgecutFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                                 LoadStrategy::kOnlyOut>;
      using AppType = WCCAuto<GraphType>;
      CreateAndQuery<GraphType, AppType>(comm_spec, efile, vfile, out_prefix,
                                         fnum, spec);
    } else if (name == "wcc") {
      using GraphType = ImmutableEdgecutFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                                 LoadStrategy::kOnlyOut>;
      using AppType = WCC<GraphType>;
      CreateAndQuery<GraphType, AppType>(comm_spec, efile, vfile, out_prefix,
                                         fnum, spec);
    } else if (name == "lcc_auto") {
      using GraphType = ImmutableEdgecutFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                                 LoadStrategy::kOnlyOut>;
      using AppType = LCCAuto<GraphType>;
      CreateAndQuery<GraphType, AppType>(comm_spec, efile, vfile, out_prefix,
                                         fnum, spec);
    } else if (name == "lcc") {
      using GraphType = ImmutableEdgecutFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                                 LoadStrategy::kOnlyOut>;
      using AppType = LCC<GraphType>;
      CreateAndQuery<GraphType, AppType>(comm_spec, efile, vfile, out_prefix,
                                         fnum, spec);
    } else if(name == "Spair"){
      cout << "create" << endl;
      using GraphType = ImmutableEdgecutFragment<unsigned int, unsigned int, string, string,
          LoadStrategy::kOnlyOut>;
      using AppType = Spair<GraphType>;
      CreateAndQuery<GraphType, AppType, string>(
          comm_spec, efile, vfile, out_prefix, fnum, spec, path_we,gd_evfile,
          g_pathfile,gd_pathfile,FLAGS_vertex_u,FLAGS_vertex_v);

    }else if(name == "Vpair"){
      using GraphType = ImmutableEdgecutFragment<OID_T, VID_T, VDATA_T, EDATA_T,
          LoadStrategy::kOnlyOut>;
      using AppType = Vpair<GraphType>;
      CreateAndQuery<GraphType, AppType, string>(
          comm_spec, efile, vfile, out_prefix, fnum, spec, path_we,gd_evfile,
          g_pathfile,gd_pathfile,FLAGS_vertex_u,FLAGS_vertex_v);

    }else if(name == "Apair"){
      using GraphType = ImmutableEdgecutFragment<OID_T, VID_T, VDATA_T, EDATA_T,
          LoadStrategy::kOnlyOut>;
      using AppType = Apair<GraphType>;
      CreateAndQuery<GraphType, AppType, string>(
          comm_spec, efile, vfile, out_prefix, fnum, spec, path_we,gd_evfile,
          g_pathfile,gd_pathfile,FLAGS_vertex_u,FLAGS_vertex_v);

    }else {
      LOG(FATAL) << "No avaiable application named [" << name << "].";
    }
  }
#ifdef GRANULA
  granula::operation offloadGraph("grape", "Id.Unique", "OffloadGraph",
                                  "Id.Unique");
#endif

#ifdef GRANULA
  if (comm_spec.worker_id() == kCoordinatorRank) {
    cout << offloadGraph.getOperationInfo("StartTime",
                                               offloadGraph.getEpoch())
              << endl;

    cout << grapeJob.getOperationInfo("EndTime", grapeJob.getEpoch())
              << endl;
  }

  granula::stopMonitorProcess(getpid());
#endif
}

}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_RUN_APP_H_
